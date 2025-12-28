from __future__ import annotations

from typing import Optional, Sequence, Union
import torch

from ..utils.tensors import flatten_2d, ensure_same_device_dtype, as_1d_tensor

DEFAULT_SIGMA_BANK_RATIOS: tuple[float, ...] = (1/8, 1/4, 1/2, 1.0, 2.0, 4.0, 8.0)


@torch.no_grad()
def sigma_median_heuristic(
    z: torch.Tensor,
    *,
    max_points: Optional[int] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Median heuristic for sigma (median of pairwise Euclidean distances).

    Args:
        z: (N, D) feature matrix, typically concat([x, y]).
        max_points: optional subsample count for speed.
        eps: lower bound for sigma.
    """
    z = flatten_2d(z)
    if not z.is_floating_point():
        raise TypeError("sigma_median_heuristic expects floating point tensors.")
    
    N = int(z.shape[0])
    if N < 2:
        return torch.tensor(1.0, device=z.device, dtype=z.dtype)

    if max_points is not None and N > int(max_points):
        idx = torch.randperm(N, device=z.device)[: int(max_points)]
        z = z[idx]
        N = int(z.shape[0])

    d = torch.cdist(z, z, p=2)  # (N,N) Euclidean distances
    iu = torch.triu_indices(N, N, offset=1, device=z.device)
    vals = d[iu[0], iu[1]]
    sigma = vals.median()
    return torch.clamp(sigma, min=eps)


def _prepare_sigmas_and_weights(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    sigmas: Optional[Union[torch.Tensor, float, Sequence[float]]],
    weights: Optional[Union[torch.Tensor, float, Sequence[float]]],
    ratios: Optional[Sequence[float]],
    median_max_points: Optional[int],
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve final sigmas (1D) and normalized weights (1D)."""
    ensure_same_device_dtype(x, y)
    device, dtype = x.device, x.dtype

    # Resolve sigmas
    if sigmas is None:
        sigma0 = sigma_median_heuristic(torch.cat([x, y], dim=0), max_points=median_max_points, eps=eps)
        if ratios is None or len(ratios) == 0:
            sigmas_t = sigma0[None]  # single sigma
        else:
            ratios_t = as_1d_tensor(ratios, device=device, dtype=dtype, name="ratios")
            sigmas_t = sigma0 * ratios_t
    else:
        sigmas_t = as_1d_tensor(sigmas, device=device, dtype=dtype, name="sigmas")

    # Clamp sigmas to avoid division by zero / numerical blowups
    sigmas_t = torch.clamp(sigmas_t, min=eps)

    # Resolve weights
    if weights is None:
        weights_t = torch.ones_like(sigmas_t) / float(sigmas_t.numel())
    else:
        weights_t = as_1d_tensor(weights, device=device, dtype=dtype, name="weights")
        if weights_t.numel() != sigmas_t.numel():
            raise ValueError(
                f"weights must match sigmas length (got weights={weights_t.numel()} vs sigmas={sigmas_t.numel()})"
            )
        if (weights_t < 0).any():
            raise ValueError("weights must be nonnegative")
        s = weights_t.sum()
        if float(s) <= 0.0:
            raise ValueError("weights must sum to a positive value")
        weights_t = weights_t / s

    return sigmas_t, weights_t


def rbf_kernel_matrix(
    x: torch.Tensor,
    y: torch.Tensor,
    sigmas: Union[torch.Tensor, float, Sequence[float]],
    *,
    weights: Optional[Union[torch.Tensor, float, Sequence[float]]] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute a (possibly multi-sigma) RBF kernel matrix.

    Args:
        x: (m, d)
        y: (n, d)
        sigmas: scalar or (L,) bandwidth(s)
        weights: optional (L,) weights that sum to 1; if None, uniform.
    """
    x = flatten_2d(x)
    y = flatten_2d(y)
    ensure_same_device_dtype(x, y)

    sigmas_t = as_1d_tensor(sigmas, device=x.device, dtype=x.dtype, name="sigmas")
    sigmas_t = torch.clamp(sigmas_t, min=eps)

    if weights is None:
        w_t = torch.ones_like(sigmas_t) / float(sigmas_t.numel())
    else:
        w_t = as_1d_tensor(weights, device=x.device, dtype=x.dtype, name="weights")

    d2 = torch.cdist(x, y, p=2) ** 2  # (m,n)

    # Vectorized multi-sigma mixture:
    # exp(-d2 / (2*sigma^2)) for each sigma, then weighted sum over sigmas.
    denom = 2.0 * (sigmas_t ** 2)              # (L,)
    k_all = torch.exp(-d2[None, :, :] / denom[:, None, None])  # (L,m,n)
    k = (w_t[:, None, None] * k_all).sum(dim=0)                # (m,n)
    return k


def _mmd2_from_grams(
    Kxx: torch.Tensor,
    Kyy: torch.Tensor,
    Kxy: torch.Tensor,
    *,
    unbiased: bool,
) -> torch.Tensor:
    m = int(Kxx.shape[0])
    n = int(Kyy.shape[0])

    if unbiased:
        if m < 2 or n < 2:
            raise ValueError("Need at least 2 samples in each set for unbiased MMD^2")
        sum_xx = (Kxx.sum() - Kxx.diag().sum()) / (m * (m - 1))
        sum_yy = (Kyy.sum() - Kyy.diag().sum()) / (n * (n - 1))
        sum_xy = Kxy.sum() / (m * n)
    else:
        if m < 1 or n < 1:
            raise ValueError("Need at least 1 sample in each set for biased MMD^2")
        sum_xx = Kxx.sum() / (m * m)
        sum_yy = Kyy.sum() / (n * n)
        sum_xy = Kxy.sum() / (m * n)

    return sum_xx + sum_yy - 2.0 * sum_xy


def mmd2_rbf(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    sigmas: Optional[Union[torch.Tensor, float, Sequence[float]]] = None,
    weights: Optional[Union[torch.Tensor, float, Sequence[float]]] = None,
    biased: bool = False,
    ratios: Optional[Sequence[float]] = DEFAULT_SIGMA_BANK_RATIOS,
    median_max_points: Optional[int] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """MMD^2 with (possibly multi-sigma) RBF kernel.

    - If sigmas is provided: uses it (ratios ignored).
    - If sigmas is None: sigma0 = median_heuristic([x;y]).
        - If ratios is None or empty: use single sigma0.
        - Else: use sigma bank sigma0 * ratios.

    Returns a scalar tensor. Unbiased estimate can be negative.
    """
    x = flatten_2d(x)
    y = flatten_2d(y)

    sigmas_t, weights_t = _prepare_sigmas_and_weights(
        x, y,
        sigmas=sigmas,
        weights=weights,
        ratios=None if sigmas is not None else ratios,  # ignore ratios if user passed sigmas
        median_max_points=median_max_points,
        eps=eps,
    )

    Kxx = rbf_kernel_matrix(x, x, sigmas_t, weights=weights_t, eps=eps)
    Kyy = rbf_kernel_matrix(y, y, sigmas_t, weights=weights_t, eps=eps)
    Kxy = rbf_kernel_matrix(x, y, sigmas_t, weights=weights_t, eps=eps)

    return _mmd2_from_grams(Kxx, Kyy, Kxy, unbiased=not biased)


class MMD:
    """Maximum Mean Discrepancy (MMD) computed on feature vectors (callable metric).

    This class measures how different two distributions are, using the **Gaussian (RBF) kernel**
    on feature vectors. It is intended to be used on features of shape (N, F), e.g. pooled
    activations from a pretrained network.

    The instance is **callable**, so you can use it like a function:

        mmd = MMD()
        score = mmd(fake_feats, real_feats)                 # returns MMD
        score2 = mmd(fake_feats, real_feats, squared=True)  # returns MMD^2

    Kernel
    ------
    Gaussian / RBF kernel:
        k(x, y) = exp( -||x-y||^2 / (2*sigma^2) )

    Multi-sigma mixture (sigma bank):
        k_mix(x, y) = sum_l w_l * exp( -||x-y||^2 / (2*sigma_l^2) ),
        where weights w_l sum to 1.

    Estimators
    ----------
    Unbiased MMD^2 (U-statistic):
        MMD^2_unb(X,Y) =
            1/(m(m-1)) * sum_{i != j} k(x_i, x_j)
          + 1/(n(n-1)) * sum_{i != j} k(y_i, y_j)
          - 2/(mn)      * sum_{i, j}  k(x_i, y_j)

    Biased MMD^2 (V-statistic):
        Same as above but includes diagonal terms in the within-set sums.

    Notes
    -----
    - The two inputs are symmetric; the names `fake_feats` and `real_feats` are used only for
      consistency with FID.
    - For the unbiased estimator, MMD^2 can be slightly negative due to finite-sample variance.
      If `clamp_min` is not None (default 0.0), MMD^2 is clamped before taking sqrt (and also
      before returning MMD^2 via `squared=True`).

    Parameters
    ----------
    sigmas:
        Kernel bandwidth(s) (sigma) for the Gaussian/RBF kernel.
        - If provided: uses these directly (ratios ignored).
        - If None: sigma0 is computed with the median heuristic from torch.cat([fake_feats, real_feats], dim=0).
    weights:
        Optional mixture weights for multi-sigma kernels.
        If provided, must match len(sigmas) and be nonnegative; will be normalized.
        If omitted, uniform weights are used.
    biased:
        If False (default) use the unbiased estimator (requires m,n >= 2).
        If True use the biased estimator (requires m,n >= 1).
    ratios:
        Only used when sigmas is None:
        - Default uses a sigma bank: sigma0 * DEFAULT_SIGMA_BANK_RATIOS
        - Set ratios=None or ratios=() to disable the bank and use only sigma0
    median_max_points:
        Optional subsampling for the median heuristic for speed.
    eps:
        Numerical safety clamp for sigma values.
    kernel:
        Kernel name. Supported: "gaussian" (alias: "rbf").
    clamp_min:
        If not None (default 0.0), clamp MMD^2 to at least `clamp_min` before returning.

    Call Parameters
    ---------------
    fake_feats:
        Feature tensor for the first sample/distribution. Shape (m, F).
        If input has more than 2 dims, it is flattened to (m, -1).
        Must be floating-point.
    real_feats:
        Feature tensor for the second sample/distribution. Shape (n, F).
        If input has more than 2 dims, it is flattened to (n, -1).
        Must be floating-point.
    squared:
        If False (default), return MMD = sqrt(MMD^2). If True, return MMD^2.

    Returns
    -------
    torch.Tensor
        Scalar tensor containing MMD or MMD^2.

    Examples
    --------
    Default (median heuristic + sigma bank):
        >>> mmd = MMD()
        >>> score = mmd(fake_feats, real_feats)

    Disable sigma bank (single sigma0):
        >>> mmd = MMD(ratios=None)
        >>> score = mmd(fake_feats, real_feats)

    Provide your own sigma bank and weights:
        >>> mmd = MMD(sigmas=[0.5, 1.0, 2.0], weights=[0.2, 0.3, 0.5])
        >>> score = mmd(fake_feats, real_feats)

    Get MMD^2:
        >>> score2 = mmd(fake_feats, real_feats, squared=True)
    """

    def __init__(
        self,
        *,
        sigmas: Optional[Union[torch.Tensor, float, Sequence[float]]] = None,
        weights: Optional[Union[torch.Tensor, float, Sequence[float]]] = None,
        biased: bool = False,
        ratios: Optional[Sequence[float]] = DEFAULT_SIGMA_BANK_RATIOS,
        median_max_points: Optional[int] = None,
        eps: float = 1e-12,
        kernel: str = "gaussian",
        clamp_min: Optional[float] = 0.0,
    ):
        self.sigmas = sigmas
        self.weights = weights
        self.biased = biased
        self.ratios = ratios
        self.median_max_points = median_max_points
        self.eps = eps
        self.kernel = kernel
        self.clamp_min = clamp_min

    def mmd2(self, fake_feats: torch.Tensor, real_feats: torch.Tensor) -> torch.Tensor:
        k = str(self.kernel).lower()
        if k == "rbf":
            k = "gaussian"
        if k != "gaussian":
            raise ValueError("Only gaussian kernel is supported right now.")

        fake_feats = flatten_2d(fake_feats)
        real_feats = flatten_2d(real_feats)
        ensure_same_device_dtype(fake_feats, real_feats)

        if (not fake_feats.is_floating_point()) or (not real_feats.is_floating_point()):
            raise TypeError(
                f"MMD expects floating point tensors (got {fake_feats.dtype} and {real_feats.dtype})."
            )

        mmd2 = mmd2_rbf(
            fake_feats,
            real_feats,
            sigmas=self.sigmas,
            weights=self.weights,
            biased=self.biased,
            ratios=self.ratios,
            median_max_points=self.median_max_points,
            eps=self.eps,
        )

        if self.clamp_min is not None:
            mmd2 = torch.clamp(mmd2, min=float(self.clamp_min))

        return mmd2

    def __call__(self, fake_feats: torch.Tensor, real_feats: torch.Tensor, *, squared: bool = False) -> torch.Tensor:
        mmd2 = self.mmd2(fake_feats, real_feats)
        if squared:
            return mmd2
        return torch.sqrt(mmd2)

