from __future__ import annotations

from typing import Optional
import torch

from ..utils.tensors import ensure_same_device_dtype, flatten_2d

try:
    from monai.metrics import FIDMetric
except Exception as e:
    FIDMetric = None
    _MONAI_IMPORT_ERR = e
else:
    _MONAI_IMPORT_ERR = None


class FID:
    """
    Frechet Inception Distance (FID) computed on feature vectors (callable metric).

    This class is a thin wrapper around MONAI's `monai.metrics.FIDMetric`.
    It expects **feature vectors**, not images.

    FID fits a multivariate Gaussian to each set of features:
        - fake_feats ~ N(mu_f, Sigma_f)
        - real_feats ~ N(mu_r, Sigma_r)

    Then computes:
        FID = ||mu_f - mu_r||^2 + Tr(Sigma_f + Sigma_r - 2 * sqrtm(Sigma_f Sigma_r))

    The instance is **callable**, so you can use it like a function:

        fid = FID()
        score = fid(fake_feats, real_feats)

    Parameters
    ----------
    clamp_min:
        If not None (default 0.0), clamps the final value to at least `clamp_min`.
        This is useful because tiny negative values can occur due to numerical precision.

    Call Parameters
    ---------------
    fake_feats:
        Features from the generated/predicted distribution. Shape (N, F).
        If input has >2 dims it will be flattened to (N, -1) (same behavior as MMD).
        Must be floating-point.
    real_feats:
        Features from the real/reference distribution. Shape (M, F).
        If input has >2 dims it will be flattened to (M, -1).
        Must be floating-point.

    Returns
    -------
    torch.Tensor
        Scalar tensor containing the FID score.

    Examples
    --------
    Basic usage:
        >>> fid = FID()
        >>> score = fid(fake_feats, real_feats)

    Disable clamping:
        >>> fid = FID(clamp_min=None)
        >>> score = fid(fake_feats, real_feats)

    Features coming from a network output (e.g. N,C,H,W):
        >>> fake = model(fake_imgs)          # (N,C,H,W)
        >>> real = model(real_imgs)          # (M,C,H,W)
        >>> fid = FID()
        >>> score = fid(fake, real)          # internally flattened to (N, C*H*W) and (M, C*H*W)
    """

    def __init__(self, *, clamp_min: Optional[float] = 0.0):
        if FIDMetric is None:
            raise ImportError(
                "MONAI is required for FID. Install with: pip install 'medmetric[monai]'"
            ) from _MONAI_IMPORT_ERR

        self.clamp_min = clamp_min
        self._metric = FIDMetric()

    def __call__(self, fake_feats: torch.Tensor, real_feats: torch.Tensor) -> torch.Tensor:
        fake_feats = flatten_2d(fake_feats)
        real_feats = flatten_2d(real_feats)

        ensure_same_device_dtype(fake_feats, real_feats)

        if not fake_feats.is_floating_point() or not real_feats.is_floating_point():
            raise TypeError(f"FID expects floating point tensors (got {fake_feats.dtype} and {real_feats.dtype}).")

        if fake_feats.shape[0] < 2 or real_feats.shape[0] < 2:
            raise ValueError("FID requires at least 2 samples in each set (N>=2, M>=2).")
        
        if fake_feats.shape[1] != real_feats.shape[1]:
            raise ValueError(
                "fake_feats and real_feats must have the same feature dimension "
                f"(got {fake_feats.shape[1]} vs {real_feats.shape[1]})."
            )

        score = self._metric(fake_feats, real_feats)

        if self.clamp_min is not None:
            score = torch.clamp(score, min=float(self.clamp_min))

        return score

    def compute(self, fake_feats: torch.Tensor, real_feats: torch.Tensor) -> torch.Tensor:
        """Alias for __call__"""
        return self(fake_feats, real_feats)
