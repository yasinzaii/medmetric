from __future__ import annotations

from typing import Sequence, Union, Optional
import torch

from ..utils.tensors import ensure_same_device_dtype

try:
    from monai.metrics import MultiScaleSSIMMetric
except Exception as e:
    MultiScaleSSIMMetric = None
    _MONAI_IMPORT_ERR = e
else:
    _MONAI_IMPORT_ERR = None


class MS_SSIM:
    """
    Multi-Scale Structural Similarity (MS-SSIM) for image tensors (callable metric).

    This class is a thin wrapper around MONAI's `monai.metrics.MultiScaleSSIMMetric`.
    It expects **images/volumes**, not feature vectors.

    MS-SSIM is a perceptual similarity measure computed across multiple scales/resolutions.
    It is commonly used to compare a prediction (`y_pred`) against a reference (`y`).

    NOTE: MS-SSIM is also used to measure diversity among generated images by computing
    pairwise similarity between generated samples. In that setting, **lower MS-SSIM**
    typically indicates **higher diversity** (sometimes reported as 1 - MS-SSIM).

    The instance is **callable**, so you can use it like a function:

        ms_ssim = MS_SSIM(spatial_dims=2, data_range=1.0)
        score = ms_ssim(y_pred, y)

    Parameters
    ----------
    spatial_dims:
        Number of spatial dimensions of the input (e.g., 2 for (H, W), 3 for (D, H, W)).
        Inputs are expected to be shaped like (B, C, *spatial).
    data_range:
        Value range of input images (usually 1.0 if normalized to [0, 1], or 255 if [0,255]).
        Inputs must be floating point tensors.
    kernel_type:
        Kernel type used in SSIM computation. Typically "gaussian" or "uniform". Forwarded to MONAI.
    kernel_size:
        Kernel size (int or per-dim sequence). Forwarded to MONAI.
    kernel_sigma:
        Standard deviation for Gaussian kernel (float or per-dim sequence). Forwarded to MONAI.
    k1, k2:
        Stability constants used in SSIM denominators. Forwarded to MONAI.
    weights:
        Per-scale weights for MS-SSIM (forwarded to MONAI).
    reduction:
        Reduction mode applied by MONAI. Common choices:
        "mean" (default), "sum", "none", "mean_batch", "sum_batch", "mean_channel", "sum_channel".
    get_not_nans:
        If True, MONAI returns (metric, not_nans). If False, returns metric only.

    Call Parameters
    ---------------
    y_pred:
        Predicted image tensor. Shape (B, C, *spatial). Must be floating-point.
    y:
        Reference/target image tensor. Shape (B, C, *spatial).
        Must be floating-point and match `y_pred` shape.

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, torch.Tensor]
        If get_not_nans=False: returns MS-SSIM reduced according to `reduction`.
        If get_not_nans=True: returns (metric, not_nans), as MONAI does.

    Examples
    --------
    2D images (B, C, H, W), normalized to [0,1]:
        >>> ms_ssim = MS_SSIM(spatial_dims=2, data_range=1.0)
        >>> score = ms_ssim(y_pred, y)

    3D volumes (B, C, D, H, W):
        >>> ms_ssim = MS_SSIM(spatial_dims=3, data_range=1.0)
        >>> score = ms_ssim(y_pred, y)

    No reduction (returns unreduced values, typically per-sample and/or per-channel):
        >>> ms_ssim = MS_SSIM(spatial_dims=2, reduction="none")
        >>> score = ms_ssim(y_pred, y)
    """

    def __init__(
        self,
        *,
        spatial_dims: int,
        data_range: float = 1.0,
        kernel_type: str = "gaussian",
        kernel_size: Union[int, Sequence[int]] = 11,
        kernel_sigma: Union[float, Sequence[float]] = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        weights: Sequence[float] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
        reduction: str = "mean",
        get_not_nans: bool = False,
    ) -> None:
        if MultiScaleSSIMMetric is None:  # pragma: no cover
            raise ImportError(
                "MONAI is required for MS-SSIM. Install with: pip install 'medmetric[monai]'"
            ) from _MONAI_IMPORT_ERR

        self.spatial_dims = int(spatial_dims)
        if self.spatial_dims < 1:
            raise ValueError(f"spatial_dims must be >= 1 (got {self.spatial_dims}).")

        self.data_range = float(data_range)
        if self.data_range <= 0:
            raise ValueError(f"data_range must be > 0 (got {self.data_range}).")

        self.get_not_nans = bool(get_not_nans)

        self._metric = MultiScaleSSIMMetric(
            spatial_dims=self.spatial_dims,
            data_range=self.data_range,
            kernel_type=kernel_type,
            kernel_size=kernel_size,
            kernel_sigma=kernel_sigma,
            k1=k1,
            k2=k2,
            weights=weights,
            reduction=reduction,
            get_not_nans=self.get_not_nans,
        )

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor):
        ensure_same_device_dtype(y_pred, y)

        if not y_pred.is_floating_point() or not y.is_floating_point():
            raise TypeError(f"MS-SSIM expects floating point tensors (got {y_pred.dtype} and {y.dtype}).")

        if y_pred.shape != y.shape:
            raise ValueError(f"y_pred and y must have the same shape (got {tuple(y_pred.shape)} vs {tuple(y.shape)}).")

        expected_min_ndim = 2 + self.spatial_dims
        if y_pred.ndim < expected_min_ndim:
            raise ValueError(
                f"Inputs must have shape (B, C, *spatial) with spatial_dims={self.spatial_dims}. "
                f"Expected ndim >= {expected_min_ndim}, got ndim={y_pred.ndim}."
            )

        
        self._metric.reset()  # No accumulation across calls
        _ = self._metric(y_pred, y)  # updates internal buffers
        return self._metric.aggregate()  # applies reduction
    
    def compute(self, y_pred: torch.Tensor, y: torch.Tensor):
        """Alias for __call__."""
        return self(y_pred, y)
