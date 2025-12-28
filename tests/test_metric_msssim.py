import pytest
import torch

pytest.importorskip("monai")

from medmetric.metrics.msssim import MS_SSIM

# MONAI MS-SSIM default settings (weights len=5, kernel_size=11) require
# spatial dims >= 176 to pass internal downsampling/kernel-size checks.
IMG = 176


def test_ms_ssim_identical_close_to_one():
    torch.manual_seed(0)
    # Use [0,1] range to match data_range=1.0
    y = torch.rand(4, 3, IMG, IMG)
    y_pred = y.clone()

    ms = MS_SSIM(spatial_dims=2, data_range=1.0, reduction="mean")
    v = ms(y_pred, y)

    assert v.ndim == 1
    assert v.shape == (1,)
    assert torch.isfinite(v)
    assert float(v) > 0.99


def test_ms_ssim_noisy_lower_than_identical():
    torch.manual_seed(0)
    y = torch.rand(4, 3, IMG, IMG)
    y_pred_same = y.clone()
    y_pred_noisy = torch.clamp(y + 0.10 * torch.randn_like(y), 0.0, 1.0)

    ms = MS_SSIM(spatial_dims=2, data_range=1.0, reduction="mean")
    v_same = ms(y_pred_same, y)
    v_noisy = ms(y_pred_noisy, y)

    assert float(v_noisy) < float(v_same)


def test_ms_ssim_shape_mismatch_raises():
    ms = MS_SSIM(spatial_dims=2, data_range=1.0)
    y = torch.rand(2, 1, IMG, IMG)
    y_pred = torch.rand(2, 1, IMG, IMG + 1)
    with pytest.raises(ValueError):
        ms(y_pred, y)


def test_ms_ssim_requires_float():
    ms = MS_SSIM(spatial_dims=2, data_range=1.0)
    y = torch.randint(0, 2, (2, 1, IMG, IMG), dtype=torch.int64)
    y_pred = y.clone()
    with pytest.raises(TypeError):
        ms(y_pred, y)


def test_ms_ssim_reduction_none_not_scalar():
    torch.manual_seed(0)
    y = torch.rand(3, 2, IMG, IMG)
    y_pred = y.clone()

    ms = MS_SSIM(spatial_dims=2, data_range=1.0, reduction="none")
    v = ms(y_pred, y)
    assert v.ndim >= 1
    assert v.shape == (3, 1)


def test_ms_ssim_get_not_nans_returns_tuple():
    torch.manual_seed(0)
    y = torch.rand(2, 1, IMG, IMG)
    y_pred = y.clone()

    ms = MS_SSIM(spatial_dims=2, data_range=1.0, get_not_nans=True)
    out = ms(y_pred, y)

    assert len(out) == 2
    metric, not_nans = out
    assert isinstance(metric, torch.Tensor)
    assert isinstance(not_nans, torch.Tensor)
