import pytest
import torch

monai = pytest.importorskip("monai")

from medmetric.metrics.fid import FID


def test_fid_identical_near_zero():
    torch.manual_seed(0)
    x = torch.randn(128, 64)
    y = x.clone()

    fid = FID(clamp_min=0.0)
    v = fid(x, y)

    assert v.ndim == 0
    assert torch.isfinite(v)
    assert float(v) < 1e-6


def test_fid_shift_larger_than_same():
    torch.manual_seed(0)
    x = torch.randn(256, 64)
    y_same = torch.randn(256, 64)
    y_shift = y_same + 1.0

    fid = FID(clamp_min=0.0)
    v_same = fid(x, y_same)
    v_shift = fid(x, y_shift)

    assert float(v_shift) > float(v_same)


def test_fid_feature_dim_mismatch_raises():
    fid = FID(clamp_min=0.0)
    x = torch.randn(32, 64)
    y = torch.randn(32, 63)
    with pytest.raises(ValueError):
        fid(x, y)


def test_fid_requires_float():
    fid = FID(clamp_min=0.0)
    x = torch.randint(0, 10, (32, 64), dtype=torch.int64)
    y = torch.randint(0, 10, (32, 64), dtype=torch.int64)
    with pytest.raises(TypeError):
        fid(x, y)

def test_fid_flattens_high_dim_inputs():
    torch.manual_seed(0)
    fid = FID(clamp_min=0.0)
    x = torch.randn(16, 4, 4)   # will flatten to (16, 16)
    y = torch.randn(20, 4, 4)   # will flatten to (20, 16)
    v = fid(x, y)
    assert v.ndim == 0
    assert torch.isfinite(v)
