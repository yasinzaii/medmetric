import inspect
import pytest
import torch

from medmetric.metrics.mmd import (
    sigma_median_heuristic,
    rbf_kernel_matrix,
    mmd2_rbf,
    MMD,
    DEFAULT_SIGMA_BANK_RATIOS,
)



def test_sigma_median_heuristic_positive():
    torch.manual_seed(0)
    z = torch.randn(64, 32)
    s = sigma_median_heuristic(z)
    assert s.ndim == 0
    assert float(s) > 0.0


def test_sigma_median_heuristic_requires_float():
    z = torch.randint(0, 10, (10, 4), dtype=torch.int64)
    with pytest.raises(TypeError):
        sigma_median_heuristic(z)


def test_rbf_kernel_matrix_shape_symmetry_and_diag_ones():
    torch.manual_seed(0)
    x = torch.randn(32, 16)
    sigma = torch.tensor(1.5)

    K = rbf_kernel_matrix(x, x, sigma)  # weights default path
    assert K.shape == (32, 32)

    # Symmetry when x == y
    assert torch.allclose(K, K.T, atol=1e-6, rtol=1e-6)

    # Diagonal should be exactly 1 (exp(0)=1); allow tiny numeric tolerance
    diag = torch.diag(K)
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-6, rtol=1e-6)


def test_mmd2_unbiased_requires_two_samples():
    torch.manual_seed(0)
    x = torch.randn(1, 8)
    y = torch.randn(2, 8)
    with pytest.raises(ValueError):
        mmd2_rbf(x, y, biased=False, sigmas=torch.tensor([1.0]))


def test_mmd2_biased_allows_one_sample():
    torch.manual_seed(0)
    x = torch.randn(1, 8)
    y = torch.randn(1, 8)
    val = mmd2_rbf(x, y, biased=True, sigmas=torch.tensor([1.0]))
    assert val.ndim == 0
    assert torch.isfinite(val)


def test_mmd_same_distribution_smaller_than_shift():
    torch.manual_seed(0)
    # Same distribution (independent draws)
    x = torch.randn(256, 64)
    y_same = torch.randn(256, 64)

    # Shifted distribution
    y_shift = y_same + 1.0

    mmd = MMD(sigmas=None)
    m_same = mmd(x, y_same)  # allow heuristic defaults
    m_shift = mmd(x, y_shift)

    assert float(m_shift) > float(m_same)


def test_ratios_default_bank_and_disable_bank():
    torch.manual_seed(0)
    x = torch.randn(64, 32)
    y = torch.randn(64, 32)

    # Default bank (ratios default)
    v_bank = mmd2_rbf(x, y, sigmas=None, ratios=DEFAULT_SIGMA_BANK_RATIOS)
    assert v_bank.ndim == 0
    assert torch.isfinite(v_bank)

    # Disable bank (single sigma0)
    v_single = mmd2_rbf(x, y, sigmas=None, ratios=None)
    assert v_single.ndim == 0
    assert torch.isfinite(v_single)

    # Also accept empty tuple/list as disable
    v_single2 = mmd2_rbf(x, y, sigmas=None, ratios=())
    assert v_single2.ndim == 0
    assert torch.isfinite(v_single2)


def test_custom_sigmas_and_weights():
    torch.manual_seed(0)
    x = torch.randn(64, 32)
    y = torch.randn(64, 32)

    sigmas = torch.tensor([0.5, 1.0, 2.0])
    weights = torch.tensor([0.2, 0.3, 0.5])

    v = mmd2_rbf(x, y, sigmas=sigmas, weights=weights)
    assert v.ndim == 0
    assert torch.isfinite(v)


def test_weights_length_mismatch_raises():
    torch.manual_seed(0)
    x = torch.randn(32, 16)
    y = torch.randn(32, 16)

    sigmas = torch.tensor([0.5, 1.0, 2.0])
    weights = torch.tensor([0.5, 0.5])  # mismatch

    with pytest.raises(ValueError):
        mmd2_rbf(x, y, sigmas=sigmas, weights=weights)


def test_negative_weights_raises():
    torch.manual_seed(0)
    x = torch.randn(32, 16)
    y = torch.randn(32, 16)

    sigmas = torch.tensor([0.5, 1.0, 2.0])
    weights = torch.tensor([0.5, -0.1, 0.6])

    with pytest.raises(ValueError):
        mmd2_rbf(x, y, sigmas=sigmas, weights=weights)


def test_kernel_selection_and_bad_kernel():
    torch.manual_seed(0)
    x = torch.randn(32, 16)
    y = torch.randn(32, 16)

    # gaussian should work
    mmd = MMD(kernel="gaussian")
    out = mmd(x, y)
    assert out.ndim == 0
    assert torch.isfinite(out)
    
    # unsupported kernel should raise
    with pytest.raises(ValueError):
        mmd = MMD(kernel="linear")
        mmd(x, y)


def test_squared_flag_behavior_if_supported():
    torch.manual_seed(0)
    x = torch.randn(64, 32)
    y = torch.randn(64, 32)

    mmd = MMD()
    m2 = mmd(x, y, squared=True)
    mmd = MMD()
    m = mmd(x, y, squared=False)

    assert m2.ndim == 0 and m.ndim == 0
    assert torch.isfinite(m2) and torch.isfinite(m)

    # non-squared should be non-negative
    assert float(m) >= 0.0
