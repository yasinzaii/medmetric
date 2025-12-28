from __future__ import annotations

from typing import Optional, Sequence, Union
import torch


def flatten_2d(z: torch.Tensor) -> torch.Tensor:
    return z if z.ndim == 2 else z.flatten(1)


def ensure_same_device_dtype(x: torch.Tensor, y: torch.Tensor) -> None:
    if x.device != y.device:
        raise ValueError(f"x and y must be on the same device (got {x.device} vs {y.device})")
    if x.dtype != y.dtype:
        raise ValueError(f"x and y must have the same dtype (got {x.dtype} vs {y.dtype})")


def as_1d_tensor(
    v: Union[torch.Tensor, float, Sequence[float]],
    *,
    ref: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    name: str = "value",
    allow_empty: bool = False,
) -> torch.Tensor:
    """Convert v to a 1D tensor on ref.device with ref.dtype.
    
    You must provide either:
      - ref=<tensor>  (uses ref.device and ref.dtype), OR
      - device=<...>, dtype=<...>

    - Scalars -> shape (1,)
    - Higher-dim -> flattened to 1D
    """
    # Resolve device/dtype
    if ref is not None:
        if device is None:
            device = ref.device
        if dtype is None:
            dtype = ref.dtype
    
    if device is None or dtype is None:
        raise TypeError(f"{name}: provide either ref=... or both device=... and dtype=...")
    
    if isinstance(v, torch.Tensor):
        t = v.to(device=device, dtype=dtype)
    else:
        t = torch.as_tensor(v, device=device, dtype=dtype)

    if t.ndim == 0:
        t = t[None]
    elif t.ndim != 1:
        t = t.flatten()

    if (not allow_empty) and t.numel() == 0:
        raise ValueError(f"{name} must have at least one element")
    return t