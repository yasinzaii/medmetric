"""MedicalNet feature extractor wrapper.

This module provides a small PyTorch `nn.Module` that:
  1) builds a MedicalNet-compatible 3D ResNet backbone (see `medmetric.models.resnet`)
  2) loads pretrained weights resolved from the YAML manifest (HF Hub cache)
  3) returns pooled feature embeddings suitable for metrics like FID/MMD.

The manifest lives at: `medmetric/weights/medicalnet.yaml`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn

from medmetric.hub.medicalnet import MedicalNetWeightEntry, get_medicalnet_entry, resolve_medicalnet_weights
from medmetric.models.resnet import medicalnet_resnet


def _get_by_dotted_path(obj: object, path: str) -> object:
    """Traverse dicts using a dotted key path like 'state_dict' or 'model.state_dict'."""
    cur = obj
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(path)
        cur = cur[key]
    return cur


def _load_state_dict(
    weights_path: str,
    *,
    state_dict_key: Optional[str] = None,
    strip_prefix: Sequence[str] = (),
) -> dict:
    """Load a checkpoint file and return a clean state_dict."""

    try:
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(weights_path, map_location="cpu")

    sd = None
    if state_dict_key is not None:
        try:
            obj = _get_by_dotted_path(ckpt, state_dict_key)
        except Exception as e:
            raise TypeError(f"The provided state_dict_key '{state_dict_key}' is bad (path not found).") from e

        if not isinstance(obj, dict):
            raise TypeError(f"The provided state_dict_key '{state_dict_key}' is bad (not a dict).")

        sd = obj

    if sd is None:
        # common fallbacks
        if isinstance(ckpt, dict):
            for k in ("state_dict", "model_state_dict", "model", "net"):
                if k in ckpt and isinstance(ckpt[k], dict):
                    sd = ckpt[k]
                    break
            if sd is None:
                raise TypeError(f"Unsupported checkpoint format at {weights_path!r}: {type(ckpt)}")

    # Strip DataParallel / wrapper prefixes
    if strip_prefix is None:
        strip_prefix = ()
    for p in strip_prefix:
        if not p:
            continue
        sd = {k[len(p):] if k.startswith(p) else k: v for k, v in sd.items()}

    return sd


class MedicalNetFeatureExtractor(nn.Module):
    """
    MedicalNet (Med3D) feature extractor for 3D medical volumes.

    This module wraps a MedicalNet-compatible 3D ResNet backbone and produces feature
    embeddings suitable for distribution-level evaluation metrics such as FID and MMD
    on 3D data (e.g., real vs synthetic MRI volumes).

    This class does **not** perform intensity normalization. For reliable evaluation,
    apply the same preprocessing pipeline to both real and synthetic volumes before
    feature extraction (e.g. per-volume z-score normalization).

    Parameters
    ----------
    depth:
        ResNet depth used to choose the architecture/checkpoint family (e.g., 10, 18,
        34, 50, 101, 152). This value is stored on the instance and is used as the
        default when calling :meth:`load_pretrained`.
    pooled:
        If True, the forward pass returns pooled feature vectors of shape (B, F).
        If False, returns the final convolutional feature map of shape (B, C, D', H', W').
    return_map:
        Only relevant when `pooled=True`. If True, returns a tuple `(feats, fmap)` where
        `feats` is (B, F) and `fmap` is the final feature map (B, C, D', H', W').
    shortcut_type:
        Residual shortcut variant expected by the MedicalNet backbone (commonly "B").
        Must match the checkpoint used.
    layers:
        Optional explicit stage configuration. When using :meth:`from_pretrained`, this
        is typically populated from the pinned manifest.
    block:
        Block type string expected by the MedicalNet builder (e.g., "BasicBlock" or
        "Bottleneck"). When using :meth:`from_pretrained`, this is typically populated
        from the pinned manifest.
    no_cuda:
        Compatibility flag used by some MedicalNet code paths for shortcut type "A".
        You generally do not need to set this manually.

    Notes
    -----
    - Input should be a 5D tensor shaped **(B, 1, D, H, W)**.
    - The extractor expects **1 input channel** (MRI volumes are typically single-channel).
    - For evaluation, call `eval()` (this is done automatically by :meth:`from_pretrained`
      and :meth:`load_pretrained` after loading weights).

    Forward Call Parameters
    -----------------------
    x:
        A tensor of shape (B, 1, D, H, W). Recommended dtype is `torch.float32`.

    Returns
    -------
    torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
        If `pooled=True` and `return_map=False`:
            `feats` with shape (B, F)
        If `pooled=True` and `return_map=True`:
            `(feats, fmap)` where `feats` is (B, F) and `fmap` is (B, C, D', H', W')
        If `pooled=False`:
            `fmap` with shape (B, C, D', H', W')

    Examples
    --------
    Load a pretrained extractor (recommended):
        >>> extractor = MedicalNetFeatureExtractor.from_pretrained(depth=50)
        >>> x = torch.randn(2, 1, 96, 96, 96)
        >>> feats = extractor(x)  # (2, F)

    In-place loading (keeps the same instance):
        >>> extractor = MedicalNetFeatureExtractor(depth=50)
        >>> extractor.load_pretrained()
        >>> feats = extractor(x)

    Return pooled features and feature map:
        >>> extractor = MedicalNetFeatureExtractor.from_pretrained(depth=50, return_map=True)
        >>> feats, fmap = extractor(x)

    Return feature map only:
        >>> extractor = MedicalNetFeatureExtractor.from_pretrained(depth=50, pooled=False)
        >>> fmap = extractor(x)
    """


    def __init__(
        self,
        *,
        depth: int = 50,
        pooled: bool = True,
        return_map: bool = False,
        shortcut_type: str = "B",
        layers: Optional[Sequence[int]] = None,
        block: Optional[str] = None,
        no_cuda: bool = False,
    ) -> None:
        super().__init__()
        self.model = medicalnet_resnet(
            layers=layers,
            block=block,
            pooled=pooled,
            return_map=return_map,
            shortcut_type=shortcut_type,
            no_cuda=no_cuda,
        )
        self.pooled = pooled
        self.return_map = return_map
        self.depth = depth

    @classmethod
    def from_pretrained(
        cls,
        *,
        depth: int = 50,
        use_23dataset: bool = True,
        pooled: bool = True,
        return_map: bool = False,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        # Rarely needed overrides (kept for flexibility)
        state_dict_key: Optional[str] = None,
        strip_prefix: Optional[Sequence[str]] = None,
        token: Optional[str] = None,
        cache_dir: Optional[str | Path] = None,
    ) -> "MedicalNetFeatureExtractor":
        """Create extractor and load pretrained weights.

        Normally you only need `depth` and `use_23dataset` — everything else is taken
        from the YAML manifest (repo_id/filename/shortcut_type/checkpoint format).
        """
        entry: MedicalNetWeightEntry = get_medicalnet_entry(depth=depth, use_23dataset=use_23dataset)

        # Build model using manifest shortcut_type (and optional self-describing arch info)
        no_cuda = not torch.cuda.is_available()
        extractor = cls(
            depth=entry.depth,
            pooled=pooled,
            return_map=return_map,
            shortcut_type=entry.shortcut_type,
            layers=list(entry.layers) if entry.layers is not None else None,
            block=entry.block,
            no_cuda=no_cuda,
        )

        extractor.load_pretrained(
            depth=entry.depth,
            use_23dataset=entry.use_23dataset,
            weights_path=weights_path,
            device=device,
            state_dict_key=state_dict_key,
            strip_prefix=strip_prefix,
            token=token,
            cache_dir=cache_dir,
        )
        return extractor

    def load_pretrained(
        self,
        *,
        depth: Optional[int] = None,
        use_23dataset: bool = True,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        state_dict_key: Optional[str] = None,
        strip_prefix: Optional[Sequence[str]] = None,
        token: Optional[str] = None,
        cache_dir: Optional[str | Path] = None,
    ) -> "MedicalNetFeatureExtractor":
        """Load pretrained MedicalNet weights into this extractor (in-place).

        This is the in-place counterpart to :meth:`from_pretrained`. It assumes the
        current instance was constructed with an architecture compatible with the
        selected checkpoint (depth/shortcut_type/layers/block). If not, you will get
        missing keys and the method will raise.

        Parameters
        ----------
        depth:
            Checkpoint depth to load. If None, uses `self.depth`.
        use_23dataset:
            Whether to use the 23-dataset MedicalNet checkpoint variant.
        weights_path:
            Optional local checkpoint file. If None, resolves from the YAML manifest.
        device:
            Optional device to move the module to after loading (e.g. "cpu", "cuda:0").
        state_dict_key, strip_prefix:
            Rare overrides for checkpoint formats. If not provided, defaults come from
            the YAML manifest.
        token, cache_dir:
            Hugging Face download options when resolving weights.

        Returns
        -------
        MedicalNetFeatureExtractor
            Returns `self` for chaining.

        Examples
        --------
        >>> extractor = MedicalNetFeatureExtractor(depth=50)
        >>> extractor.load_pretrained()
        >>> feats = extractor(x)

        Preferred one-liner:
        >>> extractor = MedicalNetFeatureExtractor.from_pretrained(depth=50)
        >>> feats = extractor(x)
        """
        d = self.depth if depth is None else int(depth)
        entry: MedicalNetWeightEntry = get_medicalnet_entry(depth=d, use_23dataset=use_23dataset)

        # Resolve weights (HF hub unless local path provided)
        if weights_path is None:
            weights_path = resolve_medicalnet_weights(
                depth=entry.depth,
                use_23dataset=entry.use_23dataset,
                token=token,
                cache_dir=cache_dir,
            )

        sd = _load_state_dict(
            weights_path,
            state_dict_key=state_dict_key or entry.state_dict_key,
            strip_prefix=tuple(strip_prefix) if strip_prefix is not None else (entry.strip_prefix or ()),
        )

        # Load into the correct module (pooled wrapper stores backbone under .backbone)
        target = self.model.backbone if hasattr(self.model, "backbone") else self.model
        missing, unexpected = target.load_state_dict(sd, strict=False)

        # Be strict about missing keys (usually indicates wrong checkpoint / wrong shortcut_type)
        if missing:
            raise RuntimeError(
                f"Missing keys when loading MedicalNet weights (showing up to 15): {missing[:15]}"
            )
        # Unexpected keys are usually harmless (e.g., optimizer states, extra heads)
        _ = unexpected

        self.eval()
        if device is not None:
            self.to(device)
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        return self.model(x)