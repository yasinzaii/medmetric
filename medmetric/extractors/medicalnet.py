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
    strip_prefix: Sequence[str] = None,
) -> dict:
    """Load a checkpoint file and return a clean state_dict."""
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)

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
    for p in strip_prefix:
        if not p:
            continue
        sd = {k[len(p):] if k.startswith(p) else k: v for k, v in sd.items()}

    return sd





class MedicalNetFeatureExtractor(nn.Module):
    """Feature extractor returning MedicalNet features.

    - If pooled=True: forward() returns (B, F) (or (feats, fmap) if return_map=True)
    - If pooled=False: forward() returns the final feature map (B, C, D', H', W')
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
            strip_prefix=tuple(strip_prefix) if strip_prefix is not None else entry.strip_prefix,
        )

        # Load into the correct module (pooled wrapper stores backbone under .backbone)
        target = extractor.model.backbone if hasattr(extractor.model, "backbone") else extractor.model
        missing, unexpected = target.load_state_dict(sd, strict=False)

        # Be strict about missing keys (usually indicates wrong checkpoint / wrong shortcut_type)
        if missing:
            raise RuntimeError(
                f"Missing keys when loading MedicalNet weights (showing up to 15): {missing[:15]}"
            )
        # Unexpected keys are usually harmless (e.g., optimizer states, extra heads)
        _ = unexpected

        extractor.eval()
        if device is not None:
            extractor.to(device)
        return extractor

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        return self.model(x)
