from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pytest
import torch
import yaml

import medmetric
from medmetric.models.resnet import medicalnet_resnet
from medmetric.extractors.medicalnet import MedicalNetFeatureExtractor


def _manifest_path() -> Path:
    return Path(medmetric.__file__).resolve().parent / "weights" / "medicalnet.yaml"


def _cache_directory() -> Path:
    return Path(__file__).resolve().parent.parent / ".cache"  # == ../.cache


def _load_manifest_items() -> List[Dict[str, Any]]:
    path = _manifest_path()
    if not path.exists():
        raise FileNotFoundError(f"Could not find MedicalNet manifest at: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid manifest format: expected mapping at root: {path}")
    items = data.get("medicalnet")
    if not isinstance(items, list):
        raise RuntimeError(f"Invalid manifest format: expected 'medicalnet: [...]' list in {path}")
    # Ensure every item is a dict
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            raise RuntimeError(f"Invalid manifest item #{i}: expected mapping, got {type(it)}")
    return items


def _combos_from_manifest(items: List[Dict[str, Any]]) -> List[Tuple[int, bool]]:
    combos: List[Tuple[int, bool]] = []
    for it in items:
        depth = int(it["depth"])
        use_23dataset = bool(it.get("use_23dataset", True))
        combos.append((depth, use_23dataset))
    # Keep deterministic ordering for stable test output
    combos = sorted(set(combos))
    return combos


def _item_lookup(items: List[Dict[str, Any]]) -> Dict[Tuple[int, bool], Dict[str, Any]]:
    out: Dict[Tuple[int, bool], Dict[str, Any]] = {}
    for it in items:
        k = (int(it["depth"]), bool(it.get("use_23dataset", True)))
        out[k] = it
    return out


def _expected_channels(block_name: str) -> int:
    # BasicBlock -> 512, Bottleneck -> 2048
    b = (block_name or "").strip().lower()
    if b.startswith("bottleneck"):
        return 2048
    return 512


def _make_input(device: torch.device, output_stride: int) -> torch.Tensor:
    # Choose input so that output fmap spatial dims are 2x2x2
    # input_size = output_stride * 2
    s = int(output_stride) * 2
    x = torch.randn(2, 1, s, s, s, device=device)
    return x


_ITEMS = _load_manifest_items()
_COMBOS = _combos_from_manifest(_ITEMS)
_LOOKUP = _item_lookup(_ITEMS)


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_all_manifest_models_forward_shapes(device: str) -> None:
    # Only run CUDA leg if available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch_device = torch.device(device)

    assert len(_ITEMS) > 0, "medicalnet.yaml has no entries"
    assert len(_COMBOS) > 0, "No (depth, use_23dataset) combos found in medicalnet.yaml"

     
    for depth, use_23dataset in _COMBOS:
        item = _LOOKUP[(depth, use_23dataset)]

        layers = item.get("layers")
        block = item.get("block")
        shortcut_type = item.get("shortcut_type", "B")

        assert isinstance(layers, list) and len(layers) == 4, f"Bad layers for {(depth, use_23dataset)}: {layers!r}"
        assert isinstance(block, str) and block, f"Bad block for {(depth, use_23dataset)}: {block!r}"

        x = _make_input(torch_device, 8)
        expected_c = _expected_channels(block)

        # Pooled model with return_map=True -> (feats, fmap)
        pooled_model = medicalnet_resnet(
            layers=layers,
            block=block,
            pooled=True,
            return_map=True,
            shortcut_type=shortcut_type,
            no_cuda=(device == "cpu"),
        ).to(torch_device).eval()

        with torch.inference_mode():
            feats, fmap = pooled_model(x)

        assert feats.shape == (2, expected_c), f"feats shape mismatch for {(depth, use_23dataset)}: {feats.shape}"
        assert fmap.shape[0] == 2 and fmap.shape[1] == expected_c, f"fmap C mismatch for {(depth, use_23dataset)}: {fmap.shape}"
        assert fmap.shape[2:] == (2, 2, 2), f"fmap spatial mismatch for {(depth, use_23dataset)}: {fmap.shape}"

        # Non-pooled model -> fmap only
        map_model = medicalnet_resnet(
            layers=layers,
            block=block,
            pooled=False,
            return_map=False,
            shortcut_type=shortcut_type,
            no_cuda=(device == "cpu"),
        ).to(torch_device).eval()

        with torch.inference_mode():
            fmap2 = map_model(x)

        assert fmap2.shape == fmap.shape, f"unpooled fmap mismatch for {(depth, use_23dataset)}: {fmap2.shape} vs {fmap.shape}"



@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_all_manifest_models_load_weights_and_forward(device: str) -> None:
    # Gate this test: it downloads HF checkpoints.
    if os.environ.get("MEDMETRIC_RUN_HF_TESTS", "").strip() != "1":
        pytest.skip("Set MEDMETRIC_RUN_HF_TESTS=1 to run HF weight download/load test.")

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch_device = torch.device(device)

    for depth, use_23dataset in _COMBOS:
        item = _LOOKUP[(depth, use_23dataset)]
        x = _make_input(torch_device, 8)

        expected_c = _expected_channels(item.get("block"))

        extractor = MedicalNetFeatureExtractor.from_pretrained(
            depth=depth,
            use_23dataset=use_23dataset,
            device=device,
            cache_dir=_cache_directory(), 
        )
        extractor.eval()

        with torch.inference_mode():
            y = extractor(x)

        assert isinstance(y, torch.Tensor), f"Extractor returned non-tensor for {(depth, use_23dataset)}: {type(y)}"
        assert y.shape == (2, expected_c), f"feats shape mismatch for {(depth, use_23dataset)}: {feats.shape}"
        assert y.shape[0] == 2
        assert y.ndim == 2

