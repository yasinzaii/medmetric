from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pytest
import torch
import yaml

import medmetric
from medmetric.models.resnet import medicalnet_resnet
from medmetric.extractors import MedicalNetFeatureExtractor
from medmetric.hub.medicalnet import get_medicalnet_entry


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


def _save_dummy_checkpoint_for_entry(
    tmp_path: Path,
    *,
    depth: int,
    use_23dataset: bool,
    state_dict_key: str = "state_dict",
    prefix: str = "module.",
) -> Path:
    """Create a local checkpoint matching the manifest entry architecture.

    This allows testing `from_pretrained(..., weights_path=...)` with zero HF/network.
    """
    entry = get_medicalnet_entry(depth=depth, use_23dataset=use_23dataset)

    # Match how from_pretrained builds the model
    no_cuda = not torch.cuda.is_available()
    m = MedicalNetFeatureExtractor(
        depth=entry.depth,
        pooled=True,
        return_map=False,
        shortcut_type=entry.shortcut_type,
        layers=list(entry.layers) if entry.layers is not None else None,
        block=entry.block,
        no_cuda=no_cuda,
    ).eval()

    target = m.model.backbone if hasattr(m.model, "backbone") else m.model
    sd = target.state_dict()
    if prefix:
        sd = {f"{prefix}{k}": v for k, v in sd.items()}

    ckpt: Dict[str, Any] = {state_dict_key: sd}
    out_path = tmp_path / f"dummy_medicalnet_{depth}_{int(use_23dataset)}.pth"
    torch.save(ckpt, out_path)
    return out_path


def test_from_pretrained_invalid_depth_raises() -> None:
    with pytest.raises(ValueError):
        MedicalNetFeatureExtractor.from_pretrained(
            depth=9999, use_23dataset=True, weights_path="/tmp/does_not_matter.pth"
        )


def test_from_pretrained_local_weights_path_loads_without_hf(tmp_path: Path) -> None:
    # Use an entry that has explicit state_dict_key/strip_prefix in the manifest.
    depth, use_23dataset = 10, False
    weights_path = _save_dummy_checkpoint_for_entry(
        tmp_path,
        depth=depth,
        use_23dataset=use_23dataset,
        state_dict_key="state_dict",
        prefix="module.",
    )

    extractor = MedicalNetFeatureExtractor.from_pretrained(
        depth=depth,
        use_23dataset=use_23dataset,
        weights_path=str(weights_path),
        device="cpu",
    )

    assert not extractor.training
    assert next(extractor.parameters()).device.type == "cpu"

    x = _make_input(torch.device("cpu"), 8)
    with torch.inference_mode():
        y = extractor(x)

    assert isinstance(y, torch.Tensor)
    assert y.ndim == 2
    assert y.shape == (2, 512)


def test_from_pretrained_state_dict_key_override_loads(tmp_path: Path) -> None:
    depth, use_23dataset = 10, False
    entry = get_medicalnet_entry(depth=depth, use_23dataset=use_23dataset)

    no_cuda = not torch.cuda.is_available()
    m = MedicalNetFeatureExtractor(
        depth=entry.depth,
        pooled=True,
        return_map=False,
        shortcut_type=entry.shortcut_type,
        layers=list(entry.layers) if entry.layers is not None else None,
        block=entry.block,
        no_cuda=no_cuda,
    ).eval()
    target = m.model.backbone if hasattr(m.model, "backbone") else m.model
    sd = {f"module.{k}": v for k, v in target.state_dict().items()}

    weights_path = tmp_path / "nested_ckpt.pth"
    torch.save({"model": {"state_dict": sd}}, weights_path)

    extractor = MedicalNetFeatureExtractor.from_pretrained(
        depth=depth,
        use_23dataset=use_23dataset,
        weights_path=str(weights_path),
        state_dict_key="model.state_dict",
        device="cpu",
    )

    x = _make_input(torch.device("cpu"), 8)
    with torch.inference_mode():
        y = extractor(x)
    assert y.shape == (2, 512)


def test_from_pretrained_strip_prefix_override_loads(tmp_path: Path) -> None:
    depth, use_23dataset = 10, False
    weights_path = _save_dummy_checkpoint_for_entry(
        tmp_path,
        depth=depth,
        use_23dataset=use_23dataset,
        state_dict_key="state_dict",
        prefix="wrap.",
    )

    extractor = MedicalNetFeatureExtractor.from_pretrained(
        depth=depth,
        use_23dataset=use_23dataset,
        weights_path=str(weights_path),
        strip_prefix=("wrap.",),
        device="cpu",
    )

    x = _make_input(torch.device("cpu"), 8)
    with torch.inference_mode():
        y = extractor(x)
    assert y.shape == (2, 512)


def test_from_pretrained_return_map_true_returns_tuple(tmp_path: Path) -> None:
    depth, use_23dataset = 10, False
    weights_path = _save_dummy_checkpoint_for_entry(
        tmp_path,
        depth=depth,
        use_23dataset=use_23dataset,
        state_dict_key="state_dict",
        prefix="module.",
    )

    extractor = MedicalNetFeatureExtractor.from_pretrained(
        depth=depth,
        use_23dataset=use_23dataset,
        weights_path=str(weights_path),
        return_map=True,
        device="cpu",
    )

    x = _make_input(torch.device("cpu"), 8)
    with torch.inference_mode():
        out = extractor(x)

    assert isinstance(out, tuple)
    feats, fmap = out
    assert feats.shape == (2, 512)
    assert fmap.shape == (2, 512, 2, 2, 2)


def test_from_pretrained_pooled_false_returns_map(tmp_path: Path) -> None:
    depth, use_23dataset = 10, False
    weights_path = _save_dummy_checkpoint_for_entry(
        tmp_path,
        depth=depth,
        use_23dataset=use_23dataset,
        state_dict_key="state_dict",
        prefix="module.",
    )

    extractor = MedicalNetFeatureExtractor.from_pretrained(
        depth=depth,
        use_23dataset=use_23dataset,
        weights_path=str(weights_path),
        pooled=False,
        device="cpu",
    )

    x = _make_input(torch.device("cpu"), 8)
    with torch.inference_mode():
        fmap = extractor(x)

    assert isinstance(fmap, torch.Tensor)
    assert fmap.shape == (2, 512, 2, 2, 2)


def test_from_pretrained_bad_state_dict_key_raises(tmp_path: Path) -> None:
    depth, use_23dataset = 10, False
    weights_path = tmp_path / "bad_key_ckpt.pth"
    torch.save({"something": {}}, weights_path)

    with pytest.raises(TypeError):
        MedicalNetFeatureExtractor.from_pretrained(
            depth=depth,
            use_23dataset=use_23dataset,
            weights_path=str(weights_path),
            state_dict_key="model.state_dict",
        )


def test_from_pretrained_uses_resolver_when_weights_path_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure from_pretrained calls the resolver and forwards cache_dir/token args.

    Offline: monkeypatch resolver to return a local weights path.
    """
    depth, use_23dataset = 10, False
    local_weights = _save_dummy_checkpoint_for_entry(
        tmp_path,
        depth=depth,
        use_23dataset=use_23dataset,
        state_dict_key="state_dict",
        prefix="module.",
    )

    seen: Dict[str, Any] = {}

def _fake_resolve_medicalnet_weights(*, depth: int, use_23dataset: bool, token=None, cache_dir=None) -> str:
    seen.update({"depth": depth, "use_23dataset": use_23dataset, "token": token, "cache_dir": cache_dir})
    return str(local_weights)

    monkeypatch.setattr(medicalnet_extractor_mod, "resolve_medicalnet_weights", _fake_resolve_medicalnet_weights)

    extractor = MedicalNetFeatureExtractor.from_pretrained(
        depth=depth,
        use_23dataset=use_23dataset,
        token="dummy_token",
        cache_dir=tmp_path / "hf_cache",
        device="cpu",
    )

    assert seen["depth"] == depth
    assert seen["use_23dataset"] == use_23dataset
    assert seen["token"] == "dummy_token"
    assert str(seen["cache_dir"]).endswith("hf_cache")

    x = _make_input(torch.device("cpu"), 8)
    with torch.inference_mode():
        y = extractor(x)
    assert y.shape == (2, 512)
    
    
def test_load_pretrained_local_weights_path_returns_self_and_runs(tmp_path: Path) -> None:
    depth, use_23dataset = 10, False
    entry = get_medicalnet_entry(depth=depth, use_23dataset=use_23dataset)

    # Build an extractor matching the manifest entry (no weights yet)
    extractor = MedicalNetFeatureExtractor(
        depth=entry.depth,
        pooled=True,
        return_map=False,
        shortcut_type=entry.shortcut_type,
        layers=list(entry.layers) if entry.layers is not None else None,
        block=entry.block,
        no_cuda=True,
    ).eval()

    # Create a local checkpoint with the SAME arch state_dict but prefixed with "module."
    # (manifest for depth=10,use_23dataset=False expects strip_prefix="module." and state_dict_key="state_dict")
    target = extractor.model.backbone if hasattr(extractor.model, "backbone") else extractor.model
    sd = {f"module.{k}": v for k, v in target.state_dict().items()}
    ckpt_path = tmp_path / "dummy_resnet10.pth"
    torch.save({"state_dict": sd}, ckpt_path)

    # Act: load_pretrained should load from local path and return self
    out = extractor.load_pretrained(
        depth=depth,
        use_23dataset=use_23dataset,
        weights_path=str(ckpt_path),
        device="cpu",
    )
    
    extractor2 = MedicalNetFeatureExtractor.from_pretrained(
        depth=depth,
        use_23dataset=use_23dataset,
        weights_path=str(ckpt_path),
        device="cpu",
    )
    
    # Forward equality check
    x = _make_input(torch.device("cpu"), 8)
    with torch.inference_mode():
        y1 = out(x)
        y2 = extractor2(x)

    assert isinstance(y1, torch.Tensor)
    assert isinstance(y2, torch.Tensor)
    assert y1.shape == y2.shape == (2, 512)
    assert torch.allclose(y1, y2, atol=1e-6, rtol=0.0)

