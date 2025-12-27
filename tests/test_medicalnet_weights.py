from __future__ import annotations

import os
import re

from pathlib import Path
from huggingface_hub import hf_hub_url, get_hf_file_metadata

import pytest

from medmetric.hub.medicalnet import (
    get_medicalnet_entry,
    load_medicalnet_manifest,
    resolve_medicalnet_weights,
)


def test_manifest_loads_and_has_expected_keys() -> None:
    manifest = load_medicalnet_manifest()

    # Ensure required combos exist in YAML
    for depth in (18, 50):
        assert (depth, False) in manifest
        assert (depth, True) in manifest

    # Example of a depth that should NOT have 23dataset in your YAML
    assert (101, False) in manifest
    assert (101, True) not in manifest


def test_missing_combo_raises_helpfully() -> None:
    # ResNet-101 has only base in our YAML
    with pytest.raises(ValueError):
        get_medicalnet_entry(depth=101, use_23dataset=True)


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def test_hub_etag_matches_yaml_sha256() -> None:
    # Integration test (network): Run explicitly:
    #   MEDMETRIC_RUN_HF_TESTS=1 pytest -m integration -q

    if os.getenv("MEDMETRIC_RUN_HF_TESTS") != "1":
        pytest.skip("Set MEDMETRIC_RUN_HF_TESTS=1 to run Hugging Face integration tests.")

    manifest = load_medicalnet_manifest()

    # Only check the ones you asked for
    targets = [
        (18, False),
        (18, True),
        (50, False),
        (50, True),
    ]

    for (depth, use_23dataset) in manifest:
        
        
        entry = manifest[(depth, use_23dataset)]

        url = hf_hub_url(
            repo_id=entry.repo_id,
            filename=entry.filename,
            revision=entry.revision,   # revision is in YAML
            repo_type="model",
        )

        meta = get_hf_file_metadata(url)
        etag = (meta.etag or "").strip('"').strip()
        
        print(etag)

        # STRICT: if sha not present, raise (no fallback)
        if not _SHA256_RE.fullmatch(etag):
            raise RuntimeError(
                f"Expected meta.etag to be SHA256 (64 hex), got {etag!r} for {entry.repo_id}/{entry.filename}"
            )

        assert etag.lower() == entry.sha256.lower(), (
            f"YAML sha256 mismatch for {entry.repo_id}/{entry.filename}\n"
            f"  yaml: {entry.sha256}\n"
            f"  hub : {etag}"
        )


def test_download_and_verify_resnet18_and_50_variants() -> None:
    # Download-heavy test. Run explicitly:
    #   MEDMETRIC_RUN_HF_TESTS=1 pytest
    if os.getenv("MEDMETRIC_RUN_HF_TESTS") != "1":
        pytest.skip("Set MEDMETRIC_RUN_HF_TESTS=1 to run Hugging Face download tests.")
        
    cache_dir = Path(__file__).resolve().parent.parent / ".cache"  # == ../.cache
    cache_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        (18, False),
        (18, True),
        (50, False),
        (50, True),
    ]

    for depth, use_23dataset in targets:
        path = resolve_medicalnet_weights(depth=depth, use_23dataset=use_23dataset, cache_dir=cache_dir)
        assert Path(path).exists(), f"Downloaded file missing: {path}"
        assert Path(path).stat().st_size > 0, f"Downloaded file is empty: {path}"