"""
MedicalNet weight resolver (Tencent Med3D).

- Reads a pinned manifest from: medmetric/weights/medicalnet.yaml
- Downloads via Hugging Face Hub caching (hf_hub_download)
- Verifies integrity by computing local SHA256 and comparing to manifest
"""

from __future__ import annotations

import hashlib
import re
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from huggingface_hub import hf_hub_download

import importlib.resources as resources


MANIFEST_FILENAME = "medicalnet.yaml"
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class MedicalNetWeightEntry:
    depth: int
    use_23dataset: bool
    repo_id: str
    filename: str
    revision: str
    sha256: str


def _normalize_sha256(s: str) -> str:
    s = s.strip().lower()
    if s.startswith("sha256:"):
        s = s.split("sha256:", 1)[1].strip()
    if not _SHA256_RE.fullmatch(s):
        raise ValueError(f"Not a valid sha256 hex digest: {s!r}")
    return s


def _sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_path() -> Path:
    return resources.files("medmetric.weights").joinpath(MANIFEST_FILENAME)


def load_medicalnet_manifest() -> Dict[Tuple[int, bool], MedicalNetWeightEntry]:
    """
    Load and validate the MedicalNet weights manifest.

    Returns:
        Dict keyed by (depth, use_23dataset) -> MedicalNetWeightEntry
    """
    path = _manifest_path()
    raw_text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw_text)

    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid manifest format: expected mapping at root: {path}")

    if data.get("schema_version") != 1:
        raise RuntimeError(
            f"Unsupported schema_version={data.get('schema_version')!r} in {path}; expected 1."
        )

    items = data.get("medicalnet")
    if not isinstance(items, list):
        raise RuntimeError(f"Invalid manifest: 'medicalnet' must be a list in {path}")

    out: Dict[Tuple[int, bool], MedicalNetWeightEntry] = {}
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise RuntimeError(f"Invalid entry at medicalnet[{i}]: expected mapping, got {type(item)}")

        try:
            depth = int(item["depth"])
            use_23dataset = bool(item["use_23dataset"])
            repo_id = str(item["repo_id"])
            filename = str(item["filename"])
            revision = str(item.get("revision", "main"))
            sha256 = _normalize_sha256(str(item["sha256"]))
        except KeyError as e:
            raise RuntimeError(f"Missing required field {e!s} in entry medicalnet[{i}]") from e

        key = (depth, use_23dataset)
        if key in out:
            raise RuntimeError(f"Duplicate manifest key {key} in {path}")

        out[key] = MedicalNetWeightEntry(
            depth=depth,
            use_23dataset=use_23dataset,
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            sha256=sha256,
        )

    return out


def get_medicalnet_entry(*, depth: int, use_23dataset: bool = True) -> MedicalNetWeightEntry:
    """
    Get a manifest entry for (depth, use_23dataset). Raises if not present.
    """
    manifest = load_medicalnet_manifest()
    key = (int(depth), bool(use_23dataset))
    if key in manifest:
        return manifest[key]

    # IF Missing
    depth_int = int(depth)
    available = sorted(k for k in manifest.keys() if k[0] == depth_int)
    if available:
        raise ValueError(
            f"No entry for depth={depth_int} with use_23dataset={use_23dataset}. "
            f"Available for this depth: {available}."
        )
    raise ValueError(
        f"No entries found for depth={depth_int}. Check medmetric/weights/{MANIFEST_FILENAME}."
    )


def resolve_medicalnet_weights(
    *,
    depth: int,
    use_23dataset: bool = True,
    token: Optional[str] = None,
    cache_dir: Optional[str | Path] = None,
) -> str:
    """
    Download (if needed) and return a local path to MedicalNet weights.
    ALWAYS verifies local SHA256 matches the manifest SHA256.

    Args:
        depth: ResNet depth (e.g. 18, 50, 101, ...)
        use_23dataset: use the 23-dataset weights (default True)
        token: HF token (optional)
        cache_dir: optional HF cache dir override

    Returns:
        Local file path as string.
    """
    entry = get_medicalnet_entry(depth=depth, use_23dataset=use_23dataset)

    local_path = hf_hub_download(
        repo_id=entry.repo_id,
        filename=entry.filename,
        revision=entry.revision,
        token=token,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )

    local_sha = _sha256_file(local_path)
    if local_sha.lower() != entry.sha256.lower():
        raise RuntimeError(
            f"SHA256 mismatch for {entry.repo_id}/{entry.filename} (revision={entry.revision}). "
            f"Expected {entry.sha256}, got {local_sha}. File: {local_path}"
        )

    return local_path
