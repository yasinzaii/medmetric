"""
End-to-end demo: 3D volume evaluation with MedicalNet features + FID/MMD + MS-SSIM.

What this script demonstrates
-----------------------------
1) Create synthetic "real" and "fake" 3D volumes (random tensors) with configurable size.
2) Z-score normalize volumes (important for MedicalNet-style inputs).
3) Extract deep features using MedicalNetFeatureExtractor.
4) Compute:
   - FID(real_feats, fake_feats)  (distribution similarity)
   - MMD(real_feats, fake_feats)  (distribution similarity)
   - MS-SSIM(fake-fake pairs)     (structural diversity of generated samples)
5) Repeat the pipeline using a minimal DataLoader + batching:
   - accumulate features across batches
   - compute final FID/MMD at the end
   - estimate fake-fake MS-SSIM over sampled pairs (K = min(target, N*(N-1)/2)).

Notes
-----
- This demo uses random data. Replace the random tensors with your real MRI volumes
  and generated MRI volumes.
- MedicalNet expects 5D tensors: (B, 1, D, H, W).
- Intensity preprocessing: In many MedicalNet pipelines, volumes are z-score normalized
  per-volume (often after percentile clipping). Here we demonstrate z-score only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


# Feature extractor
from medmetric.extractors import MedicalNetFeatureExtractor

# Metrics (features -> scalar)
from medmetric.metrics import FID
from medmetric.metrics import MMD

# Diversity (images -> scalar)
from medmetric.metrics import MS_SSIM


# ----------------------------
# Utilities
# ----------------------------

def zscore_per_volume(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Z-score normalize each volume in a batch independently.

    Parameters
    ----------
    x : torch.Tensor
        5D tensor (B, 1, D, H, W) or (B, C, D, H, W).
    eps : float
        Numerical epsilon to avoid divide-by-zero.

    Returns
    -------
    torch.Tensor
        Z-scored tensor with same shape as input.
    """
    # Compute mean/std over spatial dims (B, 1, *spatial-dims).
    # Here we keep channels separate; for MRI it's usually C=1 anyway.
    dims = (2, 3, 4)
    mean = x.mean(dim=dims, keepdim=True)
    std = x.std(dim=dims, keepdim=True, unbiased=False)
    return (x - mean) / (std + eps)


def make_random_volumes(
    *,
    n: int,
    image_size: Tuple[int, int, int],
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """
    Create random 3D volumes (B, 1, D, H, W).
    Note: we use torch.rand (0..1) so MS-SSIM data_range=1.0 makes sense in the demo.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    D, H, W = image_size
    x = torch.rand((n, 1, D, H, W), generator=g, dtype=torch.float32)
    return x.to(device)


# ----------------------------
# Scenario 1: "single-shot" evaluation
# ----------------------------

@torch.no_grad()
def evaluate_single_shot(
    *,
    real_images,
    fake_images,
    device: str = "cpu",
    kernel_size: int = 3,
) -> None:
    device_t = torch.device(device)

    # 1) Extracting Image size. (Not Important)
    image_size = real_images.shape[2:]

    # 2) IMPORTANT: z-score normalization (MedicalNet-style input) for feature extraction
    #    Keep MS-SSIM in the original image domain (e.g. [0,1]) with an appropriate data_range.
    real_z = zscore_per_volume(real_images)
    fake_z = zscore_per_volume(fake_images)

    # 3) Select feature extractor (MedicalNet)
    # Preferred usage (one-liner):
    extractor = MedicalNetFeatureExtractor.from_pretrained(depth=50, device=device)

    # Alternative usage (in-place load):
    # extractor = MedicalNetFeatureExtractor(depth=50).to(device_t)
    # extractor.load_pretrained(device=device)


    # 4) Extract features
    #    Output: (B, F) pooled embeddings by default (depending on your extractor config).
    real_feats = extractor(real_z)
    fake_feats = extractor(fake_z)

    # 5) Similarity metrics on FEATURES
    fid = FID()
    mmd = MMD()

    fid_score = fid(real_feats, fake_feats)
    mmd_score = mmd(real_feats, fake_feats)

    # 6) Diversity metric on IMAGES (fake only) via fake–fake MS-SSIM pairs
    n = fake_images.shape[0]
    k = min(5000, n * (n - 1) // 2)

    i, j = sample_random_pairs(n, k, device=device)
    ms_ssim = MS_SSIM(
        spatial_dims=fake_images.ndim - 2, 
        data_range=1.0,
        kernel_size=3 # default 11 [default require images with a dim > 180 approx.]
    )  # default reduction="mean"
    ms_mean = ms_ssim(fake_images[i], fake_images[j])  # <-- actually computes the mean over k pairs

    print("\n=== Single-shot evaluation ===")
    print(f"Image size (D,H,W): {image_size}")
    print(f"Real samples: {real_images.shape[0]}, Fake samples: {fake_images.shape[0]}")
    print(f"FID:      {float(fid_score):.6f}")
    print(f"MMD:      {float(mmd_score):.6f}")
    print(f"MS-SSIM:  {float(ms_mean):.6f}")


# ----------------------------
# Scenario 2: Batched evaluation (DataLoader) with feature accumulation - FID + MMD
# ----------------------------

class VolumePairDataset(Dataset):
    """
    Minimal dataset that serves aligned (real, fake) volume pairs.
    This is deliberately simple: it just stores tensors and returns them by index.
    """
    def __init__(self, real: torch.Tensor, fake: torch.Tensor):
        assert real.shape == fake.shape, "For this demo we assume same number/shape."
        self.real = real
        self.fake = fake

    def __len__(self) -> int:
        return self.real.shape[0]

    def __getitem__(self, idx: int):
        return self.real[idx], self.fake[idx]


@torch.no_grad()
def evaluate_batched(
    *,
    real_images,
    fake_images,
    batch_size: int = 4,
    device: str = "cpu",
) -> None:
    device_t = torch.device(device)
    image_size = real_images.shape[2:]

    # Minimal dataset/loader
    ds = VolumePairDataset(real_images, fake_images)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # Feature extractor
    extractor = MedicalNetFeatureExtractor.from_pretrained(depth=50, device=device)

    # Metrics
    fid = FID()
    mmd = MMD()

    # Accumulators for features
    all_real_feats = []
    all_fake_feats = []

    for (real_b, fake_b) in dl:
        real_b = real_b.to(device_t, non_blocking=True)
        fake_b = fake_b.to(device_t, non_blocking=True)

        # IMPORTANT: z-score normalization per-volume
        real_b = zscore_per_volume(real_b)
        fake_b = zscore_per_volume(fake_b)

        # Feature extraction
        real_f = extractor(real_b)  # (B, F)
        fake_f = extractor(fake_b)  # (B, F)

        all_real_feats.append(real_f.detach().cpu())
        all_fake_feats.append(fake_f.detach().cpu())

     # Stack all features (now we have full datasets)
    real_feats = torch.cat(all_real_feats, dim=0)
    fake_feats = torch.cat(all_fake_feats, dim=0)

    # Compute final similarity metrics on full feature sets
    fid_score = fid(real_feats, fake_feats)
    mmd_score = mmd(real_feats, fake_feats)

    print("\n=== Batched evaluation ===")
    print(f"Image size (D,H,W): {image_size}")
    print(f"Samples (real/fake): {real_images.shape[0]}")
    print(f"Batch size: {batch_size} (drop_last=False)")
    print(f"FID:      {float(fid_score):.6f}")
    print(f"MMD:      {float(mmd_score):.6f}")

    ms_ssim_score = estimate_ms_ssim_diversity(
        fake_images,
        device=device,
        max_pairs=100,  # Appox 5000 normally
        pair_batch_size=min(32, batch_size),
        data_range=1.0,
    )
    print(f"MS-SSIM:  {float(ms_ssim_score):.6f}")


# ----------------------------
# Scenario 2: Batched evaluation (DataLoader) - MS-SSIM
# ----------------------------

class FakePairDataset(Dataset):
    """Yields (fake[i], fake[j]) for precomputed index pairs."""
    def __init__(self, fake: torch.Tensor, i_idx: torch.Tensor, j_idx: torch.Tensor):
        self.fake = fake
        self.i_idx = i_idx
        self.j_idx = j_idx

    def __len__(self) -> int:
        return int(self.i_idx.numel())

    def __getitem__(self, idx: int):
        i = int(self.i_idx[idx])
        j = int(self.j_idx[idx])
        return self.fake[i], self.fake[j]


def sample_random_pairs(n: int, k: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample k ordered pairs (i, j) uniformly with j != i."""
    i = torch.randint(0, n, (k,), device=device)
    j = torch.randint(0, n - 1, (k,), device=device)
    j = j + (j >= i)  # ensures j != i
    return i, j


@torch.no_grad()
def estimate_ms_ssim_diversity(
    fake_images: torch.Tensor,
    *,
    device: str = "cpu",
    max_pairs: int = 100,  # Usually 2000-5000
    pair_batch_size: int = 16,
    data_range: float = 1.0,
    kernel_size: int = 3,
) -> float:
    device_t = torch.device(device)

    n = fake_images.shape[0]
    num_pairs = n * (n - 1) // 2
    k = min(max_pairs, num_pairs)

    # Sample index pairs for fake–fake MS-SSIM.
    i_idx, j_idx = sample_random_pairs(n, k, device_t)

    pair_ds = FakePairDataset(fake_images, i_idx, j_idx)
    pair_dl = DataLoader(pair_ds, batch_size=pair_batch_size, shuffle=False, drop_last=False)

    ms_ssim = MS_SSIM(
        spatial_dims=fake_images.ndim - 2,
        data_range=data_range,
        kernel_size=kernel_size,
        reduction="none",   # return per-pair values -> easy global average
    )

    total = 0.0
    count = 0

    for a, b in pair_dl:
        a = a.to(device_t, non_blocking=True)
        b = b.to(device_t, non_blocking=True)

        vals = ms_ssim(a, b)
        total += float(vals.sum())
        count += int(vals.numel())

    mean = float(total / max(count, 1))
    return mean


def main():
    image_size = (64, 64, 64)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    n = 16

    real_images = make_random_volumes(n=n, image_size=image_size, device=device_t, seed=999)
    fake_images = make_random_volumes(n=n, image_size=image_size, device=device_t, seed=1000)

    evaluate_single_shot(
        real_images=real_images,
        fake_images=fake_images,
        device=device,
    )

    # Scenario 2: batched - FID + MMD + MS-SSIM
    evaluate_batched(
        real_images=real_images,
        fake_images=fake_images,
        batch_size=16,
        device=device,
    )


if __name__ == "__main__":
    main()
