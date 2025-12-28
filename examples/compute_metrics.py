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
   - MS-SSIM(fake_vols)           (structural diversity of generated samples)
5) Repeat the pipeline using a minimal DataLoader + batching:
   - accumulate features across batches
   - compute MS-SSIM per batch with reduction="none" to avoid tracking batch-size remainders
   - compute final FID/MMD at the end.

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
from medmetric.extractors.medicalnet import MedicalNetFeatureExtractor

# Metrics (features -> scalar)
from medmetric.metrics.fid import FID
from medmetric.metrics.mmd import MMD

# Diversity (images -> scalar)
from medmetric.metrics.ms_ssim import MS_SSIM


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
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    D, H, W = image_size
    x = torch.randn((n, 1, D, H, W), generator=g, dtype=torch.float32)
    return x.to(device)


# ----------------------------
# Scenario 1: "single-shot" evaluation
# ----------------------------

@torch.no_grad()
def evaluate_single_shot(
    *,
    real_images=real_images,
    fake_images=fake_images,
    device: str = "cpu",
) -> None:
    device_t = torch.device(device)

    # 1) Extracting Image size. (Not Important)
    image_size = real_images.shape[2:]
    
    # 2) IMPORTANT: z-score normalization (MedicalNet-style input)
    #    In real pipelines, this can be done in dataset transforms; here we do it inline for clarity.
    real_images = zscore_per_volume(real_images)
    fake_images = zscore_per_volume(fake_images)

    # 3) Select feature extractor (MedicalNet)
    # Preferred usage (one-liner):
    extractor = MedicalNetFeatureExtractor.from_pretrained(depth=50, device=device)

    # Alternative usage (in-place load):
    # extractor = MedicalNetFeatureExtractor(depth=50).to(device_t)
    # extractor.load_pretrained(device=device)

    # Already set to Eval() - [Not Important]
    extractor.eval()

    # 4) Extract features
    #    Output: (B, F) pooled embeddings by default (depending on your extractor config).
    real_feats = extractor(real_images)
    fake_feats = extractor(fake_images)

    # 5) Similarity metrics on FEATURES
    fid = FID()
    mmd = MMD()

    fid_score = fid(real_feats, fake_feats)
    mmd_score = mmd(real_feats, fake_feats)

    # 6) Diversity metric on IMAGES (fake only)
    # MS-SSIM is used as diversity/structure metric here; higher MS-SSIM usually means more similarity among samples,
    # so lower value mean more diversity.
    ms_ssim = MS_SSIM()
    ms_ssim_score = ms_ssim(fake_images)  # by default many implementations reduce to a scalar

    print("\n=== Single-shot evaluation ===")
    print(f"Image size (D,H,W): {image_size}")
    print(f"Real samples: {n_real}, Fake samples: {n_fake}")
    print(f"FID:      {float(fid_score):.6f}")
    print(f"MMD:      {float(mmd_score):.6f}")
    print(f"MS-SSIM:  {float(ms_ssim_score):.6f}")
    print("Tip: You can report diversity as (1 - MS-SSIM) if you prefer 'higher is more diverse'.")


# ----------------------------
# Scenario 2: Batched evaluation (DataLoader) with feature accumulation
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
    real_images=real_images,
    fake_images=fake_images,
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
    # extractor.eval() <- Not Required

    # Metrics
    fid = FID()
    mmd = MMD()
    ms_ssim = MS_SSIM(reduction="none")  # IMPORTANT for batching: avoid uneven batch reduction

    # Accumulators for features
    all_real_feats = []
    all_fake_feats = []

    # Accumulator for per-sample MS-SSIM values (so we can average later)
    all_ms_ssim_vals = []

    for (real_b, fake_b) in dl:
        # Ensure on device (DataLoader might return on CPU)
        real_b = real_b.to(device_t, non_blocking=True)
        fake_b = fake_b.to(device_t, non_blocking=True)

        # IMPORTANT: z-score normalization per-volume
        # In a production setup, do this in your Dataset transforms.
        real_b = zscore_per_volume(real_b)
        fake_b = zscore_per_volume(fake_b)

        # Feature extraction
        real_f = extractor(real_b)  # (B, F)
        fake_f = extractor(fake_b)  # (B, F)

        all_real_feats.append(real_f.detach().cpu())
        all_fake_feats.append(fake_f.detach().cpu())

        # Diversity on fake only within the batch
        # Using reduction="none" means you get per-sample (or per-pair) scores without worrying
        # about last batch being smaller. We'll aggregate by a simple mean over all returned values.
        ms_vals = ms_ssim(fake_b)          # shape depends on your MS-SSIM implementation
        all_ms_ssim_vals.append(ms_vals.detach().cpu())

    # Stack all features (now we have full datasets)
    real_feats = torch.cat(all_real_feats, dim=0)
    fake_feats = torch.cat(all_fake_feats, dim=0)

    # Compute final similarity metrics on full feature sets
    fid_score = fid(real_feats, fake_feats)
    mmd_score = mmd(real_feats, fake_feats)

    # Aggregate MS-SSIM across all batches
    ms_ssim_vals = torch.cat(all_ms_ssim_vals, dim=0)
    ms_ssim_score = ms_ssim_vals.mean()

    print("\n=== Batched evaluation ===")
    print(f"Image size (D,H,W): {image_size}")
    print(f"Samples (real/fake): {n}")
    print(f"Batch size: {batch_size} (drop_last=False)")
    print(f"FID:      {float(fid_score):.6f}")
    print(f"MMD:      {float(mmd_score):.6f}")
    print(f"MS-SSIM:  {float(ms_ssim_score):.6f}")
    print("Tip: You can report diversity as (1 - MS-SSIM) if you prefer 'higher is more diverse'.")


def main():
    # Configure image size once here
    image_size = (64, 64, 64)
    
    # Pick device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Real & Fake dataset samples (16 Each)
    n = 16  
    
    # Generate common tensors once (shared by loader)
    
    real_images = make_random_volumes(n=n, image_size=image_size, device=device_t, seed=999)
    fake_images = make_random_volumes(n=n, image_size=image_size, device=device_t, seed=1000)

    # Scenario 1: single-shot
    evaluate_single_shot(
        real_images=real_images,
        fake_images=fake_images,
        device=device,
    )

    # Scenario 2: batched
    evaluate_batched(
        image_size=image_size,
        real_images=real_images,
        fake_images=fake_images,
        batch_size=16,
        device=device,
    )


if __name__ == "__main__":
    main()
