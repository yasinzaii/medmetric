# medmetric

**medmetric** is a lightweight PyTorch package for evaluating **real vs synthetic medical images**
(especially **3D volumes** like brain MRI) using:

- **MedicalNet (3D ResNet) features** (Tencent Med3D / MedicalNet)
- **FID** and **MMD** computed on **feature vectors**
- **MS-SSIM** for **pairwise similarity** (commonly used as a *diversity proxy* over fake–fake pairs)

---

## Installation

### Core install (extractor + MMD)

```bash
pip install medmetric
```

### Optional: MONAI (required for FID + MS-SSIM)

`FID` and `MS_SSIM` are thin wrappers around MONAI metrics. If MONAI is not installed, these metrics raise an informative `ImportError`.

Recommended (installs MONAI via the package extra):

```bash
pip install "medmetric[monai]"
```

If your installed `medmetric` build does not expose the extra yet, install MONAI directly:

```bash
pip install monai
```

---

## MedicalNet weights

Pretrained checkpoints are resolved from the pinned manifest:

- `medmetric/weights/medicalnet.yaml`

Each entry specifies:
- the **Hugging Face repo** (`repo_id`)
- the **filename** (`filename`)
- `shortcut_type`, and (optionally) architecture hints (`layers`, `block`)
- a pinned **SHA256** checksum, which is verified after download

### Available checkpoints (from the shipped YAML)

| depth | variant | repo_id | filename | shortcut | block |
|---:|---|---|---|:---:|---|
| 10 | standard | `TencentMedicalNet/MedicalNet-Resnet10` | `resnet_10.pth` | B | BasicBlock |
| 10 | 23dataset | `TencentMedicalNet/MedicalNet-Resnet10` | `resnet_10_23dataset.pth` | B | BasicBlock |
| 18 | standard | `TencentMedicalNet/MedicalNet-Resnet18` | `resnet_18.pth` | A | BasicBlock |
| 18 | 23dataset | `TencentMedicalNet/MedicalNet-Resnet18` | `resnet_18_23dataset.pth` | A | BasicBlock |
| 34 | standard | `TencentMedicalNet/MedicalNet-Resnet34` | `resnet_34.pth` | A | BasicBlock |
| 34 | 23dataset | `TencentMedicalNet/MedicalNet-Resnet34` | `resnet_34_23dataset.pth` | A | BasicBlock |
| 50 | standard | `TencentMedicalNet/MedicalNet-Resnet50` | `resnet_50.pth` | B | Bottleneck |
| 50 | 23dataset | `TencentMedicalNet/MedicalNet-Resnet50` | `resnet_50_23dataset.pth` | B | Bottleneck |
| 101 | standard | `TencentMedicalNet/MedicalNet-Resnet101` | `resnet_101.pth` | B | Bottleneck |
| 152 | standard | `TencentMedicalNet/MedicalNet-Resnet152` | `resnet_152.pth` | B | Bottleneck |
| 200 | standard | `TencentMedicalNet/MedicalNet-Resnet200` | `resnet_200.pth` | B | Bottleneck |

**Important behavior:**
- `use_23dataset=True` is available for depths **10/18/34/50**.
- For depths **101/152/200**, only the **standard** checkpoint is listed in the manifest, so you must call:
  `MedicalNetFeatureExtractor.from_pretrained(depth=101, use_23dataset=False, ...)`.

---

## Input normalization

For best alignment with MedicalNet training, apply **per-volume z-score normalization** before passing
volumes into the extractor (and apply the same preprocessing to *real* and *fake*).

Keep **MS-SSIM** calculations in the **original image domain** (e.g., `[0,1]`) with an appropriate `data_range`.

---

## Metrics

### FID (on features)

![FID with real/fake definitions](https://latex.codecogs.com/svg.image?\begin{aligned}\mathrm{FID}(R,F)&=\lVert\mu_r-\mu_f\rVert_2^2+\operatorname{Tr}\!\left(\Sigma_r+\Sigma_f-2(\Sigma_r\Sigma_f)^{1/2}\right)\\\mu_r&=\frac{1}{N_r}\sum_{i=1}^{N_r}r_i,\qquad \Sigma_r=\frac{1}{N_r-1}\sum_{i=1}^{N_r}(r_i-\mu_r)(r_i-\mu_r)^{\top}\\\mu_f&=\frac{1}{N_f}\sum_{i=1}^{N_f}f_i,\qquad \Sigma_f=\frac{1}{N_f-1}\sum_{i=1}^{N_f}(f_i-\mu_f)(f_i-\mu_f)^{\top}\end{aligned})

`FID(fake_feats, real_feats)` returns a scalar tensor.

### MMD (Gaussian / RBF)

**Unbiased MMD² (U-statistic)** (default):

![Unbiased MMD^2](https://latex.codecogs.com/svg.image?\widehat{\mathrm{MMD}}^2_{\mathrm{unb}}(X,Y)=\frac{1}{m(m-1)}\sum_{i\neq j}k(x_i,x_j)+\frac{1}{n(n-1)}\sum_{i\neq j}k(y_i,y_j)-\frac{2}{mn}\sum_{i=1}^m\sum_{j=1}^nk(x_i,y_j))

**Biased MMD² (V-statistic)** (`biased=True`):

![Biased MMD^2](https://latex.codecogs.com/svg.image?\widehat{\mathrm{MMD}}^2_{\mathrm{b}}(X,Y)=\frac{1}{m^2}\sum_{i=1}^m\sum_{j=1}^mk(x_i,x_j)+\frac{1}{n^2}\sum_{i=1}^n\sum_{j=1}^nk(y_i,y_j)-\frac{2}{mn}\sum_{i=1}^m\sum_{j=1}^nk(x_i,y_j))

**Gaussian (RBF) kernel**:

![RBF kernel](https://latex.codecogs.com/svg.image?k(x,y)=\exp\left(-\frac{\lVert x-y\rVert^2}{2\sigma^2}\right))

> Note: the unbiased estimator can be slightly negative due to finite-sample variance. `medmetric` clamps MMD² to `>= 0`
> before `sqrt` by default.

---

## End-to-end example (recommended workflow)

This example follows the intended pipeline:

1. Keep images in `[0,1]` for MS-SSIM  
2. Z-score normalize volumes for MedicalNet feature extraction  
3. Compute FID/MMD on **features**  
4. Compute MS-SSIM “diversity” on **fake–fake pairs**, with `K = min(target_pairs, N*(N-1)/2)`  

```python
import torch
from medmetric.extractors.medicalnet import MedicalNetFeatureExtractor
from medmetric.metrics import FID, MMD, MS_SSIM

# -----------------------------
# Config
# -----------------------------

D, H, W = 180, 180, 180
n_real, n_fake = 16, 16
target_pairs = 5000  # target number of fake–fake pairs to evaluate (upper-bounded by N choose 2)

device = "cuda" if torch.cuda.is_available() else "cpu"
device_t = torch.device(device)

# -----------------------------
# Dummy data (in [0,1])
# Volumes: (B, 1, D, H, W)
# -----------------------------
real_images = torch.rand(n_real, 1, D, H, W, device=device_t)
fake_images = torch.rand(n_fake, 1, D, H, W, device=device_t)

# -----------------------------
# IMPORTANT: z-score normalization for MedicalNet input
# (apply the same preprocessing to both real and fake)
# -----------------------------
def zscore_per_volume(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    dims = tuple(range(2, x.ndim))  # normalize per (D,H,W) per sample (and per channel if C>1)
    mean = x.mean(dim=dims, keepdim=True)
    std = x.std(dim=dims, keepdim=True).clamp_min(eps)
    return (x - mean) / std

real_z = zscore_per_volume(real_images)
fake_z = zscore_per_volume(fake_images)

# -----------------------------
# MedicalNet feature extractor
# -----------------------------
extractor = MedicalNetFeatureExtractor.from_pretrained(depth=50, use_23dataset=True, device=device)

# Alternative in-place loading (same result, avoids rebuilding):
# extractor = MedicalNetFeatureExtractor(depth=50).to(device_t)
# extractor.load_pretrained(use_23dataset=True, device=device)

real_feats = extractor(real_z)  # (N, F) pooled embeddings
fake_feats = extractor(fake_z)

# -----------------------------
# Similarity on FEATURES
# -----------------------------
fid = FID()
mmd = MMD()  # unbiased by default

fid_score = fid(fake_feats, real_feats)
mmd_score = mmd(fake_feats, real_feats)

# -----------------------------
# Diversity on IMAGES (fake–fake pairs)
# For N=16 this evaluates randomly sampled pairs (120).
# -----------------------------
n = fake_images.shape[0]
k = min(5000, n * (n - 1) // 2)

i = torch.randint(0, n, (k,), device=fake_images.device)
j = torch.randint(0, n - 1, (k,), device=fake_images.device)
j = j + (j >= i)  # ensures j != i

ms_ssim = MS_SSIM(spatial_dims=fake_images.ndim - 2, data_range=1.0)  # default reduction="mean"
ms_mean = ms_ssim(fake_images[i], fake_images[j])  # <-- actually computes the mean over k pairs


print("FID:", float(fid_score))
print("MMD:", float(mmd_score))
print("MS-SSIM(fake pairs) mean:", float(ms_mean))
```


---

## Examples

See `examples/compute_metrics.py`.

---

## Testing

```bash
pytest -q
```

---

## License

MIT (see `LICENSE`).

---

## Contributing (simple workflow)

The simplest approach is:

1. Fork the repo
2. Create a feature branch
3. Open a PR into `main`

Recommended minimal contribution checks:

```bash
pip install -e .[dev]
pytest -q
```
