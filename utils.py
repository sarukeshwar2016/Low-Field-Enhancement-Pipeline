"""
utils.py
========
Shared utility functions for the Low-Field MRI Enhancement Pipeline.
Provides SNR computation, quality metrics, and image operations
used across all pipeline scripts.
"""

import numpy as np
from scipy.stats import skew
from skimage.metrics import structural_similarity as ssim


# ---------------------------------------------------------------
# SNR
# ---------------------------------------------------------------
def compute_snr(img: np.ndarray) -> float:
    """
    Compute MRI SNR using the background-std method.

    Signal = mean of top 30% of non-zero voxels.
    Noise  = std  of bottom 30% of non-zero voxels.

    Returns np.nan when image has < 100 non-zero voxels.
    """
    img = img.astype(np.float32)
    vals = img[img > 0]
    if len(vals) < 100:
        return np.nan
    signal = np.mean(vals[vals > np.percentile(vals, 70)])
    noise  = np.std(vals[vals  < np.percentile(vals, 30)])
    return np.nan if noise < 1e-6 else float(signal / (noise + 1e-8))


# ---------------------------------------------------------------
# QUALITY METRICS
# ---------------------------------------------------------------
def compute_psnr(ref: np.ndarray, img: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio (dB). Returns 100.0 for identical images."""
    mse = np.mean((ref.astype(np.float32) - img.astype(np.float32)) ** 2)
    return 100.0 if mse < 1e-10 else float(20 * np.log10(np.max(ref) / np.sqrt(mse)))


def compute_ssim_volumetric(ref: np.ndarray, img: np.ndarray) -> float:
    """Mean SSIM across all valid slices of a 3-D volume."""
    scores = []
    for i in range(ref.shape[2]):
        r, t = ref[:, :, i], img[:, :, i]
        if np.std(r) < 1e-6 or np.std(t) < 1e-6:
            continue
        scores.append(ssim(r, t, data_range=r.max() - r.min()))
    return float(np.mean(scores)) if scores else 0.0


def compute_histogram_overlap(img1: np.ndarray, img2: np.ndarray, bins: int = 100) -> float:
    """Bhattacharyya coefficient between two image intensity histograms."""
    v1 = img1[img1 > 0]
    v2 = img2[img2 > 0]
    lo, hi = min(v1.min(), v2.min()), max(v1.max(), v2.max())
    h1, _ = np.histogram(v1, bins=bins, range=(lo, hi), density=True)
    h2, _ = np.histogram(v2, bins=bins, range=(lo, hi), density=True)
    return float(np.sum(np.sqrt(h1 * h2)))


# ---------------------------------------------------------------
# IMAGE UTILITIES
# ---------------------------------------------------------------
def match_mean(img: np.ndarray, target_mean: float) -> np.ndarray:
    """Scale image so its non-zero mean equals target_mean."""
    return img * (target_mean / (np.mean(img[img > 0]) + 1e-8))


def stats_line(name: str, img: np.ndarray) -> str:
    """One-line stage statistics string for pipeline reports."""
    v = img[img > 0]
    return (f"{name:25s} | Mean={np.mean(v):8.2f} | Std={np.std(v):8.2f} "
            f"| Skew={skew(v):6.2f} | SNR={compute_snr(img):6.2f}\n")


def is_valid_scan(img: np.ndarray, spacing: tuple,
                  min_slices: int = 10, max_thickness: float = 6.0,
                  min_std: float = 20.0) -> bool:
    """Return True if the scan passes all quality filters."""
    return (img.shape[2] >= min_slices
            and spacing[2] <= max_thickness
            and np.std(img) >= min_std)
