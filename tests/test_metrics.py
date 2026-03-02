"""
tests/test_metrics.py
=====================
Unit tests for compute_psnr, compute_ssim_volumetric,
compute_histogram_overlap, and match_mean.
Run with:  pytest tests/test_metrics.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from utils import compute_psnr, compute_ssim_volumetric, compute_histogram_overlap, match_mean


def vol(shape=(30, 30, 10), val=100.0):
    return np.full(shape, val, np.float32)


# --- PSNR ---
def test_psnr_identical_is_100():
    v = vol()
    assert compute_psnr(v, v) == 100.0

def test_psnr_returns_float():
    assert isinstance(compute_psnr(vol(val=200), vol(val=180)), float)

def test_psnr_decreases_with_noise():
    ref = vol(val=200)
    low  = ref + np.random.normal(0, 1,  ref.shape).astype(np.float32)
    high = ref + np.random.normal(0, 30, ref.shape).astype(np.float32)
    assert compute_psnr(ref, low) > compute_psnr(ref, high)

# --- SSIM ---
def test_ssim_identical_near_one():
    v = vol()
    assert abs(compute_ssim_volumetric(v, v) - 1.0) < 0.01

def test_ssim_in_valid_range():
    ref = vol(val=100)
    img = (np.random.rand(*ref.shape) * 200).astype(np.float32)
    assert 0.0 <= compute_ssim_volumetric(ref, img) <= 1.0

# --- Histogram overlap ---
def test_hist_identical_near_one():
    v = vol()
    assert abs(compute_histogram_overlap(v, v) - 1.0) < 0.05

def test_hist_different_less_than_one():
    assert compute_histogram_overlap(vol(val=50), vol(val=200)) < 0.5

# --- match_mean ---
def test_match_mean_correct():
    result = match_mean(vol(val=50), 100.0)
    assert abs(np.mean(result[result > 0]) - 100.0) < 0.1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
