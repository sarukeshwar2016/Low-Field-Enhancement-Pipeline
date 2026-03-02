"""
tests/test_snr.py
=================
Unit tests for compute_snr utility function.
Run with:  pytest tests/test_snr.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from utils import compute_snr


def phantom(shape=(50, 50, 20), signal=150.0, noise=10.0):
    img = np.zeros(shape, np.float32)
    img[10:40, 10:40, :] = signal
    img[10:40, 10:40, :] += np.random.normal(0, noise, (30, 30, shape[2])).astype(np.float32)
    return np.clip(img, 0, None)


def test_returns_float():
    assert isinstance(compute_snr(phantom()), float)

def test_positive_for_valid_image():
    assert compute_snr(phantom(signal=200, noise=5)) > 0

def test_nan_for_all_zeros():
    assert np.isnan(compute_snr(np.zeros((50, 50, 20), np.float32)))

def test_nan_for_too_few_nonzero():
    img = np.zeros((5, 5, 3), np.float32)
    img[2, 2, 1] = 100.0
    assert np.isnan(compute_snr(img))

def test_higher_snr_for_less_noise():
    assert compute_snr(phantom(noise=2)) > compute_snr(phantom(noise=40))

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
