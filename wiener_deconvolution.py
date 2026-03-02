"""
============================================================
STEP 3 — WIENER DECONVOLUTION
============================================================
Applies mild Wiener deconvolution slice-by-slice to reverse
partial-volume blur caused by the low-field scanner's larger
point spread function (PSF).

WHY: Low-field MRI produces images that are inherently
blurrier than high-field because the PSF is wider.  Wiener
deconvolution models that blur as a known Gaussian PSF and
inverts it in the frequency domain while regularising
against noise amplification.  We keep the regularisation
parameter (balance) LOW so we get modest sharpening without
ringing artefacts.  Near-empty slices (air or padding) are
skipped because deconvolving noise-only slices produces
artefacts.

Input:  bm3d_denoised.nii.gz
Output: wiener_deconvolved.nii.gz
============================================================
"""

import os
import sys
import numpy as np
import nibabel as nib
from scipy.signal import fftconvolve
from skimage.restoration import wiener
from skimage.exposure import rescale_intensity

# ============================================================
# SETTINGS — tune these parameters
# ============================================================
INPUT_DIR   = r"D:\lowfieldPipeline\inputs"
INPUT_FILE  = "bm3d_denoised.nii.gz"
OUTPUT_FILE = "wiener_deconvolved.nii.gz"

INPUT_PATH  = os.path.join(INPUT_DIR, INPUT_FILE)
OUTPUT_PATH = os.path.join(INPUT_DIR, OUTPUT_FILE)

# sigma_inplane: width of the Gaussian PSF that models the
# low-field scanner's in-plane blur (in pixels).
#   - Larger  = assumes more blur → more aggressive sharpening
#   - Smaller = milder correction
SIGMA_INPLANE = 0.9

# balance: Wiener regularisation parameter.
#   - Larger  = smoother result (less ringing, less sharpening)
#   - Smaller = sharper result (more ringing risk)
# 0.05 is a conservative starting point.
BALANCE = 0.05

# PSF_SIZE: side length (pixels) of the Gaussian PSF kernel.
# Must be odd.  A kernel of 11×11 comfortably covers a
# Gaussian with sigma ~1.
PSF_SIZE = 11

# SIGNAL_THRESHOLD_PERCENTILE: slices whose mean intensity
# is below this percentile of the whole-volume mean are
# treated as near-empty and skipped.
SIGNAL_THRESHOLD_PERCENTILE = 5

# ============================================================
# HELPER — build a normalised 2D Gaussian PSF
# ============================================================
def make_gaussian_psf(size, sigma):
    """
    Create a 2D Gaussian point-spread function.

    WHY a Gaussian: the dominant source of blur in low-field
    MRI is the broader main-lobe of the acquisition PSF,
    which is well approximated as Gaussian in-plane.
    The kernel is normalised so that its values sum to 1,
    which preserves overall image brightness during
    convolution/deconvolution.
    """
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    return kernel

# ============================================================
# LOAD
# ============================================================
try:
    print("=" * 55)
    print("STEP 3 — WIENER DECONVOLUTION")
    print("=" * 55)
    print(f"\nLoading: {INPUT_PATH}")

    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: Input file not found:\n  {INPUT_PATH}")
        sys.exit(1)

    nii = nib.load(INPUT_PATH)
    data = nii.get_fdata().astype(np.float64)
    affine = nii.affine
    header = nii.header.copy()

    print(f"  Volume shape : {data.shape}")
    print(f"  Data type    : {data.dtype}")
    print(f"  Voxel sizes  : {header.get_zooms()}")

    # ============================================================
    # STATISTICS — before deconvolution
    # ============================================================
    print("\nBefore Wiener deconvolution:")
    print(f"  Mean : {np.mean(data):.4f}")
    print(f"  Std  : {np.std(data):.4f}")
    print(f"  Min  : {np.min(data):.4f}")
    print(f"  Max  : {np.max(data):.4f}")

    # ============================================================
    # PROCESS — Wiener deconvolution slice by slice
    # ============================================================
    psf = make_gaussian_psf(PSF_SIZE, SIGMA_INPLANE)
    print(f"\nGaussian PSF: {PSF_SIZE}x{PSF_SIZE}, sigma={SIGMA_INPLANE}")
    print(f"Balance (regularisation): {BALANCE}")

    num_slices = data.shape[2]
    output = data.copy()

    # Compute a threshold to skip near-empty slices.
    # We use the percentile of per-slice means.
    slice_means = np.array([np.mean(data[:, :, i]) for i in range(num_slices)])
    signal_threshold = np.percentile(slice_means, SIGNAL_THRESHOLD_PERCENTILE)

    skipped = 0
    print(f"\nProcessing {num_slices} axial slices...")

    for i in range(num_slices):
        slice_2d = data[:, :, i]
        slice_mean = slice_means[i]

        # Skip near-empty slices (air / padding) to avoid
        # deconvolving pure noise.
        if slice_mean <= signal_threshold:
            skipped += 1
            print(f"  Slice {i + 1:3d}/{num_slices} — SKIPPED (low signal)")
            continue

        # skimage.restoration.wiener expects:
        #   image:   the degraded image (2D)
        #   psf:     the point-spread function
        #   balance: regularisation parameter (noise-to-signal power ratio)
        deconv_slice = wiener(slice_2d, psf, balance=BALANCE)

        # Clip negative values that may arise from deconvolution
        # ringing. We use the original slice min as a floor.
        deconv_slice = np.clip(deconv_slice, np.min(slice_2d), None)

        output[:, :, i] = deconv_slice
        print(f"  Slice {i + 1:3d}/{num_slices} done")

    print(f"\n  Slices processed : {num_slices - skipped}")
    print(f"  Slices skipped   : {skipped}")

    # ============================================================
    # STATISTICS — after deconvolution
    # ============================================================
    print("\nAfter Wiener deconvolution:")
    print(f"  Mean : {np.mean(output):.4f}")
    print(f"  Std  : {np.std(output):.4f}")
    print(f"  Min  : {np.min(output):.4f}")
    print(f"  Max  : {np.max(output):.4f}")

    # ============================================================
    # SAVE
    # ============================================================
    print(f"\nSaving: {OUTPUT_PATH}")

    out_nii = nib.Nifti1Image(
        output.astype(np.float64),
        affine=affine,
        header=header
    )
    nib.save(out_nii, OUTPUT_PATH)

    print("\n" + "=" * 55)
    print("WIENER DECONVOLUTION COMPLETE")
    print(f"  Input   : {os.path.basename(INPUT_PATH)}")
    print(f"  Output  : {os.path.basename(OUTPUT_PATH)}")
    print(f"  Sigma   : {SIGMA_INPLANE}")
    print(f"  Balance : {BALANCE}")
    print("=" * 55)

except Exception as e:
    print(f"\n*** ERROR in Wiener deconvolution ***")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
