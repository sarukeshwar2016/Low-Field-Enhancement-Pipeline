"""
============================================================
N4ITK BIAS FIELD CORRECTION — STANDALONE SCRIPT
============================================================
Applies N4ITK bias field correction to a NIfTI MRI volume.
Works on both high-field and simulated low-field images.

Usage:
    python n4_bias_correction.py

Edit the INPUT_PATH and OUTPUT_PATH below to point to your files.
============================================================
"""

import SimpleITK as sitk
import nibabel as nib
import numpy as np
import os

# ============================================================
# SETTINGS — EDIT THESE PATHS
# ============================================================
INPUT_PATH  = r"D:\lowfieldPipeline\inputs\simulated_low_field.nii.gz"
OUTPUT_PATH = r"D:\lowfieldPipeline\inputs\n4_corrected.nii.gz"

# N4 parameters — safe defaults, no need to change for most cases
SHRINK_FACTOR        = 2      # Downsamples image for speed. 1=full res (slow), 2=recommended, 4=fast
NUM_FITTING_LEVELS   = 4      # Number of resolution levels. 4 is standard.
NUM_ITERATIONS       = [50, 50, 50, 50]  # Iterations per level. More = more correction, slower.
CONVERGENCE_THRESH   = 0.001  # Stop early if correction stabilizes. Lower = more thorough.
SPLINE_ORDER         = 3      # B-spline order for bias field model. 3 is standard.

# ============================================================
# STEP 1 — LOAD IMAGE
# ============================================================
print("=" * 55)
print("N4ITK BIAS FIELD CORRECTION")
print("=" * 55)
print(f"\nLoading: {INPUT_PATH}")

if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

# Load with SimpleITK (handles NIfTI, DICOM, MHA etc.)
image_sitk = sitk.ReadImage(INPUT_PATH, sitk.sitkFloat32)

print(f"  Image size   : {image_sitk.GetSize()}")
print(f"  Spacing (mm) : {image_sitk.GetSpacing()}")
print(f"  Pixel type   : {image_sitk.GetPixelIDTypeAsString()}")


# ============================================================
# STEP 2 — CREATE BODY MASK
# ============================================================
# N4 works much better with a mask that excludes air background.
# We use Otsu thresholding to auto-detect tissue vs air.
print("\nGenerating body mask (Otsu threshold)...")

otsu_filter = sitk.OtsuThresholdImageFilter()
otsu_filter.SetInsideValue(0)   # air = 0
otsu_filter.SetOutsideValue(1)  # tissue = 1
mask_sitk = otsu_filter.Execute(image_sitk)

# Count masked voxels for sanity check
stats = sitk.StatisticsImageFilter()
stats.Execute(mask_sitk)
total_voxels = image_sitk.GetSize()[0] * image_sitk.GetSize()[1] * image_sitk.GetSize()[2]
mask_voxels  = int(stats.GetSum())
print(f"  Masked voxels: {mask_voxels:,} / {total_voxels:,} ({100*mask_voxels/total_voxels:.1f}%)")

if mask_voxels < 0.05 * total_voxels:
    print("  WARNING: Mask looks too small — consider checking your input image.")
if mask_voxels > 0.95 * total_voxels:
    print("  WARNING: Mask looks too large — Otsu may have failed. Proceeding without mask.")
    mask_sitk = None


# ============================================================
# STEP 3 — OPTIONAL SHRINK FOR SPEED
# ============================================================
if SHRINK_FACTOR > 1:
    print(f"\nShrinking image by factor {SHRINK_FACTOR} for speed...")
    image_small = sitk.Shrink(image_sitk, [SHRINK_FACTOR] * image_sitk.GetDimension())
    if mask_sitk is not None:
        mask_small = sitk.Shrink(mask_sitk, [SHRINK_FACTOR] * mask_sitk.GetDimension())
    else:
        mask_small = None
    print(f"  Shrunk size  : {image_small.GetSize()}")
else:
    image_small = image_sitk
    mask_small  = mask_sitk


# ============================================================
# STEP 4 — RUN N4 BIAS FIELD CORRECTION
# ============================================================
print("\nRunning N4ITK bias field correction...")
print(f"  Shrink factor     : {SHRINK_FACTOR}")
print(f"  Fitting levels    : {NUM_FITTING_LEVELS}")
print(f"  Iterations/level  : {NUM_ITERATIONS}")
print(f"  Convergence thresh: {CONVERGENCE_THRESH}")

corrector = sitk.N4BiasFieldCorrectionImageFilter()
corrector.SetMaximumNumberOfIterations(NUM_ITERATIONS)
corrector.SetConvergenceThreshold(CONVERGENCE_THRESH)
corrector.SetSplineOrder(SPLINE_ORDER)

# Execute on shrunk image
if mask_small is not None:
    corrected_small = corrector.Execute(image_small, mask_small)
else:
    corrected_small = corrector.Execute(image_small)

print("  N4 complete.")


# ============================================================
# STEP 5 — EXTRACT BIAS FIELD AND APPLY TO FULL-RES IMAGE
# ============================================================
# If we shrank the image, we need to:
# 1. Extract the log bias field from the corrected small image
# 2. Upsample it back to full resolution
# 3. Apply it to the original full-res image

if SHRINK_FACTOR > 1:
    print("\nUpsampling bias field to full resolution...")

    # Get log bias field from the filter
    log_bias_field = corrector.GetLogBiasFieldAsImage(image_sitk)

    # Apply: corrected = original / exp(log_bias_field)
    corrected_full = image_sitk / sitk.Exp(log_bias_field)

else:
    corrected_full = corrected_small


# ============================================================
# STEP 6 — SANITY CHECK: COMPARE BEFORE / AFTER STATISTICS
# ============================================================
print("\nStatistics comparison (inside mask):")

def get_stats(img_sitk, mask_sitk):
    """Get mean and std inside mask."""
    arr = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
    if mask_sitk is not None:
        msk = sitk.GetArrayFromImage(mask_sitk).astype(bool)
        # Resize mask if needed (in case of size mismatch from shrink)
        if msk.shape != arr.shape:
            msk = sitk.GetArrayFromImage(
                sitk.Resample(mask_sitk, img_sitk, sitk.Transform(),
                              sitk.sitkNearestNeighbor, 0.0, mask_sitk.GetPixelID())
            ).astype(bool)
        vals = arr[msk]
    else:
        vals = arr.flatten()
    return np.mean(vals), np.std(vals), np.min(vals), np.max(vals)

orig_mean, orig_std, orig_min, orig_max = get_stats(image_sitk, mask_sitk)
corr_mean, corr_std, corr_min, corr_max = get_stats(corrected_full, mask_sitk)

print(f"  {'':20s}  {'Before':>12s}  {'After':>12s}")
print(f"  {'Mean':20s}  {orig_mean:>12.2f}  {corr_mean:>12.2f}")
print(f"  {'Std Dev':20s}  {orig_std:>12.2f}  {corr_std:>12.2f}")
print(f"  {'Min':20s}  {orig_min:>12.2f}  {corr_min:>12.2f}")
print(f"  {'Max':20s}  {orig_max:>12.2f}  {corr_max:>12.2f}")

# Coefficient of Variation (lower after N4 = good correction)
cv_before = orig_std / (orig_mean + 1e-8)
cv_after  = corr_std / (corr_mean + 1e-8)
print(f"\n  CoV before: {cv_before:.4f}")
print(f"  CoV after : {cv_after:.4f}")
if cv_after < cv_before:
    improvement = 100 * (cv_before - cv_after) / (cv_before + 1e-8)
    print(f"  Homogeneity improved by {improvement:.1f}%  ✓")
else:
    print("  WARNING: CoV did not decrease — image may not have significant bias.")


# ============================================================
# STEP 7 — SAVE OUTPUT
# ============================================================
print(f"\nSaving corrected image to:\n  {OUTPUT_PATH}")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Save using SimpleITK (preserves spacing, origin, direction)
sitk.WriteImage(corrected_full, OUTPUT_PATH)

print("\n" + "=" * 55)
print("N4 BIAS CORRECTION COMPLETE")
print(f"  Input  : {os.path.basename(INPUT_PATH)}")
print(f"  Output : {os.path.basename(OUTPUT_PATH)}")
print("=" * 55)


# ============================================================
# OPTIONAL: ALSO SAVE THE BIAS FIELD MAP (for inspection)
# ============================================================
SAVE_BIAS_MAP = True  # Set False to skip

if SAVE_BIAS_MAP and SHRINK_FACTOR > 1:
    bias_path = OUTPUT_PATH.replace(".nii.gz", "_bias_field.nii.gz")
    print(f"\nSaving bias field map to:\n  {bias_path}")
    bias_img = sitk.Exp(log_bias_field)
    sitk.WriteImage(bias_img, bias_path)
    print("  (View this in ITK-SNAP or 3D Slicer to visualise the correction)")