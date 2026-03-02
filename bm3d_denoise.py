"""
============================================================
STEP 2 — BM3D DENOISING
============================================================
Applies BM3D denoising slice-by-slice (2D) to a low-field
MRI volume.

WHY: Low-field MRI has substantially higher thermal noise
than high-field MRI. BM3D (Block-Matching and 3D filtering)
is one of the best classical 2D image denoising algorithms.
We apply it per axial slice because the BM3D algorithm
operates on 2D images. The volume is normalized to [0,1]
before denoising because BM3D expects intensities in that
range, then rescaled back to the original range.

Input:  n4_corrected.nii.gz
Output: bm3d_denoised.nii.gz
============================================================
"""

import os
import sys
import numpy as np
import nibabel as nib
import bm3d

# ============================================================
# SETTINGS — tune these parameters
# ============================================================
INPUT_DIR  = r"D:\lowfieldPipeline\inputs"
INPUT_FILE = "n4_corrected.nii.gz"
OUTPUT_FILE = "bm3d_denoised.nii.gz"

INPUT_PATH  = os.path.join(INPUT_DIR, INPUT_FILE)
OUTPUT_PATH = os.path.join(INPUT_DIR, OUTPUT_FILE)

# sigma_psd: noise standard deviation for BM3D, expressed
# on the [0, 1] normalized scale.
#   - Higher values = more aggressive denoising (risk blur)
#   - Lower values  = gentler denoising (risk residual noise)
# For low-field MRI, 0.08-0.15 is a reasonable range.
SIGMA_PSD = 0.1

# ============================================================
# LOAD
# ============================================================
try:
    print("=" * 55)
    print("STEP 2 — BM3D DENOISING")
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
    # STATISTICS — before denoising
    # ============================================================
    print("\nBefore denoising:")
    print(f"  Mean : {np.mean(data):.4f}")
    print(f"  Std  : {np.std(data):.4f}")
    print(f"  Min  : {np.min(data):.4f}")
    print(f"  Max  : {np.max(data):.4f}")

    # ============================================================
    # PROCESS — BM3D slice by slice
    # ============================================================
    # Normalize to [0, 1] for BM3D
    # We store the original range so we can rescale after.
    vol_min = np.min(data)
    vol_max = np.max(data)
    vol_range = vol_max - vol_min

    if vol_range == 0:
        print("ERROR: Volume has zero intensity range. Cannot process.")
        sys.exit(1)

    data_norm = (data - vol_min) / vol_range

    num_slices = data_norm.shape[2]  # axial axis
    denoised = np.zeros_like(data_norm)

    print(f"\nApplying BM3D (sigma_psd={SIGMA_PSD}) to {num_slices} axial slices...")

    for i in range(num_slices):
        slice_2d = data_norm[:, :, i]

        # BM3D expects a 2D float array in [0, 1]
        denoised_slice = bm3d.bm3d(
            slice_2d,
            sigma_psd=SIGMA_PSD,
            stage_arg=bm3d.BM3DStages.ALL_STAGES
        )

        # Clip to [0, 1] in case BM3D produces slight over/undershoot
        denoised[:, :, i] = np.clip(denoised_slice, 0.0, 1.0)

        print(f"  Slice {i + 1:3d}/{num_slices} done")

    # ============================================================
    # RESCALE — back to original intensity range
    # ============================================================
    denoised_rescaled = denoised * vol_range + vol_min

    # ============================================================
    # STATISTICS — after denoising
    # ============================================================
    print("\nAfter denoising:")
    print(f"  Mean : {np.mean(denoised_rescaled):.4f}")
    print(f"  Std  : {np.std(denoised_rescaled):.4f}")
    print(f"  Min  : {np.min(denoised_rescaled):.4f}")
    print(f"  Max  : {np.max(denoised_rescaled):.4f}")

    # Noise reduction estimate (std should drop)
    std_before = np.std(data)
    std_after  = np.std(denoised_rescaled)
    reduction  = (1.0 - std_after / std_before) * 100 if std_before > 0 else 0
    print(f"\n  Std reduction : {reduction:.1f}%")

    # ============================================================
    # SAVE
    # ============================================================
    print(f"\nSaving: {OUTPUT_PATH}")

    out_nii = nib.Nifti1Image(
        denoised_rescaled.astype(np.float64),
        affine=affine,
        header=header
    )
    nib.save(out_nii, OUTPUT_PATH)

    print("\n" + "=" * 55)
    print("BM3D DENOISING COMPLETE")
    print(f"  Input  : {os.path.basename(INPUT_PATH)}")
    print(f"  Output : {os.path.basename(OUTPUT_PATH)}")
    print(f"  sigma  : {SIGMA_PSD}")
    print("=" * 55)

except Exception as e:
    print(f"\n*** ERROR in BM3D denoising ***")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
