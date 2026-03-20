import os
import numpy as np
import nibabel as nib
import bm3d

# ============================================================
# SETTINGS (STRICT PHYSICS)
# ============================================================
INPUT_PATH  = r"D:\lowfieldPipeline\inputs\n4_corrected.nii.gz"
OUTPUT_PATH = r"D:\lowfieldPipeline\inputs\bm3d_denoised.nii.gz"

# 🔥 STRICT: sigma_psd = 0.045
SIGMA_PSD = 0.045

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Error: {INPUT_PATH} not found!")
        return

    try:
        print("=" * 55)
        print("STEP 2 — BM3D DENOISING (HARD THRESHOLDING)")
        print("=" * 55)

        nii = nib.load(INPUT_PATH)
        data = nii.get_fdata().astype(np.float32)
        affine, header = nii.affine, nii.header

        print(f"Input shape: {data.shape}")
        
        # Image is already normalized [0,1] from N4
        # Apply slice-wise
        denoised = np.zeros_like(data)
        for i in range(data.shape[2]):
            sl = data[:, :, i]
            if np.max(sl) < 0.05:
                denoised[:, :, i] = sl
                continue
            
            # 🔥 STRICT: HARD_THRESHOLDING ONLY
            den = bm3d.bm3d(
                sl,
                sigma_psd=SIGMA_PSD,
                stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING
            )
            denoised[:, :, i] = np.clip(den, 0, 1)
            
            if (i+1) % 50 == 0: print(f"  Slice {i+1}/{data.shape[2]} done")

        # Save (No SNR loops, no hacks)
        nib.save(nib.Nifti1Image(denoised.astype(np.float32), affine, header), OUTPUT_PATH)
        print(f"SUCCESS: {OUTPUT_PATH}")

    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()
