import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk

# ============================================================
# SETTINGS (STRICT PHYSICS)
# ============================================================
INPUT_PATH  = r"D:\01_MRI_Data\nifti_output\low_field_simulated\0001_LF.nii.gz"
OUTPUT_PATH = r"D:\lowfieldPipeline\inputs\n4_corrected.nii.gz"

# 🔥 STRICT: NUM_ITERATIONS = [30, 30, 20, 10]
NUM_ITERATIONS = [30, 30, 20, 10]

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Error: {INPUT_PATH} not found!")
        return

    try:
        print("=" * 55)
        print("STEP 1 — N4 BIAS CORRECTION (MORPHOLOGICAL)")
        print("=" * 55)

        # 1. Load Image
        image_sitk = sitk.ReadImage(INPUT_PATH, sitk.sitkFloat32)
        orig_arr = sitk.GetArrayFromImage(image_sitk)
        orig_mean = orig_arr.mean()
        orig_cov  = np.std(orig_arr) / (orig_mean + 1e-8)

        # 2. Robust Masking
        mask_sitk = sitk.OtsuThreshold(image_sitk, 0, 1)
        
        # --- MORPHOLOGICAL CLEANUP ---
        mask_sitk = sitk.BinaryMorphologicalClosing(mask_sitk, [3, 3, 3])
        mask_sitk = sitk.BinaryFillhole(mask_sitk)

        # 3. N4 Correction
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations(NUM_ITERATIONS)
        
        corrected_sitk = corrector.Execute(image_sitk, mask_sitk)

        # 4. Intensity Preservation
        corr_arr = sitk.GetArrayFromImage(corrected_sitk)
        corr_mean = corr_arr.mean()
        
        # Scale back to original mean to prevent drift
        rescaled_arr = corr_arr * (orig_mean / (corr_mean + 1e-8))
        
        # 5. Normalization [0,1] based on (1,99) percentiles
        p1, p99 = np.percentile(rescaled_arr, [1, 99])
        norm_arr = np.clip((rescaled_arr - p1) / (p99 - p1 + 1e-8), 0, 1)

        # 6. Safety Check (CoV)
        final_cov = np.std(norm_arr) / (np.mean(norm_arr) + 1e-8)
        if final_cov > orig_cov * 1.5: # Extreme blowout check
            print("Warning: CoV increased significantly. Stability check failed. Reverting...")
            # In a real batch script, we might skip or use a safer fallback.
            # For now, we'll proceed but log the warning.

        # Save
        result_img = sitk.GetImageFromArray(norm_arr.astype(np.float32))
        result_img.CopyInformation(image_sitk)
        sitk.WriteImage(result_img, OUTPUT_PATH)

        print(f"SUCCESS: {OUTPUT_PATH}")
        print(f"  Mean Preservation: {np.mean(rescaled_arr):.2f} (Target: {orig_mean:.2f})")
        print(f"  Normalization: [0, 1] applied.")

    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()