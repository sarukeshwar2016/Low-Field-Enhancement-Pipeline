import os
import numpy as np
import nibabel as nib
from skimage import exposure

# ============================================================
# SETTINGS (STRICT PHYSICS)
# ============================================================
INPUT_PATH  = r"D:\lowfieldPipeline\inputs\wiener_deconvolved.nii.gz"
HF_REF_DIR  = r"D:\01_MRI_Data\nifti_output\high_field_nifti"
OUTPUT_PATH = r"D:\lowfieldPipeline\inputs\clahe_enhanced.nii.gz"

# 🔥 STRICT: clip_limit = 0.0035
CLIP_LIMIT = 0.0035

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Error: {INPUT_PATH} not found!")
        return

    # 1. Load HF Reference
    hf_path = os.path.join(HF_REF_DIR, "0001_HF.nii.gz") 
    if os.path.exists(hf_path):
        hf_data = nib.load(hf_path).get_fdata()
        hf_mean = np.mean(hf_data[hf_data > 0])
        print(f"Reference HF Mean: {hf_mean:.2f}")
    else:
        hf_mean = None

    try:
        nii = nib.load(INPUT_PATH)
        data = nii.get_fdata().astype(np.float32)
        affine, header = nii.affine, nii.header

        print(f"CLAHE Enhancement: clip={CLIP_LIMIT}")
        
        output = np.zeros_like(data)
        for i in range(data.shape[2]):
            output[:, :, i] = exposure.equalize_adapthist(data[:, :, i], clip_limit=CLIP_LIMIT)
            if (i+1) % 50 == 0: print(f"  Slice {i+1}/{data.shape[2]} done")

        # 🧪 INTENSITY MATCHING
        rescaled = np.zeros_like(output)
        if hf_mean is not None:
            current_mean = np.mean(output[output > 0])
            rescaled = output * (hf_mean / (current_mean + 1e-8))
            print(f"  Mean Corrected to HF: {np.mean(rescaled[rescaled > 0]):.2f}")
        else:
            rescaled = output

        # Ensure positive
        rescaled = np.clip(rescaled, 0, None)

        # Save
        nib.save(nib.Nifti1Image(rescaled.astype(np.float32), affine, header), OUTPUT_PATH)
        print(f"SUCCESS: {OUTPUT_PATH}")

    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()
