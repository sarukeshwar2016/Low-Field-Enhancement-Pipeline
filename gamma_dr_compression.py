import os
import numpy as np
import nibabel as nib

# ============================================================
# SETTINGS (STRICT PHYSICS)
# ============================================================
INPUT_PATH  = r"D:\lowfieldPipeline\inputs\clahe_enhanced.nii.gz"
HF_REF_DIR  = r"D:\01_MRI_Data\nifti_output\high_field_nifti"
OUTPUT_PATH = r"D:\lowfieldPipeline\inputs\final_enhanced.nii.gz"

# 🔥 STRICT: mild adjustment
GAMMA = 1.0 
DR_CLIP_PERCENTILE = 99.8

def compute_snr(img):
    img = img.astype(np.float32)
    vals = img[img > 0]
    if len(vals) < 100: return np.nan
    signal = np.mean(vals[vals > np.percentile(vals, 70)])
    noise  = np.std(vals[vals < np.percentile(vals, 30)])
    return signal / (noise + 1e-8)

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Error: {INPUT_PATH} not found!")
        return

    # 1. Load HF Reference
    hf_path = os.path.join(HF_REF_DIR, "0001_HF.nii.gz") 
    if os.path.exists(hf_path):
        hf_data = nib.load(hf_path).get_fdata()
        hf_snr  = compute_snr(hf_data)
        hf_std  = np.std(hf_data[hf_data > 0])
        print(f"HF Reference: SNR={hf_snr:.2f}, STD={hf_std:.2f}")
    else:
        hf_snr, hf_std = None, None

    try:
        nii = nib.load(INPUT_PATH)
        data = nii.get_fdata().astype(np.float32)
        affine, header = nii.affine, nii.header

        print("Final Gamma & Safety (Scaling-Only)")
        
        # 2. Gamma
        final_data = np.power(data, GAMMA)

        # 🧪 SCALING-ONLY SNR GUARDS (NO NOISE)
        if hf_snr is not None:
            current_snr = compute_snr(final_data)
            target_max = 0.75 * hf_snr
            target_min = 0.50 * hf_snr
            print(f"  Final SNR: {current_snr:.2f} (Window: {target_min:.2f}-{target_max:.2f})")
            
            if current_snr > target_max:
                print(f"  Scaling down to max target ({target_max:.2f})...")
                final_data *= (target_max / (current_snr + 1e-8))
            elif current_snr < target_min:
                print(f"  Scaling up to min target ({target_min:.2f})...")
                final_data *= (target_min / (current_snr + 1e-8))

        # 🧪 STD SAFEGUARD
        if hf_std is not None:
             curr_std = np.std(final_data[final_data > 0])
             if curr_std < 0.6 * hf_std:
                 print(f"  Warning: STD too low. Applying 1.1x boost.")
                 final_data *= 1.1

        # Save Final
        nib.save(nib.Nifti1Image(final_data.astype(np.float32), affine, header), OUTPUT_PATH)
        print(f"SUCCESS: {OUTPUT_PATH}")

    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()
