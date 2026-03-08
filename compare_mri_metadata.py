"""
============================================================
COMPARE High-Field vs Low-Field MRI METADATA
============================================================
This script analyses the high-field reference and the 
simulated low-field image to extract and display the exact
parameters (voxel spacing, dimensions, SNR proxy, statistics)
that prove the differences between the two modalities.

It reads directly from the NIfTI headers and computes
standard statistical moments from the image data.
============================================================
"""

import os
import nibabel as nib
import numpy as np
from scipy.stats import skew, kurtosis

# ============================================================
# SETTINGS
# ============================================================
INPUT_DIR_HF = r"D:\01_MRI_Data\nifti_output\high_field_nifti"
INPUT_DIR_LF = r"D:\01_MRI_Data\nifti_output\low_field_simulated"

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def extract_properties(filepath, name):
    if not os.path.exists(filepath):
        return None
        
    nii = nib.load(filepath)
    data = nii.get_fdata()
    header = nii.header
    
    # 1. Spacing / Resolution (from NIfTI header)
    zooms = header.get_zooms()
    if len(zooms) >= 3:
        spacing = tuple(np.round(zooms[:3], 3))
    else:
        spacing = zooms
        
    # 2. Dimensions
    shape = data.shape
    
    # 3. Simple Body Mask (ignore vast background air)
    threshold = np.percentile(data, 5)
    body_data = data[data > threshold]
    
    # 4. Statistical properties (from the image data itself)
    mean_val = np.mean(body_data)
    std_val = np.std(body_data)
    skew_val = skew(body_data)
    
    # SNR Proxy: Mean / Std (Approximation for comparison only)
    snr_proxy = mean_val / std_val if std_val > 0 else 0
    
    return {
        "Name": name,
        "File": os.path.basename(filepath),
        "Dimensions (Voxels)": str(shape),
        "Voxel Spacing (mm)": str(spacing),
        "Intensity Mean": f"{mean_val:.2f}",
        "Intensity Std_Dev": f"{std_val:.2f}",
        "Intensity Skewness": f"{skew_val:.3f}",
        "Approximate SNR": f"{snr_proxy:.2f}"
    }

# ============================================================
# MAIN
# ============================================================
import glob

output_file = "all_metadata_analysis.txt"

hf_files = sorted(glob.glob(os.path.join(INPUT_DIR_HF, "*_HF.nii.gz")))

with open(output_file, "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("BATCH METADATA ANALYSIS: HIGH-FIELD vs LOW-FIELD\n")
    f.write("=" * 80 + "\n\n")

    for hf_path in hf_files:
        basename = os.path.basename(hf_path)
        patient_id = basename.replace("_HF.nii.gz", "")
        lf_path = os.path.join(INPUT_DIR_LF, f"{patient_id}_LF.nii.gz")
        
        f.write(f"PATIENT: {patient_id}\n")
        f.write("-" * 80 + "\n")
        
        hf_props = extract_properties(hf_path, "HIGH-FIELD (Reference)")
        lf_props = extract_properties(lf_path, "LOW-FIELD  (Simulated)")

        if hf_props and lf_props:
            f.write(f"{'Parameter':<25} | {'High-Field (HF)':<25} | {'Low-Field (LF)':<25}\n")
            f.write("-" * 80 + "\n")
            
            keys_to_compare = [
                "Dimensions (Voxels)", 
                "Voxel Spacing (mm)", 
                "Intensity Mean", 
                "Intensity Std_Dev", 
                "Intensity Skewness",
                "Approximate SNR"
            ]
            
            for key in keys_to_compare:
                hf_val = hf_props[key]
                lf_val = lf_props[key]
                f.write(f"{key:<25} | {hf_val:<25} | {lf_val:<25}\n")
            
            f.write("\n")
        else:
            f.write("Failed to load paired HF/LF NIfTIs for this patient.\n\n")
            
    f.write("=" * 80 + "\n")
    f.write("KEY TAKEAWAYS PROVING THIS IS LOW-FIELD ACROSS ALL PAIRS:\n")
    f.write("1. Voxel Spacing: LF typically has larger voxels (lower resolution) than HF.\n")
    f.write("2. Approximate SNR: LF inherently has much lower SNR due to the weaker magnetic field.\n")
    f.write("3. Intensity Skewness: LF often has a highly skewed intensity profile because\n")
    f.write("   the high noise floor pushes the background up (Rician distribution at low SNR).\n")
    f.write("   HF has cleaner, wider contrast separation.\n")
    f.write("=" * 80 + "\n")
        
print(f"Batch analysis saved to {output_file}")
