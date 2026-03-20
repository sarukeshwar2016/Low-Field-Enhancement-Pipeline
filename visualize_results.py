"""
============================================================
VISUAL RESULTS COMPARISON
============================================================
Extracts the middle slice from the Low-Field source and the 
Final Enhanced result, saving them side-by-side as a PNG.
============================================================
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# Paths
LF_PATH    = r"D:\lowfieldPipeline\inputs\simulated_low_field.nii.gz"
FINAL_PATH = r"D:\lowfieldPipeline\inputs\final_enhanced.nii.gz"
OUTPUT_IMG = r"D:\lowfieldPipeline\enhancement_comparison.png"

def create_comparison():
    if not os.path.exists(LF_PATH) or not os.path.exists(FINAL_PATH):
        print("Error: Required NIfTI files not found in inputs folder.")
        return

    print("Loading images for visualization...")
    nii_lf = nib.load(LF_PATH)
    nii_final = nib.load(FINAL_PATH)

    data_lf = nii_lf.get_fdata()
    data_final = nii_final.get_fdata()

    # Get middle slice index (axial)
    mid_idx = data_lf.shape[2] // 2

    slice_lf = data_lf[:, :, mid_idx]
    slice_final = data_final[:, :, mid_idx]

    # Rotate for better viewing (usually needed for NIfTI)
    slice_lf = np.rot90(slice_lf)
    slice_final = np.rot90(slice_final)

    # Normalize for display consistency
    def norm(img):
        p1, p99 = np.percentile(img, [1, 99])
        return np.clip((img - p1) / (p99 - p1 + 1e-8), 0, 1)

    disp_lf = norm(slice_lf)
    disp_final = norm(slice_final)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    plt.subplots_adjust(wspace=0.05)

    axes[0].imshow(disp_lf, cmap='gray')
    axes[0].set_title(f"Simulated Low-Field (Source)\nSNR: 2.33", fontsize=14, color='white')
    axes[0].axis('off')

    axes[1].imshow(disp_final, cmap='gray')
    axes[1].set_title(f"Final Enhanced Pipeline\nSNR: 4.01", fontsize=14, color='cyan')
    axes[1].axis('off')

    fig.patch.set_facecolor('#111111')
    
    print(f"Saving comparison image to: {OUTPUT_IMG}")
    plt.savefig(OUTPUT_IMG, facecolor=fig.get_facecolor(), bbox_inches='tight', dpi=150)
    plt.close()
    
    print("\nVisual comparison generated successfully!")

if __name__ == "__main__":
    create_comparison()
