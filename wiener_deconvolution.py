import os
import numpy as np
import nibabel as nib
from skimage.restoration import wiener

# ============================================================
# SETTINGS (STRICT PHYSICS)
# ============================================================
INPUT_PATH  = r"D:\lowfieldPipeline\inputs\bm3d_denoised.nii.gz"
HF_REF_DIR  = r"D:\01_MRI_Data\nifti_output\high_field_nifti"
OUTPUT_PATH = r"D:\lowfieldPipeline\inputs\wiener_deconvolved.nii.gz"

# 🔥 STRICT: mild sharpening
SIGMA_INPLANE = 0.8
BALANCE       = 0.25

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
        hf_snr = compute_snr(nib.load(hf_path).get_fdata())
        print(f"Reference HF SNR: {hf_snr:.2f}")
    else:
        hf_snr = None

    try:
        nii = nib.load(INPUT_PATH)
        data = nii.get_fdata().astype(np.float32)
        affine, header = nii.affine, nii.header

        print(f"Wiener Deconvolution: sigma={SIGMA_INPLANE}, balance={BALANCE}")
        
        # PSF
        psf_size = 11
        ax = np.arange(psf_size) - psf_size // 2
        xx, yy = np.meshgrid(ax, ax)
        psf = np.exp(-(xx**2 + yy**2) / (2.0 * SIGMA_INPLANE**2))
        psf /= psf.sum()

        output = np.zeros_like(data)
        for i in range(data.shape[2]):
            sl = data[:, :, i]
            if np.mean(sl) < 0.05:
                output[:, :, i] = sl
                continue
            dec = wiener(sl, psf, balance=BALANCE)
            output[:, :, i] = np.clip(dec, 0, 1)

        # 🧪 ENERGY PRESERVATION
        orig_energy = np.mean(data**2)
        new_energy  = np.mean(output**2)
        rescaled = output * np.sqrt(orig_energy / (new_energy + 1e-8))
        
        # 🧪 STRICT SNR CAP (SCALING ONLY, NO LOOPS)
        if hf_snr is not None:
            current_snr = compute_snr(rescaled)
            limit = 0.85 * hf_snr
            print(f"Post-Wiener SNR: {current_snr:.2f} (Phys. Limit: {limit:.2f})")
            
            if current_snr > limit:
                print(f"  Warning: SNR Overshoot. Scaling down to {limit:.2f}...")
                rescaled *= (limit / (current_snr + 1e-8))
                print(f"  New Scaled SNR: {compute_snr(rescaled):.2f}")

        # Save
        nib.save(nib.Nifti1Image(rescaled.astype(np.float32), affine, header), OUTPUT_PATH)
        print(f"SUCCESS: {OUTPUT_PATH}")

    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()
