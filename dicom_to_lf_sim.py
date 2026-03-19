import os
import glob
import time
import shutil
import numpy as np
import nibabel as nib
import dicom2nifti
from nibabel.processing import resample_to_output
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from scipy.stats import skew as compute_skew

# ==================================================
# SETTINGS
# ==================================================
DICOM_ROOT_DIR = r"D:\01_MRI_Data"
HF_NIFTI_DIR   = r"D:\01_MRI_Data\nifti_output\high_field_nifti"
LF_SIM_DIR     = r"D:\01_MRI_Data\nifti_output\low_field_simulated"

LF_SPACING     = (1.6, 1.6, 5.0)
SIGMA_XY_INIT  = 0.8
SIGMA_Z_INIT   = 1.8
NOISE_STD_INIT = 0.05

MAX_SUCCESS = 20

# ==================================================
# ✅ ROBUST SNR (FINAL FIX)
# ==================================================
def compute_snr(img):
    img = img.astype(np.float32)

    non_zero = img[img > 0]
    if len(non_zero) < 100:
        return np.nan

    p20 = np.percentile(non_zero, 20)
    p80 = np.percentile(non_zero, 80)

    noise_region  = non_zero[non_zero <= p20]
    signal_region = non_zero[non_zero >= p80]

    if len(noise_region) < 50 or len(signal_region) < 50:
        return np.nan

    noise_std = np.std(noise_region)
    if noise_std < 1e-6:
        return np.nan

    return np.mean(signal_region) / (noise_std + 1e-8)

# ==================================================
# VALID SCAN CHECK
# ==================================================
def is_valid_scan(img, spacing):
    return min(img.shape) >= 10 and max(spacing) < 10

# ==================================================
# DICOM -> NIFTI
# ==================================================
def convert_dicom_to_nifti(dicom_dir, out_dir, patient_id):
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = os.path.join(out_dir, "temp_" + patient_id)
    os.makedirs(temp_dir, exist_ok=True)

    try:
        dicom2nifti.convert_directory(dicom_dir, temp_dir, compression=True, reorient=True)
        files = glob.glob(os.path.join(temp_dir, "*.nii.gz"))

        if not files:
            raise RuntimeError("No NIfTI produced")

        final_path = os.path.join(out_dir, f"{patient_id}_HF.nii.gz")
        shutil.move(files[0], final_path)
        shutil.rmtree(temp_dir)
        return final_path

    except:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None

# ==================================================
# LOW-FIELD SIMULATION
# ==================================================
def simulate_low_field(hf_path, out_dir, patient_id):
    os.makedirs(out_dir, exist_ok=True)
    final_path = os.path.join(out_dir, f"{patient_id}_LF.nii.gz")

    try:
        nii_hf = nib.load(hf_path)
        img_hf = nii_hf.get_fdata().astype(np.float32)
        spacing = nii_hf.header.get_zooms()[:3]

        if not is_valid_scan(img_hf, spacing):
            print("    Skipping invalid/localizer scan")
            return None, None, None, None, None

        hf_snr = compute_snr(img_hf)

        print(f"    HF: shape={img_hf.shape}, spacing={tuple(round(float(s),2) for s in spacing)}, SNR={hf_snr:.2f}")

        # RESAMPLE
        nii_res = resample_to_output(nii_hf, voxel_sizes=LF_SPACING)
        img_res = nii_res.get_fdata().astype(np.float32)
        affine  = nii_res.affine
        header  = nii_res.header.copy()

        print(f"    LF: shape={img_res.shape}")

        # TARGETS
        mask = img_hf > np.percentile(img_hf, 10)
        target_mean = np.mean(img_hf[mask]) * 0.97
        target_std  = np.std(img_hf[mask])  * 0.88
        target_skew = compute_skew(img_hf[mask].flatten()) * 0.91

        body_mask = img_res > np.percentile(img_res, 10)

        # SIMULATION FUNCTION
        def apply(params):
            sx, sz, noise = params
            noise = max(0.03, min(noise, 0.3))

            blurred = gaussian_filter(img_res, [sx, sx, sz])

            scale = noise * np.percentile(blurred[body_mask], 99)
            n1 = np.random.normal(0, scale, blurred.shape)
            n2 = np.random.normal(0, scale, blurred.shape)

            return np.sqrt((blurred + n1)**2 + n2**2)

        # LOSS FUNCTION (NO WRONG SNR FORCE)
        def loss(p):
            sim = apply(p)
            vals = sim[body_mask]

            lm = abs(np.mean(vals) - target_mean) / (target_mean + 1e-8)
            ls = abs(np.std(vals)  - target_std)  / (target_std  + 1e-8)
            lk = abs(compute_skew(vals.flatten()) - target_skew) / (abs(target_skew) + 1e-8)

            return lm + ls + 0.5 * lk

        print("    Optimizing...")
        res = minimize(loss,
                       [SIGMA_XY_INIT, SIGMA_Z_INIT, NOISE_STD_INIT],
                       method="Nelder-Mead",
                       options={"maxiter": 200})

        best = res.x
        print(f"    Params: sx={best[0]:.3f}, sz={best[1]:.3f}, noise={best[2]:.4f}")

        # APPLY FINAL SIMULATION
        np.random.seed(0)
        sim = apply(best)

        # RESCALE
        vals = sim[body_mask]
        img_final = (sim - np.mean(vals)) / (np.std(vals) + 1e-8) * target_std + target_mean

        # FINAL NOISE (IMPORTANT)
        noise = np.random.normal(0, 0.05 * np.max(img_final), img_final.shape)
        img_final = np.sqrt((img_final + noise)**2 + noise**2)

        img_final = np.clip(img_final, 0, None)

        lf_snr = compute_snr(img_final)

        print(f"    Output: mean={np.mean(img_final[body_mask]):.2f}, std={np.std(img_final[body_mask]):.2f}, SNR={lf_snr:.2f}")

        # SAVE
        nib.save(nib.Nifti1Image(img_final.astype(np.float32), affine, header), final_path)
        print(f"    Saved: {os.path.basename(final_path)}")

        return final_path, nii_hf, img_hf, nib.load(final_path), img_final

    except Exception as e:
        print(f"    ERROR: {e}")
        return None, None, None, None, None

# ==================================================
# MAIN LOOP
# ==================================================
if __name__ == "__main__":
    print("=" * 70)
    print("FINAL LOW-FIELD MRI PIPELINE (SNR CORRECTED)")
    print("=" * 70)

    folders = []
    for root, _, files in os.walk(DICOM_ROOT_DIR):
        if any(f.endswith((".dcm", ".ima")) for f in files):
            folders.append(root)

    print(f"Found {len(folders)} DICOM folders\n")

    success = 0
    start = time.time()

    for i, d in enumerate(folders):
        if success >= MAX_SUCCESS:
            break

        pid = d.replace("\\", "/").split("/")[-3]

        print(f"\n[{i+1}] Processing {pid}")

        hf = convert_dicom_to_nifti(d, HF_NIFTI_DIR, pid)
        if not hf:
            continue

        result = simulate_low_field(hf, LF_SIM_DIR, pid)
        if result[0]:
            success += 1

        print("-" * 50)

    print("\n" + "=" * 70)
    print("DONE")
    print(f"Success: {success}")
    print(f"Time: {(time.time()-start)/60:.1f} min")
    print("=" * 70)