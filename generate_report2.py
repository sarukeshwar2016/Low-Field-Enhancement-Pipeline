# ============================================================
# GENERATE report2.txt — PSNR & SSIM for HF, LF, Enhanced
# ============================================================
import os
import re
import glob
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from skimage.metrics import structural_similarity as ssim

LF_DIR  = r"D:\01_MRI_Data\nifti_output\low_field_simulated"
HF_DIR  = r"D:\01_MRI_Data\nifti_output\high_field_nifti"
ENH_DIR = r"D:\lowfieldPipeline\outputs"
REPORT_IN  = r"D:\lowfieldPipeline\report.txt"
REPORT_OUT = r"D:\lowfieldPipeline\report2.txt"

# Get filtered patient list from report.txt
def get_patients_from_report():
    with open(REPORT_IN, "r") as f:
        content = f.read()
    return sorted(set(re.findall(r"PATIENT (\d{4})", content)))

def compute_psnr(ref, img):
    mse = np.mean((ref - img) ** 2)
    if mse < 1e-10: return 999.99
    return 20 * np.log10(np.max(ref) / np.sqrt(mse))

def compute_ssim_vol(ref, img):
    scores = []
    for i in range(ref.shape[2]):
        r, t = ref[:,:,i], img[:,:,i]
        if np.std(r) < 1e-6 or np.std(t) < 1e-6: continue
        scores.append(ssim(r, t, data_range=r.max()-r.min()))
    return np.mean(scores) if scores else 0.0

def align_hf_to_lf(lf_path, hf_path):
    hf_sitk = sitk.ReadImage(hf_path, sitk.sitkFloat32)
    lf_sitk = sitk.ReadImage(lf_path, sitk.sitkFloat32)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(lf_sitk)
    resampler.SetInterpolator(sitk.sitkLinear)
    return np.transpose(sitk.GetArrayFromImage(resampler.Execute(hf_sitk)), (2,1,0)).astype(np.float32)

def main():
    patients = get_patients_from_report()
    print(f"Processing {len(patients)} patients...")

    lines = []
    lines.append("COMPARATIVE METRICS REPORT (PSNR & SSIM)")
    lines.append("=" * 70)
    lines.append(f"{'Patient':>8s} | {'HF PSNR':>9s} {'HF SSIM':>9s} | {'LF PSNR':>9s} {'LF SSIM':>9s} | {'Enh PSNR':>9s} {'Enh SSIM':>9s}")
    lines.append("-" * 70)

    summary = []

    for pid in patients:
        lf_path  = os.path.join(LF_DIR, f"{pid}_LF.nii.gz")
        hf_path  = os.path.join(HF_DIR, f"{pid}_T2SAG.nii.gz")
        enh_path = os.path.join(ENH_DIR, f"{pid}_enhanced.nii.gz")

        if not all(os.path.exists(p) for p in [lf_path, hf_path, enh_path]):
            print(f"  ✗ {pid}: missing files")
            continue

        try:
            lf  = nib.load(lf_path).get_fdata().astype(np.float32)
            enh = nib.load(enh_path).get_fdata().astype(np.float32)
            hf_aligned = align_hf_to_lf(lf_path, hf_path)

            # HF vs HF (reference — perfect scores)
            hf_psnr = compute_psnr(hf_aligned, hf_aligned)
            hf_ssim = compute_ssim_vol(hf_aligned, hf_aligned)

            # LF vs HF
            lf_psnr = compute_psnr(hf_aligned, lf)
            lf_ssim = compute_ssim_vol(hf_aligned, lf)

            # Enhanced vs HF
            enh_psnr = compute_psnr(hf_aligned, enh)
            enh_ssim = compute_ssim_vol(hf_aligned, enh)

            lines.append(f"{pid:>8s} | {hf_psnr:9.2f} {hf_ssim:9.4f} | {lf_psnr:9.2f} {lf_ssim:9.4f} | {enh_psnr:9.2f} {enh_ssim:9.4f}")
            summary.append((pid, hf_psnr, hf_ssim, lf_psnr, lf_ssim, enh_psnr, enh_ssim))
            print(f"  ✔ {pid}")

        except Exception as e:
            print(f"  ✗ {pid}: {e}")

    # Averages
    if summary:
        arr = np.array([(s[3], s[4], s[5], s[6]) for s in summary])
        lines.append("-" * 70)
        lines.append(f"{'AVERAGE':>8s} | {'REF':>9s} {'REF':>9s} | {arr[:,0].mean():9.2f} {arr[:,1].mean():9.4f} | {arr[:,2].mean():9.2f} {arr[:,3].mean():9.4f}")
        lines.append(f"{'STD':>8s} | {'':>9s} {'':>9s} | {arr[:,0].std():9.2f} {arr[:,1].std():9.4f} | {arr[:,2].std():9.2f} {arr[:,3].std():9.4f}")

    lines.append("=" * 70)
    lines.append("HF = High-Field Original (Ground Truth Reference)")
    lines.append("LF = Low-Field Simulated (Input)")
    lines.append("Enh = Pipeline Enhanced (Output)")

    with open(REPORT_OUT, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n✔ Report saved to {REPORT_OUT}")

if __name__ == "__main__":
    main()
