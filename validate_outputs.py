"""
validate_outputs.py
===================
Validates all enhanced NIfTI outputs in the outputs/ directory.
Checks file integrity, volume shape, and SNR feasibility.

Usage:
    python validate_outputs.py
"""

import os
import glob
import numpy as np
import nibabel as nib
from utils import compute_snr

OUT_DIR = r"D:\lowfieldPipeline\outputs"


def validate_nifti(path: str) -> dict:
    """Load a NIfTI file and return quality stats, or an error entry."""
    try:
        nii = nib.load(path)
        img = nii.get_fdata().astype(np.float32)
        return {
            "file":   os.path.basename(path),
            "shape":  img.shape,
            "snr":    round(compute_snr(img), 2),
            "mean":   round(float(np.mean(img[img > 0])), 2),
            "status": "OK",
        }
    except Exception as e:
        return {"file": os.path.basename(path), "status": f"ERROR: {e}"}


def main():
    files = sorted(glob.glob(os.path.join(OUT_DIR, "*_enhanced.nii.gz")))
    print(f"Validating {len(files)} files in {OUT_DIR}\n")
    errors = 0
    for f in files:
        r = validate_nifti(f)
        if r["status"] == "OK":
            print(f"  OK   {r['file']:40s} shape={r['shape']}  SNR={r['snr']}")
        else:
            print(f"  ERR  {r['file']} -- {r['status']}")
            errors += 1
    print(f"\nTotal={len(files)}  OK={len(files)-errors}  Errors={errors}")


if __name__ == "__main__":
    main()
