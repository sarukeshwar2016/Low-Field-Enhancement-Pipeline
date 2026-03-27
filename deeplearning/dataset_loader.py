# ============================================================
# DATASET LOADER: 3D NIfTI → 2D Slices for Deep Learning
# ============================================================
import os
import glob
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader

LF_DIR = r"D:\01_MRI_Data\nifti_output\low_field_simulated"
HF_DIR = r"D:\01_MRI_Data\nifti_output\high_field_nifti"

# Patients to exclude (same as filtered report)
EXCLUDE = {"0001", "0007", "0008", "0017", "0039", "0060"}


def align_hf_to_lf(lf_path, hf_path):
    """Resample HF volume to LF grid so slices are pixel-aligned."""
    lf_sitk = sitk.ReadImage(lf_path, sitk.sitkFloat32)
    hf_sitk = sitk.ReadImage(hf_path, sitk.sitkFloat32)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(lf_sitk)
    resampler.SetInterpolator(sitk.sitkLinear)
    hf_aligned = resampler.Execute(hf_sitk)
    return (
        np.transpose(sitk.GetArrayFromImage(lf_sitk), (2, 1, 0)).astype(np.float32),
        np.transpose(sitk.GetArrayFromImage(hf_aligned), (2, 1, 0)).astype(np.float32),
    )


def extract_all_slices(train_ratio=0.7):
    """
    Walk through all LF/HF pairs, extract axial slices,
    normalize to [0,1], and split into train/test.
    """
    lf_files = sorted(glob.glob(os.path.join(LF_DIR, "*_LF.nii.gz")))
    all_slices = []

    for lf_path in lf_files:
        pid = os.path.basename(lf_path)[:4]
        if pid in EXCLUDE:
            continue
        hf_path = os.path.join(HF_DIR, f"{pid}_T2SAG.nii.gz")
        if not os.path.exists(hf_path):
            continue

        try:
            lf_vol, hf_vol = align_hf_to_lf(lf_path, hf_path)
        except Exception as e:
            print(f"  ✗ {pid}: {e}")
            continue

        n_slices = lf_vol.shape[2]
        for i in range(n_slices):
            lf_slice = lf_vol[:, :, i]
            hf_slice = hf_vol[:, :, i]

            # Skip near-empty slices
            if np.mean(lf_slice) < 5:
                continue

            # Normalize to [0, 1]
            lf_min, lf_max = lf_slice.min(), lf_slice.max()
            hf_min, hf_max = hf_slice.min(), hf_slice.max()
            lf_norm = (lf_slice - lf_min) / (lf_max - lf_min + 1e-8)
            hf_norm = (hf_slice - hf_min) / (hf_max - hf_min + 1e-8)

            all_slices.append((lf_norm, hf_norm, pid, i))

        print(f"  ✔ {pid} ({n_slices} slices)")

    # Shuffle and split
    np.random.seed(42)
    np.random.shuffle(all_slices)
    split = int(len(all_slices) * train_ratio)
    train_slices = all_slices[:split]
    test_slices = all_slices[split:]

    print(f"\nTotal slices: {len(all_slices)} | Train: {len(train_slices)} | Test: {len(test_slices)}")
    return train_slices, test_slices


class MRISliceDataset(Dataset):
    """PyTorch Dataset wrapper for 2D MRI slices."""

    def __init__(self, slices_list):
        self.slices = slices_list

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        lf, hf, pid, slice_idx = self.slices[idx]
        # Add channel dimension: (1, H, W)
        lf_tensor = torch.from_numpy(lf[None, ...]).float()
        hf_tensor = torch.from_numpy(hf[None, ...]).float()
        return lf_tensor, hf_tensor, pid, slice_idx


def get_dataloaders(batch_size=4):
    """Build train and test DataLoaders."""
    train_slices, test_slices = extract_all_slices()
    train_ds = MRISliceDataset(train_slices)
    test_ds = MRISliceDataset(test_slices)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders()
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    for lf, hf, pid, si in train_loader:
        print(f"  Batch LF shape: {lf.shape}, HF shape: {hf.shape}")
        break
