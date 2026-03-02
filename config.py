"""
config.py
=========
Central configuration for the Low-Field MRI Enhancement Pipeline.
Edit this file to change data paths and pipeline hyperparameters.
"""

# ---------------------------------------------------------------
# DATA DIRECTORIES
# ---------------------------------------------------------------
DICOM_ROOT_DIR  = r"D:\01_MRI_Data"
HF_NIFTI_DIR    = r"D:\01_MRI_Data\nifti_output\high_field_nifti"
LF_SIM_DIR      = r"D:\01_MRI_Data\nifti_output\low_field_simulated"
OUT_DIR         = r"D:\lowfieldPipeline\outputs"
REPORT_PATH     = r"D:\lowfieldPipeline\final_batch_report.txt"

# ---------------------------------------------------------------
# SIMULATION PARAMETERS
# ---------------------------------------------------------------
LF_SPACING              = (1.5, 1.5, 3.0)   # mm - realistic low-field voxel size
MAX_SUCCESS             = 20                 # Max patients per run
TARGET_SNR_RATIO_MIN    = 0.25              # LF SNR target lower bound (% of HF)
TARGET_SNR_RATIO_MAX    = 0.40              # LF SNR target upper bound (% of HF)
NOISE_DENOMINATOR       = 0.45              # Noise injection strength
PSF_BLUR_SIGMA          = [0.6, 0.6, 1.2]  # PSF Gaussian sigma (in-plane, through-plane)
MAX_SNR_CORRECTION_ITER = 10               # Safety loop iteration limit

# ---------------------------------------------------------------
# ENHANCEMENT PARAMETERS
# ---------------------------------------------------------------
N4_MAX_ITERATIONS   = [30, 30, 20, 10]  # N4 iterations per resolution level
OTSU_BINS           = 200              # Otsu thresholding bins for mask
MORPHO_RADIUS       = [3, 3, 3]        # Morphological closing radius (voxels)
PSF_SIZE            = 9               # Wiener PSF kernel size (px)
PSF_SIGMA           = 0.8             # Wiener PSF Gaussian sigma
WIENER_BALANCE      = 0.4             # Wiener regularization parameter
RESOLUTION_SIGMA    = [0.2, 0.2, 0.6] # Resolution model blur sigma
RESOLUTION_WEIGHT   = 0.15            # Resolution model blend weight
SHARPENING_SIGMA    = 0.5             # Unsharp mask sigma
SHARPENING_STRENGTH = 0.1             # Unsharp mask strength

# ---------------------------------------------------------------
# SNR CONSTRAINTS
# ---------------------------------------------------------------
SNR_TARGET_MIN_RATIO = 0.7   # Enhanced SNR >= 70% of HF SNR
SNR_TARGET_MAX_RATIO = 0.9   # Enhanced SNR <= 90% of HF SNR

# ---------------------------------------------------------------
# SCAN QUALITY FILTER
# ---------------------------------------------------------------
MIN_SLICES      = 10    # Reject localizers (too few slices)
MAX_SLICE_THICK = 6.0   # Reject ultra-thick scans (mm)
MIN_IMG_STD     = 20.0  # Reject flat/empty scans
