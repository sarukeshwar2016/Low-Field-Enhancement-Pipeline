# Low-Field MRI Enhancement Pipeline

A physics-driven, research-grade pipeline for simulating and enhancing
low-field MRI scans from high-field DICOM data.
Built for lumbar spine imaging research.

---

## Pipeline Overview

The system operates in two main phases:

**Phase A - Simulation**
```
DICOM data  -->  NIfTI (high-field)  -->  Low-field simulation
                                          * Voxel resampling (1.5 x 1.5 x 3.0 mm)
                                          * Gaussian PSF blur
                                          * Rician noise injection
                                          * SNR enforcement (LF < HF guaranteed)
```

**Phase B - Enhancement**
```
Low-field NIfTI  -->  [Stage 1] N4 Bias Field Correction
                 -->  [Stage 2] Intensity Standardization
                 -->  [Stage 3] Wiener Deconvolution
                 -->  [Stage 4] Resolution Modeling
                 -->  [Stage 5] Structural Refinement
                 -->  [Stage 6] SNR-Constrained Scaling
                 -->  Enhanced NIfTI output
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Simulate low-field data from DICOM
python dicom_to_lf_sim.py

# 3. Run enhancement on batch
python enhanced_batch_9_100.py

# 4. Validate outputs
python validate_outputs.py

# 5. Export metrics to CSV
python export_csv.py
```

---

## Project Structure

```
lowfieldPipeline/
|
|-- Core Pipeline Scripts
|   |-- dicom_to_lf_sim.py           DICOM-to-NIfTI + LF simulation
|   |-- enhanced_batch_9_100.py      6-stage enhancement pipeline (batch)
|   |-- linear_pipeline_clean.py     Single-patient pipeline
|   |-- run_batch_fully_corrected.py Full corrected batch orchestrator
|   |-- run_batch_spine_stable.py    Spine-optimized batch runner
|   |-- simulate_batch_9_100.py      Batch simulation runner
|   |-- research_pipeline_updated.py Research variant with DL comparison
|
|-- Configuration and Utilities
|   |-- config.py                    All pipeline parameters (edit here)
|   |-- utils.py                     Shared SNR, metrics, image functions
|   |-- compare_mri_metadata.py      NIfTI metadata comparison tool
|
|-- Analysis and Reporting
|   |-- generate_plots.py            PSNR/SSIM visualization charts
|   |-- generate_report2.py          Comparative metrics report generator
|   |-- pipeline_stage_report.py     Per-stage statistics reporter
|   |-- validate_outputs.py          Output integrity checker
|   |-- summarize_results.py         Batch metrics summary printer
|   |-- export_csv.py                Metrics to CSV exporter
|   |-- check_snr_constraints.py     Physical SNR bounds verifier
|   |-- batch_stats.py               Batch-level statistics and outlier report
|
|-- Package and Build
|   |-- pipeline/                    Pipeline as importable Python package
|   |   |-- __init__.py
|   |   |-- config.py
|   |   |-- utils.py
|   |-- setup.py                     Installable package setup
|   |-- requirements.txt             Python dependencies
|
|-- Tests
|   |-- tests/
|   |   |-- test_snr.py              Unit tests for compute_snr
|   |   |-- test_metrics.py          Unit tests for PSNR, SSIM, hist overlap
|   |   |-- dncnn.py                 DnCNN denoising model
|   |   |-- unet.py                  U-Net segmentation model
|   |   |-- train.py                 Training script
|   |   |-- evaluate.py              Evaluation script
|   |   |-- dataset_loader.py        NIfTI dataset loader
|   |   |-- compare.py               DL vs classical comparison
|
|-- Data (not versioned - see .gitignore)
|   |-- inputs/                      Input NIfTI files
|   |-- outputs/                     Enhanced output NIfTI files
|
|-- Documentation
    |-- project_docs/                Architecture, Functional Spec, Test Cases
    |-- doc_images/                  Pipeline architecture diagrams
    |-- RESULTS.md                   Evaluation metrics summary
    |-- CHANGELOG.md                 Version history
```

---

## Results

| Metric    | Low-Field Input      | Enhanced Output      | Change     |
|-----------|----------------------|----------------------|------------|
| PSNR (dB) | 23.64 +/- 1.98       | 23.36 +/- 1.77       | Maintained |
| SSIM      | 0.5801 +/- 0.0805    | 0.7342 +/- 0.0528    | +26.6%     |

> **86 patients processed.** SSIM improved by **+26.6%** on average.
> Physical constraint satisfied: `0.7 * HF_SNR <= Enhanced_SNR <= 0.9 * HF_SNR`

---

## Documentation

Full project documentation in `project_docs/`:

- Architecture Document
- Functional Specification
- Test Cases
- Sprint Retrospective

---

## License

MIT License - see `LICENSE` for details.
