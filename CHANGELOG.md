# Changelog

---

## [3.0.0] - 2026-04

### Added
- Full batch processing for 86 patients (patients 2 to 112)
- final_batch_report.txt with per-stage Mean / Std / Skew / SNR statistics
- report2.txt comparative PSNR / SSIM / Hist table for all patients
- config.py centralizing all pipeline parameters
- utils.py with shared compute_snr, compute_psnr, compute_ssim_volumetric
- README.md, RESULTS.md, CHANGELOG.md project documentation
- requirements.txt for reproducible installation
- setup.py for installable package with CLI entry points
- pipeline/ package with importable modules
- Unit tests: tests/test_snr.py, tests/test_metrics.py
- validate_outputs.py, summarize_results.py, export_csv.py
- check_snr_constraints.py, batch_stats.py utility scripts

### Changed
- 6-stage enhancement pipeline replacing 4-stage prototype
- Histogram Overlap added alongside PSNR and SSIM
- N4 bias correction upgraded with morphological mask cleaning
- Wiener reconstruction includes SNR safety clamping

### Removed
- Legacy batch scripts (enhance_batch_61_112, enhance_new_batch_50)
- Duplicate documents from root directory
- Old partial batch result folders (batch_enhanced, batch_enhanced_final)
- One-off utility scripts (inspect_ppt, filter_report, rebuild_*)

---

## [2.0.0] - 2026-03

### Added
- Batch processing for patients 1 to 8
- Research pipeline with BM3D denoising variant
- Architecture diagrams in doc_images/

### Changed
- Switched from additive Gaussian to Rician noise simulation
- SNR strict enforcement loop added

---

## [1.0.0] - 2026-02

### Added
- Initial DICOM to NIfTI conversion
- Basic low-field simulation (Gaussian blur + noise)
- Single-patient pipeline (linear_pipeline_clean.py)
- Deep learning baseline models in deeplearning/
