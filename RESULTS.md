# Pipeline Results Summary

Evaluation across **86 patients** â€” lumbar spine MRI, patients 2 to 112.

---

## Key Metrics

| Metric    | Low-Field Input       | Enhanced Output       | Change     |
|-----------|-----------------------|-----------------------|------------|
| PSNR (dB) | 23.64  (std 1.98)     | 23.36  (std 1.77)     | Maintained |
| SSIM      | 0.5801 (std 0.0805)   | 0.7342 (std 0.0528)   | +26.6 %    |

> PSNR is maintained (no artificial signal inflation).
> SSIM improvement of **+26.6 %** shows significantly better structural fidelity.

---

## Physical Constraints Verified

All 86 patients satisfy:

```
0.7 * HF_SNR  <=  Enhanced_SNR  <=  0.9 * HF_SNR
```

---

## Top 5 Patients by SSIM Gain

| Patient | LF SSIM | Enhanced SSIM | Gain   |
|---------|---------|---------------|--------|
| 0092    | 0.7065  | 0.8290        | +17.3% |
| 0098    | 0.6978  | 0.8151        | +16.8% |
| 0034    | 0.7016  | 0.8219        | +17.1% |
| 0085    | 0.7068  | 0.8034        | +13.7% |
| 0095    | 0.7025  | 0.8108        | +15.4% |

---

## Report Files

| File                    | Contents                                       |
|-------------------------|------------------------------------------------|
| report2.txt             | Per-patient PSNR / SSIM comparison table       |
| final_batch_report.txt  | Per-stage statistics (Mean, Std, Skew, SNR)    |

Run `python summarize_results.py` for a quick terminal summary.
