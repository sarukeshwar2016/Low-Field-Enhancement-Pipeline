"""
batch_stats.py
==============
Computes batch-level statistics from final_batch_report.txt.
Reports PSNR / SSIM distribution and flags top / bottom performers.

Usage:
    python batch_stats.py
"""

import re

REPORT_FILE = "final_batch_report.txt"


def parse(path):
    patients, cur = [], {}
    with open(path) as f:
        for line in f:
            pm = re.match(r'PATIENT (\d+)', line)
            if pm:
                if cur and "psnr" in cur: patients.append(cur)
                cur = {"id": pm.group(1)}
            mm = re.search(r'PSNR=([\d.]+),\s*SSIM=([\d.]+)', line)
            if mm:
                cur["psnr"] = float(mm.group(1))
                cur["ssim"] = float(mm.group(2))
    if cur and "psnr" in cur: patients.append(cur)
    return patients


def stat(lst):
    n = len(lst)
    mu = sum(lst) / n
    sd = (sum((x - mu) ** 2 for x in lst) / n) ** 0.5
    return mu, sd, min(lst), max(lst)


def main():
    data = parse(REPORT_FILE)
    psnrs = [p["psnr"] for p in data]
    ssims = [p["ssim"] for p in data]
    print(f"Batch Statistics  |  {len(data)} patients")
    print("=" * 55)
    for label, vals in [("PSNR (dB)", psnrs), ("SSIM", ssims)]:
        mu, sd, lo, hi = stat(vals)
        print(f"  {label:<12}  mean={mu:.4f}  std={sd:.4f}  "
              f"min={lo:.4f}  max={hi:.4f}")
    print("\nTop 5 by SSIM:")
    for p in sorted(data, key=lambda x: x["ssim"], reverse=True)[:5]:
        print(f"  Patient {p['id']}  SSIM={p['ssim']:.4f}  PSNR={p['psnr']:.2f}")
    print("\nBottom 5 by SSIM:")
    for p in sorted(data, key=lambda x: x["ssim"])[:5]:
        print(f"  Patient {p['id']}  SSIM={p['ssim']:.4f}  PSNR={p['psnr']:.2f}")


if __name__ == "__main__":
    main()
