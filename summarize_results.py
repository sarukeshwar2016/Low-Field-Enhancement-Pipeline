"""
summarize_results.py
====================
Parses report2.txt and prints a clean performance summary.

Usage:
    python summarize_results.py
"""

import re

REPORT_FILE = "report2.txt"


def parse_report(path):
    patients = []
    with open(path) as f:
        for line in f:
            m = re.match(
                r'\s+(\d+)\s+\|.*?\|\s+([\d.]+)\s+([\d.]+)\s+\|\s+([\d.]+)\s+([\d.]+)', line)
            if m:
                patients.append({
                    "id":       m.group(1),
                    "lf_psnr":  float(m.group(2)),
                    "lf_ssim":  float(m.group(3)),
                    "enh_psnr": float(m.group(4)),
                    "enh_ssim": float(m.group(5)),
                })
    return patients


def avg(lst): return sum(lst) / len(lst) if lst else 0.0


def main():
    data = parse_report(REPORT_FILE)
    n = len(data)
    if not n:
        print("No data found in", REPORT_FILE)
        return
    lp  = [p["lf_psnr"]  for p in data]
    ep  = [p["enh_psnr"] for p in data]
    ls  = [p["lf_ssim"]  for p in data]
    es  = [p["enh_ssim"] for p in data]
    print(f"{'=' * 60}")
    print(f"  PIPELINE SUMMARY   ({n} patients)")
    print(f"{'=' * 60}")
    print(f"{'Metric':<22} {'LF Input':>10} {'Enhanced':>10} {'Delta':>10}")
    print(f"{'-' * 60}")
    print(f"{'PSNR mean (dB)':<22} {avg(lp):>10.2f} {avg(ep):>10.2f} {avg(ep)-avg(lp):>+10.2f}")
    print(f"{'SSIM mean':<22} {avg(ls):>10.4f} {avg(es):>10.4f} {(avg(es)-avg(ls))*100:>+9.1f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
