"""
export_csv.py
=============
Exports pipeline metrics from report2.txt to a CSV file.

Usage:
    python export_csv.py
    python export_csv.py --output my_metrics.csv
"""

import re
import csv
import argparse

REPORT_FILE  = "report2.txt"
DEFAULT_CSV  = "metrics_export.csv"


def parse_report(path):
    patients = []
    with open(path) as f:
        for line in f:
            m = re.match(
                r'\s+(\d+)\s+\|.*?\|\s+([\d.]+)\s+([\d.]+)\s+\|\s+([\d.]+)\s+([\d.]+)', line)
            if m:
                lp, ep = float(m.group(2)), float(m.group(4))
                ls, es = float(m.group(3)), float(m.group(5))
                patients.append({
                    "patient_id":    m.group(1),
                    "lf_psnr":       lp, "lf_ssim":       ls,
                    "enhanced_psnr": ep, "enhanced_ssim":  es,
                    "psnr_delta":    round(ep - lp, 4),
                    "ssim_delta":    round(es - ls, 4),
                })
    return patients


def main():
    ap = argparse.ArgumentParser(description="Export pipeline metrics to CSV")
    ap.add_argument("--input",  default=REPORT_FILE)
    ap.add_argument("--output", default=DEFAULT_CSV)
    args = ap.parse_args()

    data = parse_report(args.input)
    if not data:
        print("No data found."); return

    cols = ["patient_id", "lf_psnr", "lf_ssim",
            "enhanced_psnr", "enhanced_ssim", "psnr_delta", "ssim_delta"]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(data)
    print(f"Exported {len(data)} patients -> {args.output}")


if __name__ == "__main__":
    main()
