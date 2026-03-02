"""
check_snr_constraints.py
========================
Verifies all enhanced outputs satisfy the physical SNR constraint:
    0.7 * HF_SNR  <=  Enhanced_SNR  <=  0.9 * HF_SNR

Parses final_batch_report.txt and flags any violations.

Usage:
    python check_snr_constraints.py
"""

import re

REPORT_FILE      = "final_batch_report.txt"
SNR_MIN_RATIO    = 0.7
SNR_MAX_RATIO    = 0.9


def parse_snr(path):
    records, cur = [], {}
    with open(path) as f:
        for line in f:
            pm = re.match(r'PATIENT (\d+)', line)
            if pm:
                if cur: records.append(cur)
                cur = {"id": pm.group(1)}
            hf = re.search(r'HF Original.*SNR=\s*([\d.]+)', line)
            if hf: cur["hf"] = float(hf.group(1))
            en = re.search(r'Final Enhanced.*SNR=\s*([\d.]+)', line)
            if en: cur["en"] = float(en.group(1))
    if cur: records.append(cur)
    return [r for r in records if "hf" in r and "en" in r]


def main():
    data = parse_snr(REPORT_FILE)
    print(f"Checking SNR constraints for {len(data)} patients...\n")
    violations = []
    for p in data:
        ratio = p["en"] / (p["hf"] + 1e-8)
        if not (SNR_MIN_RATIO <= ratio <= SNR_MAX_RATIO):
            violations.append({**p, "ratio": ratio})
    if not violations:
        print(f"  All {len(data)} patients PASS  ({SNR_MIN_RATIO} <= ratio <= {SNR_MAX_RATIO})")
    else:
        print(f"  VIOLATIONS  ({len(violations)} / {len(data)}):")
        for v in violations:
            print(f"    Patient {v['id']:6s}  ratio={v['ratio']:.3f}"
                  f"  HF={v['hf']:.1f}  Enhanced={v['en']:.1f}")


if __name__ == "__main__":
    main()
