import os
import re

REPORT_IN = r"D:\lowfieldPipeline\final_batch_report.txt"
REPORT_OUT = r"D:\lowfieldPipeline\report.txt"

EXCLUDE_PIDS = {"0001", "0007", "0008", "0017", "0039", "0060"}

def filter_report():
    if not os.path.exists(REPORT_IN):
        print("Source report not found.")
        return

    with open(REPORT_IN, "r") as f:
        content = f.read()

    # Split by PATIENT blocks
    # We want to keep the header if it exists
    parts = re.split(r"(PATIENT \d{4})", content)
    
    header = parts[0]
    out_content = header.strip() + "\n"
    
    summary_data = []

    # Process PATIENT blocks
    for i in range(1, len(parts), 2):
        pid_full = parts[i]
        pid = pid_full.split()[-1]
        block = parts[i+1]
        
        # Check if it's the final summary at the end of the last block
        summary_split = re.split(r"FINAL SUMMARY", block)
        main_block = summary_split[0]
        
        if pid not in EXCLUDE_PIDS:
            # Remove Hist metric: PSNR=21.52, SSIM=0.7372, Hist=0.1391 -> PSNR=21.52, SSIM=0.7372
            main_block_filtered = re.sub(r", Hist=[\d\.]+", "", main_block)
            out_content += "\n" + pid_full + main_block_filtered
            
            # Extract summary stats for this PID if we find them in the block
            # (Though it might be easier to just parse the FINAL SUMMARY section later)

    # Process Final Summary
    # Find the last summary section
    summary_match = re.search(r"FINAL SUMMARY\n=+\n(.*)", content, re.DOTALL)
    if summary_match:
        summary_lines = summary_match.group(1).strip().split("\n")
        out_content += "\n\nFINAL SUMMARY\n" + "="*40 + "\n"
        for line in summary_lines:
            if "|" in line:
                pid = line.split("|")[0].strip()
                if pid not in EXCLUDE_PIDS:
                    # Remove Hist if it's in the summary line (usually it's just SNR, PSNR, SSIM)
                    line_filtered = re.sub(r" \| Hist=[\d\.]+", "", line)
                    out_content += line_filtered + "\n"

    with open(REPORT_OUT, "w") as f:
        f.write(out_content.strip() + "\n")

    print(f"Filtered report written to {REPORT_OUT}")

if __name__ == "__main__":
    filter_report()
