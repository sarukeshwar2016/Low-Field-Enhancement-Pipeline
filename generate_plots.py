import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re

# 1. Raw Data Extraction
raw_data = """
PATIENT 0002 | HF Original SNR= 55.02 | LF Simulated SNR= 29.29 | Final Enhanced SNR= 45.88
PATIENT 0003 | HF Original SNR= 38.81 | LF Simulated SNR= 20.60 | Final Enhanced SNR= 31.20
PATIENT 0004 | HF Original SNR= 63.78 | LF Simulated SNR= 38.34 | Final Enhanced SNR= 49.71
PATIENT 0005 | HF Original SNR= 44.89 | LF Simulated SNR= 29.02 | Final Enhanced SNR= 38.41
PATIENT 0006 | HF Original SNR= 44.46 | LF Simulated SNR= 25.55 | Final Enhanced SNR= 32.64
PATIENT 0009 | HF Original SNR= 86.60 | LF Simulated SNR= 52.73 | Final Enhanced SNR= 86.50
PATIENT 0010 | HF Original SNR= 78.26 | LF Simulated SNR= 53.42 | Final Enhanced SNR= 86.48
PATIENT 0011 | HF Original SNR= 46.67 | LF Simulated SNR= 25.01 | Final Enhanced SNR= 28.10
PATIENT 0012 | HF Original SNR= 43.88 | LF Simulated SNR= 27.69 | Final Enhanced SNR= 34.89
PATIENT 0013 | HF Original SNR= 53.26 | LF Simulated SNR= 27.60 | Final Enhanced SNR= 43.13
PATIENT 0014 | HF Original SNR= 63.19 | LF Simulated SNR= 36.57 | Final Enhanced SNR= 47.69
PATIENT 0015 | HF Original SNR= 50.08 | LF Simulated SNR= 32.38 | Final Enhanced SNR= 45.07
PATIENT 0016 | HF Original SNR= 62.67 | LF Simulated SNR= 40.30 | Final Enhanced SNR= 53.01
PATIENT 0019 | HF Original SNR= 52.73 | LF Simulated SNR= 32.72 | Final Enhanced SNR= 50.17
PATIENT 0021 | HF Original SNR= 54.33 | LF Simulated SNR= 26.62 | Final Enhanced SNR= 31.07
PATIENT 0022 | HF Original SNR= 64.38 | LF Simulated SNR= 29.51 | Final Enhanced SNR= 43.91
PATIENT 0023 | HF Original SNR= 55.41 | LF Simulated SNR= 36.97 | Final Enhanced SNR= 58.34
PATIENT 0024 | HF Original SNR= 40.81 | LF Simulated SNR= 24.90 | Final Enhanced SNR= 31.43
PATIENT 0025 | HF Original SNR= 63.61 | LF Simulated SNR= 32.64 | Final Enhanced SNR= 44.07
PATIENT 0026 | HF Original SNR= 66.99 | LF Simulated SNR= 37.91 | Final Enhanced SNR= 51.37
PATIENT 0029 | HF Original SNR= 71.53 | LF Simulated SNR= 37.56 | Final Enhanced SNR= 62.22
PATIENT 0030 | HF Original SNR= 63.04 | LF Simulated SNR= 34.65 | Final Enhanced SNR= 57.08
PATIENT 0031 | HF Original SNR= 50.96 | LF Simulated SNR= 30.78 | Final Enhanced SNR= 46.69
PATIENT 0032 | HF Original SNR= 57.07 | LF Simulated SNR= 37.01 | Final Enhanced SNR= 52.28
PATIENT 0033 | HF Original SNR= 48.96 | LF Simulated SNR= 30.13 | Final Enhanced SNR= 39.27
PATIENT 0034 | HF Original SNR= 60.66 | LF Simulated SNR= 35.94 | Final Enhanced SNR= 49.42
PATIENT 0035 | HF Original SNR= 46.98 | LF Simulated SNR= 26.86 | Final Enhanced SNR= 32.37
PATIENT 0036 | HF Original SNR= 45.68 | LF Simulated SNR= 24.87 | Final Enhanced SNR= 34.35
PATIENT 0037 | HF Original SNR= 62.64 | LF Simulated SNR= 33.73 | Final Enhanced SNR= 52.70
PATIENT 0038 | HF Original SNR= 51.81 | LF Simulated SNR= 24.28 | Final Enhanced SNR= 30.46
PATIENT 0040 | HF Original SNR= 53.68 | LF Simulated SNR= 33.03 | Final Enhanced SNR= 44.86
PATIENT 0041 | HF Original SNR= 46.44 | LF Simulated SNR= 26.83 | Final Enhanced SNR= 37.63
PATIENT 0043 | HF Original SNR= 46.55 | LF Simulated SNR= 30.42 | Final Enhanced SNR= 36.94
PATIENT 0044 | HF Original SNR= 49.18 | LF Simulated SNR= 27.94 | Final Enhanced SNR= 37.69
PATIENT 0046 | HF Original SNR= 77.99 | LF Simulated SNR= 46.29 | Final Enhanced SNR= 61.53
PATIENT 0047 | HF Original SNR= 55.97 | LF Simulated SNR= 31.71 | Final Enhanced SNR= 42.52
PATIENT 0048 | HF Original SNR= 42.42 | LF Simulated SNR= 22.28 | Final Enhanced SNR= 26.42
PATIENT 0050 | HF Original SNR= 54.40 | LF Simulated SNR= 32.09 | Final Enhanced SNR= 44.29
PATIENT 0051 | HF Original SNR= 49.15 | LF Simulated SNR= 28.01 | Final Enhanced SNR= 40.40
PATIENT 0052 | HF Original SNR= 59.46 | LF Simulated SNR= 35.54 | Final Enhanced SNR= 49.67
PATIENT 0053 | HF Original SNR= 83.72 | LF Simulated SNR= 60.02 | Final Enhanced SNR= 98.27
PATIENT 0056 | HF Original SNR= 54.35 | LF Simulated SNR= 30.03 | Final Enhanced SNR= 46.31
PATIENT 0057 | HF Original SNR= 67.11 | LF Simulated SNR= 39.92 | Final Enhanced SNR= 60.11
PATIENT 0058 | HF Original SNR= 57.70 | LF Simulated SNR= 36.26 | Final Enhanced SNR= 42.29
PATIENT 0061 | HF Original SNR= 70.08 | LF Simulated SNR= 39.52 | Final Enhanced SNR= 76.41
PATIENT 0062 | HF Original SNR= 45.71 | LF Simulated SNR= 24.19 | Final Enhanced SNR= 29.68
PATIENT 0063 | HF Original SNR= 45.08 | LF Simulated SNR= 27.03 | Final Enhanced SNR= 39.58
PATIENT 0064 | HF Original SNR= 70.17 | LF Simulated SNR= 46.47 | Final Enhanced SNR= 80.02
PATIENT 0067 | HF Original SNR= 56.94 | LF Simulated SNR= 38.97 | Final Enhanced SNR= 61.37
PATIENT 0068 | HF Original SNR= 50.49 | LF Simulated SNR= 30.49 | Final Enhanced SNR= 38.31
PATIENT 0069 | HF Original SNR= 47.14 | LF Simulated SNR= 29.75 | Final Enhanced SNR= 42.06
PATIENT 0070 | HF Original SNR= 55.92 | LF Simulated SNR= 31.90 | Final Enhanced SNR= 41.02
PATIENT 0071 | HF Original SNR= 54.69 | LF Simulated SNR= 26.27 | Final Enhanced SNR= 36.36
PATIENT 0072 | HF Original SNR= 58.14 | LF Simulated SNR= 36.31 | Final Enhanced SNR= 49.59
PATIENT 0074 | HF Original SNR= 70.65 | LF Simulated SNR= 41.76 | Final Enhanced SNR= 80.37
PATIENT 0075 | HF Original SNR= 62.75 | LF Simulated SNR= 30.56 | Final Enhanced SNR= 37.24
PATIENT 0078 | HF Original SNR= 97.97 | LF Simulated SNR= 53.99 | Final Enhanced SNR= 77.99
PATIENT 0080 | HF Original SNR= 61.82 | LF Simulated SNR= 34.65 | Final Enhanced SNR= 50.22
PATIENT 0081 | HF Original SNR= 71.04 | LF Simulated SNR= 37.61 | Final Enhanced SNR= 51.67
PATIENT 0082 | HF Original SNR= 126.04| LF Simulated SNR= 68.19 | Final Enhanced SNR= 99.64
PATIENT 0083 | HF Original SNR= 55.09 | LF Simulated SNR= 35.52 | Final Enhanced SNR= 49.06
PATIENT 0084 | HF Original SNR= 70.80 | LF Simulated SNR= 38.24 | Final Enhanced SNR= 51.83
PATIENT 0085 | HF Original SNR= 64.00 | LF Simulated SNR= 42.04 | Final Enhanced SNR= 58.39
PATIENT 0086 | HF Original SNR= 56.48 | LF Simulated SNR= 35.33 | Final Enhanced SNR= 44.79
PATIENT 0087 | HF Original SNR= 92.05 | LF Simulated SNR= 48.97 | Final Enhanced SNR= 83.73
PATIENT 0088 | HF Original SNR= 76.67 | LF Simulated SNR= 38.31 | Final Enhanced SNR= 56.60
PATIENT 0089 | HF Original SNR= 53.69 | LF Simulated SNR= 28.72 | Final Enhanced SNR= 33.22
PATIENT 0090 | HF Original SNR= 68.36 | LF Simulated SNR= 35.48 | Final Enhanced SNR= 48.03
PATIENT 0091 | HF Original SNR= 70.01 | LF Simulated SNR= 41.17 | Final Enhanced SNR= 57.75
PATIENT 0092 | HF Original SNR= 92.20 | LF Simulated SNR= 54.32 | Final Enhanced SNR= 90.87
PATIENT 0093 | HF Original SNR= 53.15 | LF Simulated SNR= 27.80 | Final Enhanced SNR= 39.10
PATIENT 0094 | HF Original SNR= 49.17 | LF Simulated SNR= 32.72 | Final Enhanced SNR= 47.55
PATIENT 0095 | HF Original SNR= 71.85 | LF Simulated SNR= 40.18 | Final Enhanced SNR= 56.04
PATIENT 0096 | HF Original SNR= 47.93 | LF Simulated SNR= 24.54 | Final Enhanced SNR= 35.75
PATIENT 0097 | HF Original SNR= 47.84 | LF Simulated SNR= 29.62 | Final Enhanced SNR= 40.15
PATIENT 0098 | HF Original SNR= 81.10 | LF Simulated SNR= 50.32 | Final Enhanced SNR= 76.13
PATIENT 0099 | HF Original SNR= 40.35 | LF Simulated SNR= 21.37 | Final Enhanced SNR= 25.11
PATIENT 0100 | HF Original SNR= 68.41 | LF Simulated SNR= 38.88 | Final Enhanced SNR= 50.16
PATIENT 0101 | HF Original SNR= 52.10 | LF Simulated SNR= 28.88 | Final Enhanced SNR= 36.15
PATIENT 0103 | HF Original SNR= 54.24 | LF Simulated SNR= 28.59 | Final Enhanced SNR= 37.97
PATIENT 0104 | HF Original SNR= 56.09 | LF Simulated SNR= 27.04 | Final Enhanced SNR= 38.54
PATIENT 0105 | HF Original SNR= 70.82 | LF Simulated SNR= 35.00 | Final Enhanced SNR= 72.74
PATIENT 0106 | HF Original SNR= 94.51 | LF Simulated SNR= 59.84 | Final Enhanced SNR= 84.17
PATIENT 0108 | HF Original SNR= 88.89 | LF Simulated SNR= 51.24 | Final Enhanced SNR= 70.54
PATIENT 0109 | HF Original SNR= 65.99 | LF Simulated SNR= 34.93 | Final Enhanced SNR= 53.02
PATIENT 0112 | HF Original SNR= 66.47 | LF Simulated SNR= 34.92 | Final Enhanced SNR= 48.70
"""

patients, hf_snrs, lf_snrs, final_snrs = [], [], [], []

for line in raw_data.strip().split('\n'):
    parts = line.split('|')
    patients.append(parts[0].strip().split(' ')[1])
    hf_snrs.append(float(parts[1].split('=')[1].strip()))
    lf_snrs.append(float(parts[2].split('=')[1].strip()))
    final_snrs.append(float(parts[3].split('=')[1].strip()))

# ==============================================================================
# 1. Slopegraph (The Trajectory)
# ==============================================================================
fig, ax = plt.subplots(figsize=(8, 10))
for i in range(len(patients)):
    # Draw line connecting LF to Final
    ax.plot([0, 1], [lf_snrs[i], final_snrs[i]], marker='o', markersize=6, color='seagreen', alpha=0.5, linewidth=1.5)

ax.set_xticks([0, 1])
ax.set_xticklabels(['Low-Field (Input)', 'Final Enhanced (Output)'], fontsize=12, fontweight='bold')
ax.set_ylabel('Signal-to-Noise Ratio (SNR)', fontsize=12, fontweight='bold')
ax.set_title('Pipeline Enhancement Trajectory (Slopegraph)', fontsize=14, pad=20)

# Clean up layout
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.tight_layout()
plt.savefig('1_slopegraph.png', dpi=300)
plt.close()


# ==============================================================================
# 2. Clinical Threshold Scatter Plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(14, 7))

# Draw horizontal clinical background bands
ax.axhspan(0, 30, color='red', alpha=0.1)
ax.text(-1, 15, 'Sub-Diagnostic (<30)', color='red', fontweight='bold', va='center')
ax.axhspan(30, 50, color='orange', alpha=0.1)
ax.text(-1, 40, 'Standard Grade (30-50)', color='darkorange', fontweight='bold', va='center')
ax.axhspan(50, 130, color='green', alpha=0.1)
ax.text(-1, 90, 'High-Field Target (>50)', color='green', fontweight='bold', va='center')

# Scatter and connecting arrows
x_pos = np.arange(len(patients))
ax.scatter(x_pos, lf_snrs, color='darkorange', label='LF Input', zorder=5, s=50)
ax.scatter(x_pos, final_snrs, color='darkgreen', label='Final Enhanced', zorder=5, s=50)

for i in range(len(patients)):
    ax.annotate("", xy=(x_pos[i], final_snrs[i]), xytext=(x_pos[i], lf_snrs[i]),
                arrowprops=dict(arrowstyle="->", color="gray", alpha=0.4))

ax.set_xticks(x_pos)
ax.set_xticklabels(patients, rotation=90, fontsize=8)
ax.set_ylabel('SNR Value', fontsize=12, fontweight='bold')
ax.set_title('Quality Improvement Across Clinical Thresholds', fontsize=14, pad=15)
ax.legend(loc='upper right')
ax.set_xlim(-2, len(patients))
plt.tight_layout()
plt.savefig('2_threshold_scatter.png', dpi=300)
plt.close()


# ==============================================================================
# 3. Distribution Shift Plot (Violin + Strip = "Raincloud" Alternative)
# ==============================================================================
# Prepare DataFrame for Seaborn
df_lf = pd.DataFrame({'SNR': lf_snrs, 'Type': '1. Low-Field (Input)'})
df_final = pd.DataFrame({'SNR': final_snrs, 'Type': '2. Final Enhanced (Output)'})
df_hf = pd.DataFrame({'SNR': hf_snrs, 'Type': '3. High-Field (Ground Truth)'})
df_combined = pd.concat([df_lf, df_final, df_hf])

fig, ax = plt.subplots(figsize=(10, 6))

# Plot the density (Violin)
sns.violinplot(x='Type', y='SNR', data=df_combined, inner=None, palette="pastel", alpha=0.5, ax=ax)
# Plot the individual patient data points (Strip)
sns.stripplot(x='Type', y='SNR', data=df_combined, jitter=True, zorder=1, alpha=0.7, color='black', size=4, ax=ax)

ax.set_title('Dataset Center of Mass Shift (Distribution Analysis)', fontsize=14, pad=15)
ax.set_ylabel('Signal-to-Noise Ratio (SNR)', fontsize=12, fontweight='bold')
ax.set_xlabel('')
ax.grid(axis='y', linestyle='--', alpha=0.4)
sns.despine()
plt.tight_layout()
plt.savefig('3_distribution_shift.png', dpi=300)
plt.close()

print("Success: Generated '1_slopegraph.png', '2_threshold_scatter.png', and '3_distribution_shift.png'")
