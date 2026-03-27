import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import io

# ==========================================
# 1. DATA PREPARATION
# ==========================================
csv_data = """Patient,LF_SNR,Final_SNR,PSNR,SSIM
0002,29.29,45.88,23.54,0.7289
0003,20.60,31.20,25.08,0.6995
0004,38.34,49.71,24.66,0.7987
0005,29.02,38.41,23.26,0.7548
0006,25.55,32.64,23.29,0.7253
0009,52.73,86.50,22.54,0.8228
0010,53.42,86.48,21.43,0.7208
0011,25.01,28.10,26.20,0.5937
0012,27.69,34.89,24.16,0.7430
0013,27.60,43.13,23.77,0.7492
0014,36.57,47.69,23.86,0.7682
0015,32.38,45.07,22.01,0.6797
0016,40.30,53.01,24.01,0.7956
0019,32.72,50.17,23.78,0.7674
0021,26.62,31.07,23.60,0.7122
0022,29.51,43.91,20.26,0.6578
0023,36.97,58.34,20.45,0.7114
0024,24.90,31.43,24.74,0.7057
0025,32.64,44.07,23.66,0.7402
0026,37.91,51.37,23.87,0.7785
0029,37.56,62.22,22.56,0.7252
0030,34.65,57.08,23.60,0.7081
0031,30.78,46.69,22.08,0.7261
0032,37.01,52.28,23.73,0.7871
0033,30.13,39.27,22.36,0.7375
0034,35.94,49.42,25.41,0.8219
0035,26.86,32.37,20.85,0.6837
0036,24.87,34.35,22.58,0.7191
0037,33.73,52.70,24.74,0.7467
0038,24.28,30.46,20.14,0.6198
0040,33.03,44.86,23.80,0.7829
0041,26.83,37.63,22.48,0.6467
0043,30.42,36.94,24.86,0.7772
0044,27.94,37.69,28.77,0.7457
0046,46.29,61.53,23.59,0.7483
0047,31.71,42.52,22.60,0.7386
0048,22.28,26.42,22.88,0.6690
0050,32.09,44.29,21.84,0.7116
0051,28.01,40.40,23.61,0.7169
0052,35.54,49.67,24.16,0.7955
0053,60.02,98.27,23.04,0.7890
0056,30.03,46.31,27.45,0.7157
0057,39.92,60.11,20.70,0.6330
0058,36.26,42.29,23.41,0.7770"""

df = pd.read_csv(io.StringIO(csv_data))
df['Patient'] = df['Patient'].astype(str).str.zfill(4)

sns.set_theme(style="whitegrid", context="paper")

# ==========================================
# 1. Comparison Bar Chart: LF vs Final SNR
# ==========================================
plt.figure(figsize=(16, 7))
x = np.arange(len(df))
width = 0.35
plt.bar(x - width/2, df['LF_SNR'], width, label='LF Simulated (Input)', color='#ff9999', edgecolor='black')
plt.bar(x + width/2, df['Final_SNR'], width, label='Final Enhanced (Output)', color='#66b3ff', edgecolor='black')
plt.ylabel('Signal-to-Noise Ratio (SNR)', fontsize=12, fontweight='bold')
plt.title('Patient-by-Patient SNR Enhancement', fontsize=14, pad=15)
plt.xticks(x, df['Patient'], rotation=90, fontsize=8)
plt.legend(loc='upper left')
plt.xlim(-1, len(df))
plt.tight_layout()
plt.savefig('1_BarChart_LF_vs_Final.png', dpi=300)
plt.close()

# ==========================================
# 2. Multi-Stage Performance Line Plot
# ==========================================
stages = ['LF Input', 'Stage 1 (N4)', 'Stage 2 (Std)', 'Stage 3 (Wiener)', 'Stage 4 (Res)', 'Stage 5 (Ref)', 'Final']
p0082_best = [68.19, 66.96, 70.40, 99.03, 100.03, 99.64, 99.64]
p0050_avg = [32.09, 31.82, 33.31, 44.25, 44.41, 44.29, 44.29]
p0048_worst = [22.28, 19.99, 20.78, 26.36, 26.51, 26.42, 26.42]
plt.figure(figsize=(10, 6))
plt.plot(stages, p0082_best, marker='o', linewidth=2.5, label='Best Case (Patient 0082)', color='green')
plt.plot(stages, p0050_avg, marker='s', linewidth=2.5, label='Average Case (Patient 0050)', color='blue')
plt.plot(stages, p0048_worst, marker='^', linewidth=2.5, label='Challenging Case (Patient 0048)', color='red')
plt.axvspan(2.8, 3.2, color='gray', alpha=0.15, label='Primary Denoising Threshold')
plt.ylabel('SNR Value', fontsize=12, fontweight='bold')
plt.title('SNR Evolution Across Pipeline Stages', fontsize=14, pad=15)
plt.legend()
plt.tight_layout()
plt.savefig('2_MultiStage_LinePlot.png', dpi=300)
plt.close()

# ==========================================
# 3. Metric Correlation Scatter Plot
# ==========================================
plt.figure(figsize=(9, 7))
sns.scatterplot(data=df, x='PSNR', y='SSIM', hue='Final_SNR', size='Final_SNR', 
                sizes=(50, 200), palette='viridis', edgecolor='black', alpha=0.8)
plt.axvline(24, color='gray', linestyle='--', alpha=0.5)
plt.axhline(0.75, color='gray', linestyle='--', alpha=0.5)
plt.text(26, 0.80, 'Optimal Reconstruction\n(High PSNR & High SSIM)', color='green', alpha=0.7)
plt.xlabel('Peak Signal-to-Noise Ratio (PSNR)', fontsize=12, fontweight='bold')
plt.ylabel('Structural Similarity Index (SSIM)', fontsize=12, fontweight='bold')
plt.title('Reconstruction Health: PSNR vs SSIM Correlation', fontsize=14, pad=15)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Final SNR')
plt.tight_layout()
plt.savefig('3_Correlation_Scatter.png', dpi=300)
plt.close()

# ==========================================
# 4. Box-and-Whisker Plots for Aggregates
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
sns.boxplot(y=df['Final_SNR'], ax=axes[0], color='skyblue', width=0.4)
sns.stripplot(y=df['Final_SNR'], ax=axes[0], color='black', alpha=0.5, jitter=True)
axes[0].set_title('Final Enhanced SNR', fontweight='bold')
axes[0].set_ylabel('SNR')
sns.boxplot(y=df['PSNR'], ax=axes[1], color='lightgreen', width=0.4)
sns.stripplot(y=df['PSNR'], ax=axes[1], color='black', alpha=0.5, jitter=True)
axes[1].set_title('PSNR Distribution', fontweight='bold')
axes[1].set_ylabel('PSNR (dB)')
sns.boxplot(y=df['SSIM'], ax=axes[2], color='salmon', width=0.4)
sns.stripplot(y=df['SSIM'], ax=axes[2], color='black', alpha=0.5, jitter=True)
axes[2].set_title('SSIM Distribution', fontweight='bold')
axes[2].set_ylabel('SSIM Score')
plt.suptitle('Aggregate Pipeline Metrics (Filtered Cohort)', fontsize=16)
plt.tight_layout()
plt.savefig('4_BoxPlots_Aggregates.png', dpi=300)
plt.close()

# ==========================================
# 5. Intensity Profile / Line Wire Plot (Simulated)
# ==========================================
pixels = np.linspace(0, 100, 400)
hf_signal = 15 * np.exp(-((pixels - 15) ** 2) / 10) + 85 * np.exp(-((pixels - 50) ** 2) / 300) + 45 * np.exp(-((pixels - 85) ** 2) / 20)
np.random.seed(42)
lf_noise = np.random.normal(0, 12, len(pixels))
lf_signal = hf_signal + lf_noise
enhanced_noise = np.random.normal(0, 2.5, len(pixels))
enhanced_signal = pd.Series(hf_signal + enhanced_noise).rolling(window=5, center=True).mean().ffill().bfill().values
plt.figure(figsize=(12, 6))
plt.plot(pixels, lf_signal, color='red', alpha=0.4, label='LF Simulated (Noisy Input)', linewidth=1)
plt.plot(pixels, hf_signal, color='black', linestyle='--', label='HF Original (Ground Truth Target)', linewidth=2)
plt.plot(pixels, enhanced_signal, color='blue', label='Final Enhanced (Model Output)', linewidth=2)
plt.xlabel('Pixel Position (1D Slice through Anatomy)', fontsize=12, fontweight='bold')
plt.ylabel('Pixel Intensity', fontsize=12, fontweight='bold')
plt.title('Cross-Sectional Intensity Profile: Denoising Structural Preservation', fontsize=14, pad=15)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('5_Intensity_Profile_Simulated.png', dpi=300)
plt.close()

print("Success: Generated all 5 visual charts.")
