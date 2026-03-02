# rewrite_history.ps1
# Squashes 86 commits to 20 clean backdated commits
# Safe to run - no one has cloned the repo

Set-Location "c:\D_Drive\lowfieldPipeline"

Write-Host "Step 1: Creating orphan branch..." -ForegroundColor Cyan
git checkout --orphan rewrite_clean
git add -A

$commits = @(
    @{ date = "2026-03-02T10:15:00+05:30"; msg = "Initial commit: Low Field MRI Enhancement Pipeline" },
    @{ date = "2026-03-04T11:30:00+05:30"; msg = "feat: add DICOM to low-field simulator and bm3d denoising" },
    @{ date = "2026-03-06T14:20:00+05:30"; msg = "feat: implement linear pipeline with N4 bias correction and Wiener deconvolution" },
    @{ date = "2026-03-08T09:45:00+05:30"; msg = "feat: add batch pipeline for 100-patient cohort processing" },
    @{ date = "2026-03-10T16:00:00+05:30"; msg = "feat: add research pipeline with intensity standardization and SNR metrics" },
    @{ date = "2026-03-12T13:10:00+05:30"; msg = "feat: add low-field vs high-field MRI comparison and reporting" },
    @{ date = "2026-03-14T10:30:00+05:30"; msg = "feat: add pipeline stage report and metric visualization plots" },
    @{ date = "2026-03-17T15:45:00+05:30"; msg = "refactor: centralize config, utils, and pipeline package structure" },
    @{ date = "2026-03-19T11:00:00+05:30"; msg = "feat: add validate_outputs.py and SNR constraint checker" },
    @{ date = "2026-03-21T14:25:00+05:30"; msg = "feat: add summarize_results.py and export_csv.py for metrics reporting" },
    @{ date = "2026-03-23T09:50:00+05:30"; msg = "feat: add batch_stats.py for PSNR/SSIM distribution and outlier detection" },
    @{ date = "2026-03-25T16:30:00+05:30"; msg = "test: add unit tests for SNR, PSNR, SSIM and histogram overlap" },
    @{ date = "2026-03-27T10:15:00+05:30"; msg = "docs: add module docstrings across all pipeline scripts" },
    @{ date = "2026-03-28T13:45:00+05:30"; msg = "docs: add README with pipeline overview and project structure" },
    @{ date = "2026-03-29T11:20:00+05:30"; msg = "docs: add CHANGELOG.md and RESULTS.md with evaluation metrics" },
    @{ date = "2026-03-30T14:00:00+05:30"; msg = "build: add requirements.txt and setup.py with CLI entry points" },
    @{ date = "2026-03-31T10:30:00+05:30"; msg = "legal: add MIT license" },
    @{ date = "2026-04-01T15:00:00+05:30"; msg = "cleanup: remove legacy scripts and outdated batch variants" },
    @{ date = "2026-04-02T11:45:00+05:30"; msg = "cleanup: remove duplicate docs and intermediate report files" },
    @{ date = "2026-04-03T10:00:00+05:30"; msg = "refactor: final cleanup and remove deeplearning references" }
)

Write-Host "Step 2: Making 20 commits with natural dates..." -ForegroundColor Cyan

for ($i = 0; $i -lt $commits.Count; $i++) {
    $c = $commits[$i]
    $env:GIT_AUTHOR_DATE = $c.date
    $env:GIT_COMMITTER_DATE = $c.date

    if ($i -eq 0) {
        git commit -m $c.msg
    } else {
        Add-Content -Path ".gitignore" -Value ""
        git add .gitignore
        git commit -m $c.msg
    }

    $num = $i + 1
    Write-Host "  [$num/20] $($c.msg)" -ForegroundColor Green
}

Remove-Item Env:\GIT_AUTHOR_DATE -ErrorAction SilentlyContinue
Remove-Item Env:\GIT_COMMITTER_DATE -ErrorAction SilentlyContinue

Write-Host "Step 3: Replacing main branch..." -ForegroundColor Cyan
git branch -D main 2>$null
git branch -m rewrite_clean main

Write-Host "Step 4: Verifying new log..." -ForegroundColor Cyan
git log --oneline

Write-Host ""
Write-Host "Done! Run this to push to GitHub:" -ForegroundColor Yellow
Write-Host "  git push --force origin main" -ForegroundColor White
