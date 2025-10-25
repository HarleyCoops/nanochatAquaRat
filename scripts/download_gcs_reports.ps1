#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Download training reports from Google Cloud Storage for 3D visualization.

.DESCRIPTION
    This script lists available training runs and downloads the report files
    needed for 3D visualization with W&B charts.
#>

param(
    [string]$Bucket = "nanochat-aquarat-datasets",
    [string]$Run,
    [switch]$All,
    [switch]$Latest,
    [string]$Output = "reports",
    [switch]$ListOnly
)

# Get bucket name from environment if not provided
if ($Bucket -eq "nanochat-aquarat-datasets") {
    $envBucket = $env:GCS_BUCKET -replace "gs://", ""
    if ($envBucket) {
        $Bucket = $envBucket
    }
}

Write-Host "GCS Training Reports Downloader" -ForegroundColor Cyan
Write-Host ("=" * 50) -ForegroundColor Cyan
Write-Host "Bucket: gs://$Bucket" -ForegroundColor White
Write-Host "Output: $Output" -ForegroundColor White
Write-Host ""

# Check if gcloud is available
$gcloudPath = Get-Command gcloud -ErrorAction SilentlyContinue
if (-not $gcloudPath) {
    Write-Host "ERROR: gcloud CLI not found in PATH" -ForegroundColor Red
    Write-Host ""
    Write-Host "To fix this, install Google Cloud SDK:" -ForegroundColor Yellow
    Write-Host "  https://cloud.google.com/sdk/docs/install" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Or add gcloud to your PATH if already installed." -ForegroundColor Yellow
    exit 1
}

Write-Host "SUCCESS: gcloud CLI found: $($gcloudPath.Source)" -ForegroundColor Green
Write-Host ""

# Check authentication
Write-Host "Checking GCS authentication..." -ForegroundColor Yellow
try {
    $authResult = & gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Authentication check failed"
    }

    if (-not $authResult) {
        Write-Host "WARNING: No active gcloud authentication found" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Please authenticate with one of:" -ForegroundColor Yellow
        Write-Host "  1. User account:  gcloud auth login" -ForegroundColor White
        Write-Host "  2. Service account: gcloud auth activate-service-account --key-file=key.json" -ForegroundColor White
        Write-Host ""
        $authenticate = Read-Host "Authenticate now? (y/n)"
        if ($authenticate -eq 'y') {
            & gcloud auth login
        } else {
            Write-Host "ERROR: Authentication required to access GCS" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "SUCCESS: Authenticated as: $authResult" -ForegroundColor Green
    }
} catch {
    Write-Host "ERROR: Authentication error: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""

# List available runs
Write-Host "Listing available training runs..." -ForegroundColor Yellow
try {
    $runsOutput = & gsutil ls "gs://$Bucket/runs/" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to list runs"
    }

    $runs = @()
    foreach ($line in $runsOutput) {
        if ($line -match "runs/([^/]+)/$") {
            $runs += $matches[1]
        }
    }

    $runs = $runs | Sort-Object

    if ($runs.Count -eq 0) {
        Write-Host "ERROR: No training runs found in bucket" -ForegroundColor Red
        exit 1
    }

    Write-Host "SUCCESS: Found $($runs.Count) training runs:" -ForegroundColor Green
    for ($i = 0; $i -lt $runs.Count; $i++) {
        Write-Host "  $($i + 1): $($runs[$i])" -ForegroundColor White
    }

    if ($ListOnly) {
        exit 0
    }

    Write-Host ""

    # Determine which runs to download
    $runsToDownload = @()

    if ($Run) {
        if ($runs -contains $Run) {
            $runsToDownload = @($Run)
        } else {
            Write-Host "ERROR: Run '$Run' not found in available runs" -ForegroundColor Red
            exit 1
        }
    } elseif ($Latest) {
        # Find latest run (try timestamp format first)
        $timestampRuns = @()
        $otherRuns = @()

        foreach ($run in $runs) {
            if ($run -match "^\d{8}-\d{6}") {
                $timestampRuns += $run
            } else {
                $otherRuns += $run
            }
        }

        if ($timestampRuns.Count -gt 0) {
            $timestampRuns = $timestampRuns | Sort-Object -Descending
            $runsToDownload = @($timestampRuns[0])
            Write-Host "Selected latest run: $($timestampRuns[0])" -ForegroundColor Cyan
        } else {
            $runsToDownload = @($runs[-1])
            Write-Host "Selected latest run: $($runs[-1])" -ForegroundColor Cyan
        }
    } elseif ($All) {
        $runsToDownload = $runs
        Write-Host "Will download all $($runs.Count) runs" -ForegroundColor Cyan
    } else {
        # Interactive selection
        Write-Host "Select runs to download:" -ForegroundColor Yellow
        Write-Host "  a. All runs" -ForegroundColor White
        Write-Host "  l. Latest run only" -ForegroundColor White
        Write-Host "  <number>. Specific run (1, 2, 3, etc.)" -ForegroundColor White
        Write-Host "  q. Quit" -ForegroundColor White

        $choice = Read-Host "`nEnter choice"
        $choice = $choice.ToLower()

        if ($choice -eq 'q') {
            Write-Host "Goodbye!" -ForegroundColor Cyan
            exit 0
        } elseif ($choice -eq 'a') {
            $runsToDownload = $runs
        } elseif ($choice -eq 'l') {
            if ($timestampRuns.Count -gt 0) {
                $timestampRuns = $timestampRuns | Sort-Object -Descending
                $runsToDownload = @($timestampRuns[0])
            } else {
                $runsToDownload = @($runs[-1])
            }
        } else {
            try {
                $idx = [int]$choice - 1
                if ($idx -ge 0 -and $idx -lt $runs.Count) {
                    $runsToDownload = @($runs[$idx])
                } else {
                    Write-Host "ERROR: Invalid run number" -ForegroundColor Red
                    exit 1
                }
            } catch {
                Write-Host "ERROR: Invalid choice" -ForegroundColor Red
                exit 1
            }
        }
    }

    Write-Host ""

    # Download selected runs
    $successCount = 0
    foreach ($run in $runsToDownload) {
        Write-Host "Downloading reports for: $run" -ForegroundColor Yellow

        # Create output directory
        $outputPath = Join-Path $Output $run
        New-Item -ItemType Directory -Force -Path $outputPath | Out-Null

        # Download report files
        $sourcePath = "gs://$Bucket/runs/$run/report/"
        $result = & gsutil -m cp -r $sourcePath $outputPath 2>$null

        if ($LASTEXITCODE -eq 0) {
            Write-Host "SUCCESS: Downloaded reports for run: $run" -ForegroundColor Green
            $successCount++

            # List downloaded files
            $downloadedFiles = Get-ChildItem -Recurse -File $outputPath
            Write-Host "  Downloaded $($downloadedFiles.Count) files" -ForegroundColor White
        } else {
            Write-Host "ERROR: Failed to download run: $run" -ForegroundColor Red
        }

        Write-Host ""
    }

    Write-Host "Download Summary" -ForegroundColor Cyan
    Write-Host ("=" * 30) -ForegroundColor Cyan
    Write-Host "SUCCESS: Successfully downloaded: $successCount/$($runsToDownload.Count) runs" -ForegroundColor Green
    Write-Host "Reports saved to: $Output/" -ForegroundColor White

    if ($successCount -gt 0) {
        Write-Host ""
        Write-Host "Next steps:" -ForegroundColor Cyan
        Write-Host "1. Check downloaded reports in the reports/ directory" -ForegroundColor White
        Write-Host "2. Pair with W&B 3D charts using the keys from WANDB_3D_SHOWCASE.md" -ForegroundColor White
        Write-Host "3. Run 3D visualization: python examples/showcase_3d_viz_example.py" -ForegroundColor White
    }
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
    exit 1
}
