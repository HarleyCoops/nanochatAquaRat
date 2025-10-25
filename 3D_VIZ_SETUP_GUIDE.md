# 3D Visualization Setup Guide

This guide documents the complete process for setting up 3D visualization with training data from Google Cloud Storage.

## Problem Statement

The user encountered an authentication error when trying to access GCS training data:
```
invalid_grant: Token has been expired or revoked; re-run gcloud auth application-default login
```

## Solution Overview

I've created automated tools to handle the GCS authentication and data download process. Here's how to use them:

## Method 1: Using the PowerShell Script (Recommended)

### Prerequisites
1. **Google Cloud SDK** must be installed and in PATH
2. **Authentication** must be configured

### Steps

1. **Check Authentication Status:**
   ```powershell
   gcloud auth list --filter=status:ACTIVE
   ```

2. **If authentication is expired, refresh it:**
   ```powershell
   gcloud auth application-default login
   ```

3. **List available training runs:**
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts/download_gcs_reports.ps1 -ListOnly
   ```

4. **Download the latest training reports:**
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts/download_gcs_reports.ps1 -Latest
   ```

5. **Or download all runs:**
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts/download_gcs_reports.ps1 -All
   ```

## Method 2: Using the Python Script

### Prerequisites
1. **Python environment** with google-cloud-storage installed
2. **Authentication** configured

### Steps

1. **Install dependencies:**
   ```bash
   pip install google-cloud-storage
   ```

2. **Run the downloader:**
   ```bash
   python scripts/simple_gcs_download.py
   ```

3. **Follow the interactive prompts** to select which runs to download

## Method 3: Manual Process Using gsutil

If the automated scripts don't work, use these manual commands:

### 1. Refresh Authentication
```bash
gcloud auth application-default login
```

### 2. List Available Runs
```bash
gsutil ls gs://nanochat-aquarat-datasets/runs/
```

### 3. Download Latest Run Reports
```bash
# Find the latest run (most recent timestamp)
gsutil ls gs://nanochat-aquarat-datasets/runs/ | grep -E "runs/[^/]+/$" | sort | tail -1

# Download the latest run's reports
LATEST_RUN="aquarat-20251023-143022"  # Replace with actual latest run
gsutil -m cp -r gs://nanochat-aquarat-datasets/runs/$LATEST_RUN/report ./reports/$LATEST_RUN/
```

## Expected Output Structure

After successful download, you should have:

```
reports/
└── aquarat-20251023-143022/
    └── report/
        └── report.md          # Main training report with metrics
        └── [other report files]
```

## 3D Visualization Setup

**The 3D visualizations are ALREADY INTEGRATED into your training scripts.** Here's what this means:

### What You Actually Get

When you run training with 3D visualization enabled, you get **interactive 3D charts** that show:

1. **3D Loss Landscape** - How your model's loss changes over time and learning rate
2. **3D Gradient Flow** - How gradients move through your neural network layers
3. **3D Reward Landscape** - For RL training, shows the path through loss vs reward space
4. **3D Training Dynamics** - Combined view of all training metrics in 3D space

### How to See the 3D Visualizations

**Option 1: Run New Training (Recommended)**
```bash
# This will automatically generate 3D visualizations during training
python -m scripts.chat_rl --run=my_3d_demo --eval_every=30
```

**Option 2: Use the Demo Script**
```bash
# See example 3D visualizations with fake data
python examples/showcase_3d_viz_example.py
```

**Option 3: View Existing Training Data**
The downloaded reports contain the metrics, but the 3D visualizations are generated during training. To see 3D viz of your historical data, you'd need to re-run training with the same data.

### What You'll Actually See

1. **During Training**: Every 30-60 steps, the system generates new 3D plots
2. **In W&B Dashboard**: Interactive 3D charts you can rotate, zoom, and explore
3. **Final Summary**: Complete 3D visualization of your entire training run

### Real Example Output

When working, you'll see console messages like:
```
Generating 3D visualizations at step 60...
Logged 4 3D visualizations to W&B
```

And in W&B, you'll see interactive 3D plots showing your training dynamics in ways that 2D charts can't represent.

## Available 3D Visualizations

| Visualization | W&B Path | Description |
|---------------|----------|-------------|
| **Loss Landscape** | `3d/loss_landscape` | Loss surface over training steps × learning rate |
| **Gradient Flow** | `3d/gradient_flow` | Gradient norms across layers × time |
| **Reward Landscape** | `3d/reward_landscape` | RL trajectory in loss-reward space |
| **Training Dynamics** | `3d/training_dynamics` | Unified view of all metrics |

## Integration with Training Scripts

The 3D visualizations are automatically integrated into:

- **`scripts/chat_rl.py`** - RL training with GRPO optimization
- **`scripts/chat_sft.py`** - Supervised fine-tuning

### Run Training with 3D Viz
```bash
# RL Training
python -m scripts.chat_rl --run=my_3d_showcase --eval_every=30

# SFT Training
python -m scripts.chat_sft --run=my_3d_showcase --eval_every=50
```

## Troubleshooting

### Authentication Issues
```bash
# Check current authentication
gcloud auth list

# Refresh authentication
gcloud auth application-default login

# Check bucket access
gsutil ls gs://nanochat-aquarat-datasets/
```

### Missing Dependencies
```bash
# Install visualization dependencies
pip install plotly scipy scikit-learn

# Install GCS client
pip install google-cloud-storage
```

### No Training Runs Found
1. Check if the bucket name is correct: `gs://nanochat-aquarat-datasets`
2. Verify you have access to the bucket
3. Check if training has been uploaded to GCS

### 3D Visualizations Not Appearing
1. Ensure `WANDB_API_KEY` is set
2. Check that you're not using `--run=dummy`
3. Verify W&B login: `wandb login`
4. Look for error messages in console output

## Configuration

The system uses these settings from `.env`:

```bash
GCS_BUCKET=gs://nanochat-aquarat-datasets
GCP_PROJECT_ID=n8n-automation-project-459922
GOOGLE_APPLICATION_CREDENTIALS=C:\Users\chris\AppData\Roaming\gcloud\application_default_credentials.json
```

## Next Steps After Setup

1. **Download training reports** using one of the methods above
2. **Verify data integrity** by checking the downloaded report.md files
3. **Run 3D visualization** to see training dynamics
4. **Analyze results** using the interactive W&B charts
5. **Share insights** with stakeholders using the 3D visualizations

## Support

If you encounter issues:

1. **Check the troubleshooting section** above
2. **Verify authentication** is working
3. **Ensure dependencies** are installed
4. **Check W&B configuration** in `.env`

The automated scripts handle most edge cases and provide clear error messages to help diagnose issues.
