# Google Cloud Storage Upload Guide

After training your nanochat model on Lambda Labs, use this guide to upload all weights and artifacts to Google Cloud Storage.

## Quick Start

```bash
# After training completes, SSH to your Lambda instance
ssh ubuntu@<INSTANCE_IP>

# Navigate to project directory
cd ~/nanochatAquaRat

# Run upload script
bash scripts/upload_to_gcs.sh --bucket gs://your-bucket-name
```

The script will:
1. Check/install gcloud CLI if needed
2. Verify authentication and bucket access
3. Show what will be uploaded and ask for confirmation
4. Upload all artifacts with progress
5. Ask if you want to terminate the Lambda instance

## Prerequisites

### 1. Create a GCS Bucket

```bash
# From your local machine
gcloud storage buckets create gs://your-bucket-name \
  --location=us-central1 \
  --uniform-bucket-level-access
```

Or create via console: https://console.cloud.google.com/storage/create-bucket

### 2. Set Up Authentication

#### Option A: Service Account (Recommended for Automation)

On your local machine:

```bash
# Create service account
gcloud iam service-accounts create nanochat-uploader \
  --display-name="Nanochat Model Uploader"

# Grant storage permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:nanochat-uploader@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.objectCreator"

# Create and download key
gcloud iam service-accounts keys create ~/nanochat-key.json \
  --iam-account=nanochat-uploader@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

Copy key to Lambda instance:

```bash
scp ~/nanochat-key.json ubuntu@<INSTANCE_IP>:~/
```

On Lambda instance:

```bash
gcloud auth activate-service-account --key-file=~/nanochat-key.json
```

#### Option B: User Account (Simpler for Manual Use)

On Lambda instance:

```bash
gcloud auth login
# Follow the prompts in your browser
```

## Usage

### Basic Upload

```bash
bash scripts/upload_to_gcs.sh --bucket gs://my-models
```

### Custom Run Name

```bash
bash scripts/upload_to_gcs.sh \
  --bucket gs://my-models \
  --run-name depth20-experiment1
```

### Exclude Large Dataset Files

```bash
bash scripts/upload_to_gcs.sh \
  --bucket gs://my-models \
  --exclude-data
```

### Dry Run (Preview Only)

```bash
bash scripts/upload_to_gcs.sh \
  --bucket gs://my-models \
  --dry-run
```

### Auto-Terminate After Upload

```bash
bash scripts/upload_to_gcs.sh \
  --bucket gs://my-models \
  --auto-terminate
```

## What Gets Uploaded

From `~/.cache/nanochat/`:

| Directory | Contents | Typical Size |
|-----------|----------|--------------|
| `checkpoints/` | Model weights (.pt, .pkl files) | 500MB - 2GB |
| `report/` | Training reports and markdown summaries | 1-10MB |
| `tokenizer/` | BPE tokenizer files | 10-50MB |
| `eval_bundle/` | Evaluation datasets | 50-200MB |
| `aqua/` | AQuA-RAT dataset (optional) | 100-500MB |
| `mechanistic_interpretability/` | DeepMind interp tools | 10-100MB |

**Total**: Typically 1-5 GB per training run

## Upload Structure

Files are organized in GCS as:

```
gs://your-bucket/
└── runs/
    ├── aquarat-20251023-143022/
    │   ├── checkpoints/
    │   │   ├── base_final.pt
    │   │   ├── mid_final.pt
    │   │   ├── sft_final.pt
    │   │   └── rl_final.pt
    │   ├── report/
    │   │   └── report.md
    │   ├── tokenizer/
    │   └── ...
    └── depth20-experiment1/
        └── ...
```

## Download Weights Later

### Download Entire Run

```bash
gsutil -m rsync -r \
  gs://your-bucket/runs/aquarat-20251023-143022/ \
  ./local_checkpoints/
```

### Download Just Checkpoints

```bash
gsutil -m cp -r \
  gs://your-bucket/runs/aquarat-20251023-143022/checkpoints/ \
  ./checkpoints/
```

### Download Single File

```bash
gsutil cp \
  gs://your-bucket/runs/aquarat-20251023-143022/checkpoints/rl_final.pt \
  ./rl_final.pt
```

## Cost Considerations

### Storage Costs

- Standard storage: ~$0.02/GB/month
- Nearline storage (30+ days): ~$0.01/GB/month
- Coldline storage (90+ days): ~$0.004/GB/month

**Example**: 2GB model stored for 1 month = $0.04

### Network Egress

- Upload (ingress): **Free**
- Download to same region: **Free**
- Download to internet: ~$0.12/GB

**Tip**: Keep your GCS bucket in the same region as your compute for free transfers.

### Lifecycle Management

Auto-delete or move to cheaper storage after 90 days:

```bash
cat > lifecycle.json << EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {"age": 90}
      }
    ]
  }
}
EOF

gsutil lifecycle set lifecycle.json gs://your-bucket
```

## Troubleshooting

### "gcloud: command not found"

The script auto-installs gcloud on Linux. If it fails:

```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

### "Permission denied" Error

Check your service account has `roles/storage.objectCreator`:

```bash
gcloud projects get-iam-policy YOUR_PROJECT_ID \
  --flatten="bindings[].members" \
  --filter="bindings.members:serviceAccount:nanochat-uploader*"
```

### Upload Interrupted

The script uses `gsutil rsync`, so re-running will resume:

```bash
bash scripts/upload_to_gcs.sh --bucket gs://your-bucket
# Will skip already-uploaded files
```

### Verify Upload

```bash
# List all files in the run
gsutil ls -r gs://your-bucket/runs/your-run-name/

# Check specific checkpoints
gsutil ls gs://your-bucket/runs/your-run-name/checkpoints/
```

## Integration with Lambda Launcher

You can add GCS credentials to the automated launcher:

```python
# In scripts/launch_lambda_training.py
# Add to the cloud-init user-data:

write_files:
  - path: /home/ubuntu/.config/gcloud/application_default_credentials.json
    content: |
      {your service account key JSON}
```

Or pass as environment variable:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json

python scripts/launch_lambda_training.py \
  --inject-env GOOGLE_APPLICATION_CREDENTIALS \
  ...
```

## Best Practices

1. **Name runs descriptively**: Use `--run-name depth20-lr1e4-batch32`
2. **Exclude data when iterating**: Use `--exclude-data` to save bandwidth
3. **Dry run first**: Always use `--dry-run` to preview
4. **Service accounts for automation**: Easier than user auth
5. **Regional buckets**: Match Lambda instance region when possible
6. **Lifecycle policies**: Auto-archive old models
7. **Download to Lambda**: If re-training, download previous checkpoints to Lambda first

## Security Notes

- Service account keys are sensitive - treat like passwords
- Use least-privilege IAM roles (don't grant `roles/owner`)
- Rotate service account keys regularly
- Consider Workload Identity if using GKE
- Don't commit keys to git (add to `.gitignore`)

## Support

- GCS Documentation: https://cloud.google.com/storage/docs
- gsutil Reference: https://cloud.google.com/storage/docs/gsutil
- IAM Permissions: https://cloud.google.com/storage/docs/access-control/iam-permissions

---

**Quick Reference**:
```bash
# Upload
bash scripts/upload_to_gcs.sh --bucket gs://my-bucket

# Download
gsutil -m cp -r gs://my-bucket/runs/NAME/checkpoints/ ./

# List runs
gsutil ls gs://my-bucket/runs/

# Delete old run
gsutil -m rm -r gs://my-bucket/runs/old-run/
