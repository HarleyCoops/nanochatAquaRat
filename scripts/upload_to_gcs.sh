#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Upload nanochat training artifacts to Google Cloud Storage
# =============================================================================
# Usage:
#   bash scripts/upload_to_gcs.sh --bucket gs://your-bucket-name
#   bash scripts/upload_to_gcs.sh --bucket gs://your-bucket-name --run-name my-run
#   bash scripts/upload_to_gcs.sh --bucket gs://your-bucket-name --exclude-data --dry-run

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NANOCHAT_BASE_DIR="${HOME}/.cache/nanochat"
GCS_BUCKET=""
RUN_NAME="${WANDB_RUN:-aquarat-$(date -u +'%Y%m%d-%H%M%S')}"
EXCLUDE_DATA=false
DRY_RUN=false
AUTO_TERMINATE=false

# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --bucket)
            GCS_BUCKET="$2"
            shift 2
            ;;
        --run-name)
            RUN_NAME="$2"
            shift 2
            ;;
        --exclude-data)
            EXCLUDE_DATA=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --auto-terminate)
            AUTO_TERMINATE=true
            shift
            ;;
        -h|--help)
            cat << EOF
Upload nanochat training artifacts to Google Cloud Storage

Usage:
    bash scripts/upload_to_gcs.sh --bucket gs://your-bucket-name [OPTIONS]

Required:
    --bucket BUCKET         GCS bucket path (e.g., gs://my-bucket)

Optional:
    --run-name NAME         Custom run name (default: WANDB_RUN or timestamp)
    --exclude-data          Skip uploading dataset files (saves bandwidth)
    --dry-run               Show what would be uploaded without uploading
    --auto-terminate        Automatically terminate instance after upload
    -h, --help              Show this help message

Examples:
    # Basic upload
    bash scripts/upload_to_gcs.sh --bucket gs://my-models

    # Custom run name, exclude data
    bash scripts/upload_to_gcs.sh --bucket gs://my-models --run-name depth8-test --exclude-data

    # Dry run to preview
    bash scripts/upload_to_gcs.sh --bucket gs://my-models --dry-run

Uploads from: ${NANOCHAT_BASE_DIR}
Uploads to:   gs://bucket/runs/RUN_NAME/
EOF
            exit 0
            ;;
        *)
            echo "Error: Unknown option: $1" >&2
            echo "Use --help for usage information" >&2
            exit 1
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Validate inputs
# -----------------------------------------------------------------------------
if [ -z "$GCS_BUCKET" ]; then
    echo "Error: --bucket is required" >&2
    echo "Usage: bash scripts/upload_to_gcs.sh --bucket gs://your-bucket-name" >&2
    exit 1
fi

# Normalize bucket path (remove trailing slash)
GCS_BUCKET="${GCS_BUCKET%/}"

# Ensure bucket starts with gs://
if [[ ! "$GCS_BUCKET" =~ ^gs:// ]]; then
    echo "Error: Bucket must start with gs://" >&2
    exit 1
fi

# Check if source directory exists
if [ ! -d "$NANOCHAT_BASE_DIR" ]; then
    echo "Error: Source directory not found: $NANOCHAT_BASE_DIR" >&2
    echo "Have you run training yet?" >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# Check/Install gcloud CLI
# -----------------------------------------------------------------------------
echo "==================================================================="
echo "Google Cloud Storage Upload"
echo "==================================================================="
echo ""

if ! command -v gcloud &> /dev/null; then
    echo "[info] gcloud CLI not found. Installing..."
    echo ""
    
    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "[info] Detected Linux, installing gcloud..."
        curl https://sdk.cloud.google.com | bash
        exec -l $SHELL
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "[info] Detected macOS. Please install gcloud:"
        echo "  brew install --cask google-cloud-sdk"
        exit 1
    else
        echo "[error] Unsupported OS. Please install gcloud manually:" >&2
        echo "  https://cloud.google.com/sdk/docs/install" >&2
        exit 1
    fi
fi

# Verify gcloud is working
if ! gcloud version &> /dev/null; then
    echo "[error] gcloud is installed but not working properly" >&2
    exit 1
fi

echo "[info] gcloud CLI found: $(gcloud version --format='value(core.version)')"
echo ""

# -----------------------------------------------------------------------------
# Check authentication
# -----------------------------------------------------------------------------
echo "[info] Checking GCS authentication..."
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
    echo "[warn] No active gcloud authentication found"
    echo ""
    echo "Please authenticate with one of:"
    echo "  1. User account:  gcloud auth login"
    echo "  2. Service account: gcloud auth activate-service-account --key-file=key.json"
    echo ""
    read -p "Authenticate now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        gcloud auth login
    else
        echo "[error] Authentication required to upload to GCS" >&2
        exit 1
    fi
fi

ACTIVE_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n 1)
echo "[info] Authenticated as: $ACTIVE_ACCOUNT"
echo ""

# -----------------------------------------------------------------------------
# Verify bucket access
# -----------------------------------------------------------------------------
echo "[info] Verifying bucket access: $GCS_BUCKET"
if ! gsutil ls "$GCS_BUCKET" &> /dev/null; then
    echo "[error] Cannot access bucket: $GCS_BUCKET" >&2
    echo "Please check:" >&2
    echo "  1. Bucket exists" >&2
    echo "  2. You have permissions (storage.objects.create)" >&2
    echo "  3. Bucket name is correct" >&2
    exit 1
fi
echo "[info] Bucket accessible"
echo ""

# -----------------------------------------------------------------------------
# Prepare upload manifest
# -----------------------------------------------------------------------------
DESTINATION="${GCS_BUCKET}/runs/${RUN_NAME}/"

echo "==================================================================="
echo "Upload Configuration"
echo "==================================================================="
echo "Source:      $NANOCHAT_BASE_DIR"
echo "Destination: $DESTINATION"
echo "Run name:    $RUN_NAME"
echo "Exclude data: $EXCLUDE_DATA"
echo "Dry run:     $DRY_RUN"
echo ""

# Calculate what will be uploaded
echo "[info] Scanning source directory..."
TOTAL_SIZE=$(du -sh "$NANOCHAT_BASE_DIR" | cut -f1)
echo "[info] Total size: $TOTAL_SIZE"
echo ""

# List directories that will be uploaded
echo "Directories to upload:"
for dir in checkpoints report tokenizer eval_bundle aqua mechanistic_interpretability; do
    if [ -d "$NANOCHAT_BASE_DIR/$dir" ]; then
        SIZE=$(du -sh "$NANOCHAT_BASE_DIR/$dir" 2>/dev/null | cut -f1 || echo "unknown")
        if [ "$EXCLUDE_DATA" = true ] && [ "$dir" = "aqua" ]; then
            echo "  [SKIP] $dir/ ($SIZE) - excluded by --exclude-data"
        else
            echo "  [UPLOAD] $dir/ ($SIZE)"
        fi
    fi
done
echo ""

# -----------------------------------------------------------------------------
# Confirm upload
# -----------------------------------------------------------------------------
if [ "$DRY_RUN" = false ]; then
    echo "==================================================================="
    read -p "Proceed with upload? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "[info] Upload cancelled"
        exit 0
    fi
    echo ""
fi

# -----------------------------------------------------------------------------
# Perform upload
# -----------------------------------------------------------------------------
echo "==================================================================="
echo "Uploading to GCS"
echo "==================================================================="
echo ""

GSUTIL_OPTS="-m"  # Parallel upload
if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would execute:"
    echo ""
fi

# Upload each directory
upload_dir() {
    local dir="$1"
    local src="$NANOCHAT_BASE_DIR/$dir"
    local dst="$DESTINATION$dir/"
    
    if [ ! -d "$src" ]; then
        return
    fi
    
    if [ "$EXCLUDE_DATA" = true ] && [ "$dir" = "aqua" ]; then
        echo "[skip] $dir/"
        return
    fi
    
    echo "[upload] $dir/"
    if [ "$DRY_RUN" = true ]; then
        echo "  gsutil $GSUTIL_OPTS rsync -r \"$src\" \"$dst\""
    else
        if gsutil $GSUTIL_OPTS rsync -r "$src" "$dst"; then
            echo "[✓] $dir/ uploaded successfully"
        else
            echo "[✗] $dir/ upload failed" >&2
            return 1
        fi
    fi
    echo ""
}

# Upload all directories
UPLOAD_FAILED=false
for dir in checkpoints report tokenizer eval_bundle aqua mechanistic_interpretability; do
    if ! upload_dir "$dir"; then
        UPLOAD_FAILED=true
    fi
done

# Upload any standalone files in the base directory
echo "[upload] Root files (if any)"
if [ "$DRY_RUN" = true ]; then
    echo "  gsutil $GSUTIL_OPTS cp \"$NANOCHAT_BASE_DIR/*\" \"$DESTINATION\" 2>/dev/null || true"
else
    gsutil $GSUTIL_OPTS cp "$NANOCHAT_BASE_DIR/"*.{txt,json,log} "$DESTINATION" 2>/dev/null || true
    echo "[✓] Root files uploaded"
fi
echo ""

# -----------------------------------------------------------------------------
# Upload summary
# -----------------------------------------------------------------------------
echo "==================================================================="
echo "Upload Summary"
echo "==================================================================="

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] No files were actually uploaded"
    echo "Remove --dry-run to perform the upload"
elif [ "$UPLOAD_FAILED" = true ]; then
    echo "[✗] Upload completed with errors"
    echo "Check the output above for details"
    exit 1
else
    echo "[✓] Upload completed successfully!"
    echo ""
    echo "Artifacts uploaded to:"
    echo "  $DESTINATION"
    echo ""
    echo "View in GCS console:"
    BUCKET_NAME=$(echo "$GCS_BUCKET" | sed 's|gs://||')
    echo "  https://console.cloud.google.com/storage/browser/${BUCKET_NAME}/runs/${RUN_NAME}"
fi
echo ""

# -----------------------------------------------------------------------------
# List uploaded files
# -----------------------------------------------------------------------------
if [ "$DRY_RUN" = false ]; then
    echo "[info] Verifying upload..."
    UPLOADED_COUNT=$(gsutil ls -r "$DESTINATION" | grep -c ":" || echo "0")
    echo "[info] Total objects uploaded: $UPLOADED_COUNT"
    echo ""
    
    # Show checkpoint files specifically
    echo "[info] Checkpoint files:"
    gsutil ls "${DESTINATION}checkpoints/" 2>/dev/null | grep -E "\.(pt|pkl|pth)$" || echo "  (none found)"
    echo ""
fi

# -----------------------------------------------------------------------------
# Instance termination prompt
# -----------------------------------------------------------------------------
if [ "$DRY_RUN" = false ]; then
    echo "==================================================================="
    echo "Instance Management"
    echo "==================================================================="
    
    # Check if running on Lambda Labs (by checking for Lambda-specific markers)
    IS_LAMBDA=false
    if [ -f "/etc/lambda-stack-version" ] || grep -q "lambda" /etc/hostname 2>/dev/null; then
        IS_LAMBDA=true
    fi
    
    if [ "$IS_LAMBDA" = true ]; then
        echo "[info] Running on Lambda Labs instance"
        echo ""
        echo "Options:"
        echo "  1. Keep instance running (you pay ~\$24/hour for 8x H100)"
        echo "  2. Terminate instance now (stops charges)"
        echo ""
        
        if [ "$AUTO_TERMINATE" = true ]; then
            echo "[info] Auto-terminate enabled, shutting down in 30 seconds..."
            echo "[info] Press Ctrl+C to cancel"
            sleep 30
            echo "[info] Terminating instance..."
            sudo shutdown -h now
        else
            read -p "Terminate this Lambda instance now? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo ""
                echo "[info] Terminating instance in 10 seconds..."
                echo "[info] Press Ctrl+C to cancel"
                sleep 10
                echo "[info] Shutting down..."
                sudo shutdown -h now
            else
                echo ""
                echo "[info] Instance will continue running"
                echo "[info] Remember to terminate manually when done:"
                echo "  - Via Lambda dashboard: https://cloud.lambdalabs.com/instances"
                echo "  - Via this terminal: sudo shutdown -h now"
            fi
        fi
    else
        echo "[info] Not running on Lambda Labs (or couldn't detect)"
        echo "[info] Instance management not available"
    fi
fi

echo ""
echo "==================================================================="
echo "Done!"
echo "==================================================================="
