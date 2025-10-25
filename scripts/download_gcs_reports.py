#!/usr/bin/env python3
"""
Download training reports from Google Cloud Storage for 3D visualization.

This script lists available training runs and downloads the report files
needed for 3D visualization with W&B charts.
"""

import os
import sys
from pathlib import Path
from google.cloud import storage
from google.auth.exceptions import DefaultCredentialsError
import argparse


def setup_gcs_client():
    """Set up GCS client with authentication."""
    try:
        # Try to use default credentials (ADC)
        client = storage.Client()
        return client
    except DefaultCredentialsError as e:
        print(f"Authentication error: {e}")
        print("\nTo fix this, run one of:")
        print("  gcloud auth application-default login")
        print("  gcloud auth login")
        print("  # Or set GOOGLE_APPLICATION_CREDENTIALS to a service account key")
        return None


def list_training_runs(client, bucket_name):
    """List all available training runs in the bucket."""
    try:
        bucket = client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix="runs/"))

        runs = set()
        for blob in blobs:
            # Extract run name from path like "runs/run-name/..."
            path_parts = blob.name.split("/")
            if len(path_parts) >= 2 and path_parts[0] == "runs":
                run_name = path_parts[1]
                if run_name:  # Skip empty names
                    runs.add(run_name)

        return sorted(list(runs))

    except Exception as e:
        print(f"Error listing runs: {e}")
        return []


def download_run_reports(client, bucket_name, run_name, output_dir="reports"):
    """Download report files for a specific run."""
    try:
        bucket = client.bucket(bucket_name)

        # Create output directory
        output_path = Path(output_dir) / run_name
        output_path.mkdir(parents=True, exist_ok=True)

        # List all files in the run's report directory
        prefix = f"runs/{run_name}/report/"
        blobs = list(bucket.list_blobs(prefix=prefix))

        if not blobs:
            print(f"WARNING: No report files found for run: {run_name}")
            return False

        downloaded_files = []
        for blob in blobs:
            # Download each file
            relative_path = Path(blob.name).relative_to(f"runs/{run_name}/")
            local_path = output_path / relative_path

            # Create parent directories
            local_path.parent.mkdir(parents=True, exist_ok=True)

            blob.download_to_filename(str(local_path))
            downloaded_files.append(str(relative_path))
            print(f"  ✓ Downloaded: {relative_path}")

        print(f"Downloaded {len(downloaded_files)} files for run: {run_name}")
        return True

    except Exception as e:
        print(f"Error downloading run {run_name}: {e}")
        return False


def find_latest_run(runs):
    """Find the most recent run based on naming pattern."""
    if not runs:
        return None

    # Try to find runs with timestamps (format: YYYYMMDD-HHMMSS)
    timestamp_runs = []
    other_runs = []

    for run in runs:
        if '-' in run and len(run.split('-')) >= 3:
            parts = run.split('-')
            if len(parts[0]) == 8 and len(parts[1]) == 6:  # YYYYMMDD-HHMMSS
                try:
                    timestamp_runs.append((run, parts[0], parts[1]))
                except:
                    other_runs.append(run)
            else:
                other_runs.append(run)
        else:
            other_runs.append(run)

    if timestamp_runs:
        # Sort by date and time
        timestamp_runs.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return timestamp_runs[0][0]

    # Fallback to alphabetical sorting
    return sorted(runs)[-1]


def main():
    parser = argparse.ArgumentParser(description="Download GCS training reports for 3D visualization")
    parser.add_argument("--bucket", default="nanochat-aquarat-datasets",
                       help="GCS bucket name (default: nanochat-aquarat-datasets)")
    parser.add_argument("--run", help="Specific run name to download")
    parser.add_argument("--all", action="store_true", help="Download all runs")
    parser.add_argument("--latest", action="store_true", help="Download only the latest run")
    parser.add_argument("--output", default="reports", help="Output directory")
    parser.add_argument("--list-only", action="store_true", help="Only list runs, don't download")

    args = parser.parse_args()

    # Get bucket name from environment if not provided
    if args.bucket == "nanochat-aquarat-datasets":
        env_bucket = os.getenv("GCS_BUCKET", "").replace("gs://", "")
        if env_bucket:
            args.bucket = env_bucket

    print("GCS Training Reports Downloader")
    print("=" * 50)
    print(f"Bucket: gs://{args.bucket}")
    print(f"Output: {args.output}")
    print()

    # Setup GCS client
    client = setup_gcs_client()
    if not client:
        return 1

    # List available runs
    print("Listing available training runs...")
    runs = list_training_runs(client, args.bucket)

    if not runs:
        print("No training runs found in bucket")
        return 1

    print(f"Found {len(runs)} training runs:")
    for i, run in enumerate(runs, 1):
        print(f"  {i:2d}. {run}")

    if args.list_only:
        return 0

    print()

    # Determine which runs to download
    runs_to_download = []

    if args.run:
        if args.run in runs:
            runs_to_download = [args.run]
        else:
            print(f"Run '{args.run}' not found in available runs")
            return 1
    elif args.latest:
        latest = find_latest_run(runs)
        if latest:
            runs_to_download = [latest]
            print(f"Selected latest run: {latest}")
        else:
            print("Could not determine latest run")
            return 1
    elif args.all:
        runs_to_download = runs
        print(f"Will download all {len(runs)} runs")
    else:
        # Interactive selection
        print("\nSelect runs to download:")
        print("  a. All runs")
        print("  l. Latest run only")
        print("  <number>. Specific run (1, 2, 3, etc.)")
        print("  q. Quit")

        choice = input("\nEnter choice: ").strip().lower()

        if choice == 'q':
            print("Goodbye!")
            return 0
        elif choice == 'a':
            runs_to_download = runs
        elif choice == 'l':
            latest = find_latest_run(runs)
            if latest:
                runs_to_download = [latest]
            else:
                print("Could not determine latest run")
                return 1
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(runs):
                    runs_to_download = [runs[idx]]
                else:
                    print("Invalid run number")
                    return 1
            except ValueError:
                print("Invalid choice")
                return 1

    print()

    # Download selected runs
    success_count = 0
    for run in runs_to_download:
        print(f"Downloading reports for: {run}")
        if download_run_reports(client, args.bucket, run, args.output):
            success_count += 1
        print()

    print("Download Summary")
    print("=" * 30)
    print(f"Successfully downloaded: {success_count}/{len(runs_to_download)} runs")
    print(f"Reports saved to: {args.output}/")

    if success_count > 0:
        print("\nNext steps:")
        print("1. Check downloaded reports in the reports/ directory")
        print("2. Pair with W&B 3D charts using the keys from WANDB_3D_SHOWCASE.md")
        print("3. Run 3D visualization: python examples/showcase_3d_viz_example.py")

    return 0 if success_count == len(runs_to_download) else 1


if __name__ == "__main__":
    sys.exit(main())
