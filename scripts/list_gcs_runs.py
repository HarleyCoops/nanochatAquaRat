#!/usr/bin/env python3
"""
Simple script to list GCS training runs using existing credentials.
"""

import os
import sys
from pathlib import Path

# Set up credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\chris\AppData\Roaming\gcloud\application_default_credentials.json'

try:
    from google.cloud import storage
    from google.auth.exceptions import DefaultCredentialsError
except ImportError:
    print("Installing google-cloud-storage...")
    os.system('pip install google-cloud-storage')
    from google.cloud import storage
    from google.auth.exceptions import DefaultCredentialsError

def list_training_runs():
    """List all available training runs."""
    try:
        print("Connecting to GCS...")
        # Use the project ID from .env
        client = storage.Client(project="n8n-automation-project-459922")
        bucket = client.bucket('nanochat-aquarat-datasets')

        print("Scanning for training runs...")
        blobs = list(bucket.list_blobs(prefix="runs/"))

        runs = set()
        for blob in blobs:
            path_parts = blob.name.split("/")
            if len(path_parts) >= 2 and path_parts[0] == "runs":
                run_name = path_parts[1]
                if run_name:
                    runs.add(run_name)

        runs = sorted(list(runs))

        if not runs:
            print("No training runs found in bucket")
            return []

        print(f"\nFound {len(runs)} training runs:")
        for i, run in enumerate(runs, 1):
            print(f"  {i:2d}. {run}")

        return runs

    except DefaultCredentialsError as e:
        print(f"Authentication error: {e}")
        print("Please run: gcloud auth application-default login")
        return []
    except Exception as e:
        print(f"Error: {e}")
        return []

def download_run(run_name, output_dir="reports"):
    """Download all files for a specific run."""
    try:
        print(f"\nDownloading reports for: {run_name}")

        client = storage.Client(project="n8n-automation-project-459922")
        bucket = client.bucket('nanochat-aquarat-datasets')

        # Create output directory
        output_path = Path(output_dir) / run_name
        output_path.mkdir(parents=True, exist_ok=True)

        # Download all files in the run's report directory
        prefix = f"runs/{run_name}/report/"
        blobs = list(bucket.list_blobs(prefix=prefix))

        if not blobs:
            print(f"No report files found for run: {run_name}")
            return False

        downloaded_count = 0
        for blob in blobs:
            # Download each file
            relative_path = Path(blob.name).relative_to(f"runs/{run_name}/")
            local_path = output_path / relative_path

            # Create parent directories
            local_path.parent.mkdir(parents=True, exist_ok=True)

            blob.download_to_filename(str(local_path))
            downloaded_count += 1
            print(f"  Downloaded: {relative_path}")

        print(f"Downloaded {downloaded_count} files for run: {run_name}")
        return True

    except Exception as e:
        print(f"Error downloading {run_name}: {e}")
        return False

def main():
    print("GCS Training Runs Lister")
    print("=" * 50)

    # List available runs
    runs = list_training_runs()
    if not runs:
        print("\nNo runs available to download")
        return 1

    print("\nWhat would you like to do?")
    print("  1. Download latest run only")
    print("  2. Download all runs")
    print("  3. Download specific run")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == "1":
        # Find latest run
        timestamp_runs = [r for r in runs if '-' in r and len(r.split('-')) >= 3]
        if timestamp_runs:
            latest = sorted(timestamp_runs)[-1]
        else:
            latest = runs[-1]
        print(f"Selected latest run: {latest}")
        runs_to_download = [latest]

    elif choice == "2":
        print(f"Will download all {len(runs)} runs")
        runs_to_download = runs

    elif choice == "3":
        print("\nAvailable runs:")
        for i, run in enumerate(runs, 1):
            print(f"  {i:2d}. {run}")

        try:
            run_num = int(input(f"\nEnter run number (1-{len(runs)}): ").strip())
            if 1 <= run_num <= len(runs):
                runs_to_download = [runs[run_num - 1]]
            else:
                print("Invalid run number")
                return 1
        except ValueError:
            print("Invalid input")
            return 1

    else:
        print("Invalid choice")
        return 1

    # Download selected runs
    success_count = 0
    for run in runs_to_download:
        if download_run(run):
            success_count += 1

    print(f"\nDownloaded {success_count}/{len(runs_to_download)} runs successfully!")
    print("Reports saved to: reports/")

    if success_count > 0:
        print("\nNext steps:")
        print("1. Check downloaded reports in the reports/ directory")
        print("2. Run 3D visualization: python examples/showcase_3d_viz_example.py")

    return 0 if success_count == len(runs_to_download) else 1

if __name__ == "__main__":
    sys.exit(main())
