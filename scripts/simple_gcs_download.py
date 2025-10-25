#!/usr/bin/env python3
"""
Simple GCS downloader for training reports.
Uses existing authentication from .env file.
"""

import os
import sys
from pathlib import Path

# Set up credentials from .env
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\chris\AppData\Roaming\gcloud\application_default_credentials.json'

try:
    from google.cloud import storage
    from google.auth.exceptions import DefaultCredentialsError
except ImportError:
    print("Installing google-cloud-storage...")
    os.system('pip install google-cloud-storage')
    from google.cloud import storage
    from google.auth.exceptions import DefaultCredentialsError

def list_runs():
    """List available training runs."""
    try:
        client = storage.Client()
        bucket = client.bucket('nanochat-aquarat-datasets')

        print("Listing available training runs...")
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
            print("No training runs found")
            return []

        print(f"Found {len(runs)} training runs:")
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

def download_reports(runs, output_dir="reports"):
    """Download reports for specified runs."""
    try:
        client = storage.Client()
        bucket = client.bucket('nanochat-aquarat-datasets')

        success_count = 0

        for run in runs:
            print(f"\nDownloading reports for: {run}")

            # Create output directory
            output_path = Path(output_dir) / run
            output_path.mkdir(parents=True, exist_ok=True)

            # List and download report files
            prefix = f"runs/{run}/report/"
            blobs = list(bucket.list_blobs(prefix=prefix))

            if not blobs:
                print(f"WARNING: No report files found for run: {run}")
                continue

            downloaded_count = 0
            for blob in blobs:
                # Download each file
                relative_path = Path(blob.name).relative_to(f"runs/{run}/")
                local_path = output_path / relative_path

                # Create parent directories
                local_path.parent.mkdir(parents=True, exist_ok=True)

                blob.download_to_filename(str(local_path))
                downloaded_count += 1

            print(f"Downloaded {downloaded_count} files for run: {run}")
            success_count += 1

        print(f"\nDownloaded {success_count}/{len(runs)} runs successfully!")
        print(f"Reports saved to: {output_dir}/")

        return success_count == len(runs)

    except Exception as e:
        print(f"Error downloading reports: {e}")
        return False

def main():
    print("GCS Training Reports Downloader")
    print("=" * 50)

    # List available runs
    runs = list_runs()
    if not runs:
        return 1

    print("\nOptions:")
    print("  1. Download latest run")
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
        run_num = input(f"Enter run number (1-{len(runs)}): ").strip()
        try:
            idx = int(run_num) - 1
            if 0 <= idx < len(runs):
                runs_to_download = [runs[idx]]
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
    success = download_reports(runs_to_download)

    if success:
        print("\nNext steps:")
        print("1. Check downloaded reports in the reports/ directory")
        print("2. Pair with W&B 3D charts using the keys from WANDB_3D_SHOWCASE.md")
        print("3. Run 3D visualization: python examples/showcase_3d_viz_example.py")

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
