#!/usr/bin/env python3
"""
Lambda Labs GPU Instance Launcher for AQuA-RAT Training

This script automates launching an 8x H100 GPU instance on Lambda Labs
and deploying the nanochatAquaRat training pipeline.

Prerequisites:
1. Lambda Labs API key (set as LAMBDA_API_KEY environment variable)
2. Your SSH public key added to Lambda Labs account
3. W&B API key for logging (set as WANDB_API_KEY environment variable)

Usage:
    python launch_lambda.py --instance-type gpu_8x_h100_sxm5 --region us-west-1
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

try:
    import lambda_cloud_client
    from lambda_cloud_client.rest import ApiException
except ImportError:
    print("Installing lambda-cloud-client...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lambda-cloud-client"])
    import lambda_cloud_client
    from lambda_cloud_client.rest import ApiException


def check_env_vars():
    """Check required environment variables are set"""
    required_vars = {
        'LAMBDA_API_KEY': 'Lambda Labs API key',
        'WANDB_API_KEY': 'Weights & Biases API key'
    }
    
    missing = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing.append(f"  - {var} ({description})")
    
    if missing:
        print("ERROR: Missing required environment variables:")
        print("\n".join(missing))
        print("\nSet them with:")
        print("  export LAMBDA_API_KEY='your-lambda-api-key'")
        print("  export WANDB_API_KEY='your-wandb-api-key'")
        sys.exit(1)


def get_api_client():
    """Initialize Lambda Cloud API client"""
    api_key = os.getenv('LAMBDA_API_KEY')
    configuration = lambda_cloud_client.Configuration(
        host="https://cloud.lambdalabs.com/api/v1",
        access_token=api_key
    )
    return lambda_cloud_client.ApiClient(configuration)


def list_available_instance_types(api_client):
    """List available instance types and regions"""
    api_instance = lambda_cloud_client.DefaultApi(api_client)
    
    try:
        response = api_instance.instance_types()
        print("\nAvailable Instance Types:")
        print("-" * 80)
        
        for type_name, details in response.data.items():
            if details.instance_type.regions_with_capacity_available:
                print(f"\n{type_name}:")
                print(f"  GPUs: {details.instance_type.specs.gpus}")
                print(f"  GPU Memory: {details.instance_type.specs.memory_gbs} GB")
                print(f"  Price: ${details.instance_type.specs.price_cents_per_hour / 100}/hour")
                print(f"  Available regions: {', '.join(details.instance_type.regions_with_capacity_available)}")
        
        return response.data
    except ApiException as e:
        print(f"Error fetching instance types: {e}")
        sys.exit(1)


def launch_instance(api_client, instance_type, region, name="nanochat-aquarat-training"):
    """Launch a Lambda Labs GPU instance"""
    api_instance = lambda_cloud_client.DefaultApi(api_client)
    
    # Get SSH keys
    try:
        ssh_keys_response = api_instance.list_ssh_keys()
        if not ssh_keys_response.data:
            print("ERROR: No SSH keys found in your Lambda Labs account.")
            print("Please add an SSH key at: https://cloud.lambdalabs.com/ssh-keys")
            sys.exit(1)
        
        ssh_key_names = [key.name for key in ssh_keys_response.data]
        print(f"Using SSH keys: {', '.join(ssh_key_names)}")
    except ApiException as e:
        print(f"Error fetching SSH keys: {e}")
        sys.exit(1)
    
    # Launch instance
    launch_request = lambda_cloud_client.LaunchInstanceRequest(
        region_name=region,
        instance_type_name=instance_type,
        ssh_key_names=ssh_key_names,
        name=name,
        quantity=1
    )
    
    print(f"\nLaunching {instance_type} instance in {region}...")
    
    try:
        response = api_instance.launch_instance(launch_request)
        
        if response.data and response.data.instance_ids:
            instance_id = response.data.instance_ids[0]
            print(f"✓ Instance launched successfully!")
            print(f"  Instance ID: {instance_id}")
            return instance_id
        else:
            print("ERROR: Instance launch failed")
            sys.exit(1)
            
    except ApiException as e:
        print(f"Error launching instance: {e}")
        sys.exit(1)


def wait_for_instance(api_client, instance_id, timeout=300):
    """Wait for instance to be ready"""
    api_instance = lambda_cloud_client.DefaultApi(api_client)
    
    print("\nWaiting for instance to be ready...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = api_instance.get_instance(instance_id)
            instance = response.data
            
            if instance.status == "active":
                print(f"✓ Instance is ready!")
                print(f"  IP Address: {instance.ip}")
                print(f"  SSH Command: ssh ubuntu@{instance.ip}")
                return instance
            
            print(f"  Status: {instance.status}... waiting")
            time.sleep(10)
            
        except ApiException as e:
            print(f"Error checking instance status: {e}")
            time.sleep(10)
    
    print("ERROR: Timeout waiting for instance to be ready")
    sys.exit(1)


def generate_startup_script():
    """Generate the startup script to run on the instance"""
    wandb_key = os.getenv('WANDB_API_KEY')
    
    script = f"""#!/bin/bash
set -euo pipefail

# Create .env file with credentials
cat > /home/ubuntu/nanochatAquaRat/.env << 'EOF'
WANDB_API_KEY={wandb_key}
WANDB_PROJECT=nanochat-aquarat
WANDB_ENTITY=${{WANDB_ENTITY:-}}
EOF

# Clone repository if not exists
cd /home/ubuntu
if [ ! -d "nanochatAquaRat" ]; then
    git clone https://github.com/HarleyCoops/nanochatAquaRat.git
fi

cd nanochatAquaRat

# Make script executable
chmod +x run_aquarat_small.sh

# Run training in screen session
screen -dmS training bash -c './run_aquarat_small.sh 2>&1 | tee training.log'

echo "Training started in screen session 'training'"
echo "To attach: screen -r training"
echo "To detach: Ctrl+A then D"
echo "To view log: tail -f training.log"
"""
    
    return script


def deploy_and_run(instance_ip):
    """Deploy code and start training on the instance"""
    print("\nDeploying code and starting training...")
    
    startup_script = generate_startup_script()
    
    # Save startup script locally
    script_path = Path("/tmp/lambda_startup.sh")
    script_path.write_text(startup_script)
    
    # Copy startup script to instance
    print("  Copying startup script...")
    subprocess.run([
        "scp", "-o", "StrictHostKeyChecking=no",
        str(script_path),
        f"ubuntu@{instance_ip}:/tmp/startup.sh"
    ], check=True)
    
    # Execute startup script
    print("  Starting training...")
    subprocess.run([
        "ssh", "-o", "StrictHostKeyChecking=no",
        f"ubuntu@{instance_ip}",
        "bash /tmp/startup.sh"
    ], check=True)
    
    print("\n" + "=" * 80)
    print("✓ Training deployment complete!")
    print("=" * 80)
    print("\nTo monitor your training:")
    print(f"  1. SSH: ssh ubuntu@{instance_ip}")
    print(f"  2. Attach to screen: screen -r training")
    print(f"  3. View log: tail -f ~/nanochatAquaRat/training.log")
    print(f"  4. W&B Dashboard: https://wandb.ai")
    print("\nTo detach from screen: Ctrl+A then D")
    print("\nRemember to terminate the instance when done to avoid charges!")


def main():
    parser = argparse.ArgumentParser(description="Launch Lambda Labs instance for AQuA-RAT training")
    parser.add_argument("--instance-type", default="gpu_8x_h100_sxm5", 
                       help="Instance type (default: gpu_8x_h100_sxm5)")
    parser.add_argument("--region", default="us-west-1",
                       help="Region to launch in (default: us-west-1)")
    parser.add_argument("--name", default="nanochat-aquarat-training",
                       help="Instance name (default: nanochat-aquarat-training)")
    parser.add_argument("--list-types", action="store_true",
                       help="List available instance types and exit")
    parser.add_argument("--no-deploy", action="store_true",
                       help="Launch instance but don't deploy code")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Lambda Labs GPU Instance Launcher for AQuA-RAT Training")
    print("=" * 80)
    
    # Check environment variables
    check_env_vars()
    
    # Initialize API client
    api_client = get_api_client()
    
    # List available types if requested
    if args.list_types:
        list_available_instance_types(api_client)
        return
    
    # Launch instance
    instance_id = launch_instance(api_client, args.instance_type, args.region, args.name)
    
    # Wait for instance to be ready
    instance = wait_for_instance(api_client, instance_id)
    
    # Deploy and run training
    if not args.no_deploy:
        time.sleep(5)  # Give SSH a moment to be fully ready
        try:
            deploy_and_run(instance.ip)
        except subprocess.CalledProcessError as e:
            print(f"\nWarning: Deployment encountered an error: {e}")
            print(f"You can manually SSH to the instance and run the training:")
            print(f"  ssh ubuntu@{instance.ip}")
            print(f"  cd nanochatAquaRat && bash run_aquarat_small.sh")
    
    print("\n" + "=" * 80)
    print("Instance Information")
    print("=" * 80)
    print(f"Instance ID: {instance_id}")
    print(f"IP Address: {instance.ip}")
    print(f"Status: {instance.status}")
    print("\nTo terminate this instance:")
    print(f"  python launch_lambda.py --terminate {instance_id}")


if __name__ == "__main__":
    main()
