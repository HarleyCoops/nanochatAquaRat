# Manual Setup Guide for Lambda Labs

This guide walks you through manually launching and configuring a Lambda Labs GPU instance for training the nanochatAquaRat model with RL on AQuA-RAT.

## Prerequisites

1. **Lambda Labs Account**: Sign up at https://cloud.lambdalabs.com
2. **SSH Key**: Add your SSH public key to Lambda Labs
3. **W&B Account**: Sign up at https://wandb.ai and get your API key
4. **Sufficient Credits**: Ensure you have enough credits (~$24 for 8 hours on 8x H100)

## Step 1: Add SSH Key to Lambda Labs

1. Go to https://cloud.lambdalabs.com/ssh-keys
2. Click "Add SSH Key"
3. Paste your public SSH key (from `~/.ssh/id_rsa.pub` or `~/.ssh/id_ed25519.pub`)
4. Give it a name (e.g., "my-laptop")
5. Click "Add SSH Key"

## Step 2: Launch Instance via Web Dashboard

1. Navigate to https://cloud.lambdalabs.com/instances
2. Click **"Launch instance"**
3. Configure your instance:
   - **Instance type**: Select `gpu_8x_h100_sxm5` (8x NVIDIA H100 80GB SXM5)
     - For testing: Use `gpu_1x_a10` or smaller
   - **Region**: Choose a region with availability (e.g., `us-west-1`)
   - **SSH Keys**: Select your SSH key
   - **Filesystem**: (Optional) If you have persistent storage
4. Click **"Launch instance"**
5. Wait 1-2 minutes for the instance to boot

## Step 3: Note Instance Details

Once the instance is running, note:
- **Instance ID**: (e.g., `0123456789abcdef`)
- **IP Address**: (e.g., `123.45.67.89`)
- **SSH Command**: Shown in the web interface

## Step 4: Connect to Instance

Open your terminal and connect:

```bash
ssh ubuntu@<INSTANCE_IP>
```

Example:
```bash
ssh ubuntu@123.45.67.89
```

## Step 5: Set Up Environment

Once connected, run these commands:

### 5.1 Create Environment File

```bash
# Create .env file with your credentials
cat > ~/.env << 'EOF'
WANDB_API_KEY=your-wandb-api-key-here
WANDB_PROJECT=nanochat-aquarat
WANDB_ENTITY=your-wandb-username-or-team
EOF
```

Replace `your-wandb-api-key-here` with your actual W&B API key (get it from https://wandb.ai/authorize)

### 5.2 Clone Repository

```bash
cd ~
git clone https://github.com/HarleyCoops/nanochatAquaRat.git
cd nanochatAquaRat
```

### 5.3 Copy Environment Variables

```bash
# Copy the .env file to the project directory
cp ~/.env .env
```

## Step 6: Start Training

You have two options for running the training:

### Option A: Run in Screen Session (Recommended)

This allows you to detach and let training continue even if you disconnect:

```bash
# Start a screen session
screen -S training

# Run the training script
bash run_aquarat_small.sh
```

**Screen Commands:**
- **Detach from screen**: Press `Ctrl+A` then `D`
- **Reattach to screen**: `screen -r training`
- **List all screen sessions**: `screen -ls`
- **Kill a screen session**: `screen -X -S training quit`

### Option B: Run Directly (Blocks Terminal)

```bash
# Run training directly (terminal will be blocked)
bash run_aquarat_small.sh 2>&1 | tee training.log
```

This saves output to `training.log` for later review.

## Step 7: Monitor Training

### Monitor via Terminal

If using screen:
```bash
# Reattach to see live output
screen -r training

# Or tail the report
tail -f ~/.cache/nanochat/report/report.md
```

### Monitor via Weights & Biases

1. Go to https://wandb.ai
2. Navigate to your project: `nanochat-aquarat`
3. View real-time metrics, losses, and generated samples

Key metrics to watch:
- `rl/acc` - Accuracy on AQuA-RAT
- `rl/mean_reward` - Average reward per sample
- `rl/kl_letter_mean` - KL divergence from initial policy
- `rl/letter_margin_mean` - Confidence in letter choices
- `attn/entropy_mean` - Attention mechanism entropy

## Step 8: Training Timeline

For the **small (depth=8) model**:
- **Base pretraining**: ~1-2 hours
- **Mid-training**: ~30 minutes
- **SFT**: ~30 minutes
- **RL**: ~30 minutes
- **Total**: ~3-4 hours

For the **d-20 model** (561M params):
- **Base pretraining**: ~3-4 hours
- **Mid-training**: ~1 hour
- **SFT**: ~1 hour
- **RL**: ~1 hour
- **Total**: ~6-8 hours

## Step 9: Check Results

After training completes:

```bash
# View the final report
cat ~/.cache/nanochat/report/report.md

# Check RL checkpoint
ls -lh ~/.cache/nanochat/checkpoints/

# View evaluation results
cat ~/.cache/nanochat/evals/
```

## Step 10: Download Artifacts (Optional)

If you want to save the trained model locally:

```bash
# From your local machine (not on the Lambda instance):
# Download checkpoints
scp -r ubuntu@<INSTANCE_IP>:~/.cache/nanochat/checkpoints ./local_checkpoints/

# Download reports
scp -r ubuntu@<INSTANCE_IP>:~/.cache/nanochat/report ./local_reports/

# Download training logs
scp ubuntu@<INSTANCE_IP>:~/nanochatAquaRat/training.log ./training.log
```

## Step 11: Terminate Instance

**IMPORTANT**: Remember to terminate your instance when done to avoid charges!

### Via Web Dashboard:
1. Go to https://cloud.lambdalabs.com/instances
2. Find your instance
3. Click the **"..."** menu
4. Select **"Terminate"**
5. Confirm termination

### Via SSH (before disconnecting):
```bash
# Shutdown the instance (will auto-terminate if configured)
sudo shutdown -h now
```

## Troubleshooting

### Issue: "Out of memory" Error

**Solution**: Reduce batch size in the training script
```bash
# Edit run_aquarat_small.sh and add these flags to the torchrun commands:
--device_batch_size=2  # Reduce from default
```

### Issue: W&B Not Logging

**Solution**: Check your API key
```bash
# Test W&B login
wandb login

# Verify environment variable
echo $WANDB_API_KEY

# Re-run with explicit login
export WANDB_API_KEY=your-key-here
bash run_aquarat_small.sh
```

### Issue: Screen Session Lost

**Solution**: Reattach to screen
```bash
# List all screen sessions
screen -ls

# Reattach to the training session
screen -r training

# If screen says "Detached", force attach
screen -d -r training
```

### Issue: Dataset Download Slow

**Solution**: The script downloads data in parallel. Wait for completion or reduce number of shards.

### Issue: SSH Connection Drops

**Solution**: Use `screen` or `tmux` to keep processes running
```bash
# If you didn't use screen initially and got disconnected:
# Reconnect and check if the process is still running
ps aux | grep python

# If running, you can monitor the log files:
tail -f ~/.cache/nanochat/report/report.md
tail -f ~/nanochatAquaRat/training.log
```

## Cost Estimation

**8x H100 SXM5** pricing (as of reference):
- ~$3.00/hour per GPU
- 8 GPUs = $24/hour
- Small model (4 hours) = ~$96
- d-20 model (8 hours) = ~$192

**Budget-friendly testing options:**
- 1x A10 (24GB): ~$0.60/hour - Good for testing pipeline
- 1x A6000 (48GB): ~$0.80/hour - Can run small model
- 2x A100 (40GB): ~$2.20/hour - Can run d-20 with reduced batch size

## Quick Reference Commands

```bash
# SSH to instance
ssh ubuntu@<INSTANCE_IP>

# Start training in screen
screen -S training
bash run_aquarat_small.sh

# Detach from screen
Ctrl+A then D

# Reattach to screen
screen -r training

# Monitor W&B
Open: https://wandb.ai

# View live report
tail -f ~/.cache/nanochat/report/report.md

# Check GPU usage
nvidia-smi

# Terminate instance (via dashboard)
https://cloud.lambdalabs.com/instances
```

## Support

- **Lambda Labs Support**: https://lambdalabs.com/support
- **W&B Support**: https://docs.wandb.ai
- **nanochat Issues**: https://github.com/HarleyCoops/nanochatAquaRat/issues

## Next Steps

After your model is trained:
1. Download checkpoints for inference
2. Use the web interface: `python -m scripts.chat_web`
3. Test via CLI: `python -m scripts.chat_cli`
4. Share your results on W&B
5. Fine-tune on additional datasets if desired
