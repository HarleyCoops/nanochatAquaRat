# Lambda Labs Training Quickstart

This project provides two ways to run model training on Lambda Labs:

1. **Automated API Script** (Recommended) - Fully automated deployment
2. **Manual Setup** - Step-by-step web dashboard approach

## Prerequisites

Both methods require:
- Lambda Labs account with API key
- SSH key added to Lambda Labs
- Weights & Biases account with API key
- Sufficient credits (~$24/hour for 8x H100)

## Method 1: Automated API Script (Recommended)

### Setup

1. **Set environment variables:**

```bash
export LAMBDA_API_KEY='your-lambda-api-key'
export WANDB_API_KEY='your-wandb-api-key'
```

Get your Lambda API key from: https://cloud.lambdalabs.com/api-keys
Get your W&B API key from: https://wandb.ai/authorize

2. **Install dependencies:**

```bash
pip install lambda-cloud-client
```

### Usage

**Check available instances:**

```bash
python launch_lambda.py --list-types
```

**Launch and start training:**

```bash
# Launch 8x H100 instance (recommended for d-20 model)
python launch_lambda.py --instance-type gpu_8x_h100_sxm5 --region us-west-1

# Launch smaller instance for testing (depth-8 model)
python launch_lambda.py --instance-type gpu_1x_a100 --region us-west-1
```

**Just launch without deploying:**

```bash
python launch_lambda.py --instance-type gpu_8x_h100_sxm5 --no-deploy
```

The script will:
1. ✓ Launch the instance
2. ✓ Wait for it to be ready
3. ✓ Deploy the code
4. ✓ Start training in a screen session
5. ✓ Provide connection details

### Monitor Training

After launching, SSH to the instance:

```bash
ssh ubuntu@<INSTANCE_IP>
```

Then attach to the screen session:

```bash
screen -r training
```

Or view logs:

```bash
tail -f ~/nanochatAquaRat/training.log
```

## Method 2: Manual Setup

For detailed step-by-step instructions, see [LAMBDA_MANUAL_SETUP.md](LAMBDA_MANUAL_SETUP.md)

**Quick summary:**
1. Go to https://cloud.lambdalabs.com/instances
2. Launch instance manually
3. SSH to instance
4. Clone repo and set up .env
5. Run `bash run_aquarat_small.sh`

## Training Configuration

The `run_aquarat_small.sh` script trains a **depth-8 (smaller) model** which takes approximately **3-4 hours** on 8x H100.

### What Gets Trained:

1. **Base Model** (depth-8, ~60M params)
   - Pretrained on limited corpus (24 shards)
   - Faster iteration for testing

2. **Mid-Training**
   - Conversation format adaptation
   - Tool use capabilities

3. **Supervised Fine-Tuning (SFT)**
   - Fine-tuned on AQuA-RAT dataset
   - Multiple-choice math reasoning

4. **Reinforcement Learning (RL)**
   - GRPO-style RL on AQuA-RAT
   - KL divergence tracking
   - Letter-choice logit margin analysis
   - Attention mechanism logging

### W&B Metrics Logged:

- `rl/acc` - Answer accuracy
- `rl/mean_reward` - Average reward
- `rl/kl_letter_mean` - Policy drift (letter-level)
- `rl/kl_sequence_mean` - Policy drift (sequence-level)
- `rl/letter_margin_mean` - Confidence in answers
- `attn/entropy_mean` - Attention patterns

## Model Sizes Available

You can modify `run_aquarat_small.sh` to change the model depth:

| Depth | Params | Training Time | Recommended Instance |
|-------|--------|---------------|---------------------|
| 8     | ~60M   | 3-4 hours     | 1x A100 / 2x A100  |
| 12    | ~180M  | 4-5 hours     | 4x A100             |
| 20    | ~561M  | 6-8 hours     | 8x H100             |
| 26    | ~1.1B  | 10-12 hours   | 8x H100             |

To change depth, edit the `--depth` parameter in `run_aquarat_small.sh`:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20 --run=$WANDB_RUN
```

## Cost Estimates

Based on Lambda Labs pricing:

| Instance Type    | GPUs         | Cost/Hour | Small (4h) | d-20 (8h) |
|------------------|--------------|-----------|------------|-----------|
| gpu_8x_h100_sxm5 | 8x H100 80GB | ~$24.00   | ~$96       | ~$192     |
| gpu_4x_a100      | 4x A100 40GB | ~$8.80    | ~$35       | ~$70      |
| gpu_2x_a100      | 2x A100 40GB | ~$4.40    | ~$18       | ~$35      |
| gpu_1x_a100      | 1x A100 40GB | ~$2.20    | ~$9        | ~$18      |

## Monitoring Options

### 1. SSH + Screen

```bash
ssh ubuntu@<INSTANCE_IP>
screen -r training
# Ctrl+A then D to detach
```

### 2. Weights & Biases

Dashboard: https://wandb.ai

Real-time metrics, attention heatmaps, sample completions

### 3. Log Files

```bash
# Training log
tail -f ~/nanochatAquaRat/training.log

# Progress report
tail -f ~/.cache/nanochat/report/report.md
```

## After Training

### Download Checkpoints

From your local machine:

```bash
scp -r ubuntu@<INSTANCE_IP>:~/.cache/nanochat/checkpoints ./checkpoints/
```

### Run Inference

On the Lambda instance:

```bash
# Web interface
python -m scripts.chat_web

# CLI interface
python -m scripts.chat_cli -p "What is 25 * 37?"

# Evaluate on test set
python -m scripts.chat_eval -- -i rl -a AQUA
```

### Don't Forget to Terminate!

**Via Dashboard:**
https://cloud.lambdalabs.com/instances → Terminate

**Via CLI:**
```bash
sudo shutdown -h now
```

## Troubleshooting

### Issue: API Key Not Working

```bash
# Verify keys are set
echo $LAMBDA_API_KEY
echo $WANDB_API_KEY

# Re-export if needed
export LAMBDA_API_KEY='your-key'
export WANDB_API_KEY='your-key'
```

### Issue: No Available Instances

Lambda Labs instances can be in high demand. Try:
- Different regions (`--region us-east-1`)
- Smaller instance types (`gpu_1x_a100`)
- Check availability: `python launch_lambda.py --list-types`

### Issue: Out of Memory

Edit `run_aquarat_small.sh` and reduce batch size:

```bash
# Add to torchrun commands:
--device_batch_size=2
```

### Issue: Training Stuck

Check GPU utilization:

```bash
nvidia-smi
```

If GPUs are idle, check for errors:

```bash
tail -100 ~/nanochatAquaRat/training.log
```

## Files in This Repository

- `launch_lambda.py` - Automated Lambda Labs launcher
- `run_aquarat_small.sh` - Training script (depth-8 model)
- `LAMBDA_MANUAL_SETUP.md` - Detailed manual setup guide
- `QUICKSTART.md` - This file
- `.env.template` - Environment variable template

## Support

- **Lambda Labs**: https://lambdalabs.com/support
- **Weights & Biases**: https://docs.wandb.ai
- **Project Issues**: https://github.com/HarleyCoops/nanochatAquaRat/issues

## Next Steps

1. ✓ Set up Lambda Labs account and API key
2. ✓ Set up Weights & Biases account
3. ✓ Choose your method (API script or manual)
4. ✓ Launch instance and start training
5. ✓ Monitor via W&B dashboard
6. ✓ Download checkpoints when complete
7. ✓ Terminate instance to stop charges

Happy training!
