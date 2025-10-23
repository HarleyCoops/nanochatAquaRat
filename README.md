# nanochatAquaRat

nanochat RL tuned on the [DeepMind AQuA-RAT dataset](https://huggingface.co/datasets/deepmind/aqua_rat).

## Quick Start

### Option 1: Lambda Labs (Recommended)

Run training on Lambda Labs cloud GPUs with full automation:

```bash
# Set up credentials
export LAMBDA_API_KEY='your-lambda-api-key'
export WANDB_API_KEY='your-wandb-api-key'

# Launch and start training
python launch_lambda.py --instance-type gpu_8x_h100_sxm5 --region us-west-1
```

**See [QUICKSTART.md](QUICKSTART.md) for detailed Lambda Labs instructions.**

### Option 2: Local/Custom Setup

Run the lightweight end-to-end pipeline locally (depth=8 base, RL on AQuA-RAT):

```bash
# Set up environment
cp .env.template .env
# Edit .env with your WANDB_API_KEY

# Run training
bash run_aquarat_small.sh
```

## What This Does

The training pipeline will:

1. Bootstrap the Python environment with `uv` and activate the project virtualenv
2. Authenticate with Weights & Biases (using credentials from `.env`)
3. Build the `rustbpe` tokenizer and fetch a reduced pretraining corpus
4. Prepare the [DeepMind AQuA-RAT dataset](https://huggingface.co/datasets/deepmind/aqua_rat) for supervised fine-tuning and RL
5. Clone or update Google DeepMind's mechanistic interpretability tooling for attention diagnostics
6. Train a depth-8 base model (~60M params)
7. Run mid-stage training + supervised fine-tuning (SFT)
8. Launch RL session on AQuA-RAT with:
   - KL divergence tracking (letter-level and sequence-level)
   - Letter-choice logit margin analysis
   - Attention mechanism logging
   - Real-time W&B metrics
9. Evaluate the RL checkpoint and generate comprehensive reports

**Training Time:**
- Depth-8 model: ~3-4 hours on 8x H100
- Depth-20 model (561M params): ~6-8 hours on 8x H100

## Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Fast track guide for Lambda Labs deployment
- **[LAMBDA_MANUAL_SETUP.md](LAMBDA_MANUAL_SETUP.md)** - Detailed manual setup walkthrough
- **[.env.template](.env.template)** - Environment variable configuration template

## Key Features

### AQuA-RAT Multiple Choice Math Reasoning
- Fine-tuned on [DeepMind's AQuA-RAT dataset](https://huggingface.co/datasets/deepmind/aqua_rat)
- Multiple choice (A-E) mathematical reasoning
- ~97,000+ algebra and word problems with rationales

### Advanced RL Implementation
- **GRPO-style on-policy RL** (Group Relative Policy Optimization)
- **KL divergence tracking** at both letter-level and sequence-level
- **Letter-choice logit margin analysis** for confidence measurement
- **Reward shaping** for stable early-stage learning

### Mechanistic Interpretability
- Integration with Google DeepMind's interpretability toolkit
- **Attention pattern logging** during training
- **Per-layer entropy metrics** to track attention evolution
- **W&B visualization** of attention heatmaps and distributions

### Comprehensive Monitoring
All metrics logged to Weights & Biases:
- `rl/acc` - Answer accuracy on AQuA-RAT
- `rl/mean_reward` - Average reward per generation
- `rl/kl_letter_mean` - Policy drift (letter-level)
- `rl/kl_sequence_mean` - Policy drift (sequence-level)
- `rl/letter_margin_mean` - Confidence in answer selection
- `attn/entropy_mean` - Attention mechanism patterns
- `rl/letter_pred_counts/*` - Distribution over A-E choices

## Model Configurations

| Depth | Parameters | Training Time | Best Instance Type | Estimated Cost |
|-------|------------|---------------|-------------------|----------------|
| 8     | ~60M       | 3-4 hours     | 1-2x A100        | ~$18-35        |
| 12    | ~180M      | 4-5 hours     | 4x A100          | ~$35-45        |
| 20    | ~561M      | 6-8 hours     | 8x H100          | ~$144-192      |
| 26    | ~1.1B      | 10-12 hours   | 8x H100          | ~$240-288      |

To change model depth, edit the `--depth` parameter in `run_aquarat_small.sh`.

## Requirements

- Python 3.8+
- CUDA-capable GPU(s)
- 40GB+ GPU memory (for depth-8)
- 80GB+ GPU memory per GPU (for depth-20)
- Lambda Labs account (for cloud deployment)
- Weights & Biases account (for logging)

## Expected Results

After training, you should see:

**Base Model Performance:**
- CORE benchmark evaluation results
- Pretraining loss curves
- Sample generations

**After RL:**
- AQuA-RAT dev set accuracy: 30-50% (depth-8), 40-60% (depth-20)
- Improved reasoning coherence
- Better multiple-choice selection confidence
- Stable attention patterns

## Important Notes

### For Lambda Labs Users
- **Always terminate instances** after training to avoid unnecessary charges
- Monitor your spending in the Lambda Labs dashboard
- Check instance availability before launching (can be limited during peak times)

### Cost Management
- Start with depth-8 model to validate pipeline (~$18-35 total)
- Use smaller instance types for testing (1x A10 at ~$0.60/hr)
- Monitor training via W&B to catch issues early

### Known Limitations
- RL on AQuA-RAT is experimental and results may vary
- Attention logging adds ~5-10% overhead
- KL computation can be expensive with large batch sizes

## Contributing

This project is based on the nanochat framework. For issues specific to:
- **AQuA-RAT training**: Open an issue in this repository
- **Base nanochat framework**: Refer to the upstream nanochat project
- **Lambda Labs deployment**: Check [LAMBDA_MANUAL_SETUP.md](LAMBDA_MANUAL_SETUP.md)

## License

This project inherits the license from the base nanochat project.

## Acknowledgments

- **nanochat framework** - Base training pipeline
- **DeepMind** - AQuA-RAT dataset and mechanistic interpretability tools
- **Lambda Labs** - Cloud GPU infrastructure
- **Weights & Biases** - Experiment tracking and visualization

## Support

- **Lambda Labs Support**: https://lambdalabs.com/support
- **Weights & Biases Docs**: https://docs.wandb.ai
- **Project Issues**: https://github.com/HarleyCoops/nanochatAquaRat/issues
