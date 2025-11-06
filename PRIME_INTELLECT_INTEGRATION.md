# Prime Intellect RL Integration Guide

This document describes how to use nanochatAquaRat with Prime Intellect's distributed reinforcement learning framework.

## Overview

NanochatAquaRat now integrates with [Prime Intellect's prime-rl framework](https://github.com/PrimeIntellect-ai/prime-rl) and [verifiers environments](https://github.com/PrimeIntellect-ai/verifiers) to enable:

- **Distributed Asynchronous RL Training**: Scale training across multiple nodes and GPUs
- **Environment Hub Integration**: Use the nanochatAquaRat verifier environment from Prime Intellect's hub
- **Enhanced W&B Tracking**: Comprehensive metrics and 3D visualizations for RL training
- **New Dataset Support**: Easily integrate new datasets for further training and refinement

## Features

### 1. Prime Intellect Integration

- **Verifiers Environment**: Pre-built environment at `harleycooper/nanochatAquaRat` on Prime Intellect's Environments Hub
- **GRPO Algorithm**: Group Relative Policy Optimization for efficient RL training
- **Distributed Training**: Multi-GPU and multi-node support via PyTorch DDP
- **Async Rollouts**: Non-blocking rollout generation for improved throughput

### 2. Enhanced W&B Tracking

The integration includes comprehensive Weights & Biases logging:

#### Standard Metrics
- Loss, learning rate, gradient norms
- Training step timing and throughput

#### RL-Specific Metrics
- Rollout rewards (mean, max, min)
- Sequence lengths
- Advantage statistics (mean, std)
- Policy loss and value loss
- Token-level log probabilities and entropy

#### Evaluation Metrics
- Pass@k accuracy (k=1,2,4,8,...)
- Categorical accuracy
- Exact match rewards

#### Distributed Training Metrics
- World size tracking
- Synchronization times
- Per-rank statistics

#### 3D Visualizations
- **Reward Landscape**: 3D trajectory of rewards over training
- **Advantage Distribution**: Temporal evolution of advantage statistics
- **Gradient Flow**: Per-layer gradient norms across training
- **Loss Surface**: 3D visualization of loss landscape
- **Pass@k Curves**: Multi-line plots of pass@k metrics

### 3. Flexible Configuration

Four pre-configured training setups:

1. **default.toml**: Standard training (2000 examples, 400 steps)
2. **small_scale.toml**: Quick debugging (500 examples, 100 steps)
3. **distributed.toml**: Multi-GPU optimized (10k examples, 1000 steps)
4. **full_dataset.toml**: Complete corpus (~97k examples, 2000 steps)

## Installation

### 1. Install Dependencies

```bash
# Install base nanochat dependencies
pip install -e .

# Install Prime Intellect verifiers library
pip install verifiers

# Optional: Install prime-rl framework (for full integration)
pip install git+https://github.com/PrimeIntellect-ai/prime-rl.git
```

### 2. Configure Environment

Copy and configure the environment template:

```bash
cp .env.template .env
```

Edit `.env` to add:

```bash
# Weights & Biases (required for tracking)
WANDB_API_KEY=your-wandb-api-key
WANDB_PROJECT=nanochat-prime-rl
WANDB_ENTITY=your-username

# Prime Intellect (optional, for advanced features)
PRIME_INTELLECT_API_KEY=your-api-key
PRIME_INTELLECT_ENV_ID=harleycooper/nanochatAquaRat
```

### 3. Verify Environment Setup

Check that the verifiers environment is available:

```bash
python -c "import verifiers; print('Verifiers available:', verifiers.__version__)"
```

Test the local environment:

```bash
python -c "from environments.nanochatAquaRat.nanochatAquaRat import load_environment; env = load_environment(); print('Environment loaded:', env)"
```

## Usage

### Quick Start

Train with default configuration on a single GPU:

```bash
python -m scripts.prime_rl_train
```

### Multi-GPU Training

Use `torchrun` for distributed training:

```bash
# 8 GPUs on single node
torchrun --standalone --nproc_per_node=8 -m scripts.prime_rl_train -- --run=my_experiment

# 4 nodes x 8 GPUs each (32 GPUs total)
torchrun --nnodes=4 --nproc_per_node=8 \
  --master_addr=$MASTER_ADDR --master_port=29500 \
  -m scripts.prime_rl_train -- --run=distributed_32gpu
```

### Using Configuration Files

Train with a specific configuration:

```bash
# Small-scale for debugging
python -m scripts.prime_rl_train --config configs/prime_rl/small_scale.toml

# Full dataset training
python -m scripts.prime_rl_train --config configs/prime_rl/full_dataset.toml

# Distributed training on 8 GPUs
torchrun --standalone --nproc_per_node=8 \
  -m scripts.prime_rl_train -- --config configs/prime_rl/distributed.toml
```

### Custom Configuration

Override specific parameters via CLI:

```bash
python -m scripts.prime_rl_train \
  --run=custom_experiment \
  --num_train_examples=5000 \
  --learning_rate=1e-4 \
  --num_samples=32 \
  --eval_every=50
```

## Advanced Features

### 1. Using the Prime Intellect Adapter

The `PrimeIntellectRLAdapter` bridges nanochat models with Prime Intellect:

```python
from nanochat.prime_intellect_integration import PrimeIntellectRLAdapter
from nanochat.checkpoint_manager import load_model

# Load your trained model
model, tokenizer, _ = load_model("sft", "cuda", phase="train")

# Create adapter
adapter = PrimeIntellectRLAdapter(
    model=model,
    tokenizer=tokenizer,
    env_id="harleycooper/nanochatAquaRat",
    env_args={
        "num_train_examples": 2000,
        "num_eval_examples": 254,
    }
)

# Load environment
env = adapter.load_environment()

# Generate responses
messages = [{"role": "user", "content": "What is 2+2?"}]
responses = adapter.generate_response(messages, num_samples=4)
```

### 2. Custom Reward Functions

Extend the verifiers environment with custom rewards:

```python
from environments.nanochatAquaRat.nanochatAquaRat import load_environment
import verifiers as vf

# Load base environment
env = load_environment()

# Add custom reward function
def custom_format_reward(completion, parser, **kwargs):
    # Reward for showing work
    if "because" in completion.lower() or "therefore" in completion.lower():
        return 0.2
    return 0.0

# Add to rubric
env.rubric.add_reward_func(custom_format_reward, weight=0.2)
```

### 3. Integrating New Datasets

Add a new dataset while maintaining the RL infrastructure:

```python
from datasets import load_dataset
import verifiers as vf

# Load your dataset
new_dataset = load_dataset("your/dataset", split="train")

# Format to match verifiers expectations
def format_example(example):
    return {
        "prompt": [{"role": "user", "content": example["question"]}],
        "answer": example["answer"],
        "metadata": {}
    }

formatted_ds = new_dataset.map(format_example)

# Create new environment
def custom_reward(completion, answer, **kwargs):
    # Your reward logic
    return 1.0 if answer in completion else 0.0

rubric = vf.Rubric()
rubric.add_reward_func(custom_reward)

custom_env = vf.SingleTurnEnv(
    dataset=formatted_ds,
    rubric=rubric,
    message_type="chat",
    env_id="your/environment-name"
)

# Now train with this environment
# (modify scripts/prime_rl_train.py to use custom_env)
```

### 4. Advanced W&B Logging

Use the enhanced logging utilities:

```python
from nanochat.prime_rl_wandb import (
    RLMetricsTracker,
    setup_prime_rl_wandb,
    log_prime_rl_metrics_to_wandb
)

# Initialize W&B with custom config
wandb_run = setup_prime_rl_wandb(
    project_name="my-rl-project",
    run_name="experiment-1",
    config={"custom_param": 42},
    tags=["custom", "experiment"]
)

# Track metrics
rl_tracker = RLMetricsTracker()

# During training loop
rl_tracker.log_rollout(
    step=step,
    rewards=rewards_list,
    sequence_lengths=lengths,
    advantages=advantages
)

# Log to W&B with visualizations
log_prime_rl_metrics_to_wandb(
    wandb_run=wandb_run,
    step=step,
    rl_tracker=rl_tracker,
    create_visualizations=True
)
```

## Configuration Reference

### Environment Arguments

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `system_prompt` | str | Default tutor prompt | System message for algebra tutor |
| `train_split` | str | "train" | Dataset split for training |
| `eval_split` | str | "validation" | Dataset split for evaluation |
| `num_train_examples` | int | -1 | Number of training examples (-1 = all) |
| `num_eval_examples` | int | -1 | Number of eval examples (-1 = all) |
| `seed` | int | 42 | Random seed for reproducibility |
| `include_rationale_metadata` | bool | true | Include human rationales in metadata |
| `data_dir` | str | null | Local directory with preprocessed data |
| `cache_dir` | str | null | Hugging Face cache directory |

### Training Arguments

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `run` | str | "prime_rl_default" | W&B run name |
| `source` | str | "sft" | Checkpoint source (base/mid/sft) |
| `device_batch_size` | int | 8 | Max batch size per device |
| `examples_per_step` | int | 16 | Training examples per step |
| `num_samples` | int | 16 | Rollouts per example |
| `max_new_tokens` | int | 256 | Max tokens to generate |
| `temperature` | float | 1.0 | Sampling temperature |
| `top_k` | int | 50 | Top-k filtering |
| `unembedding_lr` | float | 0.004 | LR for output head |
| `embedding_lr` | float | 0.2 | LR for embeddings |
| `matrix_lr` | float | 0.02 | LR for transformer matrices |
| `weight_decay` | float | 0.0 | Weight decay coefficient |
| `num_epochs` | int | 1 | Number of training epochs |
| `save_every` | int | 60 | Save checkpoint every N steps |
| `eval_every` | int | 60 | Evaluate every N steps |

## Monitoring Training

### W&B Dashboard

Access your training metrics at: `https://wandb.ai/<entity>/<project>/runs/<run-id>`

Key sections to monitor:

1. **rollout/** - Rollout statistics (rewards, lengths)
2. **policy/** - Policy gradient metrics
3. **eval/** - Evaluation accuracy and pass@k
4. **advantages/** - Advantage distribution
5. **viz/** - 3D visualizations

### Example Queries

View reward trends:
```python
import wandb
api = wandb.Api()
runs = api.runs("entity/nanochat-prime-rl")
for run in runs:
    print(f"{run.name}: final reward = {run.summary.get('rollout/mean_reward')}")
```

### Checkpoints

Checkpoints are saved to:
```
~/.cache/nanochat/prime_rl_checkpoints/d{depth}/step_{N}/
├── model.pt           # Model state dict
├── metadata.json      # Training metadata
```

Load a checkpoint:
```python
from nanochat.checkpoint_manager import load_model
model, tokenizer, metadata = load_model(
    source="prime_rl",
    device="cuda",
    model_tag="d20",
    step=400
)
```

## Troubleshooting

### Common Issues

#### 1. Verifiers import error
```
ImportError: No module named 'verifiers'
```
**Solution**: Install verifiers: `pip install verifiers`

#### 2. CUDA out of memory
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce `device_batch_size` or `num_samples`:
```bash
python -m scripts.prime_rl_train --device_batch_size=4 --num_samples=8
```

#### 3. Distributed training hangs
**Solution**: Check that all processes can communicate:
```bash
# Set environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=8
export RANK=0  # Set appropriately for each process
```

#### 4. W&B login issues
**Solution**: Login explicitly:
```bash
wandb login
# Or set API key
export WANDB_API_KEY=your-api-key
```

### Debug Mode

Run in debug mode with minimal configuration:

```bash
python -m scripts.prime_rl_train \
  --run=debug \
  --num_train_examples=10 \
  --num_samples=2 \
  --eval_every=5 \
  --save_every=5
```

## Performance Tips

### 1. Optimize Batch Sizes

Balance GPU memory and throughput:
```python
# High memory GPU (A100 80GB)
device_batch_size = 16
num_samples = 32

# Medium memory GPU (A100 40GB)
device_batch_size = 8
num_samples = 16

# Low memory GPU (RTX 3090 24GB)
device_batch_size = 4
num_samples = 8
```

### 2. Use Mixed Precision

Already enabled by default with `bfloat16`. To use float32:
```bash
python -m scripts.prime_rl_train --dtype=float32
```

### 3. Gradient Accumulation

Simulate larger batch sizes:
```python
# Effective batch size = examples_per_step * num_samples
examples_per_step = 16
num_samples = 16
# Total = 256 sequences per step
```

### 4. Async Evaluation

Evaluation runs synchronously by default. For faster training, reduce evaluation frequency:
```bash
python -m scripts.prime_rl_train --eval_every=200
```

## Integration with Prime RL Framework

For full Prime Intellect prime-rl integration:

### 1. Install Prime RL
```bash
pip install git+https://github.com/PrimeIntellect-ai/prime-rl.git
```

### 2. Use Prime RL CLI
```bash
# Install environment to Prime Intellect hub
cd environments/nanochatAquaRat
uv run vf-install . --from-local

# Run evaluation
uv run vf-eval harleycooper/nanochatAquaRat -m gpt-4o-mini -n 100

# Launch RL training with prime-rl
uv run vf-rl @ ../../configs/prime_rl/default.toml
```

### 3. Distributed Async Training

See [prime-rl documentation](https://github.com/PrimeIntellect-ai/prime-rl) for:
- Multi-node orchestration
- Async rollout generation
- FSDP2 training
- vLLM inference backend

## Next Steps

1. **Experiment with hyperparameters**: Try different learning rates, rollout counts
2. **Add custom rewards**: Implement domain-specific reward functions
3. **Integrate new datasets**: Extend to other reasoning tasks
4. **Scale up training**: Use distributed training on multiple nodes
5. **Analyze results**: Use W&B visualizations to understand training dynamics

## Resources

- [Prime Intellect Environments Hub](https://app.primeintellect.ai/dashboard/environments)
- [Prime RL GitHub](https://github.com/PrimeIntellect-ai/prime-rl)
- [Verifiers GitHub](https://github.com/PrimeIntellect-ai/verifiers)
- [NanochatAquaRat Environment](https://app.primeintellect.ai/environments/harleycooper/nanochatAquaRat)
- [W&B Documentation](https://docs.wandb.ai/)

## Support

For issues or questions:
- Open an issue on the nanochatAquaRat GitHub repository
- Check Prime Intellect documentation
- Join Prime Intellect Discord community

## License

This integration maintains the same license as the base nanochatAquaRat project.
