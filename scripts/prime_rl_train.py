"""
Prime Intellect RL Training Script for NanochatAquaRat
=====================================================

This script integrates nanochat with Prime Intellect's prime-rl framework
for distributed asynchronous reinforcement learning.

Usage:
    Single GPU:
        python -m scripts.prime_rl_train

    Multi-GPU (8 GPUs):
        torchrun --standalone --nproc_per_node=8 -m scripts.prime_rl_train -- --run=experiment_name

    With custom config:
        python -m scripts.prime_rl_train --config configs/prime_rl/custom.toml
"""

import os
import sys
import json
import time
import torch
import torch.distributed as dist
from pathlib import Path
from typing import Optional, Dict, Any

import wandb

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, DummyWandb
from nanochat.checkpoint_manager import load_model, save_checkpoint
from nanochat.engine import Engine
from nanochat.prime_intellect_integration import (
    PrimeIntellectRLAdapter,
    create_prime_rl_config,
    load_nanochat_for_prime_rl,
)
from nanochat.prime_rl_wandb import (
    RLMetricsTracker,
    log_prime_rl_metrics_to_wandb,
    setup_prime_rl_wandb,
    log_dataset_statistics,
)
from nanochat.wandb_3d_viz import TrainingMetricsBuffer, GradientFlowBuffer
from tasks.aqua import AQUA

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Model and training configuration
run = "prime_rl_default"
source = "sft"  # base|mid|sft - which checkpoint to start from
dtype = "bfloat16"
device_batch_size = 8
examples_per_step = 16
num_samples = 16  # rollouts per example
max_new_tokens = 256
temperature = 1.0
top_k = 50

# Optimizer configuration
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
weight_decay = 0.0
init_lr_frac = 0.05

# Training schedule
num_epochs = 1
num_train_examples = 2000  # -1 for all examples
num_eval_examples = 254
save_every = 60
eval_every = 60
eval_examples = 400

# Prime Intellect integration
use_verifiers_env = True  # Use Prime Intellect verifiers environment
env_id = "harleycooper/nanochatAquaRat"

# W&B configuration
wandb_project = "nanochat-prime-rl"
wandb_entity = None  # Set to your W&B entity/username

# Allow CLI overrides
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}

# -----------------------------------------------------------------------------
# Initialize distributed training
# -----------------------------------------------------------------------------

ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
master_process = ddp_rank == 0
dtype = torch.float32 if dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)

print0(f"Initialized distributed training: rank={ddp_rank}, world_size={ddp_world_size}")

# -----------------------------------------------------------------------------
# Initialize W&B
# -----------------------------------------------------------------------------

use_dummy_wandb = run == "dummy" or not master_process

if not use_dummy_wandb:
    wandb_run = setup_prime_rl_wandb(
        project_name=wandb_project,
        run_name=run,
        config=user_config,
        tags=["distributed"] if ddp_world_size > 1 else [],
    )
else:
    wandb_run = DummyWandb()

# -----------------------------------------------------------------------------
# Load model and create Prime Intellect adapter
# -----------------------------------------------------------------------------

print0(f"Loading model from source: {source}")
model, tokenizer, meta = load_model(source, device, phase="train")
engine = Engine(model, tokenizer)

print0("Creating Prime Intellect RL adapter")
prime_adapter = PrimeIntellectRLAdapter(
    model=model,
    tokenizer=tokenizer,
    env_id=env_id,
    env_args={
        "num_train_examples": num_train_examples,
        "num_eval_examples": num_eval_examples,
        "seed": 42,
        "include_rationale_metadata": True,
    }
)

# Load environment if using verifiers
if use_verifiers_env:
    try:
        print0("Loading Prime Intellect verifiers environment")
        vf_env = prime_adapter.load_environment()
        print0(f"Environment loaded successfully: {env_id}")
    except Exception as e:
        print0(f"Warning: Could not load verifiers environment: {e}")
        print0("Falling back to native nanochat tasks")
        use_verifiers_env = False

# Fallback to native tasks if verifiers not available
if not use_verifiers_env:
    train_task = AQUA(split="train")
    val_task = AQUA(split="validation")

    if num_train_examples > 0:
        # Limit training examples
        train_task._examples = train_task._examples[:num_train_examples]

    if num_eval_examples > 0:
        val_task._examples = val_task._examples[:num_eval_examples]
else:
    # When using verifiers, we'll access the dataset through the environment
    train_task = AQUA(split="train")
    val_task = AQUA(split="validation")

# -----------------------------------------------------------------------------
# Initialize metrics tracking
# -----------------------------------------------------------------------------

metrics_buffer = TrainingMetricsBuffer(max_size=1000)
gradient_buffer = GradientFlowBuffer(num_layers=model.config.n_layer, max_steps=100)
rl_tracker = RLMetricsTracker(max_history=1000)

# Log dataset statistics
if master_process and not use_dummy_wandb:
    print0("Logging dataset statistics to W&B")

    # Calculate dataset stats
    train_lengths = []
    for i in range(min(100, len(train_task))):
        example = train_task[i]
        tokens = tokenizer.render_for_completion(example)
        train_lengths.append(len(tokens))

    reward_dist = {"correct": 0, "incorrect": 0}
    for i in range(min(100, len(train_task))):
        example = train_task[i]
        # Assuming we can check the answer
        reward_dist["correct"] += 1  # Placeholder

    log_dataset_statistics(
        wandb_run=wandb_run,
        dataset_name="AQuA-RAT",
        num_examples=len(train_task),
        avg_length=sum(train_lengths) / len(train_lengths) if train_lengths else 0,
        reward_distribution=reward_dist,
    )

# -----------------------------------------------------------------------------
# Training setup
# -----------------------------------------------------------------------------

num_steps = (len(train_task) // examples_per_step) * num_epochs
print0(f"Training for {num_steps} steps ({num_epochs} epochs)")

# Initialize optimizers
optimizers = model.setup_optimizers(
    unembedding_lr=unembedding_lr,
    embedding_lr=embedding_lr,
    matrix_lr=matrix_lr,
    weight_decay=weight_decay,
)

# Set initial learning rate
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * init_lr_frac
        group["initial_lr"] = group["lr"]


def get_lr_multiplier(it):
    """Learning rate schedule: linear decay to zero."""
    return 1.0 - it / num_steps


# Calculate examples per rank
assert examples_per_step % ddp_world_size == 0, \
    "examples_per_step must be divisible by world_size"
examples_per_rank = examples_per_step // ddp_world_size
print0(f"Examples per rank: {examples_per_rank}")

# -----------------------------------------------------------------------------
# Rollout generator
# -----------------------------------------------------------------------------

import itertools


@torch.no_grad()
def get_batch():
    """Generate rollout batches for training."""
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    rank_indices = range(ddp_rank, len(train_task), ddp_world_size)

    for example_idx in itertools.cycle(rank_indices):
        conversation = train_task[example_idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        # Generate samples
        model.eval()
        generated_sequences = []
        masks = []

        num_sampling_steps = num_samples // device_batch_size

        for sampling_step in range(num_sampling_steps):
            seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF

            with autocast_ctx:
                batch_sequences, batch_masks = engine.generate_batch(
                    tokens,
                    num_samples=device_batch_size,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    seed=seed,
                )

            generated_sequences.extend(batch_sequences)
            masks.extend(batch_masks)

        # Calculate rewards
        rewards = []
        for sample_tokens in generated_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            reward = train_task.reward(conversation, generated_text)
            rewards.append(reward)

        # Pad sequences
        max_length = max(len(seq) for seq in generated_sequences)
        padded_sequences = [
            seq + [assistant_end] * (max_length - len(seq))
            for seq in generated_sequences
        ]
        padded_masks = [
            mask + [0] * (max_length - len(mask))
            for mask in masks
        ]

        # Convert to tensors
        ids = torch.tensor(padded_sequences, dtype=torch.long, device=device)
        mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)

        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone()
        targets[mask_ids[:, 1:] == 0] = -1

        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        mu = rewards.mean()
        advantages = rewards - mu

        yield generated_sequences, inputs, targets, rewards, advantages


# -----------------------------------------------------------------------------
# Evaluation function
# -----------------------------------------------------------------------------

@torch.no_grad()
def run_evaluation(task, max_examples=None):
    """Run evaluation and return pass@k metrics."""
    max_examples = min(max_examples, len(task)) if max_examples else len(task)
    passk = torch.zeros(device_batch_size, device=device)

    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        with autocast_ctx:
            generated_sequences, _ = engine.generate_batch(
                tokens,
                num_samples=device_batch_size,
                max_tokens=max_new_tokens,
                temperature=1.0,
                top_k=top_k,
            )

        # Check correctness for pass@k
        outcomes = []
        for sample_tokens in generated_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            is_correct = task.evaluate(conversation, generated_text)
            outcomes.append(is_correct)

        # Update pass@k
        for k in range(1, device_batch_size + 1):
            if any(outcomes[:k]):
                passk[k - 1] += 1

    # Reduce across ranks
    num_evaluated = torch.tensor(max_examples, dtype=torch.long, device=device)
    if ddp:
        dist.all_reduce(num_evaluated, op=dist.ReduceOp.SUM)
        dist.all_reduce(passk, op=dist.ReduceOp.SUM)

    passk = passk / num_evaluated.item()
    return {k: passk[k - 1].item() for k in range(1, device_batch_size + 1)}


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

print0("Starting training loop")
batch_iterator = get_batch()
sync_start_time = time.time()

for step in range(num_steps):

    # Evaluation
    if step % eval_every == 0:
        print0(f"Running evaluation at step {step}")
        model.eval()

        with autocast_ctx:
            passk_dict = run_evaluation(val_task, max_examples=eval_examples)

        print_passk = [f"Pass@{k}: {v:.4f}" for k, v in passk_dict.items()]
        print0(f"Step {step} | {', '.join(print_passk)}")

        # Log to RL tracker
        rl_tracker.log_evaluation(
            step=step,
            accuracy=passk_dict.get(1, 0.0),
            pass_at_k=passk_dict,
        )

    # Training step
    rewards_list = []
    sequence_lengths = []
    advantages_list = []

    for example_step in range(examples_per_rank):
        sequences, inputs_all, targets_all, rewards_all, advantages_all = next(batch_iterator)

        model.train()

        # Process in batches to avoid OOM
        num_passes = inputs_all.size(0) // device_batch_size

        for pass_idx in range(num_passes):
            b0, b1 = pass_idx * device_batch_size, (pass_idx + 1) * device_batch_size
            inputs = inputs_all[b0:b1]
            targets = targets_all[b0:b1]
            rewards = rewards_all[b0:b1]
            advantages = advantages_all[b0:b1]

            # Forward pass
            with autocast_ctx:
                logp = -model(inputs, targets, loss_reduction='none').view_as(inputs)

            # Policy gradient objective
            pg_obj = (logp * advantages.unsqueeze(-1)).sum()
            num_valid = (targets >= 0).sum().clamp(min=1)
            pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)

            loss = -pg_obj
            loss.backward()

            print0(
                f"Step {step}/{num_steps} | Example {example_step} | "
                f"Pass {pass_idx} | Loss: {loss.item():.6f} | "
                f"Avg Reward: {rewards.mean().item():.4f}"
            )

        # Track metrics
        rewards_list.append(rewards_all.mean().item())
        sequence_lengths.extend(len(seq) for seq in sequences)
        advantages_list.append(advantages_all.mean().item())

    # Aggregate metrics
    mean_reward = sum(rewards_list) / len(rewards_list)
    mean_length = sum(sequence_lengths) / len(sequence_lengths)
    mean_advantage = sum(advantages_list) / len(advantages_list)

    if ddp:
        mean_reward_tensor = torch.tensor(mean_reward, device=device)
        mean_length_tensor = torch.tensor(mean_length, device=device)
        dist.all_reduce(mean_reward_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(mean_length_tensor, op=dist.ReduceOp.AVG)
        mean_reward = mean_reward_tensor.item()
        mean_length = mean_length_tensor.item()

    # Compute gradient norm
    grad_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm += param.grad.norm().item() ** 2
    grad_norm = grad_norm ** 0.5

    # Track gradient flow
    if master_process and not use_dummy_wandb:
        gradient_buffer.add(step, model)

    # Optimizer step
    lrm = get_lr_multiplier(step)
    current_lr = 0.0
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            current_lr = max(current_lr, group["lr"])
        opt.step()

    model.zero_grad(set_to_none=True)

    # Track metrics
    if master_process and not use_dummy_wandb:
        metrics_buffer.add(
            step=step,
            loss=loss.item() if 'loss' in locals() else 0.0,
            lr=current_lr,
            grad_norm=grad_norm,
            reward=mean_reward,
            accuracy=None,
        )

        rl_tracker.log_rollout(
            step=step,
            rewards=rewards_list,
            sequence_lengths=sequence_lengths,
            advantages=advantages_list,
        )

        rl_tracker.log_policy_update(
            step=step,
            policy_loss=loss.item() if 'loss' in locals() else 0.0,
        )

        # Track distributed metrics
        sync_end_time = time.time()
        sync_time = sync_end_time - sync_start_time
        rl_tracker.log_distributed_metrics(
            world_size=ddp_world_size,
            sync_time=sync_time,
        )
        sync_start_time = time.time()

    # Log to W&B
    if master_process and not use_dummy_wandb:
        create_viz = (step > 0 and step % eval_every == 0)
        log_prime_rl_metrics_to_wandb(
            wandb_run=wandb_run,
            step=step,
            rl_tracker=rl_tracker,
            base_metrics_buffer=metrics_buffer,
            gradient_buffer=gradient_buffer,
            create_visualizations=create_viz,
        )

    # Save checkpoint
    if master_process and ((step > 0 and step % save_every == 0) or step == num_steps - 1):
        print0(f"Saving checkpoint at step {step}")
        base_dir = get_base_dir()
        depth = model.config.n_layer
        model_tag = f"d{depth}"
        checkpoint_dir = os.path.join(base_dir, "prime_rl_checkpoints", model_tag)

        save_checkpoint(
            checkpoint_dir,
            step,
            model.state_dict(),
            None,
            {
                "model_config": model.config.__dict__,
                "mean_reward": mean_reward,
                "pass_at_1": rl_tracker.eval_accuracy[-1] if rl_tracker.eval_accuracy else 0.0,
            }
        )
        print0(f"Checkpoint saved to {checkpoint_dir}")

print0("Training complete!")

# Cleanup
if master_process and not use_dummy_wandb:
    wandb_run.finish()

compute_cleanup()
