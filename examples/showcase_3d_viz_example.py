"""
Example: 3D Visualization Showcase for W&B

This script demonstrates how to use the 3D visualization features
with minimal setup and synthetic data for testing.

Usage:
    python examples/showcase_3d_viz_example.py
"""

import sys
import os
import site
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import wandb
from wandb.errors import CommError


def _ensure_python_paths():
    """Ensure site-packages directories are available when running from CLI."""
    candidate_paths = []

    # Collect system site-packages if available.
    try:
        # site.getsitepackages() can return a list depending on environment.
        sys_site = site.getsitepackages()
        if isinstance(sys_site, str):
            candidate_paths.append(sys_site)
        elif sys_site:
            candidate_paths.extend(sys_site)
    except AttributeError:
        # getsitepackages is not available in some virtualenv configurations.
        pass

    # Always include user site-packages in case pip --user was used.
    user_site = site.getusersitepackages()
    if isinstance(user_site, str):
        candidate_paths.append(user_site)
    elif user_site:
        candidate_paths.extend(user_site)

    for path in candidate_paths:
        if path and path not in sys.path:
            sys.path.append(path)


_ensure_python_paths()

# Force UTF-8 stdout on Windows consoles to avoid encode errors with rich text.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Optional defaults sourced from .env when available.
os.environ.setdefault("WANDB_ENTITY", os.environ.get("WANDB_ENTITY", "christian-cooper-us"))
os.environ.setdefault("WANDB_PROJECT", os.environ.get("WANDB_PROJECT", "nanochat-3d-showcase"))

# ---------------------------------------------------------------------------
# W&B replay helpers
# ---------------------------------------------------------------------------

def _extract_column_name(frame: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first non-empty column matching the candidate list."""
    for col in candidates:
        if col in frame.columns and frame[col].notna().any():
            return col
    return None


def _resolve_wandb_run(entity: str, project: str, run_ref: str):
    """Resolve a W&B run by id or display name."""
    api = wandb.Api()
    run_path = f"{entity}/{project}/{run_ref}"
    try:
        return api.run(run_path)
    except (ValueError, CommError):
        pass

    # Attempt lookup by display name.
    try:
        matches = api.runs(f"{entity}/{project}", filters={"display_name": run_ref}, per_page=100)
        for run in matches:
            if run.name == run_ref:
                return run
    except Exception:
        # Fall back to exhaustive search if filters unsupported.
        candidates = api.runs(f"{entity}/{project}", order="-created_at", per_page=200)
        for run in candidates:
            if run.name == run_ref or run.id == run_ref:
                return run

    raise ValueError(f"Could not find W&B run '{run_ref}' in project {entity}/{project}")


def _history_to_metrics(history: pd.DataFrame,
                        limit: Optional[int] = None,
                        overrides: Optional[Dict[str, Optional[str]]] = None
                        ) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    """Convert a W&B history dataframe into TrainingMetricsBuffer-ready arrays."""
    if history.empty:
        raise ValueError("Run history is empty; nothing to visualize.")

    overrides = overrides or {}

    def pick_column(key: str, candidates: List[str]) -> Optional[str]:
        override = overrides.get(key)
        if override:
            if override not in history.columns:
                raise ValueError(f"Requested column '{override}' for '{key}' not found in W&B history.")
            if not history[override].notna().any():
                raise ValueError(f"Override column '{override}' for '{key}' contains only missing values.")
            return override
        return _extract_column_name(history, candidates)

    step_col = pick_column("step", ["step", "_step", "global_step", "trainer/global_step"])
    loss_col = pick_column(
        "loss",
        [
            "train_loss",
            "loss",
            "train/loss",
            "training/loss",
            "train_total_loss",
            "objective",
        ],
    )
    if loss_col is None:
        # Allow reward-only runs to derive pseudo loss later.
        loss_from_reward = True
    else:
        loss_from_reward = False

    lr_col = pick_column("learning_rate", ["lrm", "learning_rate", "lr", "optimizer/learning_rate"])
    grad_col = pick_column("grad", ["grad_norm", "grad_norms", "gradient_norm", "train/grad_norm"])
    reward_col = pick_column("reward", ["reward", "eval/reward", "mean_reward", "train/reward", "pass@1"])
    accuracy_col = pick_column(
        "accuracy",
        [
            "accuracy",
            "eval/accuracy",
            "train/accuracy",
            "val_accuracy",
            "val_acc",
            "arc_easy_acc",
            "mmlu_acc",
            "pass@1",
        ],
    )

    if loss_col is None:
        if reward_col is None:
            raise ValueError("Could not find a loss or reward column in the W&B history.")
        loss_col = reward_col

    working_cols: List[str] = []
    for candidate in [loss_col, step_col, lr_col, grad_col, reward_col, accuracy_col]:
        if candidate and candidate not in working_cols:
            working_cols.append(candidate)

    df = history[working_cols].copy()
    df = df.dropna(subset=[loss_col])

    if step_col is None:
        df = df.reset_index().rename(columns={"index": "auto_step"})
        step_col = "auto_step"
    df = df[df[step_col].notna()]
    df = df.sort_values(step_col)

    if limit is not None and limit > 0:
        df = df.tail(limit)

    df[step_col] = pd.to_numeric(df[step_col], errors="coerce")
    df[loss_col] = pd.to_numeric(df[loss_col], errors="coerce")

    if lr_col:
        df[lr_col] = pd.to_numeric(df[lr_col], errors="coerce").ffill().bfill()
    else:
        # Provide a synthetic learning rate schedule when unavailable.
        lr_col = "__synthetic_lr__"
        df[lr_col] = np.linspace(1.0, 0.1, len(df))

    if grad_col:
        df[grad_col] = pd.to_numeric(df[grad_col], errors="coerce")

    if reward_col:
        df[reward_col] = pd.to_numeric(df[reward_col], errors="coerce")

    if accuracy_col:
        df[accuracy_col] = pd.to_numeric(df[accuracy_col], errors="coerce")

    df = df.dropna(subset=[step_col, loss_col])
    df = df.ffill()

    steps = df[step_col].to_numpy(dtype=float)
    raw_loss_values = df[loss_col].to_numpy(dtype=float)
    losses = raw_loss_values.copy()
    if loss_from_reward:
        finite_mask = np.isfinite(losses)
        if finite_mask.any():
            min_val = np.nanmin(losses[finite_mask])
            max_val = np.nanmax(losses[finite_mask])
            if max_val - min_val > 1e-9:
                losses = 1.0 - ((losses - min_val) / (max_val - min_val))
            else:
                losses = 1.0 - losses
        else:
            losses = np.ones_like(losses)
    learning_rates = df[lr_col].to_numpy(dtype=float)

    if grad_col and df[grad_col].notna().any():
        grad_values = df[grad_col].to_numpy(dtype=float)
        finite_mask = np.isfinite(grad_values)
        if finite_mask.any():
            fill_value = np.nanmean(grad_values[finite_mask])
            if np.isnan(fill_value):
                fill_value = np.mean(np.abs(np.gradient(losses)))
            grad_norms = np.where(finite_mask, grad_values, fill_value)
        else:
            grad_norms = np.abs(np.gradient(losses))
    else:
        grad_norms = np.abs(np.gradient(losses))
        # Avoid zeros which can flatten the visualization.
        grad_norms = np.maximum(grad_norms, 1e-6)

    rewards = None
    if reward_col and df[reward_col].notna().any():
        rewards = df[reward_col].to_numpy(dtype=float)

    accuracies = None
    if accuracy_col and df[accuracy_col].notna().any():
        accuracies = df[accuracy_col].to_numpy(dtype=float)

    loss_label = loss_col
    if loss_from_reward:
        loss_label = f"{loss_col} (inverted from reward)"

    lr_label = lr_col if lr_col != "__synthetic_lr__" else "synthetic"

    used_columns = {
        "step": step_col,
        "loss": loss_label,
        "learning_rate": lr_label,
        "grad_norm": grad_col or "derived_from_loss",
        "reward": reward_col or "not_available",
        "accuracy": accuracy_col or "not_available",
    }

    metrics = {
        "steps": steps,
        "losses": losses,
        "learning_rates": learning_rates,
        "grad_norms": grad_norms,
        "rewards": rewards,
        "accuracies": accuracies,
    }

    return metrics, used_columns


def replay_wandb_run(entity: str,
                     project: str,
                     run_ref: str,
                     *,
                     samples: Optional[int] = None,
                     column_overrides: Optional[Dict[str, Optional[str]]] = None,
                     output_entity: Optional[str] = None,
                     output_project: Optional[str] = None,
                     output_run_name: Optional[str] = None,
                     log_to_wandb: bool = True):
    """Replay metrics from an existing W&B run and generate 3D visualizations."""
    entity = entity or os.environ.get("WANDB_ENTITY")
    project = project or os.environ.get("WANDB_PROJECT")
    if not entity or not project:
        raise ValueError("Both entity and project are required to replay a W&B run.")

    run = _resolve_wandb_run(entity, project, run_ref)
    run_path = "/".join(run.path)

    print("\n" + "=" * 60)
    print("Replaying W&B Run For 3D Visualization")
    print("=" * 60)
    print(f"[INFO] Source run: {run_path}")
    print(f"[INFO] State: {run.state}")
    print(f"[INFO] Created: {run.created_at}")

    history = run.history(samples=samples or 2000)
    metrics, used_columns = _history_to_metrics(history, limit=samples, overrides=column_overrides)

    print("\n[INFO] Columns used for visualization:")
    for key, value in used_columns.items():
        print(f"  - {key.replace('_', ' ').title()}: {value}")

    steps = metrics["steps"]
    total_points = len(steps)
    print(f"\n[INFO] Data points after cleaning: {total_points}")
    if total_points < 10:
        print("[WARN] Not enough datapoints to build rich 3D plots. Results may look sparse.")

    metrics_buffer = TrainingMetricsBuffer(max_size=total_points)
    rewards = metrics["rewards"]
    accuracies = metrics["accuracies"]

    for idx in range(total_points):
        reward_val = float(rewards[idx]) if rewards is not None and np.isfinite(rewards[idx]) else None
        accuracy_val = float(accuracies[idx]) if accuracies is not None and np.isfinite(accuracies[idx]) else None

        metrics_buffer.add(
            step=int(steps[idx]),
            loss=float(metrics["losses"][idx]),
            lr=float(metrics["learning_rates"][idx]),
            grad_norm=float(metrics["grad_norms"][idx]),
            reward=reward_val,
            accuracy=accuracy_val,
        )

    viz_dict = create_checkpoint_3d_summary(
        metrics_buffer=metrics_buffer,
        gradient_buffer=None,
        checkpoint_step=int(steps[-1]) if total_points else 0,
    )

    if log_to_wandb:
        out_entity = output_entity or entity
        out_project = output_project or project
        run_name = output_run_name or f"3d-replay-{run.name or run.id}"

        print(f"\n[INFO] Logging 3D visualizations to {out_entity}/{out_project} as '{run_name}'")
        replay_run = wandb.init(
            entity=out_entity,
            project=out_project,
            name=run_name,
            config={
                "source_run": run_path,
                "source_state": run.state,
                "samples": samples,
            },
            reinit=True,
        )

        source_step = viz_dict.get("step", None)
        for viz_name, viz_obj in viz_dict.items():
            if viz_name == "step":
                continue
            wandb.log({viz_name: viz_obj, "source_step": source_step, "source_run": run_path})

        wandb.finish()
    else:
        print("\n[INFO] Skipping W&B logging (--no-log specified).")

    print("\n[OK] Finished replay. Visualizations generated for:")
    for key in viz_dict.keys():
        if key != "step":
            print(f"  - {key}")
    print("")

    return viz_dict

# Add the parent directory to Python path so we can import nanochat
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from nanochat.wandb_3d_viz import (
    TrainingMetricsBuffer,
    GradientFlowBuffer,
    create_3d_loss_landscape,
    create_3d_gradient_flow,
    create_3d_rl_reward_landscape,
    create_3d_loss_gradient_reward_space,
    create_3d_attention_visualization,
    create_3d_embedding_trajectory,
    create_checkpoint_3d_summary,
)


def generate_synthetic_training_data(num_steps=200):
    """Generate synthetic training data for demonstration."""
    steps = np.arange(num_steps)

    # Simulate loss decay
    losses = 3.0 * np.exp(-steps / 50) + 0.5 + np.random.normal(0, 0.1, num_steps)

    # Simulate learning rate schedule
    learning_rates = 0.001 * (1 - steps / num_steps) + 0.0001

    # Simulate gradient norms
    grad_norms = 0.5 * np.exp(-steps / 100) + 0.1 + np.random.normal(0, 0.05, num_steps)

    # Simulate RL rewards (increasing)
    rewards = 0.3 + 0.4 * (1 - np.exp(-steps / 50)) + np.random.normal(0, 0.05, num_steps)
    rewards = np.clip(rewards, 0, 1)

    # Simulate accuracy
    accuracies = 0.2 + 0.5 * (1 - np.exp(-steps / 60)) + np.random.normal(0, 0.03, num_steps)
    accuracies = np.clip(accuracies, 0, 1)

    return {
        'steps': steps,
        'losses': losses,
        'learning_rates': learning_rates,
        'grad_norms': grad_norms,
        'rewards': rewards,
        'accuracies': accuracies,
    }


def demo_buffers_and_auto_viz():
    """Demo: Using buffers for automatic visualization."""
    print("\n" + "="*60)
    print("DEMO 1: Automatic 3D Visualization with Buffers")
    print("="*60)

    # Initialize W&B (use dummy mode for testing)
    print("\n[INFO] Initializing W&B...")
    run = wandb.init(
        project="nanochat-3d-showcase",
        name="demo_auto_viz",
        config={"demo": "automatic_visualization"}
    )

    # Create buffers
    metrics_buffer = TrainingMetricsBuffer(max_size=1000)
    gradient_buffer = GradientFlowBuffer(num_layers=12, max_steps=100)

    # Generate synthetic data
    data = generate_synthetic_training_data(num_steps=200)

    # Simulate training by adding metrics to buffers
    print("[INFO] Simulating training and collecting metrics...")
    for i in range(len(data['steps'])):
        metrics_buffer.add(
            step=int(data['steps'][i]),
            loss=float(data['losses'][i]),
            lr=float(data['learning_rates'][i]),
            grad_norm=float(data['grad_norms'][i]),
            reward=float(data['rewards'][i]),
            accuracy=float(data['accuracies'][i])
        )

        # Simulate gradient flow (create fake per-layer gradients)
        if i % 2 == 0:  # Every other step
            # Create synthetic per-layer gradient norms
            layer_grads = np.random.exponential(0.5, 12) * (1 - i / 200)
            # Mock model with gradients
            class MockModel:
                def named_parameters(self):
                    for layer_idx in range(12):
                        for param_idx in range(3):  # 3 params per layer
                            yield f"layers.{layer_idx}.param{param_idx}", MockParam(layer_grads[layer_idx])

            class MockParam:
                def __init__(self, grad_val):
                    self.grad = MockGrad(grad_val)

            class MockGrad:
                def __init__(self, val):
                    self.val = val
                def norm(self):
                    return MockNorm(self.val)

            class MockNorm:
                def __init__(self, val):
                    self.val = val
                def item(self):
                    return self.val

            gradient_buffer.add(i, MockModel())

    print(f"[OK] Collected {len(data['steps'])} steps of metrics")

    # Generate all 3D visualizations at once
    print("\n[INFO] Generating 3D visualizations...")
    viz_dict = create_checkpoint_3d_summary(
        metrics_buffer=metrics_buffer,
        gradient_buffer=gradient_buffer,
        checkpoint_step=200
    )

    print(f"[OK] Generated {len(viz_dict)-1} 3D visualizations")

    # Log to W&B
    print("\n[LOG] Logging to W&B...")
    for viz_name, viz_obj in viz_dict.items():
        if viz_name != 'step':
            wandb.log({viz_name: viz_obj, "step": 200})
            print(f"  - Logged: {viz_name}")

    print("\n[OK] Demo complete! Check W&B for visualizations.")
    print(f"   URL: {run.get_url()}")

    wandb.finish()


def demo_manual_viz():
    """Demo: Manual creation of individual visualizations."""
    print("\n" + "="*60)
    print("DEMO 2: Manual 3D Visualization Creation")
    print("="*60)

    # Initialize W&B
    print("\n[INFO] Initializing W&B...")
    run = wandb.init(
        project="nanochat-3d-showcase",
        name="demo_manual_viz",
        config={"demo": "manual_visualization"}
    )

    # Generate synthetic data
    data = generate_synthetic_training_data(num_steps=150)

    # 1. Loss Landscape
    print("\n[INFO] Creating 3D Loss Landscape...")
    loss_landscape_fig = create_3d_loss_landscape(data, window_size=30)
    wandb.log({"manual/loss_landscape": loss_landscape_fig})
    print("  [OK] Loss landscape created")

    # 2. RL Reward Landscape
    print("\n[INFO] Creating 3D RL Reward Landscape...")
    reward_landscape_fig = create_3d_rl_reward_landscape(data, window_size=30)
    wandb.log({"manual/reward_landscape": reward_landscape_fig})
    print("  [OK] Reward landscape created")

    # 3. Unified Training Dynamics
    print("\n[INFO] Creating 3D Training Dynamics...")
    dynamics_fig = create_3d_loss_gradient_reward_space(data)
    wandb.log({"manual/training_dynamics": dynamics_fig})
    print("  [OK] Training dynamics created")

    # 4. Gradient Flow (manual)
    print("\n[INFO] Creating 3D Gradient Flow...")
    # Create synthetic gradient flow data
    steps_array = np.arange(50)
    layers_array = np.arange(12)
    X, Y = np.meshgrid(steps_array, layers_array)

    # Synthetic gradient decay: higher gradients in early layers, decay over time
    Z = np.zeros_like(X, dtype=float)
    for i, layer in enumerate(layers_array):
        for j, step in enumerate(steps_array):
            # Gradient magnitude decreases with layer depth and training time
            Z[i, j] = (1.0 - layer / 12) * (1.0 - step / 50) * np.random.uniform(0.3, 1.0)

    grad_flow_fig = create_3d_gradient_flow(X, Y, Z)
    wandb.log({"manual/gradient_flow": grad_flow_fig})
    print("  [OK] Gradient flow created")

    print("\n[OK] Demo complete! Check W&B for visualizations.")
    print(f"   URL: {run.get_url()}")

    wandb.finish()


def demo_attention_viz():
    """Demo: 3D Attention pattern visualization."""
    print("\n" + "="*60)
    print("DEMO 3: 3D Attention Visualization")
    print("="*60)

    # Initialize W&B
    print("\n[INFO] Initializing W&B...")
    run = wandb.init(
        project="nanochat-3d-showcase",
        name="demo_attention_viz",
        config={"demo": "attention_visualization"}
    )

    # Create synthetic attention weights
    print("\n[INFO] Creating synthetic attention patterns...")
    seq_len = 32
    num_heads = 4

    # Create attention pattern: causal + some diagonal patterns
    attention = torch.zeros(1, num_heads, seq_len, seq_len)

    for head in range(num_heads):
        for i in range(seq_len):
            for j in range(seq_len):
                if j <= i:  # Causal attention
                    # Add different patterns per head
                    if head == 0:
                        # Uniform causal
                        attention[0, head, i, j] = 1.0 / (i + 1)
                    elif head == 1:
                        # Local attention (window of 5)
                        if i - j < 5:
                            attention[0, head, i, j] = 1.0 / min(i + 1, 5)
                    elif head == 2:
                        # Attending to beginning
                        if j < 3:
                            attention[0, head, i, j] = 0.3
                        else:
                            attention[0, head, i, j] = 0.1 / (i - 2)
                    else:
                        # Diagonal attention
                        if abs(i - j) < 3:
                            attention[0, head, i, j] = 0.3

    # Normalize each row to sum to 1
    for head in range(num_heads):
        for i in range(seq_len):
            row_sum = attention[0, head, i].sum()
            if row_sum > 0:
                attention[0, head, i] /= row_sum

    # Create visualizations for each head
    print("\n[INFO] Creating 3D attention visualizations for each head...")
    for head in range(num_heads):
        attn_viz = create_3d_attention_visualization(attention, layer_idx=0, head_idx=head)
        if attn_viz is not None:
            wandb.log({f"attention/layer0_head{head}": attn_viz})
            print(f"  [OK] Head {head} visualization created")

    print("\n[OK] Demo complete! Check W&B for attention visualizations.")
    print(f"   URL: {run.get_url()}")

    wandb.finish()


def demo_embedding_trajectory():
    """Demo: 3D Embedding space trajectory."""
    print("\n" + "="*60)
    print("DEMO 4: 3D Embedding Trajectory Visualization")
    print("="*60)

    # Initialize W&B
    print("\n[INFO] Initializing W&B...")
    run = wandb.init(
        project="nanochat-3d-showcase",
        name="demo_embedding_trajectory",
        config={"demo": "embedding_trajectory"}
    )

    # Create synthetic embedding evolution
    print("\n[INFO] Creating synthetic embedding evolution...")
    num_steps = 50
    embedding_dim = 128

    # Create embeddings that gradually cluster
    embeddings_list = []
    labels = []

    for step in range(num_steps):
        # Create embedding that moves toward a target
        progress = step / num_steps

        # Start random, end clustered
        if step < 10:
            emb = torch.randn(embedding_dim)
        else:
            # Move toward one of 3 cluster centers
            cluster = step % 3
            centers = {
                0: torch.randn(embedding_dim) * 0.5 + 1.0,
                1: torch.randn(embedding_dim) * 0.5 - 1.0,
                2: torch.randn(embedding_dim) * 0.5
            }
            target = centers[cluster]
            noise = torch.randn(embedding_dim) * (1 - progress) * 0.5
            emb = target + noise

        embeddings_list.append(emb)
        labels.append(f"Step {step}")

    # Create visualization with each method
    methods = ['pca', 'tsne']

    for method in methods:
        print(f"\n[INFO] Creating 3D embedding trajectory with {method.upper()}...")
        try:
            fig = create_3d_embedding_trajectory(
                embeddings_list,
                labels,
                method=method
            )
            wandb.log({f"embeddings/trajectory_{method}": fig})
            print(f"  [OK] {method.upper()} trajectory created")
        except Exception as e:
            print(f"  [WARN]  Failed to create {method.upper()} trajectory: {e}")

    print("\n[OK] Demo complete! Check W&B for embedding trajectories.")
    print(f"   URL: {run.get_url()}")

    wandb.finish()


def run_showcase_demos():
    """Run all demos."""
    print("\n" + "="*60)
    print("W&B 3D Visualization Showcase - Example Demonstrations")
    print("="*60)
    print("\nThis script demonstrates all 3D visualization features")
    print("using synthetic data for testing purposes.\n")

    # Run demos
    try:
        demo_buffers_and_auto_viz()
    except Exception as e:
        print(f"\n[ERROR] Demo 1 failed: {e}")

    try:
        demo_manual_viz()
    except Exception as e:
        print(f"\n[ERROR] Demo 2 failed: {e}")

    try:
        demo_attention_viz()
    except Exception as e:
        print(f"\n[ERROR] Demo 3 failed: {e}")

    try:
        demo_embedding_trajectory()
    except Exception as e:
        print(f"\n[ERROR] Demo 4 failed: {e}")

    print("\n" + "="*60)
    print("All Demos Complete!")
    print("="*60)
    print("\n[INFO] View your visualizations at: https://wandb.ai")
    print("   Project: nanochat-3d-showcase")
    print("\n[TIP] Tip: Run actual training with:")
    print("   python -m scripts.chat_rl --run=my_run")
    print("   python -m scripts.chat_sft --run=my_run")
    print("\n")


def parse_cli_args() -> argparse.Namespace:
    """Parse CLI arguments for the showcase script."""
    parser = argparse.ArgumentParser(description="NanoChat 3D visualization showcase.")
    parser.add_argument(
        "--run",
        dest="run_ref",
        help="Replay an existing W&B run (accepts run id or display name).",
    )
    parser.add_argument(
        "--entity",
        help="W&B entity for the source run (defaults to WANDB_ENTITY or 'christian-cooper-us').",
    )
    parser.add_argument(
        "--project",
        help="W&B project for the source run (defaults to WANDB_PROJECT or 'nanochat-3d-showcase').",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Limit the number of history rows to fetch when replaying a run.",
    )
    parser.add_argument(
        "--step-key",
        help="Override the column used for training steps.",
    )
    parser.add_argument(
        "--loss-key",
        help="Override the column used as the loss metric.",
    )
    parser.add_argument(
        "--lr-key",
        help="Override the column used for learning rate.",
    )
    parser.add_argument(
        "--grad-key",
        help="Override the column used for gradient norms.",
    )
    parser.add_argument(
        "--reward-key",
        help="Override the column used for reward metrics.",
    )
    parser.add_argument(
        "--accuracy-key",
        help="Override the column used for accuracy metrics.",
    )
    parser.add_argument(
        "--output-entity",
        help="W&B entity for logging replay visualizations (defaults to source entity).",
    )
    parser.add_argument(
        "--output-project",
        help="W&B project for logging replay visualizations (defaults to source project).",
    )
    parser.add_argument(
        "--output-run-name",
        help="Custom run name for the replay logging run.",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Generate plots locally but skip logging them back to W&B.",
    )
    parser.add_argument(
        "--include-synthetic",
        action="store_true",
        help="Also execute the synthetic showcase demos after replaying a real run.",
    )
    return parser.parse_args()


def main():
    """CLI entry point with optional W&B replay support."""
    args = parse_cli_args()
    replay_ran = False

    if args.run_ref:
        overrides = {
            "step": args.step_key,
            "loss": args.loss_key,
            "learning_rate": args.lr_key,
            "grad": args.grad_key,
            "reward": args.reward_key,
            "accuracy": args.accuracy_key,
        }
        overrides = {key: val for key, val in overrides.items() if val}

        replay_wandb_run(
            entity=args.entity,
            project=args.project,
            run_ref=args.run_ref,
            samples=args.samples,
             column_overrides=overrides if overrides else None,
            output_entity=args.output_entity,
            output_project=args.output_project,
            output_run_name=args.output_run_name,
            log_to_wandb=not args.no_log,
        )
        replay_ran = True

    if not replay_ran or args.include_synthetic:
        run_showcase_demos()


if __name__ == "__main__":
    main()
