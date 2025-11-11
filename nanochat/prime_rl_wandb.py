"""
Enhanced W&B Logging for Prime Intellect RL Training
====================================================

This module extends the existing W&B visualization capabilities with
Prime Intellect RL-specific metrics and visualizations.
"""

import wandb
import numpy as np
import torch
from typing import Dict, List, Optional, Any
import plotly.graph_objects as go
from collections import deque, defaultdict

from nanochat.wandb_3d_viz import TrainingMetricsBuffer, GradientFlowBuffer


class RLMetricsTracker:
    """
    Comprehensive RL metrics tracker for Prime Intellect training.
    Tracks and logs RL-specific metrics to W&B.
    """

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history

        # Rollout metrics
        self.rollout_rewards = deque(maxlen=max_history)
        self.rollout_lengths = deque(maxlen=max_history)
        self.rollout_steps = deque(maxlen=max_history)

        # Policy metrics
        self.policy_losses = deque(maxlen=max_history)
        self.value_losses = deque(maxlen=max_history)
        self.entropy_values = deque(maxlen=max_history)

        # Advantage metrics
        self.advantages = deque(maxlen=max_history)
        self.advantage_std = deque(maxlen=max_history)

        # Evaluation metrics
        self.eval_accuracy = deque(maxlen=max_history)
        self.eval_pass_at_k = defaultdict(lambda: deque(maxlen=max_history))

        # Token-level metrics
        self.token_logprobs = deque(maxlen=max_history)
        self.token_entropies = deque(maxlen=max_history)

        # Distributed training metrics
        self.world_size_history = deque(maxlen=max_history)
        self.sync_times = deque(maxlen=max_history)

    def log_rollout(
        self,
        step: int,
        rewards: List[float],
        sequence_lengths: List[int],
        advantages: Optional[List[float]] = None,
    ):
        """Log metrics from a rollout batch."""
        self.rollout_steps.append(step)
        self.rollout_rewards.append(np.mean(rewards))
        self.rollout_lengths.append(np.mean(sequence_lengths))

        if advantages is not None:
            self.advantages.append(np.mean(advantages))
            self.advantage_std.append(np.std(advantages))

    def log_policy_update(
        self,
        step: int,
        policy_loss: float,
        value_loss: Optional[float] = None,
        entropy: Optional[float] = None,
    ):
        """Log policy update metrics."""
        self.policy_losses.append(policy_loss)
        if value_loss is not None:
            self.value_losses.append(value_loss)
        if entropy is not None:
            self.entropy_values.append(entropy)

    def log_evaluation(
        self,
        step: int,
        accuracy: float,
        pass_at_k: Optional[Dict[int, float]] = None,
    ):
        """Log evaluation metrics."""
        self.eval_accuracy.append(accuracy)

        if pass_at_k:
            for k, value in pass_at_k.items():
                self.eval_pass_at_k[k].append(value)

    def log_token_metrics(
        self,
        logprobs: List[float],
        entropies: List[float],
    ):
        """Log token-level metrics."""
        self.token_logprobs.append(np.mean(logprobs))
        self.token_entropies.append(np.mean(entropies))

    def log_distributed_metrics(
        self,
        world_size: int,
        sync_time: float,
    ):
        """Log distributed training metrics."""
        self.world_size_history.append(world_size)
        self.sync_times.append(sync_time)

    def get_wandb_summary(self) -> Dict[str, Any]:
        """Get current metrics summary for W&B logging."""
        summary = {}

        if self.rollout_rewards:
            summary["rollout/mean_reward"] = np.mean(list(self.rollout_rewards)[-10:])
            summary["rollout/max_reward"] = max(list(self.rollout_rewards)[-10:])
            summary["rollout/min_reward"] = min(list(self.rollout_rewards)[-10:])

        if self.rollout_lengths:
            summary["rollout/mean_length"] = np.mean(list(self.rollout_lengths)[-10:])

        if self.policy_losses:
            summary["policy/loss"] = list(self.policy_losses)[-1]

        if self.value_losses:
            summary["policy/value_loss"] = list(self.value_losses)[-1]

        if self.entropy_values:
            summary["policy/entropy"] = list(self.entropy_values)[-1]

        if self.advantages:
            summary["advantages/mean"] = np.mean(list(self.advantages)[-10:])
            summary["advantages/std"] = np.mean(list(self.advantage_std)[-10:])

        if self.eval_accuracy:
            summary["eval/accuracy"] = list(self.eval_accuracy)[-1]

        for k, values in self.eval_pass_at_k.items():
            if values:
                summary[f"eval/pass@{k}"] = list(values)[-1]

        if self.token_logprobs:
            summary["tokens/mean_logprob"] = np.mean(list(self.token_logprobs)[-10:])

        if self.token_entropies:
            summary["tokens/mean_entropy"] = np.mean(list(self.token_entropies)[-10:])

        if self.world_size_history:
            summary["distributed/world_size"] = list(self.world_size_history)[-1]

        if self.sync_times:
            summary["distributed/mean_sync_time"] = np.mean(list(self.sync_times)[-10:])

        return summary

    def create_reward_landscape_3d(self) -> Optional[go.Figure]:
        """Create 3D visualization of reward landscape over training."""
        if len(self.rollout_steps) < 2:
            return None

        steps = np.array(list(self.rollout_steps))
        rewards = np.array(list(self.rollout_rewards))

        # Create a smooth surface by interpolating
        if len(steps) > 10:
            window_size = min(10, len(steps) // 5)
            rewards_smooth = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            steps_smooth = steps[:len(rewards_smooth)]
        else:
            rewards_smooth = rewards
            steps_smooth = steps

        # Create 3D surface
        fig = go.Figure(data=[
            go.Scatter3d(
                x=steps_smooth,
                y=rewards_smooth,
                z=np.arange(len(rewards_smooth)),
                mode='lines+markers',
                marker=dict(
                    size=5,
                    color=rewards_smooth,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Reward"),
                ),
                line=dict(color='darkblue', width=2),
                name='Reward Trajectory'
            )
        ])

        fig.update_layout(
            title='RL Reward Landscape (3D Trajectory)',
            scene=dict(
                xaxis_title='Training Step',
                yaxis_title='Mean Reward',
                zaxis_title='Time Sequence',
            ),
            width=900,
            height=700,
        )

        return fig

    def create_advantage_distribution_plot(self) -> Optional[go.Figure]:
        """Create visualization of advantage distribution over time."""
        if len(self.advantages) < 2:
            return None

        steps = np.array(list(self.rollout_steps))
        advantages = np.array(list(self.advantages))
        advantage_stds = np.array(list(self.advantage_std))

        fig = go.Figure()

        # Add mean line
        fig.add_trace(go.Scatter(
            x=steps,
            y=advantages,
            mode='lines',
            name='Mean Advantage',
            line=dict(color='blue', width=2),
        ))

        # Add std dev bands
        fig.add_trace(go.Scatter(
            x=np.concatenate([steps, steps[::-1]]),
            y=np.concatenate([
                advantages + advantage_stds,
                (advantages - advantage_stds)[::-1]
            ]),
            fill='toself',
            fillcolor='rgba(0,100,255,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='±1 Std Dev',
        ))

        fig.update_layout(
            title='Advantage Distribution Over Training',
            xaxis_title='Training Step',
            yaxis_title='Advantage',
            width=900,
            height=500,
        )

        return fig

    def create_pass_at_k_plot(self) -> Optional[go.Figure]:
        """Create pass@k metric visualization."""
        if not self.eval_pass_at_k:
            return None

        fig = go.Figure()

        for k in sorted(self.eval_pass_at_k.keys()):
            values = list(self.eval_pass_at_k[k])
            if values:
                fig.add_trace(go.Scatter(
                    y=values,
                    mode='lines+markers',
                    name=f'Pass@{k}',
                    marker=dict(size=6),
                ))

        fig.update_layout(
            title='Pass@k Metrics During Training',
            xaxis_title='Evaluation Step',
            yaxis_title='Pass@k Accuracy',
            yaxis=dict(range=[0, 1]),
            width=900,
            height=500,
        )

        return fig


def log_prime_rl_metrics_to_wandb(
    wandb_run,
    step: int,
    rl_tracker: RLMetricsTracker,
    base_metrics_buffer: Optional[TrainingMetricsBuffer] = None,
    gradient_buffer: Optional[GradientFlowBuffer] = None,
    create_visualizations: bool = True,
):
    """
    Comprehensive W&B logging function for Prime Intellect RL training.

    Args:
        wandb_run: Active W&B run instance
        step: Current training step
        rl_tracker: RL metrics tracker
        base_metrics_buffer: Optional base training metrics buffer
        gradient_buffer: Optional gradient flow buffer
        create_visualizations: Whether to create and log 3D visualizations
    """
    # Log current RL metrics
    rl_summary = rl_tracker.get_wandb_summary()
    wandb_run.log({"step": step, **rl_summary})

    # Create and log visualizations periodically
    if create_visualizations and step > 0:
        try:
            # Reward landscape
            reward_fig = rl_tracker.create_reward_landscape_3d()
            if reward_fig:
                wandb_run.log({
                    "step": step,
                    "viz/reward_landscape_3d": wandb.Plotly(reward_fig)
                })

            # Advantage distribution
            advantage_fig = rl_tracker.create_advantage_distribution_plot()
            if advantage_fig:
                wandb_run.log({
                    "step": step,
                    "viz/advantage_distribution": wandb.Plotly(advantage_fig)
                })

            # Pass@k metrics
            passk_fig = rl_tracker.create_pass_at_k_plot()
            if passk_fig:
                wandb_run.log({
                    "step": step,
                    "viz/pass_at_k": wandb.Plotly(passk_fig)
                })

        except Exception as e:
            print(f"Warning: Failed to create RL visualizations: {e}")


def create_model_comparison_table(
    model_checkpoints: Dict[str, Dict[str, float]],
) -> wandb.Table:
    """
    Create a W&B table comparing different model checkpoints.

    Args:
        model_checkpoints: Dict mapping checkpoint names to metrics

    Returns:
        W&B Table object
    """
    columns = ["Checkpoint", "Step", "Reward", "Accuracy", "Pass@1", "Pass@8"]
    data = []

    for checkpoint_name, metrics in model_checkpoints.items():
        data.append([
            checkpoint_name,
            metrics.get("step", 0),
            metrics.get("reward", 0.0),
            metrics.get("accuracy", 0.0),
            metrics.get("pass@1", 0.0),
            metrics.get("pass@8", 0.0),
        ])

    return wandb.Table(columns=columns, data=data)


def log_dataset_statistics(
    wandb_run,
    dataset_name: str,
    num_examples: int,
    avg_length: float,
    reward_distribution: Dict[str, int],
):
    """
    Log dataset statistics to W&B.

    Args:
        wandb_run: Active W&B run
        dataset_name: Name of the dataset
        num_examples: Number of examples
        avg_length: Average sequence length
        reward_distribution: Distribution of rewards in the dataset
    """
    wandb_run.log({
        f"dataset/{dataset_name}/num_examples": num_examples,
        f"dataset/{dataset_name}/avg_length": avg_length,
    })

    # Create reward distribution chart
    fig = go.Figure(data=[
        go.Bar(
            x=list(reward_distribution.keys()),
            y=list(reward_distribution.values()),
        )
    ])
    fig.update_layout(
        title=f'{dataset_name} Reward Distribution',
        xaxis_title='Reward Value',
        yaxis_title='Count',
    )

    wandb_run.log({
        f"dataset/{dataset_name}/reward_distribution": wandb.Plotly(fig)
    })


def setup_prime_rl_wandb(
    project_name: str = "nanochat-prime-rl",
    run_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
) -> wandb.sdk.wandb_run.Run:
    """
    Initialize W&B run with Prime Intellect RL configuration.

    Args:
        project_name: W&B project name
        run_name: Optional run name
        config: Configuration dictionary to log
        tags: Optional list of tags

    Returns:
        W&B run instance
    """
    default_tags = ["prime-rl", "nanochat", "aquarat", "reinforcement-learning"]
    if tags:
        default_tags.extend(tags)

    wandb_run = wandb.init(
        project=project_name,
        name=run_name,
        config=config or {},
        tags=default_tags,
        notes="Prime Intellect RL training with enhanced W&B tracking",
    )

    # Define custom metrics
    wandb.define_metric("step")
    wandb.define_metric("rollout/*", step_metric="step")
    wandb.define_metric("policy/*", step_metric="step")
    wandb.define_metric("eval/*", step_metric="step")
    wandb.define_metric("advantages/*", step_metric="step")
    wandb.define_metric("tokens/*", step_metric="step")
    wandb.define_metric("distributed/*", step_metric="step")

    return wandb_run
