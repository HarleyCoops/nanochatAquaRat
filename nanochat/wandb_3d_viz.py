"""
W&B 3D Visualization Utilities for NanoChat Training
=====================================================

This module provides advanced 3D visualizations for model training using
Weights & Biases latest features (v0.69.x+).

Features:
- 3D Loss Surface Landscapes
- 3D Gradient Flow Visualization
- 3D RL Reward Landscapes
- 3D Attention Pattern Visualization
- 3D Embedding Space Trajectories
"""

import numpy as np
import torch
import wandb
from typing import Dict, List, Optional, Tuple, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import deque


class TrainingMetricsBuffer:
    """Buffer for storing training metrics to create 3D visualizations."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.losses = deque(maxlen=max_size)
        self.learning_rates = deque(maxlen=max_size)
        self.grad_norms = deque(maxlen=max_size)
        self.steps = deque(maxlen=max_size)
        self.rewards = deque(maxlen=max_size)
        self.accuracies = deque(maxlen=max_size)

    def add(self, step: int, loss: float, lr: float, grad_norm: float,
            reward: Optional[float] = None, accuracy: Optional[float] = None):
        """Add metrics for a training step."""
        self.steps.append(step)
        self.losses.append(loss)
        self.learning_rates.append(lr)
        self.grad_norms.append(grad_norm)
        if reward is not None:
            self.rewards.append(reward)
        if accuracy is not None:
            self.accuracies.append(accuracy)

    def get_arrays(self) -> Dict[str, np.ndarray]:
        """Get numpy arrays of all metrics."""
        return {
            'steps': np.array(list(self.steps)),
            'losses': np.array(list(self.losses)),
            'learning_rates': np.array(list(self.learning_rates)),
            'grad_norms': np.array(list(self.grad_norms)),
            'rewards': np.array(list(self.rewards)) if self.rewards else None,
            'accuracies': np.array(list(self.accuracies)) if self.accuracies else None,
        }


class GradientFlowBuffer:
    """Buffer for tracking gradient flow across layers."""

    def __init__(self, num_layers: int, max_steps: int = 100):
        self.num_layers = num_layers
        self.max_steps = max_steps
        self.gradient_norms = deque(maxlen=max_steps)  # List of arrays [num_layers]
        self.steps = deque(maxlen=max_steps)

    def add(self, step: int, model: torch.nn.Module):
        """Extract and store gradient norms for each layer."""
        layer_grads = []
        for name, param in model.named_parameters():
            if param.grad is not None and 'layers.' in name:
                # Extract layer number and compute grad norm
                layer_grads.append(param.grad.norm().item())

        if layer_grads:
            # Group by layer and compute mean norm per layer
            # Assuming parameters are ordered by layer
            num_params_per_layer = len(layer_grads) // self.num_layers
            layer_norms = []
            for i in range(self.num_layers):
                start_idx = i * num_params_per_layer
                end_idx = start_idx + num_params_per_layer
                layer_mean = np.mean(layer_grads[start_idx:end_idx]) if end_idx <= len(layer_grads) else 0
                layer_norms.append(layer_mean)

            self.steps.append(step)
            self.gradient_norms.append(np.array(layer_norms))

    def get_3d_surface(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get X, Y, Z arrays for 3D surface plot."""
        if not self.gradient_norms:
            return None, None, None

        steps_array = np.array(list(self.steps))
        layers_array = np.arange(self.num_layers)

        # Create meshgrid
        X, Y = np.meshgrid(steps_array, layers_array)

        # Z is gradient norms (layers x steps)
        Z = np.array(list(self.gradient_norms)).T

        return X, Y, Z


def create_3d_loss_landscape(metrics: Dict[str, np.ndarray],
                             window_size: int = 50) -> go.Figure:
    """
    Create a 3D loss landscape visualization.

    Args:
        metrics: Dictionary with 'steps', 'losses', 'learning_rates', 'grad_norms'
        window_size: Size of sliding window for surface generation

    Returns:
        Plotly figure object
    """
    steps = metrics['steps']
    losses = metrics['losses']
    lrs = metrics['learning_rates']
    grad_norms = metrics['grad_norms']

    if len(steps) < window_size:
        window_size = len(steps) // 2

    # Create 3D surface: X=step, Y=learning_rate, Z=loss
    # Use sliding windows to create a surface
    X, Y, Z = [], [], []

    for i in range(0, len(steps) - window_size, max(1, window_size // 4)):
        window_steps = steps[i:i+window_size]
        window_losses = losses[i:i+window_size]
        window_lrs = lrs[i:i+window_size]

        X.append(window_steps)
        Y.append(window_lrs)
        Z.append(window_losses)

    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)

    # Create the 3D surface plot
    fig = go.Figure(data=[
        go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Viridis',
            colorbar=dict(title='Loss'),
            name='Loss Surface'
        )
    ])

    fig.update_layout(
        title='3D Training Loss Landscape',
        scene=dict(
            xaxis_title='Training Step',
            yaxis_title='Learning Rate',
            zaxis_title='Loss',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        width=1000,
        height=800,
    )

    return fig


def create_3d_gradient_flow(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> go.Figure:
    """
    Create 3D gradient flow visualization across layers over time.

    Args:
        X: Steps meshgrid (layers x steps)
        Y: Layer indices meshgrid (layers x steps)
        Z: Gradient norms (layers x steps)

    Returns:
        Plotly figure object
    """
    if X is None or Y is None or Z is None:
        # Return empty figure
        return go.Figure()

    # Create surface plot
    fig = go.Figure(data=[
        go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Plasma',
            colorbar=dict(title='Gradient Norm'),
            name='Gradient Flow'
        )
    ])

    fig.update_layout(
        title='3D Gradient Flow Across Layers',
        scene=dict(
            xaxis_title='Training Step',
            yaxis_title='Layer Index',
            zaxis_title='Gradient Norm',
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=1.2)
            )
        ),
        width=1000,
        height=800,
    )

    return fig


def create_3d_rl_reward_landscape(metrics: Dict[str, np.ndarray],
                                   window_size: int = 50) -> go.Figure:
    """
    Create 3D reward landscape for RL training.

    Args:
        metrics: Dictionary with 'steps', 'rewards', 'losses', 'learning_rates'
        window_size: Size of sliding window

    Returns:
        Plotly figure object
    """
    if metrics.get('rewards') is None or len(metrics['rewards']) == 0:
        return go.Figure()

    steps = metrics['steps']
    rewards = metrics['rewards']
    losses = metrics['losses']

    if len(steps) < window_size:
        window_size = max(2, len(steps) // 2)

    # Create 3D scatter plot with trajectory
    # X=step, Y=loss, Z=reward, colored by step progression

    fig = go.Figure()

    # Add trajectory line
    fig.add_trace(go.Scatter3d(
        x=steps,
        y=losses,
        z=rewards,
        mode='lines+markers',
        marker=dict(
            size=4,
            color=steps,
            colorscale='Turbo',
            colorbar=dict(title='Step'),
            showscale=True,
        ),
        line=dict(
            color=steps,
            colorscale='Turbo',
            width=2
        ),
        name='Training Trajectory',
        text=[f'Step: {s}<br>Loss: {l:.4f}<br>Reward: {r:.4f}'
              for s, l, r in zip(steps, losses, rewards)],
        hoverinfo='text'
    ))

    # Add surface representing reward landscape
    if len(steps) >= window_size:
        # Create interpolated surface
        from scipy.interpolate import griddata

        # Create grid
        xi = np.linspace(steps.min(), steps.max(), 30)
        yi = np.linspace(losses.min(), losses.max(), 30)
        X_grid, Y_grid = np.meshgrid(xi, yi)

        # Interpolate rewards on grid
        points = np.column_stack([steps, losses])
        Z_grid = griddata(points, rewards, (X_grid, Y_grid), method='linear')

        fig.add_trace(go.Surface(
            x=X_grid,
            y=Y_grid,
            z=Z_grid,
            colorscale='RdYlGn',
            opacity=0.6,
            name='Reward Landscape',
            showscale=False,
        ))

    fig.update_layout(
        title='3D RL Reward Landscape (Training Trajectory)',
        scene=dict(
            xaxis_title='Training Step',
            yaxis_title='Loss',
            zaxis_title='Reward',
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.5)
            )
        ),
        width=1200,
        height=900,
    )

    return fig


def create_3d_attention_visualization(attention_weights: torch.Tensor,
                                       layer_idx: int,
                                       head_idx: int) -> wandb.Object3D:
    """
    Create 3D point cloud visualization of attention patterns.

    Args:
        attention_weights: Attention tensor [batch, heads, seq_len, seq_len]
        layer_idx: Layer index for labeling
        head_idx: Head index for labeling

    Returns:
        W&B Object3D for logging
    """
    # Extract single attention matrix
    if attention_weights.dim() == 4:
        attn = attention_weights[0, head_idx].detach().cpu().numpy()  # [seq_len, seq_len]
    else:
        attn = attention_weights.detach().cpu().numpy()

    seq_len = attn.shape[0]

    # Create 3D point cloud: (query_pos, key_pos, attention_value)
    points = []
    colors = []

    for i in range(seq_len):
        for j in range(seq_len):
            attention_val = attn[i, j]
            if attention_val > 0.01:  # Filter low attention values
                points.append([i, j, attention_val])

                # Color by attention strength (RGB)
                r = int(255 * attention_val)
                g = int(255 * (1 - attention_val))
                b = 128
                colors.append([r, g, b])

    # Convert to numpy arrays with RGB colors (nx6 format)
    if points:
        points_array = np.array(points)
        colors_array = np.array(colors)
        point_cloud = np.concatenate([points_array, colors_array], axis=1)

        return wandb.Object3D(point_cloud)
    else:
        return None


def create_3d_embedding_trajectory(embeddings: List[torch.Tensor],
                                   labels: List[str],
                                   method: str = 'pca') -> go.Figure:
    """
    Create 3D embedding space trajectory using dimensionality reduction.

    Args:
        embeddings: List of embedding tensors across training
        labels: List of labels for each point
        method: 'pca', 'tsne', or 'umap'

    Returns:
        Plotly figure object
    """
    # Stack embeddings
    emb_array = torch.stack(embeddings).cpu().numpy()

    # Reshape if needed
    if emb_array.ndim > 2:
        batch_size = emb_array.shape[0]
        emb_array = emb_array.reshape(batch_size, -1)

    # Apply dimensionality reduction to 3D
    if method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=3)
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=3, random_state=42)
    elif method == 'umap':
        try:
            from umap import UMAP
            reducer = UMAP(n_components=3, random_state=42)
        except ImportError:
            # Fallback to PCA if UMAP not available
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=3)
            method = 'pca'
    else:
        raise ValueError(f"Unknown method: {method}")

    # Reduce to 3D
    emb_3d = reducer.fit_transform(emb_array)

    # Create 3D scatter plot with trajectory
    fig = go.Figure()

    # Color by time/step
    colors = np.arange(len(emb_3d))

    fig.add_trace(go.Scatter3d(
        x=emb_3d[:, 0],
        y=emb_3d[:, 1],
        z=emb_3d[:, 2],
        mode='lines+markers',
        marker=dict(
            size=6,
            color=colors,
            colorscale='Rainbow',
            colorbar=dict(title='Training Progress'),
            showscale=True,
        ),
        line=dict(
            color=colors,
            colorscale='Rainbow',
            width=3
        ),
        text=labels,
        name='Embedding Trajectory'
    ))

    fig.update_layout(
        title=f'3D Embedding Space Trajectory ({method.upper()})',
        scene=dict(
            xaxis_title=f'{method.upper()} Component 1',
            yaxis_title=f'{method.upper()} Component 2',
            zaxis_title=f'{method.upper()} Component 3',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1000,
        height=800,
    )

    return fig


def create_3d_loss_gradient_reward_space(metrics: Dict[str, np.ndarray]) -> go.Figure:
    """
    Create unified 3D visualization of loss, gradient, and reward space.

    Args:
        metrics: Dictionary with 'losses', 'grad_norms', 'rewards'

    Returns:
        Plotly figure object
    """
    losses = metrics['losses']
    grad_norms = metrics['grad_norms']
    rewards = metrics.get('rewards')

    if rewards is None or len(rewards) == 0:
        # Create without rewards
        steps = np.arange(len(losses))

        fig = go.Figure(data=[
            go.Scatter3d(
                x=steps,
                y=losses,
                z=grad_norms,
                mode='markers+lines',
                marker=dict(
                    size=4,
                    color=losses,
                    colorscale='Viridis',
                    colorbar=dict(title='Loss'),
                    showscale=True,
                ),
                line=dict(color='cyan', width=2),
                name='Training Path'
            )
        ])

        fig.update_layout(
            title='3D Training Dynamics (Loss vs Gradient)',
            scene=dict(
                xaxis_title='Training Step',
                yaxis_title='Loss',
                zaxis_title='Gradient Norm',
            )
        )
    else:
        # Ensure all arrays have same length
        min_len = min(len(losses), len(grad_norms), len(rewards))
        losses = losses[:min_len]
        grad_norms = grad_norms[:min_len]
        rewards = rewards[:min_len]

        fig = go.Figure(data=[
            go.Scatter3d(
                x=losses,
                y=grad_norms,
                z=rewards,
                mode='markers+lines',
                marker=dict(
                    size=5,
                    color=np.arange(min_len),
                    colorscale='Turbo',
                    colorbar=dict(title='Step'),
                    showscale=True,
                ),
                line=dict(
                    color=np.arange(min_len),
                    colorscale='Turbo',
                    width=2
                ),
                name='Training Trajectory'
            )
        ])

        fig.update_layout(
            title='3D Training Dynamics (Loss-Gradient-Reward Space)',
            scene=dict(
                xaxis_title='Loss',
                yaxis_title='Gradient Norm',
                zaxis_title='Reward',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            )
        )

    fig.update_layout(
        width=1000,
        height=800,
    )

    return fig


def log_3d_plotly_to_wandb(fig: go.Figure, name: str, step: Optional[int] = None):
    """
    Log Plotly 3D figure to W&B.

    Args:
        fig: Plotly figure object
        name: Name for the visualization
        step: Optional step number
    """
    if step is not None:
        wandb.log({name: fig, "step": step})
    else:
        wandb.log({name: fig})


def create_checkpoint_3d_summary(metrics_buffer: TrainingMetricsBuffer,
                                 gradient_buffer: Optional[GradientFlowBuffer] = None,
                                 checkpoint_step: int = 0) -> Dict[str, Any]:
    """
    Create comprehensive 3D visualization summary at checkpoint.

    Args:
        metrics_buffer: Buffer with training metrics
        gradient_buffer: Optional gradient flow buffer
        checkpoint_step: Current training step

    Returns:
        Dictionary of W&B loggable objects
    """
    metrics = metrics_buffer.get_arrays()

    viz_dict = {}

    # 1. Loss Landscape
    if len(metrics['steps']) >= 50:
        loss_landscape = create_3d_loss_landscape(metrics)
        viz_dict['3d/loss_landscape'] = loss_landscape

    # 2. RL Reward Landscape (if rewards available)
    if metrics['rewards'] is not None and len(metrics['rewards']) >= 20:
        reward_landscape = create_3d_rl_reward_landscape(metrics)
        viz_dict['3d/reward_landscape'] = reward_landscape

    # 3. Unified Dynamics
    unified = create_3d_loss_gradient_reward_space(metrics)
    viz_dict['3d/training_dynamics'] = unified

    # 4. Gradient Flow (if available)
    if gradient_buffer is not None:
        X, Y, Z = gradient_buffer.get_3d_surface()
        if X is not None:
            grad_flow = create_3d_gradient_flow(X, Y, Z)
            viz_dict['3d/gradient_flow'] = grad_flow

    # Add checkpoint step
    viz_dict['step'] = checkpoint_step

    return viz_dict
