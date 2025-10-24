"""
Example: 3D Visualization Showcase for W&B

This script demonstrates how to use the 3D visualization features
with minimal setup and synthetic data for testing.

Usage:
    python examples/showcase_3d_viz_example.py
"""

import numpy as np
import torch
import wandb
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
    print("\n📊 Initializing W&B...")
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
    print("🎯 Simulating training and collecting metrics...")
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

    print(f"✅ Collected {len(data['steps'])} steps of metrics")

    # Generate all 3D visualizations at once
    print("\n🎨 Generating 3D visualizations...")
    viz_dict = create_checkpoint_3d_summary(
        metrics_buffer=metrics_buffer,
        gradient_buffer=gradient_buffer,
        checkpoint_step=200
    )

    print(f"✅ Generated {len(viz_dict)-1} 3D visualizations")

    # Log to W&B
    print("\n📤 Logging to W&B...")
    for viz_name, viz_obj in viz_dict.items():
        if viz_name != 'step':
            wandb.log({viz_name: viz_obj, "step": 200})
            print(f"  - Logged: {viz_name}")

    print("\n✅ Demo complete! Check W&B for visualizations.")
    print(f"   URL: {run.get_url()}")

    wandb.finish()


def demo_manual_viz():
    """Demo: Manual creation of individual visualizations."""
    print("\n" + "="*60)
    print("DEMO 2: Manual 3D Visualization Creation")
    print("="*60)

    # Initialize W&B
    print("\n📊 Initializing W&B...")
    run = wandb.init(
        project="nanochat-3d-showcase",
        name="demo_manual_viz",
        config={"demo": "manual_visualization"}
    )

    # Generate synthetic data
    data = generate_synthetic_training_data(num_steps=150)

    # 1. Loss Landscape
    print("\n🌄 Creating 3D Loss Landscape...")
    loss_landscape_fig = create_3d_loss_landscape(data, window_size=30)
    wandb.log({"manual/loss_landscape": loss_landscape_fig})
    print("  ✅ Loss landscape created")

    # 2. RL Reward Landscape
    print("\n🎁 Creating 3D RL Reward Landscape...")
    reward_landscape_fig = create_3d_rl_reward_landscape(data, window_size=30)
    wandb.log({"manual/reward_landscape": reward_landscape_fig})
    print("  ✅ Reward landscape created")

    # 3. Unified Training Dynamics
    print("\n🔄 Creating 3D Training Dynamics...")
    dynamics_fig = create_3d_loss_gradient_reward_space(data)
    wandb.log({"manual/training_dynamics": dynamics_fig})
    print("  ✅ Training dynamics created")

    # 4. Gradient Flow (manual)
    print("\n📊 Creating 3D Gradient Flow...")
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
    print("  ✅ Gradient flow created")

    print("\n✅ Demo complete! Check W&B for visualizations.")
    print(f"   URL: {run.get_url()}")

    wandb.finish()


def demo_attention_viz():
    """Demo: 3D Attention pattern visualization."""
    print("\n" + "="*60)
    print("DEMO 3: 3D Attention Visualization")
    print("="*60)

    # Initialize W&B
    print("\n📊 Initializing W&B...")
    run = wandb.init(
        project="nanochat-3d-showcase",
        name="demo_attention_viz",
        config={"demo": "attention_visualization"}
    )

    # Create synthetic attention weights
    print("\n🧠 Creating synthetic attention patterns...")
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
    print("\n🎨 Creating 3D attention visualizations for each head...")
    for head in range(num_heads):
        attn_viz = create_3d_attention_visualization(attention, layer_idx=0, head_idx=head)
        if attn_viz is not None:
            wandb.log({f"attention/layer0_head{head}": attn_viz})
            print(f"  ✅ Head {head} visualization created")

    print("\n✅ Demo complete! Check W&B for attention visualizations.")
    print(f"   URL: {run.get_url()}")

    wandb.finish()


def demo_embedding_trajectory():
    """Demo: 3D Embedding space trajectory."""
    print("\n" + "="*60)
    print("DEMO 4: 3D Embedding Trajectory Visualization")
    print("="*60)

    # Initialize W&B
    print("\n📊 Initializing W&B...")
    run = wandb.init(
        project="nanochat-3d-showcase",
        name="demo_embedding_trajectory",
        config={"demo": "embedding_trajectory"}
    )

    # Create synthetic embedding evolution
    print("\n📈 Creating synthetic embedding evolution...")
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
        print(f"\n🎨 Creating 3D embedding trajectory with {method.upper()}...")
        try:
            fig = create_3d_embedding_trajectory(
                embeddings_list,
                labels,
                method=method
            )
            wandb.log({f"embeddings/trajectory_{method}": fig})
            print(f"  ✅ {method.upper()} trajectory created")
        except Exception as e:
            print(f"  ⚠️  Failed to create {method.upper()} trajectory: {e}")

    print("\n✅ Demo complete! Check W&B for embedding trajectories.")
    print(f"   URL: {run.get_url()}")

    wandb.finish()


def main():
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
        print(f"\n❌ Demo 1 failed: {e}")

    try:
        demo_manual_viz()
    except Exception as e:
        print(f"\n❌ Demo 2 failed: {e}")

    try:
        demo_attention_viz()
    except Exception as e:
        print(f"\n❌ Demo 3 failed: {e}")

    try:
        demo_embedding_trajectory()
    except Exception as e:
        print(f"\n❌ Demo 4 failed: {e}")

    print("\n" + "="*60)
    print("All Demos Complete!")
    print("="*60)
    print("\n📊 View your visualizations at: https://wandb.ai")
    print("   Project: nanochat-3d-showcase")
    print("\n💡 Tip: Run actual training with:")
    print("   python -m scripts.chat_rl --run=my_run")
    print("   python -m scripts.chat_sft --run=my_run")
    print("\n")


if __name__ == "__main__":
    main()
