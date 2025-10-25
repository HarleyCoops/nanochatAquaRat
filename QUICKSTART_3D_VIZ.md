# Quick Start: W&B 3D Visualization Showcase

Get up and running with 3D training visualizations in 5 minutes!

## Quick Setup

### 1. Install Dependencies

```bash
# Navigate to the repository root
cd nanochatAquaRat  # Or your repo path

# Install visualization dependencies
pip install -r requirements_3d_viz.txt
```

Required packages:
- `plotly>=5.18.0` - Interactive 3D plots
- `scipy>=1.10.0` - Interpolation for reward landscapes
- `scikit-learn>=1.3.0` - Dimensionality reduction (PCA, t-SNE)
- `umap-learn>=0.5.0` - UMAP embedding (optional)

### 2. Login to W&B

```bash
wandb login
```

Or set your API key:
```bash
export WANDB_API_KEY=your_api_key_here
```

### 3. Run Example Demo

Test the visualizations with synthetic data:

```bash
# Make sure you're in the repo root directory
cd nanochatAquaRat  # Or your repo path

# Run the demo
python examples/showcase_3d_viz_example.py
```

**Note**: The script automatically adds the repo to your Python path, so you can run it from the repo root directory.

This will create a W&B project called `nanochat-3d-showcase` with example 3D visualizations.

## Run Real Training with 3D Viz

### Option 1: RL Training (Recommended for Showcase)

Single GPU:
```bash
python -m scripts.chat_rl --run=my_3d_showcase --eval_every=30
```

Multi-GPU (8 GPUs):
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl --run=my_3d_showcase --eval_every=30
```

**What you'll see**:
- 3D Loss Landscape
- 3D Gradient Flow across layers
- 3D RL Reward Landscape (trajectory through loss-reward space)
- Unified 3D Training Dynamics

**W&B Project**: `nanochat-rl`

---

### Option 2: SFT Training

Single GPU:
```bash
python -m scripts.chat_sft --run=my_3d_showcase --eval_every=50
```

Multi-GPU (8 GPUs):
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft --run=my_3d_showcase --eval_every=50
```

**What you'll see**:
- 3D Loss Landscape
- 3D Gradient Flow across layers
- Unified 3D Training Dynamics (loss-gradient space)

**W&B Project**: `nanochat-sft`

---

## View Your Visualizations

1. Go to [https://wandb.ai](https://wandb.ai)
2. Navigate to your project (`nanochat-rl` or `nanochat-sft`)
3. Click on your run (e.g., `my_3d_showcase`)
4. Go to the **Charts** tab
5. Look for visualizations starting with `3d/`

### Interact with 3D Plots

- **Rotate**: Click and drag
- **Zoom**: Scroll wheel
- **Pan**: Right-click and drag (or Shift + drag)
- **Reset view**: Double-click
- **Save image**: Click camera icon in top-right

---

## Available Visualizations

### During Training (every `eval_every` steps)

| Visualization | Path | What it shows |
|--------------|------|---------------|
| **Loss Landscape** | `3d/loss_landscape` | Loss surface over steps × learning rate |
| **Gradient Flow** | `3d/gradient_flow` | Gradient norms across layers × time |
| **Reward Landscape** | `3d/reward_landscape` | RL trajectory in loss-reward space (RL only) |
| **Training Dynamics** | `3d/training_dynamics` | Unified view of all metrics |

### At Training Completion

All visualizations are re-logged with `final/` prefix showing the complete training history.

---

## Quick Tips

### Adjust Visualization Frequency

**More frequent** (every 30 steps):
```bash
python -m scripts.chat_rl --run=my_run --eval_every=30
```

**Less frequent** (every 100 steps, lower overhead):
```bash
python -m scripts.chat_rl --run=my_run --eval_every=100
```

### Test Without GPU

Use dummy mode (no W&B logging):
```bash
python -m scripts.chat_rl --run=dummy
```

### View Console Output

Watch for these messages:
```
Generating 3D visualizations at step 60...
Logged 4 3D visualizations to W&B
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'nanochat'"

This happens when Python can't find the nanochat module. **Solution**:

```bash
# Make sure you're running from the repository root directory
cd nanochatAquaRat  # Or wherever you cloned the repo

# Then run the example
python examples/showcase_3d_viz_example.py
```

The example script automatically adds the repo root to Python's path, but you need to run it from the repo root directory.

**Alternative**: Install the package in development mode (if you have a setup.py):
```bash
pip install -e .
```

### "No module named 'plotly'"

```bash
pip install plotly scipy scikit-learn
```

### "No visualizations appearing in W&B"

Check that:
1. `--run` is NOT "dummy"
2. You're logged into W&B (`wandb login`)
3. Master process (rank 0) is running
4. Look for error messages in console

### "Out of memory when generating viz"

Reduce buffer sizes in the training scripts:
```python
metrics_buffer = TrainingMetricsBuffer(max_size=500)  # Default: 1000
gradient_buffer = GradientFlowBuffer(num_layers=model.config.n_layer, max_steps=50)  # Default: 100
```

Or increase `eval_every` to generate visualizations less frequently.

---

## What Was Added to Your Codebase

### New Files

1. **`nanochat/wandb_3d_viz.py`** (571 lines)
   - All 3D visualization functions
   - Buffers for metric tracking
   - Checkpoint summary generation

2. **`examples/showcase_3d_viz_example.py`** (433 lines)
   - Demo script with synthetic data
   - Examples of all visualization types

3. **`WANDB_3D_SHOWCASE.md`**
   - Complete documentation
   - Technical details
   - Scientific insights

4. **`QUICKSTART_3D_VIZ.md`** (this file)
   - Quick setup guide
   - Command examples

5. **`requirements_3d_viz.txt`**
   - Visualization dependencies

### Modified Files

1. **`scripts/chat_rl.py`**
   - Added 3D viz imports
   - Initialize buffers
   - Track metrics and gradients
   - Log 3D visualizations every `eval_every` steps
   - Final visualization summary

2. **`scripts/chat_sft.py`**
   - Same modifications as `chat_rl.py`

**Total changes**: ~50 lines per training script

---

## Next Steps

1. **Run the demo**: `python examples/showcase_3d_viz_example.py`
2. **Start real training**: `python -m scripts.chat_rl --run=showcase_test`
3. **View in W&B**: Open your browser and explore the 3D plots
4. **Read full docs**: See `WANDB_3D_SHOWCASE.md` for details
5. **Customize**: Adjust buffer sizes, visualization frequency, etc.

---

## Key Documentation

- **Full Showcase**: `WANDB_3D_SHOWCASE.md` - Complete technical documentation
- **Example Script**: `examples/showcase_3d_viz_example.py` - Working demo code
- **Viz Module**: `nanochat/wandb_3d_viz.py` - All visualization functions
- **W&B Docs**: https://docs.wandb.ai/ref/python/data-types/object3d/

---

## You're Ready!

You now have state-of-the-art 3D training visualizations integrated into your model training pipeline!

```bash
# Let's go!
python -m scripts.chat_rl --run=my_first_3d_showcase --eval_every=30
```

Check W&B to see your beautiful 3D visualizations!

---

*Questions? See `WANDB_3D_SHOWCASE.md` for detailed documentation.*
