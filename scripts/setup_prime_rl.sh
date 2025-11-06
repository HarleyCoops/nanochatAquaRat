#!/bin/bash
# Setup script for Prime Intellect RL integration

set -e

echo "========================================="
echo "Prime Intellect RL Integration Setup"
echo "========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Must run from nanochatAquaRat root directory"
    exit 1
fi

# Install base dependencies
echo "1. Installing base nanochat dependencies..."
pip install -e . || {
    echo "Error: Failed to install base dependencies"
    exit 1
}

# Install Prime RL dependencies
echo ""
echo "2. Installing Prime Intellect dependencies..."
pip install verifiers || {
    echo "Warning: Failed to install verifiers. You may need to install it manually."
}

# Optional: Install full prime-rl framework
read -p "Do you want to install the full prime-rl framework? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing prime-rl framework..."
    pip install git+https://github.com/PrimeIntellect-ai/prime-rl.git || {
        echo "Warning: Failed to install prime-rl. You can install it later if needed."
    }
fi

# Create .env if it doesn't exist
echo ""
echo "3. Setting up environment configuration..."
if [ ! -f ".env" ]; then
    echo "Creating .env from template..."
    cp .env.template .env
    echo "✓ Created .env file. Please edit it to add your API keys."
else
    echo "✓ .env file already exists."
fi

# Create config directory if it doesn't exist
echo ""
echo "4. Setting up configuration files..."
mkdir -p configs/prime_rl
echo "✓ Configuration directory ready."

# Check W&B login
echo ""
echo "5. Checking Weights & Biases setup..."
if command -v wandb &> /dev/null; then
    if wandb status &> /dev/null; then
        echo "✓ W&B is already logged in."
    else
        read -p "Do you want to login to Weights & Biases now? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            wandb login || echo "Warning: W&B login failed. You can login later with: wandb login"
        fi
    fi
else
    echo "Warning: wandb CLI not found. Install with: pip install wandb"
fi

# Test verifiers environment
echo ""
echo "6. Testing environment setup..."
python -c "
try:
    import verifiers
    print('✓ Verifiers library is available.')
except ImportError:
    print('✗ Verifiers library not found. Install with: pip install verifiers')

try:
    from environments.nanochatAquaRat.nanochatAquaRat import load_environment
    env = load_environment(num_train_examples=10)
    print('✓ NanochatAquaRat environment loaded successfully.')
except Exception as e:
    print(f'✗ Environment loading failed: {e}')

try:
    import plotly
    print('✓ Plotly is available for visualizations.')
except ImportError:
    print('✗ Plotly not found. Install with: pip install plotly')

try:
    import wandb
    print('✓ Weights & Biases is available.')
except ImportError:
    print('✗ W&B not found. Install with: pip install wandb')
"

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Edit .env to add your API keys:"
echo "     - WANDB_API_KEY (required for tracking)"
echo "     - PRIME_INTELLECT_API_KEY (optional)"
echo ""
echo "  2. Run a test training:"
echo "     python -m scripts.prime_rl_train --run=test --num_train_examples=100"
echo ""
echo "  3. For distributed training:"
echo "     torchrun --standalone --nproc_per_node=8 -m scripts.prime_rl_train -- --run=distributed"
echo ""
echo "  4. See PRIME_INTELLECT_INTEGRATION.md for detailed documentation"
echo ""
