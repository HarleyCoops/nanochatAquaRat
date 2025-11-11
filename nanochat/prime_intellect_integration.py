"""
Prime Intellect RL Integration Module

This module provides integration between nanochat and Prime Intellect's prime-rl framework,
enabling distributed asynchronous reinforcement learning with the verifiers environment.
"""

import os
import torch
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

try:
    import verifiers as vf
    VERIFIERS_AVAILABLE = True
except ImportError:
    VERIFIERS_AVAILABLE = False
    print("Warning: verifiers library not available. Install with: pip install verifiers")


class PrimeIntellectRLAdapter:
    """
    Adapter class to bridge nanochat models with Prime Intellect's RL framework.
    """

    def __init__(
        self,
        model,
        tokenizer,
        env_id: str = "harleycooper/nanochatAquaRat",
        env_args: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Prime Intellect RL adapter.

        Args:
            model: nanochat GPT model instance
            tokenizer: nanochat tokenizer instance
            env_id: Prime Intellect environment ID from the Environments Hub
            env_args: Optional arguments to pass to the environment loader
        """
        if not VERIFIERS_AVAILABLE:
            raise ImportError(
                "verifiers library is required for Prime Intellect integration. "
                "Install with: pip install verifiers"
            )

        self.model = model
        self.tokenizer = tokenizer
        self.env_id = env_id
        self.env_args = env_args or {}
        self.environment = None

    def load_environment(self) -> Any:
        """
        Load the Prime Intellect verifiers environment.

        Returns:
            Loaded verifiers environment
        """
        if self.environment is None:
            # Check if we should load from local or hub
            local_env_path = Path(__file__).parent.parent / "environments" / "nanochatAquaRat"

            if local_env_path.exists():
                # Load local environment
                import sys
                sys.path.insert(0, str(local_env_path.parent))
                from nanochatAquaRat.nanochatAquaRat import load_environment
                self.environment = load_environment(**self.env_args)
            else:
                # Load from Prime Intellect hub
                self.environment = vf.load_environment(self.env_id, **self.env_args)

        return self.environment

    def prepare_model_for_prime_rl(self) -> Dict[str, Any]:
        """
        Prepare model configuration for Prime Intellect's prime-rl framework.

        Returns:
            Configuration dictionary for prime-rl
        """
        config = {
            "model_type": "custom_nanochat_gpt",
            "vocab_size": self.model.config.vocab_size,
            "n_layer": self.model.config.n_layer,
            "n_head": self.model.config.n_head,
            "n_embd": self.model.config.n_embd,
            "sequence_len": self.model.config.sequence_len,
        }

        return config

    def convert_messages_to_tokens(self, messages: list) -> torch.Tensor:
        """
        Convert Prime Intellect message format to nanochat tokens.

        Args:
            messages: List of message dictionaries with 'role' and 'content'

        Returns:
            Tensor of token IDs
        """
        tokens = self.tokenizer.render_for_completion(messages)
        return torch.tensor(tokens, dtype=torch.long)

    def generate_response(
        self,
        messages: list,
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        num_samples: int = 1,
    ) -> list:
        """
        Generate responses compatible with Prime Intellect's expected format.

        Args:
            messages: Input messages in Prime Intellect format
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering parameter
            num_samples: Number of samples to generate

        Returns:
            List of generated response strings
        """
        from nanochat.engine import Engine

        tokens = self.tokenizer.render_for_completion(messages)
        engine = Engine(self.model, self.tokenizer)

        # Generate samples
        with torch.no_grad():
            generated_sequences, masks = engine.generate_batch(
                tokens,
                num_samples=num_samples,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
            )

        # Decode responses
        prefix_length = len(tokens)
        responses = []
        for seq in generated_sequences:
            generated_tokens = seq[prefix_length:]
            response_text = self.tokenizer.decode(generated_tokens)
            responses.append(response_text)

        return responses

    def compute_reward(
        self,
        messages: list,
        response: str,
        use_environment: bool = True
    ) -> float:
        """
        Compute reward using the Prime Intellect environment or native nanochat reward.

        Args:
            messages: Input messages
            response: Generated response
            use_environment: Whether to use the verifiers environment for reward

        Returns:
            Reward value (typically 0.0 or 1.0 for AQuA-RAT)
        """
        if use_environment and self.environment:
            # Use verifiers environment reward
            # This will be called by prime-rl automatically
            raise NotImplementedError("Direct environment reward should be called by prime-rl")
        else:
            # Use native nanochat reward function
            from tasks.aqua import AQUA
            task = AQUA(split="train")

            # Convert messages to conversation format
            conversation = {"messages": messages}
            return task.reward(conversation, response)


def create_prime_rl_config(
    model_path: str,
    output_dir: str = "./prime_rl_output",
    num_train_examples: int = 2000,
    num_eval_examples: int = 254,
    learning_rate: float = 2e-5,
    rollouts_per_example: int = 8,
    max_steps: int = 400,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    eval_every: int = 50,
    save_every: int = 100,
    wandb_project: str = "nanochat-prime-rl",
    wandb_run_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a configuration dictionary for Prime Intellect RL training.

    Args:
        model_path: Path to the model checkpoint or HuggingFace model ID
        output_dir: Directory to save outputs and checkpoints
        num_train_examples: Number of training examples to use
        num_eval_examples: Number of evaluation examples
        learning_rate: Learning rate for training
        rollouts_per_example: Number of rollouts per training example
        max_steps: Maximum training steps
        batch_size: Batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        eval_every: Evaluate every N steps
        save_every: Save checkpoint every N steps
        wandb_project: W&B project name
        wandb_run_name: W&B run name (optional)

    Returns:
        Configuration dictionary
    """
    config = {
        "model": model_path,
        "output_dir": output_dir,

        "env": {
            "id": "harleycooper/nanochatAquaRat",
            "args": {
                "num_train_examples": num_train_examples,
                "num_eval_examples": num_eval_examples,
                "seed": 42,
                "include_rationale_metadata": True,
            }
        },

        "trainer": {
            "args": {
                "learning_rate": learning_rate,
                "rollouts_per_example": rollouts_per_example,
                "max_steps": max_steps,
                "per_device_train_batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "eval_steps": eval_every,
                "save_steps": save_every,
                "logging_steps": 10,

                # RL-specific parameters
                "temperature": 1.0,
                "top_k": 50,
                "max_new_tokens": 256,

                # Optimization
                "weight_decay": 0.0,
                "warmup_steps": 50,
                "max_grad_norm": 1.0,

                # W&B integration
                "report_to": ["wandb"],
                "run_name": wandb_run_name or f"prime_rl_{Path(model_path).name}",
            }
        },

        "wandb": {
            "project": wandb_project,
            "name": wandb_run_name,
            "tags": ["prime-rl", "nanochat", "aquarat"],
        }
    }

    return config


def export_model_for_prime_rl(
    model,
    tokenizer,
    output_dir: str,
    model_name: str = "nanochat_aquarat",
) -> str:
    """
    Export nanochat model in a format compatible with Prime Intellect.

    Args:
        model: nanochat GPT model
        tokenizer: nanochat tokenizer
        output_dir: Output directory path
        model_name: Name for the exported model

    Returns:
        Path to the exported model directory
    """
    output_path = Path(output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Save model state
    model_state_path = output_path / "model.pt"
    torch.save(model.state_dict(), model_state_path)

    # Save configuration
    config_dict = {
        "model_type": "nanochat_gpt",
        "vocab_size": model.config.vocab_size,
        "n_layer": model.config.n_layer,
        "n_head": model.config.n_head,
        "n_kv_head": model.config.n_kv_head,
        "n_embd": model.config.n_embd,
        "sequence_len": model.config.sequence_len,
    }

    import json
    with open(output_path / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Save tokenizer info
    tokenizer_info = {
        "vocab_size": tokenizer.n_vocab,
        "special_tokens": {
            "bos": "<|bos|>",
            "user_start": "<|user_start|>",
            "user_end": "<|user_end|>",
            "assistant_start": "<|assistant_start|>",
            "assistant_end": "<|assistant_end|>",
        }
    }

    with open(output_path / "tokenizer_info.json", "w") as f:
        json.dump(tokenizer_info, f, indent=2)

    print(f"Model exported to: {output_path}")
    return str(output_path)


def load_nanochat_for_prime_rl(
    checkpoint_path: str,
    device: str = "cuda",
) -> Tuple[Any, Any]:
    """
    Load a nanochat model checkpoint for use with Prime Intellect.

    Args:
        checkpoint_path: Path to nanochat checkpoint
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer)
    """
    from nanochat.checkpoint_manager import load_model

    # Extract source type from path
    if "sft" in checkpoint_path.lower():
        source = "sft"
    elif "mid" in checkpoint_path.lower():
        source = "mid"
    elif "base" in checkpoint_path.lower():
        source = "base"
    else:
        source = "sft"  # default to sft

    model, tokenizer, metadata = load_model(source, device, phase="train")

    return model, tokenizer
