# nanochatAquaRat

nanochat RL tuned on the [DeepMind AQuA-RAT dataset](https://huggingface.co/datasets/deepmind/aqua_rat).

## Quickstart

Run the lightweight end-to-end pipeline (depth=8 base, RL on AQuA-RAT with W&B logging and mechanistic interpretability hooks):

```bash
bash run_aquarat_small.sh
```

The script will:

1. Bootstrap the Python environment with `uv` and activate the project virtualenv.
2. Authenticate with Weights & Biases (using credentials from `.env`, if present).
3. Build the `rustbpe` tokenizer and fetch a reduced pretraining corpus.
4. Prepare the [DeepMind AQuA-RAT dataset](https://huggingface.co/datasets/deepmind/aqua_rat) for supervised fine-tuning and RL.
5. Clone or update Google DeepMind's mechanistic interpretability tooling for attention diagnostics.
6. Train a depth-8 base model, run mid-stage + SFT, then launch a short RL session on AQuA-RAT with KL/letter-margins logged to W&B.
7. Evaluate the RL checkpoint and summarize the run report.

Ensure you have your W&B API key available in `.env` (e.g., `WANDB_API_KEY=...`) before running the script.
