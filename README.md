---
language: en
license: mit
datasets:
  - deepmind/aqua_rat
metrics:
  - accuracy
tags:
  - reinforcement-learning
  - autoformalization
  - education
  - nanochat
---

# nanochat AQuA-RL 🐠  
_A multiple-choice math apprentice that actually remembers its answer key._

![AQuA Fish](aquarat2.png)

## 1. Pipeline at a Glance

| Stage | What happens | Notes |
|-------|---------------|-------|
| Base  | Original nanochat pretraining | vanilla corpus |
| Mid   | Conversational alignment | SmolTalk + MMLU + GSM8K |
| SFT   | Supervised fine-tune on AQuA | `scripts.chat_sft` |
| RL    | Policy-gradient finetune on AQuA | `scripts.chat_rl`, 8× H100 (~30 min, 1,560 steps) |

**Debugging log:** the first RL attempt left rewards at zero. We later realised the reward function still looked for GSM8K numeric answers. Swapping to AQuA's letter-based reward fixed the issue—reward curves now oscillate around 0.2–0.4 and pass@1 plateaued near 0.27.

## 2. Key Metrics (AQuA validation)

| Stage | Accuracy |
|-------|----------|
| SFT   | **27.56 %** |
| RL    | **27.17 %** |

SFT gives the big jump; RL keeps accuracy in range while stabilising formatting ("Answer: X") and sampling.

## 3. Where to find the artifacts

| Item | Location |
|------|----------|
| RL checkpoint (`model_001560.pt`) | [HF repo](https://huggingface.co/HarleyCooper/nanochatAquaRat) & `gs://nanochat-aquarat-datasets/runs/aquarat-20251024-033245/chatrl_checkpoints/d8/` |
| Tokenizer (`tokenizer.pkl`, `token_bytes.pt`) | Same locations as above |
| Full report (`report.md`) | Same locations |
| Latest git commit | `git rev-parse HEAD` in this repo |

To pull everything from GCS:
```bash
gsutil -m cp -r gs://nanochat-aquarat-datasets/runs/aquarat-20251024-033245 ./aquarat_run
```

## 4. Inference quick-start

### CPU example
```python
from tasks.aqua import AQUA
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

model, tokenizer, _ = load_model("rl", device="cpu", phase="eval")
engine = Engine(model, tokenizer)

task = AQUA(split="validation", shuffle=False)
convo = task[0]
prompt = tokenizer.render_for_completion(convo)

generated, _ = engine.generate_batch(
    prompt,
    num_samples=1,
    max_tokens=64,
    temperature=0.7,
    top_k=50,
)
completion = tokenizer.decode(generated[0][len(prompt):])
print("Model output:\n", completion)
```

### GPU tip
The checkpoint is in bfloat16. When you generate on GPU, wrap the call in:

```python
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    generated, _ = engine.generate_batch(...)
```

## 5. Students: swapping in a new dataset

1. Prepare your dataset in the same conversation schema (`scripts/prepare_<your_dataset>.py`).
2. Update `scripts/chat_sft.py` & `scripts/chat_rl.py` (see how we switched to AQUA).
3. Rerun `run_aquarat_small.sh` (or your own driver) on multi-GPU hardware.
4. Replace `hf_release/` contents and rerun the HF upload.

## 6. Visuals & next steps

- Drop W&B reward/pass@1 graphs here once exported.
- Want to show pipeline plumbing? Draft a TKZ diagram, render it to PNG, and place it next to the hero image.

## 7. Credits & License

MIT License.
Built on nanochat by @HarleyCoops.
Questions? PRs welcome!