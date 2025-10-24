# nanochat training report

Generated: 2025-10-23 22:26:58

## Environment

### Git Information
- Branch: main
- Commit: f1e8954 (dirty)
- Message: Document Hyperbolic setup and fix lite RL script

### Hardware
- Platform: Linux
- CPUs: 104 cores (104 logical)
- Memory: 1007.4 GB
- GPUs: 8x NVIDIA H100 80GB HBM3
- GPU Memory: 633.7 GB total
- CUDA Version: 12.8
- Hourly Rate: $24.00/hour

### Software
- Python: 3.10.12
- PyTorch: 2.8.0+cu128


### Bloat
- Characters: 474,203
- Lines: 12,350
- Files: 57
- Tokens (approx): 118,550
- Dependencies (uv.lock lines): 2,220

Run started: 2025-10-23 22:27:01

---

## Tokenizer training
timestamp: 2025-10-24 01:20:23

- max_chars: 10,000,000,000
- doc_cap: 10,000
- vocab_size: 65,536
- train_time: 167.3429
- num_special_tokens: 9
- token_bytes_min: 1
- token_bytes_max: 32
- token_bytes_mean: 6.9125
- token_bytes_std: 2.8738


## Tokenizer evaluation
timestamp: 2025-10-24 01:20:28

### Comparison with GPT-2

| Text Type | Bytes | GPT-2 Tokens | GPT-2 Ratio | Ours Tokens | Ours Ratio | Relative Diff % |
|-----------|-------|--------------|--------------|-------------|------------|-----------------|
| news | 1819 | 404 | 4.50 | 371 | 4.90 | +8.2% |
| korean | 893 | 745 | 1.20 | 723 | 1.24 | +3.0% |
| code | 1259 | 576 | 2.19 | 492 | 2.56 | +14.6% |
| math | 1834 | 936 | 1.96 | 966 | 1.90 | -3.2% |
| science | 1112 | 260 | 4.28 | 223 | 4.99 | +14.2% |
| fwe-train | 4208518 | 900364 | 4.67 | 856938 | 4.91 | +4.8% |
| fwe-val | 5028883 | 1083776 | 4.64 | 1033017 | 4.87 | +4.7% |

### Comparison with GPT-4

| Text Type | Bytes | GPT-4 Tokens | GPT-4 Ratio | Ours Tokens | Ours Ratio | Relative Diff % |
|-----------|-------|--------------|--------------|-------------|------------|-----------------|
| news | 1819 | 387 | 4.70 | 371 | 4.90 | +4.1% |
| korean | 893 | 364 | 2.45 | 723 | 1.24 | -98.6% |
| code | 1259 | 309 | 4.07 | 492 | 2.56 | -59.2% |
| math | 1834 | 832 | 2.20 | 966 | 1.90 | -16.1% |
| science | 1112 | 249 | 4.47 | 223 | 4.99 | +10.4% |
| fwe-train | 4208518 | 874799 | 4.81 | 856938 | 4.91 | +2.0% |
| fwe-val | 5028883 | 1054265 | 4.77 | 1033017 | 4.87 | +2.0% |


## Base model training
timestamp: 2025-10-23 22:28:41

- run: aquarat-20251023-222655
- device_type: 
- depth: 8
- max_seq_len: 2048
- num_iterations: 200
- target_flops: -1.0000
- target_param_data_ratio: 20
- device_batch_size: 32
- total_batch_size: 524,288
- embedding_lr: 0.2000
- unembedding_lr: 0.0040
- weight_decay: 0.0000
- matrix_lr: 0.0200
- grad_clip: 1.0000
- eval_every: 250
- eval_tokens: 10,485,760
- core_metric_every: 2000
- core_metric_max_per_task: 500
- sample_every: 2000
- model_tag: 
- Number of parameters: 92,274,688
- Number of FLOPs per token: 4.529848e+08
- Calculated number of iterations: 200
- Number of training tokens: 104,857,600
- Tokens : Params ratio: 1.1364
- DDP world size: 8
- warmup_ratio: 0.0000
- warmdown_ratio: 0.2000
- final_lr_frac: 0.0000
- Minimum validation bpb: 1.2992
- Final validation bpb: 1.2992
- CORE metric estimate: 0.0135
- MFU %: 21.07%
- Total training flops: 4.749890e+16
- Total training time: 0.39m
- Peak memory usage: 19176.26MiB


## Base model loss
timestamp: 2025-10-24 01:21:08

- train bpb: 1.2967
- val bpb: 1.3001
- sample 0: <|bos|>The capital of France is the capital of the city of the city of the city of the city of the
- sample 1: <|bos|>The chemical symbol of gold is a symbol of the world’s most important symbol of the world’s most important symbol
- sample 2: <|bos|>If yesterday was Friday, then tomorrow will be a bit more than a few years, and then the next day, the next
- sample 3: <|bos|>The opposite of hot is the same as the one of the two of the same kind of cold weather.
- sample 4: <|bos|>The planets of the solar system are: 1.5 billion years ago, and the 1.5 billion years
- sample 5: <|bos|>My favorite color is the color of the color of the color of the color of the color of the
- sample 6: <|bos|>If 5*x + 3 = 13, then x is 3.5.5.5.5.5.5.5


## Base model evaluation
timestamp: 2025-10-24 01:32:41

- Model: base_model (step 200)
- CORE metric: 0.0155
- hellaswag_zeroshot: -0.0018
- jeopardy: 0.0000
- bigbench_qa_wikidata: 0.0005
- arc_easy: 0.1246
- arc_challenge: -0.0364
- copa: -0.0400
- commonsense_qa: -0.0053
- piqa: 0.0794
- openbook_qa: -0.0373
- lambada_openai: 0.0190
- hellaswag: -0.0067
- winograd: 0.0842
- winogrande: 0.0024
- bigbench_dyck_languages: 0.0080
- agi_eval_lsat_ar: 0.0326
- bigbench_cs_algorithms: 0.0235
- bigbench_operators: 0.0619
- bigbench_repeat_copy_logic: 0.0000
- squad: 0.0004
- coqa: 0.0009
- boolq: -0.1468
- bigbench_language_identification: 0.1770


## Midtraining
timestamp: 2025-10-23 22:30:54

- run: aquarat-20251023-222655
- device_type: 
- dtype: bfloat16
- num_iterations: 200
- max_seq_len: 2048
- device_batch_size: 32
- unembedding_lr: 0.0040
- embedding_lr: 0.2000
- matrix_lr: 0.0200
- init_lr_frac: 1.0000
- weight_decay: 0.0000
- eval_every: 150
- eval_tokens: 10,485,760
- total_batch_size: 524,288
- dry_run: 0
- Number of iterations: 199
- DDP world size: 8
- Minimum validation bpb: 0.6738


## Chat evaluation mid
timestamp: 2025-10-24 01:52:32

- source: mid
- task_name: GSM8K
- dtype: bfloat16
- temperature: 0.0000
- max_new_tokens: 512
- num_samples: 1
- top_k: 50
- batch_size: 8
- model_tag: None
- step: None
- max_problems: None
- device_type: 
- GSM8K: 0.0023


## Chat SFT
timestamp: 2025-10-23 22:46:36

- run: aquarat-20251023-224449
- source: mid
- device_type: 
- dtype: bfloat16
- device_batch_size: 4
- num_epochs: 1
- num_iterations: -1
- target_examples_per_step: 32
- unembedding_lr: 0.0040
- embedding_lr: 0.2000
- matrix_lr: 0.0200
- weight_decay: 0.0000
- init_lr_frac: 0.0200
- aqua_train_examples: 20,000
- aqua_val_examples: 254
- eval_every: 100
- eval_steps: 100
- eval_metrics_every: 200
- eval_metrics_max_problems: 1024
- Training rows: 41,839
- Number of iterations: 1307
- Training loss: 2.9144
- Validation loss: 1.9120


## Chat evaluation sft
timestamp: 2025-10-24 01:52:40

- source: sft
- task_name: AQUA
- dtype: bfloat16
- temperature: 0.0000
- max_new_tokens: 512
- num_samples: 1
- top_k: 50
- batch_size: 8
- model_tag: None
- step: None
- max_problems: None
- device_type: 
- AQUA: 0.2756


## Chat RL
timestamp: 2025-10-24 01:01:09

- run: aquarat-20251023-224750-rl
- source: sft
- dtype: bfloat16
- device_batch_size: 1
- examples_per_step: 16
- num_samples: 4
- max_new_tokens: 64
- temperature: 0.7000
- top_k: 50
- unembedding_lr: 0.0040
- embedding_lr: 0.2000
- matrix_lr: 0.0200
- weight_decay: 0.0000
- init_lr_frac: 0.0500
- num_epochs: 1
- save_every: 60
- eval_every: 60
- eval_examples: 400


## Chat evaluation rl
timestamp: 2025-10-24 03:52:41

- source: rl
- task_name: AQUA
- dtype: bfloat16
- temperature: 0.0000
- max_new_tokens: 64
- num_samples: 1
- top_k: 50
- batch_size: 8
- model_tag: None
- step: None
- max_problems: None
- device_type: 
- AQUA: 0.2717


## Summary

- Characters: 474,203
- Lines: 12,350
- Files: 57
- Tokens (approx): 118,550
- Dependencies (uv.lock lines): 2,220

| Metric          | BASE     | MID      | SFT      | RL       |
|-----------------|----------|----------|----------|----------|
| CORE            | 0.0155   | -        | -        | -        |
| GSM8K           | -        | 0.0023   | -        | -        |

Total wall clock time: 3h25m
