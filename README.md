# nanochatAquaRat
nanochat RL trained on the [Chinar/AQuA-RAT dataset](https://huggingface.co/datasets/deepmind/aqua_rat)


from datasets import load_dataset

ds = load_dataset("deepmind/aqua_rat", "raw")
-or-

from datasets import load_dataset

ds = load_dataset("deepmind/aqua_rat", "tokenized")
