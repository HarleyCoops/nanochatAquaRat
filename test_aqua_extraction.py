from tasks.aqua import AQUA
from nanochat.checkpoint_manager import load_model
from nanochat.tokenizer import get_tokenizer
from nanochat.engine import Engine

task = AQUA(split="validation", shuffle=False)
convo = task[0]
tokenizer = get_tokenizer()
model, tok, _ = load_model("sft", "cuda", phase="eval")
engine = Engine(model, tok)

prompt = tok.render_for_completion(convo)
out, _ = engine.generate_batch(prompt, num_samples=1, max_tokens=64, temperature=0.7, top_k=50)
text = tok.decode(out[0][len(prompt):])
print("completion:", text)
print("reward:", task.reward(convo, text))
print("correct answer:", convo["answer_letter"])
