from tasks.aqua import AQUA
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

task = AQUA(split="validation", shuffle=False)
indices = [0, 10, 50]
model, tok, _ = load_model("rl", device="cuda", phase="eval")
engine = Engine(model, tok)

for i in indices:
    convo = task[i]
    prompt = tok.render_for_completion(convo)
    out, _ = engine.generate_batch(prompt, num_samples=1, max_tokens=64, temperature=0.7, top_k=50)
    completion = tok.decode(out[0][len(prompt):])
    print(f"Question {i+1}:\n{convo['messages'][0]['content']}\n")
    print("Model output:", completion)
    print("Reward:", task.reward(convo, completion))
    print("-" * 60)

