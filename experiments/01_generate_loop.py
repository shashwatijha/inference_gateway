"""Milestone 1: text generation by hand.

model.generate() hides the whole inference loop. This script writes it out:
tokenize -> forward pass -> logits -> pick a token -> append -> repeat.

    pip install -r experiments/requirements.txt
    python experiments/01_generate_loop.py --prompt "Why is the sky blue?"
    python experiments/01_generate_loop.py --temperature 0      # greedy, deterministic
    python experiments/01_generate_loop.py --temperature 1.2 --top-p 0.9

Things to try: run the same prompt twice at temperature 0 and at 1.0; look at
how many tokens different words and languages take; change top-p and watch
the candidate list shrink.
"""
import argparse
import time

try:
    import torch
except ImportError:
    raise SystemExit("PyTorch is not installed. Run: make install-experiments")
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # Apple Silicon GPU
        return "mps"
    return "cpu"


def sample(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    """logits: [vocab]. Returns one token id."""
    if temperature == 0:
        return int(torch.argmax(logits))
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    # Nucleus (top-p): keep the smallest set of tokens whose probability sums to top_p.
    sorted_probs, sorted_ids = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    sorted_probs[cumulative - sorted_probs > top_p] = 0
    sorted_probs /= sorted_probs.sum()
    choice = torch.multinomial(sorted_probs, 1)
    return int(sorted_ids[choice])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--prompt", default="Explain what a GPU does in one paragraph.")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    args = ap.parse_args()

    device = pick_device()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16 if device != "cpu" else torch.float32)
    model.to(device).eval()

    # 1. Tokenize. The chat template wraps the prompt in the special tokens the model was trained on.
    messages = [{"role": "user", "content": args.prompt}]
    input_ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
    pieces = [tok.decode([t]) for t in input_ids[0]]
    print(f"device={device}  prompt tokens={input_ids.shape[1]}  vocab={model.config.vocab_size}")
    print("tokens:", pieces[:60], "..." if len(pieces) > 60 else "")
    print()

    # 2. Loop. The first call processes the whole prompt (prefill); later calls feed one token
    #    and reuse cached keys/values for everything before it (decode).
    past = None
    ids = input_ids
    generated: list[int] = []
    printed = 0
    t0 = time.perf_counter()
    ttft = None
    with torch.no_grad():
        for step in range(args.max_new_tokens):
            out = model(input_ids=ids, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1]  # scores for the next token only

            if step == 0:
                top = torch.topk(torch.softmax(logits.float(), -1), 5)
                cands = ", ".join(f"{tok.decode([i])!r} {p:.2f}" for p, i in zip(top.values, top.indices))
                print(f"first-token candidates: {cands}\n")

            next_id = sample(logits, args.temperature, args.top_p)
            if ttft is None:
                ttft = time.perf_counter() - t0
            if next_id == tok.eos_token_id or next_id == tok.convert_tokens_to_ids("<|im_end|>"):
                break
            generated.append(next_id)
            # Decode the whole list and print the new part: one token can be half a character.
            text = tok.decode(generated)
            print(text[printed:], end="", flush=True)
            printed = len(text)
            ids = torch.tensor([[next_id]], device=device)

    total = time.perf_counter() - t0
    n = len(generated)
    print(f"\n\ntokens={n}  TTFT={ttft * 1000:.0f} ms  "
          f"decode={((total - ttft) / max(1, n - 1)) * 1000:.1f} ms/token  ({n / total:.1f} tokens/s overall)")


if __name__ == "__main__":
    main()
