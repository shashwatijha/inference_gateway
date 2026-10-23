"""Milestone 2: prefill vs decode, and what the KV cache buys.

Three measurements:
  1. Prefill: one forward pass over prompts of growing length. Tokens/s is high
     because all prompt tokens are processed in parallel (compute-bound).
  2. Decode with the KV cache: one token per step; each step reads all the
     weights to produce a single token (memory-bandwidth-bound).
  3. Decode WITHOUT the cache: every step re-processes the whole sequence, so
     step time grows with length. This is why every server keeps a KV cache.
Then it computes KV cache memory from the model config, the number that limits
how many requests fit on a GPU at once.

    python experiments/02_prefill_vs_decode.py
    python experiments/02_prefill_vs_decode.py --model Qwen/Qwen2.5-1.5B-Instruct
"""
import argparse
import time

try:
    import torch
except ImportError:
    raise SystemExit("PyTorch is not installed. Run: make install-experiments")
from transformers import AutoConfig, AutoModelForCausalLM


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device: str) -> None:
    # GPU work is asynchronous; without a sync you time the launch, not the work.
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def timed(fn, device: str, repeats: int = 3) -> float:
    fn()  # warm-up
    sync(device)
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        sync(device)
        best = min(best, time.perf_counter() - t)
    return best


def kv_bytes_per_token(cfg, dtype_bytes: int) -> tuple[int, int]:
    """Returns (bytes with grouped-query attention, bytes if every head had its own K/V)."""
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    per_head = 2 * head_dim * dtype_bytes  # K and V
    return (cfg.num_hidden_layers * kv_heads * per_head,
            cfg.num_hidden_layers * cfg.num_attention_heads * per_head)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--prompt-lens", type=int, nargs="+", default=[16, 64, 256, 1024])
    ap.add_argument("--decode-steps", type=int, default=64)
    args = ap.parse_args()

    device = pick_device()
    dtype = torch.float16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    vocab = model.config.vocab_size
    print(f"model={args.model} device={device} dtype={dtype}\n")

    print("1. Prefill (one forward pass over the prompt)")
    print(f"{'prompt tokens':>14} {'time ms':>9} {'tokens/s':>10}")
    with torch.no_grad():
        for n in args.prompt_lens:
            ids = torch.randint(0, vocab, (1, n), device=device)
            s = timed(lambda: model(input_ids=ids, use_cache=False), device)
            print(f"{n:>14} {s * 1000:>9.1f} {n / s:>10.0f}")

        print(f"\n2/3. Decode {args.decode_steps} tokens after a 256-token prompt")
        prompt = torch.randint(0, vocab, (1, 256), device=device)

        def decode_with_cache():
            out = model(input_ids=prompt, use_cache=True)
            past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
            for _ in range(args.decode_steps):
                out = model(input_ids=nxt, past_key_values=past, use_cache=True)
                past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)

        def decode_without_cache():
            seq = prompt
            model(input_ids=seq, use_cache=False)
            for _ in range(args.decode_steps):
                nxt = model(input_ids=seq, use_cache=False).logits[:, -1:].argmax(-1)
                seq = torch.cat([seq, nxt], dim=1)

        prefill = timed(lambda: model(input_ids=prompt, use_cache=True), device)
        with_cache = timed(decode_with_cache, device, repeats=2) - prefill
        without = timed(decode_without_cache, device, repeats=1) - prefill
        steps = args.decode_steps
        print(f"   with KV cache:    {with_cache / steps * 1000:7.1f} ms/token  ({steps / with_cache:.0f} tokens/s)")
        print(f"   without KV cache: {without / steps * 1000:7.1f} ms/token  ({steps / without:.0f} tokens/s)")
        print(f"   prefill of 256 tokens took {prefill * 1000:.1f} ms = "
              f"{prefill / 256 * 1000:.2f} ms/token: prefill is far cheaper per token than decode")

    cfg = AutoConfig.from_pretrained(args.model)
    dtype_bytes = 2  # fp16/bf16 KV cache, the common default on GPU
    gqa, mha = kv_bytes_per_token(cfg, dtype_bytes)
    print("\n4. KV cache memory (fp16)")
    print(f"   layers={cfg.num_hidden_layers} attention heads={cfg.num_attention_heads} "
          f"kv heads={getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)}")
    print(f"   per token: {gqa / 1024:.1f} KiB (would be {mha / 1024:.1f} KiB without grouped-query attention)")
    for ctx, batch in ((4096, 1), (4096, 16), (32768, 16)):
        print(f"   {batch:>3} requests x {ctx:>6} tokens: {gqa * ctx * batch / 2**30:6.2f} GiB")
    params = sum(p.numel() for p in model.parameters())
    print(f"   model weights: {params * dtype_bytes / 2**30:.2f} GiB at fp16")


if __name__ == "__main__":
    main()
