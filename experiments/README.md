# Experiments

Small scripts, one per concept. Each prints numbers; write them down in `docs/notes.md` with one sentence on what they mean. Those notes are your interview material.

| # | Script | Concept | Runs on |
|---|---|---|---|
| 1 | `01_generate_loop.py` | tokenization, logits, sampling | M3 (MPS) |
| 2 | `02_prefill_vs_decode.py` | prefill vs decode, KV cache, memory math | M3 (MPS) |
| 3 | commands below | quantization | M3 (llama.cpp, Metal) |
| 4 | `04_batching_sim.py` | static vs continuous batching | anywhere, pure Python |
| 5 | commands below | speculative decoding | M3 or rented GPU |

Setup: `python -m venv .venv && source .venv/bin/activate && pip install -r experiments/requirements.txt`. The first run of 1 and 2 downloads the model (about 1 GB).

## 3. Quantization (llama.cpp)

```bash
brew install llama.cpp huggingface-cli
# Check the repo's file list if a name differs.
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  qwen2.5-1.5b-instruct-fp16.gguf qwen2.5-1.5b-instruct-q8_0.gguf qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --local-dir models

# Speed: pp = prompt processing (prefill) tokens/s, tg = text generation (decode) tokens/s
for q in fp16 q8_0 q4_k_m; do llama-bench -m models/qwen2.5-1.5b-instruct-$q.gguf -p 512 -n 128; done

# Quality: perplexity (lower is better); compare quantizations on the same text
# Use the WikiText-2 test split, or any long plain-text file (e.g. a Project Gutenberg book) saved as models/eval.txt
for q in fp16 q8_0 q4_k_m; do llama-perplexity -m models/qwen2.5-1.5b-instruct-$q.gguf -f models/eval.txt --chunks 50; done
```

Record a table: file size, pp tokens/s, tg tokens/s, perplexity. What to explain afterwards: decode speed tracks file size closely, because decode reads every weight once per token, so fewer bytes per weight means faster decode. Prefill changes less, because it is compute-bound.

## 5. Speculative decoding (llama.cpp)

A small draft model guesses several tokens ahead, and the big model checks all of them in one forward pass. That pass costs about the same as generating one token, because decode is memory-bound. Accepted guesses are free tokens.

```bash
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q8_0.gguf --local-dir models
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF --include "*q4_k_m*" --local-dir models   # ~5 GB; fits in 18 GB+ unified memory

# Baseline, then with the draft model
llama-server -m models/<7b file>.gguf --port 8080
llama-server -m models/<7b file>.gguf -md models/qwen2.5-0.5b-instruct-q8_0.gguf --draft-max 16 --draft-min 4 --port 8080
```

Point the gateway at it (`MODEL_URL=http://host.docker.internal:8080`) and run `loadtest/bench.py --temperature 0` against both. The server log prints the draft acceptance rate. Compare code prompts, which are predictable and get high acceptance, with creative prompts, which get low acceptance. Then explain why the speedup shrinks as batch size grows: at large batch sizes the GPU is no longer idle waiting on memory.
