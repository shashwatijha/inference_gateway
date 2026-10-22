"""Milestone 6: a discrete-event simulation of LLM batching policies.

No model, no GPU: just the scheduling arithmetic. Requests arrive as a Poisson
process with long-tailed output lengths. Each decode step produces one token
for every running request and takes a bit longer as the batch grows.

  sequential  one request at a time
  static      take up to B requests, run until ALL of them finish, then refill
  continuous  refill free slots every step (Orca / vLLM style)

    python experiments/04_batching_sim.py
    python experiments/04_batching_sim.py --rates 1 2 4 8 --batch 8 --plot

Things to look for: where each policy's latency explodes (its capacity), and
how much throughput static batching wastes on idle slots.
"""
import argparse
import random
from collections import deque
from dataclasses import dataclass


@dataclass
class Req:
    idx: int
    arrival: float
    length: int
    remaining: int
    first_token: float | None = None


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def simulate(policy: str, batch: int, rate: float, n: int, out_mean: float,
             decode_ms: float, penalty: float, prefill_ms: float, seed: int = 0) -> dict:
    rng = random.Random(seed)
    arrivals, t = [], 0.0
    for _ in range(n):
        t += rng.expovariate(rate)
        arrivals.append(t)
    lengths = [max(1, int(rng.expovariate(1 / out_mean))) for _ in range(n)]
    limit = 1 if policy == "sequential" else batch

    t, nxt = 0.0, 0
    queue: deque[Req] = deque()
    running: list[Req] = []
    finished: list[tuple[Req, float]] = []
    busy_slot_steps = total_slot_steps = 0

    while len(finished) < n:
        while nxt < n and arrivals[nxt] <= t:
            queue.append(Req(nxt, arrivals[nxt], lengths[nxt], lengths[nxt]))
            nxt += 1
        admitted = 0
        if policy == "continuous" or not running:
            while queue and len(running) < limit:
                running.append(queue.popleft())
                admitted += 1
        if not running:
            t = arrivals[nxt]  # idle until the next arrival
            continue
        # One step: prefill for newly admitted requests (stalls the batch), then one decode token each.
        t += (admitted * prefill_ms + decode_ms * (1 + penalty * (len(running) - 1))) / 1000
        busy_slot_steps += len(running)
        total_slot_steps += limit
        still = []
        for r in running:
            r.remaining -= 1
            if r.first_token is None:
                r.first_token = t
            if r.remaining == 0:
                finished.append((r, t))
            else:
                still.append(r)
        running = still

    makespan = max(end for _, end in finished) - arrivals[0]
    ttft = [r.first_token - r.arrival for r, _ in finished]
    e2e = [end - r.arrival for r, end in finished]
    return {
        "policy": policy, "rate": rate,
        "tok_s": sum(lengths) / makespan,
        "ttft_p50": pct(ttft, 0.5), "ttft_p95": pct(ttft, 0.95),
        "e2e_p50": pct(e2e, 0.5), "e2e_p95": pct(e2e, 0.95),
        "slot_util": busy_slot_steps / total_slot_steps,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--requests", type=int, default=1000)
    ap.add_argument("--out-mean", type=float, default=128, help="mean output tokens (exponential)")
    ap.add_argument("--decode-ms", type=float, default=25, help="one decode step at batch size 1")
    ap.add_argument("--penalty", type=float, default=0.08, help="extra step time per extra request")
    ap.add_argument("--prefill-ms", type=float, default=40, help="prefill cost per admitted request")
    ap.add_argument("--plot", action="store_true", help="save results/batching_sim.png (needs matplotlib)")
    args = ap.parse_args()

    rows = []
    print(f"{'policy':<11} {'req/s':>6} {'tok/s':>7} {'TTFT p50':>9} {'TTFT p95':>9} "
          f"{'E2E p95':>8} {'slot use':>9}")
    for policy in ("sequential", "static", "continuous"):
        for rate in args.rates:
            r = simulate(policy, args.batch, rate, args.requests, args.out_mean,
                         args.decode_ms, args.penalty, args.prefill_ms)
            rows.append(r)
            print(f"{policy:<11} {rate:>6.1f} {r['tok_s']:>7.0f} {r['ttft_p50']:>8.2f}s {r['ttft_p95']:>8.2f}s "
                  f"{r['e2e_p95']:>7.1f}s {r['slot_util']:>8.0%}")
        print()

    if args.plot:
        import matplotlib.pyplot as plt
        from pathlib import Path

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for policy in ("sequential", "static", "continuous"):
            pts = [r for r in rows if r["policy"] == policy]
            ax.plot([r["tok_s"] for r in pts], [r["e2e_p95"] for r in pts], marker="o", label=policy)
        ax.set_xlabel("throughput (output tokens/s)")
        ax.set_ylabel("p95 end-to-end latency (s)")
        ax.set_yscale("log")
        ax.set_title(f"Throughput vs tail latency, batch={args.batch}")
        ax.legend()
        Path("results").mkdir(exist_ok=True)
        fig.savefig("results/batching_sim.png", dpi=150, bbox_inches="tight")
        print("saved results/batching_sim.png")


if __name__ == "__main__":
    main()
