"""Exact-match response cache.

Only deterministic requests (temperature == 0) are cached: with sampling, two
identical prompts are supposed to give different answers.
"""
import hashlib
import json


def cache_key(model: str, messages: list[dict], max_tokens: int) -> str:
    blob = json.dumps(
        {"model": model, "messages": messages, "max_tokens": max_tokens},
        sort_keys=True,
        separators=(",", ":"),
    )
    return "cache:" + hashlib.sha256(blob.encode()).hexdigest()
