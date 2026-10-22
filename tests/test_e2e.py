"""Gateway + worker + Redis + fake model server, all as real processes."""
import json
import os
import subprocess
import sys

import httpx
import pytest

from .conftest import free_port, start_uvicorn


@pytest.fixture(scope="module", params=["continuous", "static"])
def stack(request, redis_url):
    model_port, gw_port = free_port(), free_port()
    env = {**os.environ, "REDIS_URL": redis_url, "MODEL_URL": f"http://127.0.0.1:{model_port}",
           "FAKE_DECODE_MS": "2", "RATE_PER_SEC": "1000", "RATE_BURST": "1000",
           "METRICS_PORT": str(free_port()), "API_KEYS": "test-key",
           "BATCH_MODE": request.param}
    procs = [start_uvicorn("tools.fake_model_server:app", model_port, env)]
    procs.append(subprocess.Popen([sys.executable, "-m", "worker.main"], env=env))
    procs.append(start_uvicorn("gateway.main:app", gw_port, env))
    yield f"http://127.0.0.1:{gw_port}"
    for p in procs:
        p.terminate()
        p.wait()


HEADERS = {"Authorization": "Bearer test-key"}
BODY = {"messages": [{"role": "user", "content": "hello there"}], "max_tokens": 5}


def test_rejects_bad_key(stack):
    r = httpx.post(f"{stack}/v1/chat/completions", json=BODY, headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


def test_non_streaming(stack):
    r = httpx.post(f"{stack}/v1/chat/completions", json=BODY, headers=HEADERS, timeout=10)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["choices"][0]["message"]["content"].split() == "the quick brown fox jumps".split()
    assert out["usage"]["completion_tokens"] == 5


def test_streaming(stack):
    deltas, finish = [], None
    with httpx.stream("POST", f"{stack}/v1/chat/completions", json={**BODY, "stream": True},
                      headers=HEADERS, timeout=10) as r:
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            choice = json.loads(line[6:])["choices"][0]
            deltas.append(choice["delta"].get("content", ""))
            finish = choice["finish_reason"] or finish
    assert "".join(deltas).split() == "the quick brown fox jumps".split()
    assert finish == "length"


def test_cache_hit_for_greedy_requests(stack):
    body = {**BODY, "temperature": 0, "messages": [{"role": "user", "content": "cache me"}]}
    first = httpx.post(f"{stack}/v1/chat/completions", json=body, headers=HEADERS, timeout=10).json()
    second = httpx.post(f"{stack}/v1/chat/completions", json=body, headers=HEADERS, timeout=10).json()
    assert first["choices"][0]["message"] == second["choices"][0]["message"]
    metrics = httpx.get(f"{stack}/metrics").text
    assert 'gateway_cache_total{result="hit"}' in metrics


def test_concurrent_requests(stack):
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(16) as ex:
        codes = list(ex.map(lambda _: httpx.post(f"{stack}/v1/chat/completions", json=BODY,
                                                 headers=HEADERS, timeout=30).status_code, range(32)))
    assert codes == [200] * 32
