import shutil
import socket
import subprocess
import sys
import time

import httpx
import pytest


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(check, timeout=15.0):
    end = time.time() + timeout
    while time.time() < end:
        try:
            if check():
                return
        except Exception:
            pass
        time.sleep(0.1)
    raise TimeoutError("service did not come up")


@pytest.fixture(scope="session")
def redis_url():
    if not shutil.which("redis-server"):
        pytest.skip("redis-server not installed")
    port = free_port()
    proc = subprocess.Popen(["redis-server", "--port", str(port), "--save", "", "--appendonly", "no"],
                            stdout=subprocess.DEVNULL)
    import redis
    wait_for(lambda: redis.Redis(port=port).ping())
    yield f"redis://127.0.0.1:{port}/0"
    proc.terminate()
    proc.wait()


def start_uvicorn(app: str, port: int, env: dict) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", app, "--port", str(port), "--log-level", "warning"],
        env=env,
    )
    wait_for(lambda: httpx.get(f"http://127.0.0.1:{port}/docs").status_code == 200)
    return proc
