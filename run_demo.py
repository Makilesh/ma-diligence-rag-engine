"""
One-command local launcher — brings the whole stack up ready to demo.

    python run_demo.py

Starts Postgres and Qdrant in Docker, starts the API, ingests the sample data
room if the index is empty, and opens the Streamlit UI. Everything runs on this
machine: no cloud services, no hosting cost, no memory or CPU ceiling beyond the
hardware.

It exists because the manual sequence has real ordering constraints that are
easy to get wrong under pressure — Qdrant must be healthy before the API opens
its client, the API must be up before ingestion, and the index must be populated
before the UI is worth showing. Every failure mode here has actually happened
during this project, so each step is checked rather than assumed:

  * Docker Desktop not running          → detected, with the fix printed
  * Qdrant client/server version skew   → the API refuses to start, by design
  * Empty index after a volume reset    → re-ingested automatically
  * Cold models on the first question   → warmed during startup

Ctrl+C stops the API and UI. The containers are left running; `python
run_demo.py --stop` shuts everything down.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Windows consoles default to cp1252, which cannot encode the em dashes and
# check marks below and raises UnicodeEncodeError mid-startup.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
if not VENV_PYTHON.exists():  # POSIX layout
    VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

API_URL = "http://127.0.0.1:8000"
UI_URL = "http://localhost:8501"
QDRANT_URL = "http://localhost:6333"
DEAL_ID = "aurora_vertex_2024"
CORPUS_DIR = PROJECT_ROOT / "data" / "sample_deal"

# Ingestion classifies most documents correctly on filename and content; these
# three are ambiguous enough to be worth pinning so the demo corpus is
# categorised the same way the evaluation corpus is.
CATEGORY_OVERRIDES = {
    "board_deck_strategic_review_mar2024.txt": "board",
    "regulatory_and_data_privacy_memo.txt": "regulatory",
    "employment_and_retention_agreements.txt": "legal",
}


def say(message: str, symbol: str = "•") -> None:
    print(f" {symbol} {message}", flush=True)


def fail(message: str, remedy: str = "") -> None:
    print(f"\n  FAILED: {message}", flush=True)
    if remedy:
        print(f"  Fix:    {remedy}", flush=True)
    sys.exit(1)


def get_json(url: str, timeout: float = 5.0):
    """GETs JSON, returning None on any failure — used for polling."""
    import json

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None


def wait_for(label: str, check, timeout_s: int, interval_s: float = 2.0) -> bool:
    """Polls `check` until it returns truthy, printing progress on one line."""
    deadline = time.monotonic() + timeout_s
    start = time.monotonic()
    while time.monotonic() < deadline:
        if check():
            print(f"\r {label}: ready ({time.monotonic() - start:.0f}s)      ", flush=True)
            return True
        print(f"\r   waiting for {label}… {time.monotonic() - start:.0f}s", end="", flush=True)
        time.sleep(interval_s)
    print(flush=True)
    return False


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(
        ["docker", "info"], capture_output=True, text=True
    ).returncode == 0


def try_start_docker_desktop() -> bool:
    """
    Launches Docker Desktop on Windows and waits for the daemon.

    Worth automating specifically because it is the one prerequisite that is
    invisible until it fails, and it fails at the worst moment: Docker Desktop
    does not survive a reboot unless configured to, so "it worked yesterday" is
    not evidence it is running now.

    Returns:
        True if the daemon is responding by the end of the wait.
    """
    if sys.platform != "win32":
        return False

    candidates = [
        Path(r"C:\Program Files\Docker\Docker\Docker Desktop.exe"),
        Path(os.environ.get("LOCALAPPDATA", "")) / "Docker" / "Docker Desktop.exe",
    ]
    executable = next((p for p in candidates if p.exists()), None)
    if executable is None:
        return False

    say("Docker is not running — starting Docker Desktop")
    try:
        subprocess.Popen([str(executable)])
    except OSError:
        return False

    return wait_for("   Docker", docker_available, 180, interval_s=4.0)


def start_containers() -> None:
    say("Starting Postgres and Qdrant", "1.")
    if not docker_available() and not try_start_docker_desktop():
        fail(
            "Docker is not running.",
            "Launch Docker Desktop, wait for the whale icon to settle, then re-run.",
        )

    result = subprocess.run(
        ["docker", "compose", "up", "-d", "postgres", "qdrant"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        fail(f"docker compose failed:\n{result.stderr.strip()}")

    if not wait_for("   Qdrant", lambda: get_json(f"{QDRANT_URL}/collections") is not None, 120):
        fail(
            "Qdrant did not become reachable.",
            "Check `docker logs manda-qdrant`. If it panicked on a segment format, "
            "the volume was written by an older Qdrant — `docker volume rm ma_qdrant_data`.",
        )


def start_api() -> subprocess.Popen:
    say("Starting the API (loads ~5GB of models — this is the slow step)", "2.")
    log_path = PROJECT_ROOT / "demo_api.log"
    log = open(log_path, "w", encoding="utf-8")
    process = subprocess.Popen(
        [PYTHON, "run_api.py"],
        cwd=PROJECT_ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
    )

    def healthy() -> bool:
        if process.poll() is not None:
            return False
        return get_json(f"{API_URL}/health") is not None

    if not wait_for("   API", healthy, 420, interval_s=3.0):
        if process.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
            fail(f"The API exited during startup. Last output:\n{tail}")
        fail(f"The API did not come up. See {log_path}")

    return process


def ingest_if_empty() -> None:
    say("Checking the index", "3.")
    deals = get_json(f"{API_URL}/api/v1/deals", timeout=30) or []
    deal = next((d for d in deals if d.get("deal_id") == DEAL_ID), None)
    indexed = (deal or {}).get("document_count", 0)

    files = sorted(CORPUS_DIR.glob("*.txt"))
    if not files:
        say(f"No corpus found at {CORPUS_DIR} — skipping ingestion", "!")
        return

    if indexed >= len(files):
        say(f"{indexed} documents already indexed for {DEAL_ID}", "✓")
        return

    say(f"Index has {indexed}/{len(files)} documents — ingesting the sample data room")
    for path in files:
        ok = _ingest_one(path)
        print(f"      {'✓' if ok else '✗'} {path.name}", flush=True)

    deals = get_json(f"{API_URL}/api/v1/deals", timeout=30) or []
    deal = next((d for d in deals if d.get("deal_id") == DEAL_ID), None)
    final = (deal or {}).get("document_count", 0)
    if final < len(files):
        fail(
            f"Only {final}/{len(files)} documents indexed.",
            "See demo_api.log. A Qdrant client/server version skew makes every "
            "write fail while reads keep working.",
        )
    say(f"{final} documents indexed", "✓")


def _ingest_one(path: Path) -> bool:
    """Posts one document to the ingest endpoint as multipart/form-data."""
    import mimetypes
    import uuid

    boundary = uuid.uuid4().hex
    category = CATEGORY_OVERRIDES.get(path.name)
    content_type = mimetypes.guess_type(path.name)[0] or "text/plain"

    parts: list[bytes] = []

    def field(name: str, value: str) -> None:
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
            f"{value}\r\n".encode("utf-8")
        )

    field("deal_id", DEAL_ID)
    field("is_current_version", "true")
    if category:
        field("document_category", category)

    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"{path.name}\"\r\nContent-Type: {content_type}\r\n\r\n".encode("utf-8")
    )
    parts.append(path.read_bytes())
    parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))

    request = urllib.request.Request(
        f"{API_URL}/api/v1/ingest",
        data=b"".join(parts),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as resp:
            return resp.status == 200
    except Exception:
        return False


def start_ui() -> subprocess.Popen:
    say("Starting the Streamlit UI", "4.")
    log = open(PROJECT_ROOT / "demo_ui.log", "w", encoding="utf-8")
    process = subprocess.Popen(
        [PYTHON, "-m", "streamlit", "run", "app/streamlit_app.py",
         "--server.port", "8501", "--server.headless", "true"],
        cwd=PROJECT_ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    wait_for("   UI", lambda: get_json(f"{UI_URL}/_stcore/health") is not None
             or _plain_ok(f"{UI_URL}/_stcore/health"), 120)
    return process


def _plain_ok(url: str) -> bool:
    """Streamlit's health endpoint returns bare "ok", which is not JSON."""
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def stop_everything() -> None:
    print("\nStopping containers…", flush=True)
    subprocess.run(["docker", "compose", "down"], cwd=PROJECT_ROOT)
    print("Stopped. Vector and budget data are preserved in the named volumes.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stop", action="store_true", help="stop the containers and exit")
    parser.add_argument("--no-ui", action="store_true", help="start the API only")
    args = parser.parse_args()

    if args.stop:
        stop_everything()
        return

    print("\n  M&A Due Diligence Intelligence Engine — local demo\n")
    start_containers()
    api = start_api()
    ingest_if_empty()
    ui = None if args.no_ui else start_ui()

    print(f"""
  Ready.

    UI       {UI_URL}
    API docs {API_URL}/docs

  The deal is preselected and the models are warm — the first question runs at
  the same speed as every one after it. Logs: demo_api.log, demo_ui.log.

  Ctrl+C stops the API and UI. `python run_demo.py --stop` also stops the
  containers.
""", flush=True)

    try:
        while True:
            time.sleep(1)
            if api.poll() is not None:
                print("\n  The API exited — see demo_api.log", flush=True)
                break
            if ui is not None and ui.poll() is not None:
                print("\n  The UI exited — see demo_ui.log", flush=True)
                break
    except KeyboardInterrupt:
        print("\n  Shutting down…", flush=True)
    finally:
        for process in (ui, api):
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
        print("  API and UI stopped. Containers are still running.", flush=True)


if __name__ == "__main__":
    main()
