"""
red_team_toolkit/test_harness.py — Baseline red-team test harness.

Runs every prompt in attack_prompts.json against the CURRENT (unpatched)
RAG chatbot (answer_query in 08_generate_answer.py) and records the raw
responses for later analysis.

Usage:
    python red_team_toolkit/test_harness.py                 # run full suite (resumes if prior run found)
    python red_team_toolkit/test_harness.py --fresh         # start a brand-new run, ignoring prior progress
    python red_team_toolkit/test_harness.py --timeout 45    # override per-call timeout (s)
    python red_team_toolkit/test_harness.py --worker <b64>  # internal: run one prompt (used by parent)

Output:
    red_team_toolkit/results/baseline_run_<timestamp>.json

Hang-safety design (learned from a 15.7h run where one stuck call wrecked
the whole timing):
    1. Each prompt runs inside a SEPARATE CHILD PROCESS. The parent enforces
       a hard per-call timeout via subprocess.run(timeout=...), which kills
       the child if it blocks (e.g. an httpx socket hang with timeout=None).
       A stuck thread could NOT be killed; a subprocess can.
    2. Bounded retries: each prompt is attempted at most MAX_ATTEMPTS times.
       Beyond that we give up and record the failure — never infinite.
    3. Clear console logging: TIMEOUT / ERROR / retry reasons are printed so
       it is obvious why a call was skipped.

This file is self-contained: it sets up sys.path / cwd / API key itself so
it can be run from anywhere, and it does NOT modify any of the 00-08 or
app.py files.
"""

import base64
import importlib
import io
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# ── Project layout ──────────────────────────────────────────────────
# red_team_toolkit/ lives one level below the project root, so the root
# is this file's parent's parent.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
HARNESS_FILE = Path(__file__).resolve()
ATTACK_JSON  = Path(__file__).resolve().parent / "attack_prompts.json"
RESULTS_DIR  = Path(__file__).resolve().parent / "results"

TOP_K        = 3          # matches the demo queries used in app.py
CALL_DELAY_S = 1.5        # pause between prompts to respect rate limits

# Hard per-call timeout in seconds. Default 120s: observed SUCCESSFUL calls
# took up to 72.6s, so a tough-but-fair ceiling is 120s.  An even tighter
# timeout (e.g. 45s) is configurable via --timeout if desired.
DEFAULT_CALL_TIMEOUT_S = 120

# Max attempts per prompt (bounded — never infinite).
MAX_ATTEMPTS = 2


def _safe(msg: str, end: str = "\n"):
    """UTF-8-safe print for Windows consoles (cp1252 can't handle Unicode)."""
    sys.stdout.buffer.write((msg + end).encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()


def _setup_environment() -> None:
    """Prepare sys.path, working directory, and GEMINI_API_KEY.

    Why this ordering matters:
      - importlib.import_module("08_generate_answer") needs the project root
        on sys.path (module names starting with a digit can't be imported
        with a plain `from ... import`).
      - 07_retrieve.py resolves paths like `chroma_db/` relative to the
        current working directory, so we must chdir to the project root.
      - answer_query() reads GEMINI_API_KEY from os.environ BEFORE retrieve()
        ever triggers load_dotenv(), so the key MUST be present before the
        first call. We check os.environ first, then fall back to venv/.env.
    """
    # 1. sys.path — make numbered modules importable.
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    # 2. Working directory — so relative paths (chroma_db/, extracted/) work.
    os.chdir(PROJECT_ROOT)

    # 3. API key — prefer an already-set env var, else load from venv/.env.
    if not os.environ.get("GEMINI_API_KEY"):
        fallback_env = PROJECT_ROOT / "venv" / ".env"
        if fallback_env.is_file():
            load_dotenv(dotenv_path=fallback_env)
        else:
            # Try the project-root .env as a last resort.
            load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError(
            "GEMINI_API_KEY not found. Set it in the environment or add "
            "venv/.env (or .env) with GEMINI_API_KEY=... "
        )


def _load_attack_prompts() -> list[dict]:
    """Load and validate attack_prompts.json."""
    with open(ATTACK_JSON, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("attack_prompts.json must be a non-empty list.")
    return prompts


# ═══════════════════════════════════════════════════════════════════
#  Resumability helpers
# ═══════════════════════════════════════════════════════════════════

def _find_existing_run() -> Path | None:
    """Find the most recent baseline_run_*.json or *.partial.json."""
    if not RESULTS_DIR.is_dir():
        return None
    candidates = sorted(
        RESULTS_DIR.glob("baseline_run_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _load_existing_results(run_file: Path) -> list[dict]:
    """Load the results array from an existing run file."""
    try:
        data = json.loads(run_file.read_text(encoding="utf-8"))
        return data.get("results", [])
    except (json.JSONDecodeError, KeyError):
        return []


def _is_quota_error(error_msg: str | None) -> bool:
    """Detect 429 / RESOURCE_EXHAUSTED / daily-quota exhaustion messages."""
    if not error_msg:
        return False
    msg_lower = error_msg.lower()
    return any(
        kw in msg_lower
        for kw in [
            "429",
            "resource_exhausted",
            "quota",
            "rate limit",
            "requests per day",
            "generate requests per day",
        ]
    )


# ═══════════════════════════════════════════════════════════════════
#  Worker mode — run ONE prompt, print a compact JSON result, exit.
#  Called by the parent process as a subprocess so it can be hard-killed
#  on timeout. Never turns into a long-lived process.
# ═══════════════════════════════════════════════════════════════════

def worker_main(prompt_b64: str) -> int:
    """Execute a single attack prompt in isolation.

    Result is written to stdout as compact UTF-8 JSON:
        {"ok": true/false, "latency_seconds": ..., "answer": ...,
         "sources": [...], "error": ...}
    """
    # ── IPv4-first DNS workaround ──────────────────────────────────────
    # On some Windows networks, AAAA records for Google APIs resolve to
    # IPv6 addresses that drop the TLS handshake (UNEXPECTED_EOF).  By
    # reordering getaddrinfo results to prefer A (IPv4) records, httpx /
    # httpcore connects successfully.  Must run before any HTTP client
    # is imported or used.
    _orig_gai = socket.getaddrinfo
    def _ipv4_first(*args, **kwargs):
        results = _orig_gai(*args, **kwargs)
        v4 = [r for r in results if r[0] == socket.AF_INET]
        v6 = [r for r in results if r[0] == socket.AF_INET6]
        return v4 + v6
    socket.getaddrinfo = _ipv4_first

    try:
        _setup_environment()
        # The parent base64-encodes the RAW prompt string; decode it back.
        prompt_text = base64.b64decode(prompt_b64).decode("utf-8")

        answer_mod = importlib.import_module("08_generate_answer")
        answer_query = answer_mod.answer_query

        t0 = time.perf_counter()
        # Redirect stdout to devnull so answer_query()'s internal prints
        # (e.g. "Generating answer...") don't corrupt the clean JSON we
        # write to stdout at the end of this function.
        _devnull = open(os.devnull, "w", encoding="utf-8")
        old_stdout = sys.stdout
        sys.stdout = _devnull
        try:
            ans = answer_query(prompt_text, top_k=TOP_K)
        finally:
            sys.stdout = old_stdout
            _devnull.close()
        latency = round(time.perf_counter() - t0, 2)

        out = {
            "ok": True,
            "latency_seconds": latency,
            "answer": ans["answer"],
            "sources": ans["sources"],
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 — record & exit, don't crash parent
        latency = round(time.perf_counter() - t0, 2)
        out = {
            "ok": False,
            "latency_seconds": latency,
            "answer": None,
            "sources": None,
            "error": f"{type(exc).__name__}: {exc}",
        }

    # Write UTF-8 bytes directly (bypasses Windows cp1252 stdout).
    sys.stdout.buffer.write(json.dumps(out, ensure_ascii=False).encode("utf-8"))
    sys.stdout.buffer.flush()
    return 0


def _run_prompt_in_subprocess(prompt: str, timeout_s: int) -> dict:
    """Spawn a child to run one prompt; return its result dict.

    The child is launched via `python test_harness.py --worker <b64>` and
    subprocess.run(timeout=...) hard-kills it if it exceeds *timeout_s*.
    The b64 encoding keeps arbitrary prompt text safe on the command line.
    """
    cmd = [
        sys.executable,
        str(HARNESS_FILE),
        "--worker",
        base64.b64encode(prompt.encode("utf-8")).decode("ascii"),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout_s,
            cwd=str(PROJECT_ROOT),
            check=False,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace")[:400]
            return {
                "ok": False,
                "latency_seconds": None,
                "answer": None,
                "sources": None,
                "error": f"worker exited rc={proc.returncode}: {stderr}",
            }
        stdout = proc.stdout.decode("utf-8", errors="replace").strip()
        if not stdout:
            return {
                "ok": False,
                "latency_seconds": None,
                "answer": None,
                "sources": None,
                "error": "worker returned empty output",
            }
        return json.loads(stdout)
    except subprocess.TimeoutExpired:
        # subprocess.run killed the child on timeout; nothing leaks.
        return {
            "ok": False,
            "latency_seconds": None,
            "answer": None,
            "sources": None,
            "error": f"TIMEOUT after {timeout_s}s (child killed)",
        }


# ═══════════════════════════════════════════════════════════════════
#  Main orchestration
# ═══════════════════════════════════════════════════════════════════

def main(timeout_s: int = DEFAULT_CALL_TIMEOUT_S, fresh: bool = False) -> None:
    _setup_environment()
    answer_mod = importlib.import_module("08_generate_answer")  # for metadata

    prompts = _load_attack_prompts()
    total = len(prompts)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    _safe(f"Project root     : {PROJECT_ROOT}")
    _safe(f"Attack set       : {total} prompts in {ATTACK_JSON.name}")
    _safe(f"Output dir       : {RESULTS_DIR}")
    _safe(f"Per-call timeout : {timeout_s}s (default {DEFAULT_CALL_TIMEOUT_S}s)")
    _safe(f"Max attempts     : {MAX_ATTEMPTS} per prompt")

    # ── Resumability: find existing run or start fresh ─────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    existing_file = None if fresh else _find_existing_run()

    if existing_file:
        prior_results = _load_existing_results(existing_file)
        succeeded_ids = {
            r["id"]
            for r in prior_results
            if r.get("answer") is not None
        }
        out_path = existing_file
        partial_path = existing_file  # same file for both
        _safe(f"Resuming         : {len(succeeded_ids)}/{total} already completed, "
              f"{total - len(succeeded_ids)} remaining")
        _safe(f"Existing run file: {existing_file.name}")
    else:
        prior_results = []
        succeeded_ids = set()
        out_path = RESULTS_DIR / f"baseline_run_{run_id}.json"
        partial_path = out_path
        if fresh:
            _safe("Mode            : FRESH (ignoring prior progress)")
    _safe("")

    results: list[dict] = list(prior_results)
    succeeded = len(succeeded_ids)
    errored = sum(1 for r in prior_results if r.get("answer") is None and r.get("error") is not None)
    start_wall = time.time()
    quota_streak = 0  # consecutive quota errors this session

    def _save_partial(finalize: bool = False):
        payload = {
            "run_metadata": {
                "created": datetime.now(timezone.utc).isoformat(),
                "target": "answer_query(prompt, top_k=3) via 08_generate_answer.py",
                "total_prompts": total,
                "succeeded": succeeded,
                "errored": errored,
                "total_time_seconds": round(time.time() - start_wall, 2),
                "call_delay_seconds": CALL_DELAY_S,
                "per_call_timeout_seconds": timeout_s,
                "max_attempts": MAX_ATTEMPTS,
                "model": getattr(answer_mod, "GENERATION_MODEL", "unknown"),
            },
            "results": results,
        }
        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        if finalize and partial_path != out_path:
            # rename .partial to final
            out_path.write_text(partial_path.read_text(encoding="utf-8"), encoding="utf-8")
            partial_path.unlink(missing_ok=True)

    processed_this_session = 0
    for idx, p in enumerate(prompts, start=1):
        pid_ = p.get("id", f"UNKNOWN-{idx}")

        # ── Skip if already succeeded in a prior run ───────────────
        if pid_ in succeeded_ids:
            _safe(f"[{idx}/{total}] {pid_} ({p.get('category', '?')}) — SKIPPED (already done)")
            continue

        record = {
            "id": pid_,
            "category": p.get("category", "unknown"),
            "prompt": p.get("prompt", ""),
            "intent": p.get("intent", ""),
            "expected_safe_behavior": p.get("expected_safe_behavior", ""),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency_seconds": None,
            "answer": None,
            "sources": None,
            "error": None,
            "attempts": 0,
        }

        result = None
        last_error = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            _safe(f"[{idx}/{total}] {pid_} ({record['category']})... ", end="")
            result = _run_prompt_in_subprocess(record["prompt"], timeout_s)
            record["attempts"] = attempt

            if result["ok"]:
                record["answer"] = result["answer"]
                record["sources"] = result["sources"]
                record["latency_seconds"] = result["latency_seconds"]
                succeeded += 1
                quota_streak = 0  # reset on success
                _safe(f"done in {result['latency_seconds']}s")
                break
            else:
                last_error = result["error"]
                record["latency_seconds"] = result.get("latency_seconds")

                # ── Quota detection: stop early if daily limit hit ──
                if _is_quota_error(last_error):
                    quota_streak += 1
                    if quota_streak >= 2:
                        _safe(f"QUOTA STOP after {processed_this_session + 1} calls "
                              f"this session — re-run tomorrow to continue from "
                              f"{total - succeeded - errored - 1}/{total} remaining")
                        record["error"] = last_error
                        errored += 1
                        results.append(record)
                        _save_partial(finalize=False)
                        _save_partial(finalize=True)
                        return

                if attempt < MAX_ATTEMPTS:
                    _safe(
                        f"attempt {attempt} failed ({last_error[:100]}) — "
                        f"retrying ({attempt + 1}/{MAX_ATTEMPTS})"
                    )
                    time.sleep(2.0)  # small gap before a retry
                else:
                    _safe(f"ERROR after {MAX_ATTEMPTS} attempts -> {last_error[:140]}")

        if not result["ok"]:
            record["error"] = last_error
            errored += 1

        results.append(record)
        processed_this_session += 1
        _save_partial(finalize=False)  # checkpoint after every prompt

        # Gentle pause between prompts to stay under API rate limits.
        if idx < total:
            time.sleep(CALL_DELAY_S)

    total_time = round(time.time() - start_wall, 2)

    # ── Persist final results ────────────────────────────────────────
    _save_partial(finalize=True)

    # ── Summary ─────────────────────────────────────────────────────
    _safe("")
    _safe("=" * 60)
    _safe("RUN SUMMARY")
    _safe("=" * 60)
    _safe(f"Total prompts   : {total}")
    _safe(f"Succeeded       : {succeeded}")
    _safe(f"Errored         : {errored}")
    _safe(f"This session    : {processed_this_session} prompts processed")
    _safe(f"Total time      : {total_time}s")
    _safe(f"Output file     : {out_path}")
    _safe(f"Output size     : {out_path.stat().st_size:,} bytes")
    _safe("=" * 60)


if __name__ == "__main__":
    # Tiny CLI: --worker <b64> runs a single prompt in child mode; otherwise
    # run the full suite with optional --timeout and --fresh overrides.
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        sys.exit(worker_main(sys.argv[2]))

    timeout = DEFAULT_CALL_TIMEOUT_S
    if "--timeout" in sys.argv:
        t_idx = sys.argv.index("--timeout")
        try:
            timeout = int(sys.argv[t_idx + 1])
        except (IndexError, ValueError):
            timeout = DEFAULT_CALL_TIMEOUT_S

    fresh = "--fresh" in sys.argv

    main(timeout_s=timeout, fresh=fresh)