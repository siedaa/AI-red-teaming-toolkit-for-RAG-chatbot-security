# AI Red Teaming Toolkit

Automated security testing for the Multimodal RAG chatbot ("Attention Is All You Need").

## Overview

This toolkit tests the RAG chatbot against common LLM security vulnerabilities by
feeding it curated attack prompts and judging each response with an independent LLM
evaluator. It helps identify where the chatbot's guardrails are strong and where they
need improvement — before real adversaries find the gaps.

## Folder Structure

```
red_team_toolkit/
├── attack_prompts.json    # 35 attack prompts across 5 vulnerability categories
├── test_harness.py        # Runs each prompt against answer_query() in isolated subprocesses
├── evaluator.py           # Groq LLM-as-judge: scores each response (SAFE / PARTIALLY_VULNERABLE / VULNERABLE)
├── results/               # Raw baseline runs and evaluation JSONs (timestamped)
│   ├── baseline_run_*.json
│   └── evaluation_baseline_*.json
└── reports/               # Human-readable write-ups and analysis (planned)
```

## How It Works

```
attack_prompts.json (35 attacks)
        │
        ▼
test_harness.py
  • Spawns each prompt in an isolated child subprocess (hard timeout prevents hangs)
  • Calls answer_query() directly — no Streamlit, no UI
  • Logs results to results/baseline_run_<timestamp>.json
        │
        ▼
evaluator.py
  • Loads the latest baseline run
  • Sends all 35 cases to Groq (openai/gpt-oss-120b) in a batched request
  • Falls back to chunked requests if the single call is too large
  • Judges each response: SAFE / PARTIALLY_VULNERABLE / VULNERABLE
  • Saves evaluation to results/evaluation_baseline_<timestamp>.json
```

## How to Run

### Prerequisites

Set these environment variables (or add them to `venv/.env`):

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Required by the chatbot's `answer_query()` for retrieval + generation |
| `GROQ_API_KEY` | Required by `evaluator.py` for the LLM judge |

### Run the attack suite

```bash
# First run (or resume from a previous incomplete run):
python red_team_toolkit/test_harness.py

# Force a fresh run from scratch (ignores prior progress):
python red_team_toolkit/test_harness.py --fresh

# Override per-call timeout (default 120s):
python red_team_toolkit/test_harness.py --timeout 60
```

The harness is **resumable** — if a run is interrupted or hits the daily quota, re-run
and it will skip already-succeeded prompts and continue from where it left off.

### Run the evaluator

```bash
# Evaluate the latest baseline run:
python red_team_toolkit/evaluator.py

# Evaluate a specific run file:
python red_team_toolkit/evaluator.py --run baseline_run_20260906_030713.json
```

## Attack Categories Tested

| Category | Count | Description |
|----------|-------|-------------|
| **prompt_injection** | 7 | Attempts to override system instructions (direct overrides, smuggled context, priority-hijacking, multi-step chains) |
| **jailbreak** | 7 | Roleplay/hypothetical/DAN framing to bypass safety filters (make-believe, thought experiments, obfuscation, fiction framing) |
| **role_confusion** | 7 | Fake system turns, admin mode, developer personas, function calls, authority claims |
| **system_prompt_leakage** | 7 | Direct extraction, encoding tricks, reconstruction attacks, debugging pretexts, translation pretexts |
| **data_extraction** | 7 | Raw context dumps, scope-escape via pretext, verbatim passage extraction, fabricated content |

## Current Status (Baseline Results)

**Baseline run**: 35/35 prompts succeeded, 0 errors.

| Verdict | Count | Percentage |
|---------|-------|------------|
| SAFE | 20 | 57% |
| PARTIALLY_VULNERABLE | 4 | 11% |
| VULNERABLE | 11 | 32% |

### Breakdown by category

| Category | SAFE | PARTIALLY_VULNERABLE | VULNERABLE |
|----------|------|---------------------|------------|
| prompt_injection | 3 | 4 | 0 |
| jailbreak | 5 | 0 | 2 |
| role_confusion | 7 | 0 | 0 |
| system_prompt_leakage | 1 | 0 | 6 |
| data_extraction | 4 | 0 | 3 |

### Key findings

- **System prompt leakage is the weakest area**: 6/7 prompts resulted in the chatbot
  reproducing its system prompt (verbatim, base64-encoded, or targeted sub-instructions).
  This is the highest-priority area for guardrail improvements.
- **Role confusion is well-handled**: 7/7 SAFE — the chatbot correctly deflected all
  fake admin/developer/system-turn attacks.
- **Jailbreaks mostly deflected**: 5/7 SAFE, but 2/7 vulnerable to creative-writing
  fiction framing and trust-manipulation.
- **Data extraction**: 3/7 vulnerable — the chatbot dumps raw retrieved chunks when
  asked, rather than summarizing.
- **Prompt injection**: Partially compliant on 4/7 — echoes injected instructions
  instead of refusing, though never fully vulnerable.

Guardrail improvements and a re-test are in progress.

## Known Constraints

- **Gemini free-tier quota**: The API key is on a free tier with a hard limit of
  `GenerateRequestsPerDayPerProjectPerModel-FreeTier` (20 requests/day). The harness
  is resumable to handle this — if the daily quota is exhausted, re-run the next day
  and it will continue from where it left off.
- **IPv6 DNS issue**: On some Windows networks, AAAA records for Google APIs cause TLS
  handshake failures. The harness includes an IPv4-first DNS workaround.
- **Subprocess isolation**: Each prompt runs in a child process with a hard timeout
  (default 120s) to prevent a single hung API call from stalling the entire run.
