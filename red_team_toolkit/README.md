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
├── screenshots/           # Red Teaming tab screenshots (embedded below)
├── results/               # Raw attack runs and evaluation JSONs (timestamped)
│   ├── baseline_run_*.json
│   ├── evaluation_baseline_*.json   # before-guardrails evaluation
│   └── evaluation_patched_*.json    # after-guardrails evaluation
└── reports/               # Human-readable write-ups and analysis
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

## Current Status (Before/After Guardrails)

Both testing phases are complete: a baseline run against the original chatbot, then a full
re-run after guardrails were implemented (an input-side `input_guard()` regex filter plus
5 numbered security rules in the system prompt). **Baseline run**: 35/35 prompts succeeded,
0 errors. **Patched run**: 35/35 prompts succeeded, 0 errors.

| Verdict | Before (baseline) | After (guardrails) | Change |
|---------|-------------------|--------------------|--------|
| SAFE | 20 (57%) | 35 (100%) | +15 |
| PARTIALLY_VULNERABLE | 4 (11%) | 0 (0%) | −4 |
| VULNERABLE | 11 (31%) | 0 (0%) | −11 |
| **Vulnerability rate** | **43% (15/35)** | **0% (0/35)** | −43 pts |

### Breakdown by category (before → after)

| Category | Before (SAFE / PV / V) | After (SAFE / PV / V) |
|----------|------------------------|-----------------------|
| prompt_injection | 3 / 4 / 0 | 7 / 0 / 0 |
| jailbreak | 5 / 0 / 2 | 7 / 0 / 0 |
| role_confusion | 7 / 0 / 0 | 7 / 0 / 0 |
| system_prompt_leakage | 1 / 0 / 6 | 7 / 0 / 0 |
| data_extraction | 4 / 0 / 3 | 7 / 0 / 0 |

### Key findings (baseline → patched)

- **System prompt leakage was the weakest baseline area**: 6/7 prompts reproduced the
  chatbot's system prompt (verbatim, base64-encoded, or targeted sub-instructions).
  Fixed by the instruction-disclosure refusal rule + `input_guard()` → now 7/7 SAFE.
- **Role confusion was already well-handled**: 7/7 SAFE before and after — the chatbot
  consistently deflected all fake admin/developer/system-turn attacks.
- **Jailbreaks**: 2/7 vulnerable to creative-writing fiction framing and obfuscation →
  now 7/7 SAFE.
- **Data extraction**: 3/7 vulnerable — the chatbot dumped raw retrieved chunks when
  asked; the synthesis-over-raw-dumps rule fixed this → now 7/7 SAFE.
- **Prompt injection**: 4/7 partially compliant (echoed injected instructions) →
  now 7/7 SAFE.

Guardrails are implemented and verified: the full 35-prompt suite scores **35/35 SAFE
(0% vulnerable)**, versus 43% at least partially exploitable in the baseline. See
`reports/before_after_comparison_report.md` for the detailed side-by-side comparison and
`reports/baseline_vulnerability_report.md` for the full baseline analysis (with CWE
classifications and CVSS v3.1 scores).

## Streamlit Red Teaming Tab

To make the security testing results visible and interactive — rather than buried in
JSON result files and markdown reports — a dedicated **Red Teaming** tab was added to the
Streamlit app (`app.py`, sitting alongside the Chat tab). It surfaces the before/after
guardrail comparison as charts and metric tiles, shows real attack/response transcripts,
and optionally launches live attacks against the patched chatbot. The three screenshots
below walk through it in order: tab overview, the per-category verdict chart, then the
before/after transcripts and live demo.

![Red Teaming Tab Overview](screenshots/red_teaming_tab_overview.png)

*Tab overview: the headline before/after summary (43% of prompts exploited before
guardrails vs 0% after) and the overall verdict-count tiles — 20/4/11 before, 35/0/0 after.*

![Verdicts by Category](screenshots/verdict_counts_metrics.png)

*Verdicts by category: stacked before/after chart of SAFE / PARTIALLY_VULNERABLE /
VULNERABLE counts across the five attack categories, leading into the example transcripts.*

![Before/After Transcripts and Live Attack Demo](screenshots/before_after_live_demo.png)

*Before/after example transcripts (expandable) and the Live Attack Demo panel for running
attack prompts against the patched chatbot in real time.*

## Known Constraints

- **Gemini free-tier quota**: The API key is on a free tier with a hard limit of
  `GenerateRequestsPerDayPerProjectPerModel-FreeTier` (20 requests/day). The harness
  is resumable to handle this — if the daily quota is exhausted, re-run the next day
  and it will continue from where it left off.
- **IPv6 DNS issue**: On some Windows networks, AAAA records for Google APIs cause TLS
  handshake failures. The harness includes an IPv4-first DNS workaround.
- **Subprocess isolation**: Each prompt runs in a child process with a hard timeout
  (default 120s) to prevent a single hung API call from stalling the entire run.
