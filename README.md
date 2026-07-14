# Multimodal RAG — "Attention Is All You Need"

> A multimodal Retrieval-Augmented Generation system over the Transformer paper.
> Extracts text, tables, and figures from the PDF, embeds everything with Google
> Gemini, stores vectors in Chroma DB, and answers questions via grounded
> generation with figure re-attachment for vision-rich answers.

**Built by Maha Fatima** | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) | Google Gemini API | ChromaDB | Streamlit

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Setup](#setup)
- [Script-by-Script Guide](#script-by-script-guide)
- [Querying the System](#querying-the-system)
- [Streamlit App](#streamlit-app)
- [Technical Highlights](#technical-highlights)
- [Edge Cases & Error Handling](#edge-cases--error-handling)
- [Known Limitations](#known-limitations)

---

## Overview

This project implements a full RAG pipeline over the seminal "Attention Is All
You Need" paper. Unlike many RAG demos that handle only text, this system
processes **three distinct modalities** from a single PDF:

| Modality | What is extracted | How it is used |
|----------|------------------|----------------|
| **Text** | Section-aware chunks (~700 words) | Semantic search via embedding |
| **Tables** | Borderless tabular data (4 tables) | Semantic search + displayed as formatted content |
| **Figures** | Raster images + full-page renders (5 figures) | Text embedding for retrieval; actual image attached at generation time |

All three are unified into a single vector store, so a single query can retrieve
evidence across any combination of modalities. The LLM then generates an answer
grounded **only** in the retrieved context.

---

## Pipeline Architecture

```
PDF ("Attention Is All You Need")
│
├─── 00_inspect_pdf.py ──────── Console output (manual inspection)
│
├─── 01_extract_text.py ─────── extracted/text_chunks.json
│     (PyMuPDF block extraction, section-aware chunking)
│
├─── 02_extract_tables.py ───── extracted/tables.json
│     (pdfplumber word-level extraction, borderless-table heuristics)
│
├─── 03_extract_figures.py ──── extracted/images/*.png
│     (CASE A: embedded raster; CASE B: full-page vector render)
│
├─── 04_caption_figures.py ──── extracted/figures.json (enriched)
│     (Gemini vision descriptions for each figure)
│
├─── 05_unify_elements.py ───── extracted/unified_elements.json
│     (Normalises all three modalities into one schema)
│
├─── 06_build_vectorstore.py ── chroma_db/
│     (Gemini embeddings → Chroma HNSW index with cosine distance)
│
├─── 07_retrieve.py ─────────── Reusable retrieve() function
│     (Embed query → query Chroma → return clean results)
│
├─── 08_generate_answer.py ──── extracted/demo_results.json
│     (Retrieve → build context → Gemini generation → save)
│
└─── app.py ─────────────────── Streamlit browser UI
      (Interactive Q&A with source cards for text/table/figure)
```

Each script writes its output to disk and can be re-run independently. The
pipeline is designed so you can inspect or fix any stage without repeating
earlier steps.

---

## Setup

### Prerequisites

- Python 3.12+
- A Google Gemini API key (free tier works; get one at
  [aistudio.google.com/apikey](https://aistudio.google.com/apikey))

### Installation

```bash
# 1. Clone the repo
git clone <repo-url>
cd multimodal-rag-attention

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Gemini API key
echo GEMINI_API_KEY=your_key_here > .env
```

### Download the source PDF

The PDF is excluded from version control (see [.gitignore](#version-control)).
Download it from arXiv:

```bash
curl -L -o "attention is all you need.pdf" \
  https://arxiv.org/pdf/1706.03762.pdf
```

Alternatively, rename your downloaded copy to `attention is all you need.pdf`
and place it in the project root.

---

## Script-by-Script Guide

### 00 — Inspect PDF (`00_inspect_pdf.py`)

Opens the PDF and prints basic metadata: page count, text preview per page,
number of embedded images per page. Use this to understand the document
structure before extraction.

```bash
python 00_inspect_pdf.py
```

### 01 — Extract Text (`01_extract_text.py`)

Uses PyMuPDF (`fitz`) block-level extraction to extract body text. Key details:

- **Header/footer filtering**: blocks in the top 8% or bottom 6% of page height
  shorter than 25 characters are discarded.
- **Boilerplate rejection**: known arXiv copyright lines and common strings are
  filtered via `SKIP_PATTERNS`.
- **Section-aware chunking**: a `current_section` tracker carries section
  headings across pages. Body blocks are grouped into ~700-word chunks, each
  tagged with its section name.
- **Heading detection**: special handling so short headings are not rejected by
  the margin heuristic.

**Output:** `extracted/text_chunks.json` (25 chunks)

### 02 — Extract Tables (`02_extract_tables.py`)

Extracts the paper's tables (pages 6, 8, 9, 10) using pdfplumber. These tables
have **no visible borders**, so the standard `extract_tables()` fails.

**Key technique — word-level reconstruction:**

1. Extract individual words with `page.extract_words(x_tolerance=1)` (lowered
   from the default `x_tolerance=3` to prevent justified-text word merging).
2. Group words by rounded y-position (binned at 4pt) and sort by x-position.
3. Join each group with spaces to reconstruct clean text lines.
4. Detect "Table N:" captions and isolate the tabular region.
5. Heuristics detect where table data ends and body prose resumes (section
   number patterns, sentence-boundary detection).
6. False-positive filter removes attention-visualisation pages (pages 13–15)
   that contain `<pad>`/`<EOS>` tokens.

**Output:** `extracted/tables.json` (4 tables)

### 03 — Extract Figures (`03_extract_figures.py`)

Extracts 5 figures from the paper using two strategies:

- **CASE A (pages 3–4)**: Figures are embedded as raster images directly in the
  PDF objects. Extracted via `doc.extract_image(xref)` at original quality.
- **CASE B (pages 13–15)**: Figures are vector-drawn (no embedded raster).
  Detected by `page.get_images()` returning empty. Falls back to a full-page
  render via `page.get_pixmap(matrix=fitz.Matrix(2,2))` at 144 DPI.

Caption detection uses `page.get_text("text", clip=bbox)` on the approximate
caption region below each figure image.

**Output:** `extracted/images/*.png`, `extracted/figures.json`

### 04 — Caption Figures (`04_caption_figures.py`)

Sends each extracted figure image to **Gemini 2.5 Flash** with a vision prompt
asking for a 4–6 sentence description. The resulting `vision_description` is
added to each entry in `figures.json`.

A 2-second delay between API calls avoids rate limits during batch processing.

**Output:** `extracted/figures.json` (enriched with `vision_description`)

### 05 — Unify Elements (`05_unify_elements.py`)

Combines text chunks, tables, and figures into a single unified schema:

```json
{
  "id": "figure_2",
  "type": "figure",
  "page": 4,
  "title": "Figure 2: Scaled Dot-Product Attention",
  "embedding_text": "Figure 2: Scaled Dot-Product Attention... [caption + vision description]",
  "display_content": "This figure illustrates... [vision description only]",
  "image_path": "extracted/images/page4_fig2.png"
}
```

**Dual-content strategy**: `embedding_text` is optimised for semantic search
(richer, includes caption and description), while `display_content` is
optimised for human reading (cleaner, description only).

A validation pass warns if any element has an empty `embedding_text` (which
would produce a meaningless vector).

**Output:** `extracted/unified_elements.json` (35 elements: 25 text, 4 tables,
6 figures — one extra figure from splitting a grouped image)

### 06 — Build Vector Store (`06_build_vectorstore.py`)

1. Embeds every element using **Gemini Embedding 001** (3072-dimensional
   vectors — higher quality than the default all-MiniLM ONNX model).
2. Stores vectors in a persistent **Chroma DB** collection (`attention_paper`)
   configured with cosine distance (`hnsw:space`).
3. `embedding_function=None` disables Chroma's default embedding model since
   we supply our own vectors.
4. Runs a sanity query (`"transformer architecture"`) and prints the top-4
   results so you can manually verify retrieval quality.

1-second delay between embedding calls prevents per-minute rate limit errors.

**Output:** `chroma_db/` (persistent vector store directory)

### 07 — Retrieve (`07_retrieve.py`)

A reusable `retrieve(query, top_k=4)` function used by the generation script
and the Streamlit app:

- **Lazy initialisation**: Chroma client and Gemini client are created on the
  first call, not at import time.
- **Result adapter**: converts Chroma's raw parallel-list format into clean
  dicts with named fields (`id`, `type`, `page`, `title`, `display_content`,
  `image_path`, `distance`, `similarity`).
- **Similarity conversion**: cosine distance (lower = better) is converted to
  `similarity = 1 - distance` (higher = better) for intuitive display.
- **Quota resilience**: the embedding API call is wrapped with `_api_retry` —
  exponential backoff (capped at 60s), hard timeout of 180s, interruptible
  polling sleep for responsive Ctrl+C.
- **CLI mode**: `python 07_retrieve.py` runs three test queries and prints
  ranked results for manual inspection.

### 08 — Generate Answer (`08_generate_answer.py`)

The full RAG generation pipeline:

1. **Retrieve**: calls `retrieve(query)` to get relevant chunks.
2. **Build context**: constructs a grounded prompt with each source labelled
   by type, title, and page. If any source is a figure with an `image_path`,
   the actual image is loaded with PIL.
3. **Generate**: sends the prompt + any figure images to Gemini 2.5 Flash.
   The prompt explicitly instructs the model to NOT insert `[Source N]` markers
   and instead reference sources naturally (e.g. "as shown in Table 2").
4. **Retry logic**: a shared `_api_retry` helper handles 429 quota errors with
   full error printing, exponential backoff, and a 180-second hard cap.
5. **Incremental saving**: results are saved to `demo_results.json` after
   *each* query, so partial output survives a Ctrl+C or crash mid-run.
6. **Multi-query mode**: runs three test queries (table, architecture, attention)
   with a 5-second delay between them, saving to JSON and Markdown.

**Key design decision — figure re-attachment at generation time:**

> Vector databases cannot index pixels. Retrieval is done on text embeddings
> (caption + description). But once a figure is retrieved, the actual image is
> sent to the LLM so it can read diagrams and plots directly — producing richer
> answers than from the text description alone.

**Output:** `extracted/demo_results.json`, `extracted/demo_results.md`

---

## Querying the System

### Via Streamlit (recommended)

```bash
python -m streamlit run app.py
```

Opens a browser UI where you can:
- Type any question about the paper
- Click one of four example queries (table, architecture, attention, training)
- View the generated answer with expandable source cards
- See actual figure images in the sources

### Via CLI (for testing)

```bash
# Test retrieval only
python 07_retrieve.py

# Run full RAG with 3 demo queries
python 08_generate_answer.py
```

---

## Streamlit App (`app.py`)

The frontend is a **thin UI layer** — it imports `retrieve()` and
`answer_query()` from the backend scripts rather than duplicating any pipeline
logic.

Key features:

| Feature | Implementation |
|---------|---------------|
| **Example query buttons** | 4 pre-set queries (one per modality) for one-click testing |
| **Spinner** | `st.spinner` while the pipeline runs |
| **Quota error display** | `st.error()` for rate-limit and other failures |
| **Answer display** | `st.info` container for the generated answer |
| **Source cards** | Expandable section showing each retrieved source with type-specific rendering |
| **Figure images** | `st.image()` displays actual diagrams in source cards |
| **Sidebar** | About section with project description |
| **Footer** | Attribution to the paper, Gemini API, and ChromaDB |

Session state (`st.session_state`) preserves results across Streamlit
re-renders so the answer does not disappear when interacting with widgets.

---

## Technical Highlights

### Dual-Content Strategy
What gets vectorised (`embedding_text`) differs from what gets displayed
(`display_content`). For figures, the embedding includes both the caption and
the vision description (richer search), while display shows only the description
(cleaner reading).

### Borderless Table Extraction
pdfplumber's `extract_tables()` relies on visible ruled lines — absent in this
paper. The system falls back to low-level word extraction (`x_tolerance=1`),
groups words by y-position, and applies heuristic table-boundary detection.

### Figure Extraction: Two Strategies
Figures can be embedded raster images (direct object extraction via xref) or
vector-drawn diagrams (full-page pixmap render at 144 DPI). The code
auto-detects which case applies per page.

### Shared Quota-Resilient API Retry
A single `_api_retry()` helper in `07_retrieve.py` handles rate-limit errors
for both the embedding API and the generation API. Features:
- Catches only `429`/`RESOURCE_EXHAUSTED` — other errors propagate immediately
- Prints the full API error message on every attempt
- Exponential backoff capped at 60s per wait
- Hard timeout of 180s total retry time
- Interruptible polling sleep (0.2s increments) for responsive Ctrl+C

### Importing Digit-Starting Modules
Python forbids `from 07_retrieve import retrieve`. The project uses
`importlib.import_module("07_retrieve")` as a workaround, repeated in three
files.

### Unicode-Safe Console Output
Windows `cp1252` cannot encode many Unicode characters. A `_safe()` helper
writes UTF-8 bytes directly to `sys.stdout.buffer` with `errors="replace"`.

---

## Edge Cases & Error Handling

The pipeline explicitly handles 28+ edge cases across the extraction,
retrieval, and UI stages, including:

| Category | Edge Case | How Handled |
|----------|-----------|-------------|
| Text extraction | Page numbers in margins | `text.isdigit()` + margin-position check |
| Text extraction | ArXiv boilerplate | `SKIP_PATTERNS` string-list filter |
| Text extraction | Section headings too short for margin heuristic | Prioritised heading detection before short-block filter |
| Text extraction | "Figure N:" captions misidentified as headings | `NOT_HEADING_PREFIXES` exclusion |
| Table extraction | Borderless tables | Fallback to `extract_words()` word-level reconstruction |
| Table extraction | Justified text merging words | Lowered `x_tolerance` from 3 to 1 |
| Table extraction | Caption bleeding into table headers | Heuristic: >=3 words, >=50% capitalised, no trailing period |
| Table extraction | Attention heatmaps with `<pad>`/`<EOS>` tokens | Explicit false-positive string filter |
| Figure extraction | Embedded raster images | Direct `doc.extract_image(xref)` |
| Figure extraction | Vector-drawn figures (no raster) | Full-page `pixmap` render fallback |
| Figure extraction | Missing "Figure N:" caption | Fallback caption: `"Figure (page N)"` |
| Unicode | Windows console cp1252 encoding | `sys.stdout.buffer.write(utf8_bytes)` |
| Quota | Gemini 429 rate limit | Exponential backoff + 180s hard cap + interruptible sleep |
| Quota | Partial batch failure | Incremental saving after each query |
| UI | Figure image missing from disk | Silent skip in context builder; `"(image file not found)"` in UI |
| UI | Empty table / empty text chunk | Fallback captions in source cards |

---

## Known Limitations

| Limitation | Explanation |
|-----------|-------------|
| **PDF-specific tuning** | Extraction heuristics are tuned to this paper's structure. Reprocessing a different PDF would require adjustment. |
| **English-only text** | Sentence-boundary detection assumes English punctuation patterns. |
| **No OCR fallback** | Scanned/image-based PDFs are not supported (no Tesseract integration). |
| **Gemini API dependency** | Generation, embedding, and captioning all require the Gemini API. No offline fallback. |
| **Rate limits** | Free tier allows ~20 generation requests/day. Exceeding this pauses the pipeline for hours. |
| **No embedding caching** | Rebuilding the vector store re-embeds every element, even if only one extraction step changed. |
| **No query decomposition** | Each question is a single retrieval step. Multi-hop or decomposed queries are not supported. |
| **No streaming** | The Streamlit app waits for full generation before displaying results. |
| **Chroma metadata size limits** | `display_content` (potentially hundreds of words) is stored as metadata, which may hit Chroma's string-size limits for very large chunks. |

---

## Version Control

The following are excluded from git (see `.gitignore`):

| Path | Reason |
|------|--------|
| `.env` | Contains API key — never commit |
| `venv/` | Virtual environment — recreated locally |
| `chroma_db/` | Vector store — regenerated by running steps 00–06 |
| `extracted/` | All generated data (JSON, images, demo results) — pipeline artifacts |
| `attention is all you need.pdf` | Large binary — download from arXiv |
| `__pycache__/`, `*.pyc` | Python bytecode |
| `.vscode/`, `.idea/` | IDE configuration |

---

## Demo

Three example queries demonstrating the system:

| Query | Modality Focus |
|-------|---------------|
| "What BLEU score did the Transformer achieve on English-to-French translation, and how does it compare to the training cost of previous models?" | **Table** lookup |
| "Describe the encoder-decoder architecture shown in the Transformer diagram." | **Figure** + text |
| "What is multi-head attention and why is it used instead of single-head attention?" | **Text** explanation |

Run `python 08_generate_answer.py` to generate fresh results, or use the
Streamlit app to ask your own questions.
