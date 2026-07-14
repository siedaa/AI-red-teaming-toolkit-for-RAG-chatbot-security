"""
01_extract_text.py — Extract body text with section metadata from the PDF.

Usage:
    python 01_extract_text.py

What it does:
    1. Opens the PDF with PyMuPDF.
    2. Reads text blocks (with bounding-box positions) via page.get_text("blocks").
    3. Skips headers / footers / page numbers using simple heuristics.
    4. Detects section headings (numbered patterns, known names).
    5. Groups body-text blocks under the nearest heading.
    6. Splits long sections into ~700-word chunks.
    7. Saves the result to extracted/text_chunks.json.

Learnings:
    - page.get_text("blocks") returns a list of (x0, y0, x1, y1, text, block_no, block_type)
      tuples — much richer than page.get_text() alone.
    - In PyMuPDF 1.28, "blocks" only returns text blocks (type=0); images are
      retrieved separately via page.get_images().  The tuple is
      (x0, y0, x1, y1, text, block_no, block_type).
    - We can use y-coordinates to guess whether a block is a page header/footer.
    - Section-aware chunking preserves context (each chunk knows what section it belongs to).
"""

import json
import os
import re

import fitz


# ═══════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════

PDF_PATH = "attention is all you need.pdf"
OUTPUT_DIR = "extracted"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "text_chunks.json")

# A section longer than this many words is split into multiple chunks.
MAX_WORDS_PER_CHUNK = 700

# Blocks shorter than this (in characters) are *candidates* for headings.
HEADING_MAX_CHARS = 60

# Top / bottom margins (as fraction of page height) used to detect
# running headers, page numbers, etc.
TOP_MARGIN_RATIO = 0.08
BOTTOM_MARGIN_RATIO = 0.06

# ── Patterns for blocks we want to skip entirely ──────────────────
# If any of these substrings appears (case‑insensitive) the block is dropped.
SKIP_PATTERNS = [
    "arxiv:",
    "provided proper attribution",
    "google hereby grants permission",
    "scholarly works",
    "31st conference on neural information processing systems",
]
# Very short blocks (< 25 chars) that sit in the margin are also skipped
# (this catches things like page numbers).

# ── Known section heading names (compared case‑insensitively)  ────
KNOWN_HEADINGS = [
    "abstract",
    "introduction",
    "conclusion",
    "references",
    "background",
    "related work",
]

# ── Prefixes that make a short block *not* a heading (captions, etc.)
NOT_HEADING_PREFIXES = [
    "figure",
    "table",
    "algorithm",
]


# ═══════════════════════════════════════════════════════════════════
#  Helper functions
# ═══════════════════════════════════════════════════════════════════

def is_skip_block(text: str, y0: float, y1: float, page_height: float) -> bool:
    """
    Return True if this block looks like a page header, footer, or
    other decorative element that should be ignored.
    """
    stripped = text.strip()
    if not stripped:
        return True  # empty block

    lower = stripped.lower()

    # 1) Known skip patterns (arXiv ID, copyright notice, …)
    for pat in SKIP_PATTERNS:
        if pat in lower:
            return True

    # 2) Page number (just digits)
    if stripped.isdigit():
        return True

    # 3) Very short block sitting in the top or bottom margin →
    #    likely a running header or page number.
    if len(stripped) < 25:
        top_edge = page_height * TOP_MARGIN_RATIO
        bot_edge = page_height * (1 - BOTTOM_MARGIN_RATIO)
        if y0 < top_edge or y1 > bot_edge:
            return True

    return False


def is_heading(text: str) -> bool:
    """
    Return True if the block looks like a section heading.
    """
    stripped = text.strip()
    if not stripped:
        return False
    if len(stripped) > HEADING_MAX_CHARS:
        return False

    lower = stripped.lower().rstrip(".: ")

    # ── A) Known section name (e.g. "Abstract", "3 Introduction")
    if lower in KNOWN_HEADINGS:
        return True

    # ── B) Starts with a citation like "3" or "3.2" or "3.2.1"
    if re.match(r"^[A-Z]?\d+(\.\d+)*[\.\s]", stripped):
        return True

    # ── C) Not a caption
    if any(lower.startswith(p) for p in NOT_HEADING_PREFIXES):
        return False

    return False


def split_into_chunks(text: str, section: str, page: int) -> list[dict]:
    """
    Split a long body of text into ~MAX_WORDS_PER_CHUNK-word chunks.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), MAX_WORDS_PER_CHUNK):
        chunk_text = " ".join(words[i : i + MAX_WORDS_PER_CHUNK])
        chunks.append({
            "type": "text",
            "page": page,
            "section": section,
            "content": chunk_text,
        })
    return chunks


# ═══════════════════════════════════════════════════════════════════
#  Main extraction pipeline
# ═══════════════════════════════════════════════════════════════════

def main():
    doc = fitz.open(PDF_PATH)

    # State carried across pages
    current_section = "unknown"            # most recently seen heading
    current_body_blocks: list[tuple[str, int]] = []  # (text, page_num)
    all_chunks: list[dict] = []

    for page_num, page in enumerate(doc, start=1):
        page_height = page.rect.height
        # Blocks: (x0, y0, x1, y1, text, block_no, block_type)
        #   block_type 0 = text, 1 = image (images usually absent here)
        raw_blocks = page.get_text("blocks")

        for (x0, y0, x1, y1, text, block_no, block_type) in raw_blocks:
            # ── Skip embedded images (if any appear as blocks) ──────
            # NOTE: In PyMuPDF 1.28, get_text("blocks") only returns
            # text blocks (type=0).  Images are accessed via
            # page.get_images() separately, so this check is rarely hit.
            if block_type == 1:
                continue

            text = text.strip()
            if not text:
                continue

            # ── 1) Check for headers / footers ────────────────────
            if is_skip_block(text, y0, y1, page_height):
                continue

            # ── 2) Detect section headings ────────────────────────
            if is_heading(text):
                # Flush any body text accumulated under the previous section
                if current_body_blocks:
                    all_text = " ".join(t for t, _ in current_body_blocks)
                    first_page = current_body_blocks[0][1]
                    all_chunks.extend(
                        split_into_chunks(all_text, current_section, first_page)
                    )
                    current_body_blocks.clear()
                # Replace newlines with spaces so the section name
                # is a clean single-line string.
                current_section = text.replace("\n", " ")
            else:
                # ── 3) Regular body text → accumulate ────────────
                current_body_blocks.append((text, page_num))

    # ── Flush remaining body blocks after the last page ────────────
    if current_body_blocks:
        all_text = " ".join(t for t, _ in current_body_blocks)
        first_page = current_body_blocks[0][1]
        all_chunks.extend(
            split_into_chunks(all_text, current_section, first_page)
        )

    doc.close()

    # ── Save to JSON ──────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    # ── Print summary ─────────────────────────────────────────────
    print(f"Total chunks: {len(all_chunks)}\n")
    for i, chunk in enumerate(all_chunks, start=1):
        preview = chunk["content"][:60].replace("\n", " ")
        print(
            f"  Chunk {i:>3} | "
            f"section='{chunk['section']}' | "
            f"page={chunk['page']} | "
            f"preview='{preview}...'"
        )


if __name__ == "__main__":
    main()
