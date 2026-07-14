"""
02_extract_tables.py — Extract tables from the PDF using pdfplumber.

Usage:
    python 02_extract_tables.py

What it does:
    1. Opens the PDF with pdfplumber.
    2. For each page, uses page.extract_words(x_tolerance=1) to get
       individual words with proper spacing (the default tolerance of 3
       merges characters that the PDF renders without explicit spaces).
    3. Groups the words into lines based on y-position, sorts each line
       by x-position, and joins words with spaces to produce clean text.
    4. Searches for "Table N:" captions and collects lines from the
       caption through the table data, stopping when body text begins.
    5. Saves the result to extracted/tables.json.

Learnings:
    - pdfplumber's default extract_tables() relies on visible lines/borders.
      The tables in this paper are borderless, so default detection fails.
    - page.extract_words(x_tolerance=N) controls how close characters must
      be to be merged into a word.  The default is 3, but justified PDF
      text can have gaps < 3 between words, so lowering to 1 separates
      them properly.
    - To separate table data from body text, we look for section headings
      or prose patterns (period + space + capital letter).
    - Pages 13–15 contain attention-visualization figures — we filter by
      checking for "<pad>" / "<EOS>" tokens inside the content.
"""

import json
import os
import re
from collections import defaultdict

import pdfplumber


# ═══════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════

PDF_PATH = "attention is all you need.pdf"
OUTPUT_DIR = "extracted"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "tables.json")

# Regex for section headings like "3.5 Positional Encoding" or "6 Results".
# Requires a LETTER after the number + space (not another digit), so table
# data like "1 512 512 ..." does NOT match.
SECTION_HEADING_RE = re.compile(r"^\s*\d+(\.\d+)*\.?\s+[A-Za-z]")

# Words/phrases that signal we have left the table and entered body text.
TABLE_END_PATTERNS = [
    "residual", "label smoothing", "label", "acknowledgements",
    "references", "the code", "we are grateful", "we excited",
]

# x_tolerance to use for word extraction.  The default (3) merges words
# in justified text; lowering to 1 keeps them separate so we can insert
# proper spaces when joining.
WORD_X_TOLERANCE = 1


# ═══════════════════════════════════════════════════════════════════
#  Helper functions
# ═══════════════════════════════════════════════════════════════════

def is_table_end_line(line: str) -> bool:
    """
    Return True if *line* looks like body text rather than table data.

    Heuristics used:
        - Starts with a section-number pattern (e.g. "3.5 ", "6 Results")
        - Starts with a known end-of-section keyword
        - Contains a period followed by space + capital letter (sentence)
    """
    stripped = line.strip()
    if not stripped:
        return False

    lower = stripped.lower()

    # 1) Section heading like "3.5 PositionalEncoding", "6 Results"
    if SECTION_HEADING_RE.match(stripped):
        return True

    # 2) Known body-text starters
    for pat in TABLE_END_PATTERNS:
        if lower.startswith(pat):
            return True

    # 3) Prose sentence pattern: period → space → capital letter
    #    (table data like "4.92" does NOT have a space after the period)
    if re.search(r"\.\s[A-Z]", stripped):
        return True

    return False


def get_clean_lines(page) -> list[str]:
    """
    Extract page text as properly-spaced lines using word-level extraction.

    The PDF uses justified text where spaces between words are narrow.
    pdfplumber's default word-extraction tolerance (x_tolerance=3) merges
    adjacent characters into a single word, losing word boundaries.  By
    lowering x_tolerance to WORD_X_TOLERANCE (1), we obtain individual
    words and can reconstruct lines with correct spaces.

    Returns: list of strings, one per visual line on the page.
    """
    words = page.extract_words(x_tolerance=WORD_X_TOLERANCE)
    if not words:
        return []

    # Group words by rounded y-position (bin size = 4 pt).
    # This collapses words on the same baseline (e.g. "O(n²" and "· d)"
    # from a table cell spanning two rendered lines) into one line.
    y_groups = defaultdict(list)
    for w in words:
        y_key = round(w["top"] / 4) * 4
        y_groups[y_key].append(w)

    # Sort each group by x (left-to-right), join words with space.
    lines = []
    for y_key in sorted(y_groups):
        group = sorted(y_groups[y_key], key=lambda w: w["x0"])
        text = " ".join(w["text"] for w in group)
        if text.strip():
            lines.append(text)

    return lines


def extract_page_tables(page, page_num: int) -> list[dict]:
    """
    Try to extract tables from a single page.

    Returns a list of dicts, one per table found on the page.
    """
    lines = get_clean_lines(page)

    # ── Find all "Table N:" caption lines ───────────────────────────
    caption_indices = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"^Table\s*\d+\s*:", stripped, re.IGNORECASE):
            caption_indices.append(i)

    if not caption_indices:
        return []

    tables = []

    MAX_CAPTION_CHARS = 400

    for cap_idx in caption_indices:
        # ── Extract the caption sentence(s) ─────────────────────────
        caption_parts = []
        for j in range(cap_idx, min(cap_idx + 5, len(lines))):
            s = lines[j].strip()
            if not s:
                break

            # Detect table-header rows (e.g. "Layer Type Complexity per Layer")
            # so we stop BEFORE they pollute the caption string.
            #
            # Heuristic: ≥ 3 words, ≥ 50 % of alphabetic words start with
            # an uppercase letter, and the line does NOT end with a period
            # (genuine caption sentences do).
            tokens = [t for t in s.split() if t]
            is_header = False
            if len(tokens) >= 3 and not s.rstrip().endswith("."):
                n_upper = sum(1 for t in tokens if t[0].isalpha() and t[0].isupper())
                n_alpha = sum(1 for t in tokens if t[0].isalpha())
                if n_alpha > 0 and (n_upper / n_alpha) >= 0.50:
                    is_header = True

            if is_header and len(caption_parts) >= 1:
                break

            caption_parts.append(s)

            # Hard character cap so that overly long captions don't
            # bleed into the table data.
            if len(" ".join(caption_parts)) > MAX_CAPTION_CHARS:
                break

        caption = " ".join(caption_parts)

        # ── Collect table data lines ────────────────────────────────
        data_lines: list[str] = []
        for j in range(cap_idx, len(lines)):
            line = lines[j]
            if is_table_end_line(line) and len(data_lines) > max(2, len(caption_parts)):
                break
            data_lines.append(line)

        # Remove trailing blank lines
        while data_lines and not data_lines[-1].strip():
            data_lines.pop()

        if len(data_lines) <= max(1, len(caption_parts)):
            continue

        content = "\n".join(data_lines).strip()

        # ── Filter false positives ──────────────────────────────────
        if not re.match(r"^Table\s*\d+\s*:", content, re.IGNORECASE):
            continue
        if "<pad>" in content or "<EOS>" in content:
            continue
        num_cols_estimate = max(len(row) for row in content.split("\n")[:5]) if content else 0
        if num_cols_estimate < 40:
            continue
        if len(content.split("\n")) < 5:
            continue

        tables.append({
            "type": "table",
            "page": page_num,
            "caption": caption,
            "content": content,
        })

    return tables


# ═══════════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════════

def main():
    pdf = pdfplumber.open(PDF_PATH)
    all_tables: list[dict] = []

    for page_num, page in enumerate(pdf.pages, start=1):
        tables = extract_page_tables(page, page_num)
        all_tables.extend(tables)

    pdf.close()

    # ── Save to JSON ──────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_tables, f, indent=2, ensure_ascii=False)

    # ── Print summary ─────────────────────────────────────────────
    print(f"Total tables found: {len(all_tables)}\n")
    for t in all_tables:
        # Preview: first 3 lines of content
        preview_lines = t["content"].split("\n")[:3]
        preview = " | ".join(l.strip() for l in preview_lines)
        preview = preview[:120]
        print(f"  Page {t['page']:>2} | caption='{t['caption']}'")
        print(f"          preview='{preview}...'")
        print()


if __name__ == "__main__":
    main()
