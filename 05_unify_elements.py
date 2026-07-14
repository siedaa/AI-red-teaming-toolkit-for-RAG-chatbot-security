"""
05_unify_elements.py — Merge text chunks, tables, and figures into a single
unified list with a consistent schema.

Usage:
    python 05_unify_elements.py

What it does:
    1. Loads extracted/text_chunks.json, extracted/tables.json, and
       extracted/figures.json.
    2. Converts every element into the common dict shape defined below.
    3. For each element, constructs an "embedding_text" (the text that will
       be embedded and used for similarity search) and a "display_content"
       (the human-readable content shown when the element is retrieved).
    4. Validates that every element has a non-empty embedding_text.
    5. Saves the unified list to extracted/unified_elements.json.
"""

import json
import os
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════

EXTRACTED_DIR = Path("extracted")

TEXT_CHUNKS_PATH   = EXTRACTED_DIR / "text_chunks.json"
TABLES_PATH        = EXTRACTED_DIR / "tables.json"
FIGURES_PATH       = EXTRACTED_DIR / "figures.json"
OUTPUT_PATH        = EXTRACTED_DIR / "unified_elements.json"


# ═══════════════════════════════════════════════════════════════════
#  Loading helpers
# ═══════════════════════════════════════════════════════════════════

def load_json(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════
#  Conversion functions — one per source type
# ═══════════════════════════════════════════════════════════════════

def convert_text_chunk(entry: dict, index: int) -> dict:
    """
    Convert a single text-chunk entry to the unified schema.

    Input schema (from 01_extract_text.py):
        { "type": "text", "page": int, "section": str, "content": str }

    Unified schema:
        id              → "text_{index}"
        type            → "text"
        page            → page (passthrough)
        title           → section name
        embedding_text  → the full chunk content (it's already plain text)
        display_content → the full chunk content
        image_path      → None
    """
    return {
        "id": f"text_{index}",
        "type": "text",
        "page": entry["page"],
        "title": entry.get("section") or "(no section)",
        "embedding_text": entry["content"],
        "display_content": entry["content"],
        "image_path": None,
    }


def convert_table(entry: dict, index: int) -> dict:
    """
    Convert a single table entry to the unified schema.

    Input schema (from 02_extract_tables.py):
        { "type": "table", "page": int, "caption": str, "content": str }

    Unified schema:
        id              → "table_{index}"
        type            → "table"
        page            → page (passthrough)
        title           → caption
        embedding_text  → caption + "\n" + table content
                          (both matter for matching: the caption says what
                           the table is about, the content has the numbers)
        display_content → the table content only (the raw data the user
                          wants to see)
        image_path      → None
    """
    caption = entry.get("caption") or ""
    content = entry.get("content") or ""
    return {
        "id": f"table_{index}",
        "type": "table",
        "page": entry["page"],
        "title": caption,
        "embedding_text": f"{caption}\n{content}".strip(),
        "display_content": content,
        "image_path": None,
    }


def convert_figure(entry: dict, index: int) -> dict:
    """
    Convert a single figure entry to the unified schema.

    Input schema (from 03_extract_figures.py, after 04_caption_figures.py):
        { "type": "figure", "page": int, "image_path": str,
          "caption": str, "vision_description": str, ... }

    Unified schema:
        id              → "figure_{index}"
        type            → "figure"
        page            → page (passthrough)
        title           → caption
        embedding_text  → caption + "\n" + vision_description
                          (the vision description is the text that represents
                           the image's content for vector search — we cannot
                           embed pixels directly)
        display_content → vision_description (the rich interpretation,
                          more useful to a reader than the short caption)
        image_path      → image_path (so downstream code can load the image)
    """
    caption = entry.get("caption") or ""
    vision_desc = entry.get("vision_description") or ""
    return {
        "id": f"figure_{index}",
        "type": "figure",
        "page": entry["page"],
        "title": caption,
        "embedding_text": f"{caption}\n{vision_desc}".strip(),
        "display_content": vision_desc,
        "image_path": entry.get("image_path"),
    }


# ═══════════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════════

def main():
    # 1. Load all three source files.
    text_chunks: list[dict] = load_json(TEXT_CHUNKS_PATH)
    tables:      list[dict] = load_json(TABLES_PATH)
    figures:     list[dict] = load_json(FIGURES_PATH)

    print(f"Loaded: {len(text_chunks)} text chunks, {len(tables)} tables, {len(figures)} figures")

    # 2. Convert every element into the unified schema.
    unified: list[dict] = []

    # Text chunks — local index to keep ids sequential per type
    for i, chunk in enumerate(text_chunks):
        unified.append(convert_text_chunk(chunk, i))

    # Tables
    for i, table in enumerate(tables):
        unified.append(convert_table(table, i))

    # Figures
    for i, figure in enumerate(figures):
        unified.append(convert_figure(figure, i))

    # 3. Save to unified JSON.
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(unified, f, indent=2, ensure_ascii=False)

    # 4. Print summary.
    n_text   = sum(1 for e in unified if e["type"] == "text")
    n_table  = sum(1 for e in unified if e["type"] == "table")
    n_figure = sum(1 for e in unified if e["type"] == "figure")

    print(f"\nUnified elements: {len(unified)} total")
    print(f"  {n_text} text chunks")
    print(f"  {n_table} tables")
    print(f"  {n_figure} figures")

    # Validate: every element must have a non-empty embedding_text.
    empty_ids = [
        e["id"] for e in unified
        if not e["embedding_text"] or not e["embedding_text"].strip()
    ]
    if empty_ids:
        print(f"\nWARNING: {len(empty_ids)} elements have empty embedding_text:")
        for eid in empty_ids:
            print(f"  - {eid}")
    else:
        print("\nValidation passed: all elements have non-empty embedding_text.")

    # Validate: image_path exists for figures, is None for others.
    fig_no_img = [
        e["id"] for e in unified
        if e["type"] == "figure" and not e["image_path"]
    ]
    non_fig_with_img = [
        e["id"] for e in unified
        if e["type"] != "figure" and e["image_path"] is not None
    ]
    if fig_no_img:
        print(f"WARNING: {len(fig_no_img)} figures missing image_path: {fig_no_img}")
    if non_fig_with_img:
        print(f"WARNING: {len(non_fig_with_img)} non-figures have image_path: {non_fig_with_img}")

    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
