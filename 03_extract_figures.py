"""
03_extract_figures.py — Extract figures from the PDF as standalone images.

Usage:
    python 03_extract_figures.py

What it does:
    1. Opens the PDF with PyMuPDF (fitz).
    2. For each page, checks whether it contains embedded raster images.

       CASE A — Pages with embedded images (e.g. Figures 1 and 2 on pages 3-4):
           - Extracts the embedded image data directly via doc.extract_image(xref).
           - Saves as page{N}_fig{i}.png.
           - This avoids re-encoding losses and gives the original image quality.

       CASE B — Pages where the figure is vector-drawn (no embedded image).
           These are typically figures drawn with PDF path/stroke operations
           (e.g. Figures 3, 4, 5 on pages 13-15, attention heatmap visualisations).
           There is no image object to extract, so we render the entire page
           to a raster image with page.get_pixmap().

    3. For every figure page, searches for a "Figure N:" caption in the
       page's text blocks.
    4. Saves metadata to extracted/figures.json.
    5. Prints a summary table.

Learnings:
    - PyMuPDF's page.get_images() returns a list of (xref, ...) tuples for
      each embedded raster image on the page.  The xref is the cross-reference
      number used to retrieve the binary image data.
    - doc.extract_image(xref) returns a dict with "image" (bytes), "ext"
      (file extension like "png"), "width", "height".
    - For vector-drawn figures (no raster image), page.get_images() returns
      an empty list.  We fall back to page.get_pixmap() which rasterizes the
      entire page at a given resolution.
    - fitz.Matrix(2, 2) renders at 2x the default 72 DPI → 144 DPI output.
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
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")

# Render resolution multiplier for CASE B (full-page renders).
# Matrix(2, 2) → 144 DPI (2 × the default 72 DPI).
RENDER_ZOOM = 2


# ═══════════════════════════════════════════════════════════════════
#  Caption detection
# ═══════════════════════════════════════════════════════════════════

def find_figure_caption(page) -> str | None:
    """
    Search *page* for a text block whose text starts with "Figure N:".

    Returns the full text of that block, or None if no caption is found.
    Uses get_text("text", clip=bbox) to extract block text with correct
    spacing, rather than manually joining spans which introduces artifacts.
    """
    blocks = page.get_text("dict")["blocks"]
    for block in blocks:
        if block["type"] != 0:          # 0 = text block, 1 = image block
            continue

        # Clean, properly-spaced text within this block's bounding box.
        bbox = block["bbox"]
        text = page.get_text("text", clip=bbox).strip()

        if re.match(r"^Figure\s*\d+\s*:", text, re.IGNORECASE):
            return text

    return None


# ═══════════════════════════════════════════════════════════════════
#  CASE A — Extract embedded raster images
# ═══════════════════════════════════════════════════════════════════

def extract_embedded_images(page, page_num: int, images_dir: str) -> list[dict]:
    """
    Extract all embedded raster images from *page*.

    Uses page.get_images() to find image xrefs, then doc.extract_image(xref)
    to get the binary data.  Saves each as page{page_num}_fig{i}.png.

    Returns a list of result dicts, one per extracted image.
    """
    doc = page.parent  # the fitz.Document that owns this page
    images_on_page = page.get_images()

    results: list[dict] = []

    for i, img_info in enumerate(images_on_page, start=1):
        xref = img_info[0]

        # extract_image returns a dict with keys:
        #   "image" → raw bytes of the image
        #   "ext"   → file extension (e.g. "png", "jpeg")
        #   "width", "height" → dimensions in pixels
        img_data = doc.extract_image(xref)
        img_bytes: bytes = img_data["image"]
        ext: str = img_data["ext"]

        # Save as PNG regardless of the original extension for consistency.
        filename = f"page{page_num}_fig{i}.png"
        filepath = os.path.join(images_dir, filename)

        with open(filepath, "wb") as f:
            f.write(img_bytes)

        results.append({
            "type": "figure",
            "page": page_num,
            "image_path": filepath,
            "extraction_method": "embedded",
            "original_ext": ext,
            "width_px": img_data["width"],
            "height_px": img_data["height"],
        })

    return results


# ═══════════════════════════════════════════════════════════════════
#  CASE B — Render full page as image (vector-drawn figures)
# ═══════════════════════════════════════════════════════════════════

def render_full_page(page, page_num: int, images_dir: str, zoom: int = RENDER_ZOOM) -> dict:
    """
    Render *page* to a raster image at *zoom* × 72 DPI.

    This is used when the figure is drawn with PDF vector operations
    rather than embedded as a raster image.

    Returns a single result dict.
    """
    matrix = fitz.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=matrix)

    filename = f"page{page_num}_full.png"
    filepath = os.path.join(images_dir, filename)

    pixmap.save(filepath)

    return {
        "type": "figure",
        "page": page_num,
        "image_path": filepath,
        "extraction_method": "full_page_render",
        "width_px": pixmap.width,
        "height_px": pixmap.height,
    }


# ═══════════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════════

def main():
    doc = fitz.open(PDF_PATH)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    all_figures: list[dict] = []

    for page_num in range(1, doc.page_count + 1):
        page = doc[page_num - 1]

        embedded = page.get_images()
        has_embedded_images = len(embedded) > 0

        # Try to find a "Figure N:" caption on this page.
        caption = find_figure_caption(page)

        if has_embedded_images:
            # ── CASE A — extract embedded raster images ─────────────
            if caption is None:
                caption = f"Figure (page {page_num})"
            results = extract_embedded_images(page, page_num, IMAGES_DIR)
            for r in results:
                r["caption"] = caption
            all_figures.extend(results)
        elif caption is not None:
            # ── CASE B — vector-drawn figure, render full page ─────
            result = render_full_page(page, page_num, IMAGES_DIR)
            result["caption"] = caption
            all_figures.append(result)
        # else: no embedded images, no "Figure N:" caption → skip

    doc.close()

    # ── Save metadata to JSON ─────────────────────────────────────
    metadata_path = os.path.join(OUTPUT_DIR, "figures.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_figures, f, indent=2, ensure_ascii=False)

    # ── Print summary ─────────────────────────────────────────────
    print(f"Total figures extracted: {len(all_figures)}\n")

    # Column widths for alignment
    print(f"{'Page':>5} | {'Method':<18} | {'File':<30} | {'Caption'}")
    print("-" * 120)
    for fig in all_figures:
        fname = os.path.basename(fig["image_path"])
        caption_preview = fig["caption"][:80]
        print(f"  {fig['page']:>2}   | {fig['extraction_method']:<18} | {fname:<30} | {caption_preview}")


if __name__ == "__main__":
    main()
