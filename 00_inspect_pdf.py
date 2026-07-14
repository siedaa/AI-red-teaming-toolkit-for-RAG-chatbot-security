"""
00_inspect_pdf.py - Inspect the "Attention Is All You Need" PDF

Usage:
    python 00_inspect_pdf.py

Learnings:
    - PyMuPDF (fitz) is a fast PDF library that gives access to text, images,
      metadata, and more on a per-page basis.
    - doc.page_count → total number of pages.
    - page.get_text() → extracts all text from a page as a plain string.
    - page.get_images() → returns a list of image references embedded on a page.
"""

import fitz  # PyMuPDF – the leading Python PDF library

# 1. Open the PDF file
pdf_path = "attention is all you need.pdf"
doc = fitz.open(pdf_path)

# 2. Print the total number of pages
print(f"Total pages: {doc.page_count}\n")

# 3. Loop through every page of the document
for i, page in enumerate(doc):
    # --- Text preview -------------------------------------------------
    text = page.get_text()                     # extract all text on this page
    # First 100 characters, whitespace collapsed, for a compact preview
    preview = text.strip()[:100].replace("\n", " ")

    # --- Embedded images ----------------------------------------------
    # get_images(full=True) returns a list of tuples,
    # one per image reference on the page.
    images = page.get_images(full=True)

    print(
        f"Page {i+1:>2}: "
        f"preview='{preview}...' | "
        f"images={len(images)}"
    )

# 4. Clean up – close the document
doc.close()
