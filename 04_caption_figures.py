"""
04_caption_figures.py — Use Gemini's vision model to generate rich text
descriptions for each extracted figure image.

Usage:
    python 04_caption_figures.py

What it does:
    1. Loads GEMINI_API_KEY from .env and creates a genai.Client.
    2. Reads extracted/figures.json (produced by 03_extract_figures.py).
    3. For each figure:
       a. Opens the image with PIL.
       b. Sends the image + its detected caption to Gemini 2.5 Flash.
       c. Stores the returned description as "vision_description".
       d. Prints a review log.
    4. Saves the enriched list back to extracted/figures.json.
    5. Inserts a small delay between API calls to avoid rate limits.

Learnings:
    - google-genai's client accepts a PIL.Image object directly as part
      of the contents list for multimodal requests.
    - Gemini 2.5 Flash (gemini-2.5-flash) is well-suited for vision tasks
      that require detailed reasoning about diagrams and plots.
    - The free tier has ~10 requests/minute — a 2-second delay between
      calls keeps us safely within that limit.
"""

import json
import os
import time
from pathlib import Path

import PIL.Image
from dotenv import load_dotenv
from google import genai

# ═══════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════

FIGURES_JSON = Path("extracted") / "figures.json"
MODEL_NAME = "gemini-2.5-flash"
API_DELAY_SECONDS = 2  # delay between API calls to stay under rate limits

# ═══════════════════════════════════════════════════════════════════
#  Prompt engineering
# ═══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are a helpful research assistant describing scientific figures from a machine-learning paper ("Attention Is All You Need").

I will give you a figure image and its caption. Write a detailed but concise description (4-6 sentences) of what the figure shows.

- If it is an architecture diagram: describe the components, their arrangement, and how data flows through them.
- If it is a chart / table / plot: describe what is plotted on each axis, what trends or patterns are visible, and any notable data points.
- If it is an attention-visualisation heatmap: describe what the coloured lines or connections represent, which layers or heads are shown, and what linguistic phenomenon is being illustrated.

Keep the description self-contained so someone who has not seen the paper can understand the key takeaway from your description alone."""


# ═══════════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════════

def main():
    # 1. Load the API key from .env and create the Gemini client.
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in .env file.")
        return

    client = genai.Client(api_key=api_key)

    # 2. Load the figure metadata produced by 03_extract_figures.py.
    with open(FIGURES_JSON, "r", encoding="utf-8") as f:
        figures: list[dict] = json.load(f)

    print(f"Loaded {len(figures)} figures from {FIGURES_JSON}\n")

    # 3. Process each figure one by one.
    for i, fig in enumerate(figures, start=1):
        img_path = fig["image_path"]
        caption = fig["caption"]

        print(f"[{i}/{len(figures)}]  Page {fig['page']} — {os.path.basename(img_path)}")
        print(f"      Caption: {caption[:100]}...")

        # 3a. Open the image with PIL.
        image = PIL.Image.open(img_path)

        # 3b. Send the image + caption to Gemini and ask for a description.
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                SYSTEM_PROMPT,
                f"Figure caption: {caption}",
                image,
            ],
        )

        description = response.text.strip()
        # 3c. Store the description on the figure dict.
        fig["vision_description"] = description

        # 3d. Print for immediate review.
        print(f"      Description: {description[:200]}...")
        print()

        # 3e. Small delay between calls so we don't hit the free-tier rate
        #     limit (~10 requests / minute → one every 6 s, but 2 s + network
        #     latency is usually fine for 6 figures).
        if i < len(figures):
            time.sleep(API_DELAY_SECONDS)

    # 4. Save the enriched list back to extracted/figures.json.
    with open(FIGURES_JSON, "w", encoding="utf-8") as f:
        json.dump(figures, f, indent=2, ensure_ascii=False)

    print(f"Done. Enriched {len(figures)} figures saved to {FIGURES_JSON}")


if __name__ == "__main__":
    main()
