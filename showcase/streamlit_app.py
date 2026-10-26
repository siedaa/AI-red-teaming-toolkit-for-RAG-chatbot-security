"""
showcase/streamlit_app.py — entry point for a hosted, read-only copy of app.py.

Streamlit Community Cloud: set "Main file path" to showcase/streamlit_app.py.
It installs showcase/requirements.txt (streamlit + pandas only) and runs app.py
with SHOWCASE_MODE=true: the Red Teaming tab shows the saved results, while the
chat and the live attack demo are off, so no API key or vector store is needed.
"""

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

os.environ["SHOWCASE_MODE"] = "true"
os.chdir(ROOT)  # app.py reads red_team_toolkit/results/ relative to the repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
