import runpy
from pathlib import Path

# Streamlit names the main nav entry after this file, so the entry point is
# renamed here rather than by touching app.py.
runpy.run_path(str(Path(__file__).parent / "app.py"), run_name="__main__")
