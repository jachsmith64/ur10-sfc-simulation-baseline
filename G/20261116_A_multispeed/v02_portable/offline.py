"""Launch the inherited offline guard with an explicitly chosen local scientific runtime."""
import os
from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
science = os.environ.get("FILTER_SFC_SCIENCE_DIR")
if science:
    sys.path.insert(0, str(Path(science).resolve(strict=True)))
sys.argv[0] = str(root / "guarded_run.py")
runpy.run_path(sys.argv[0], run_name="__main__")
