"""Named entrypoint for Yuekai's main Streamlit dashboard.

Keeps `gui_demo.py` as the implementation module while giving this app
its own unambiguous filename in a multi-GUI repository.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo.gui_demo import main


if __name__ == "__main__":
    main()
