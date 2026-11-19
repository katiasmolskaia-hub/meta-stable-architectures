"""Run the fixed single-world example from the extracted companion root."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parent /
                      'publication/recovery-note-2026-11-17/quick_demo.py'), run_name='__main__')
