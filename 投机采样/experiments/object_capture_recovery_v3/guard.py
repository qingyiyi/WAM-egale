"""Delegate unchanged birth/UID/PGID guard and original absolute deadlines."""
from pathlib import Path
import runpy
import sys

sys.dont_write_bytecode = True
runpy.run_path(str(Path(__file__).resolve().parent.parent / 'tf_suite_pipeline_v1/guard.py'), run_name='__main__')
