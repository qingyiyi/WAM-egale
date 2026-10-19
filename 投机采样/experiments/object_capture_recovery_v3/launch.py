"""Existing pinned launcher/guard lifecycle, separate migration worker entry."""
from pathlib import Path
import sys

OWN = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(OWN.parent))
from tf_suite_pipeline_v1 import launch

if __name__ == '__main__':
    launch.OWN = OWN
    raise SystemExit(launch.main())
