"""Opt-in canonical WAM server with all 60 attention operators replaced.

No global source/config mutation. This is an operator-alignment entry, not a
new speculative quality/performance permission or candidate certificate.
"""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "投机采样/wam-p0-worktree"))
from wan_va import wan_va_server as source
from integration import install, restore

_original_server = source.run_async_server_mode

def run_aligned(server, rank, host, port):
    bindings = install(server.transformer)
    source.logger.info("Installed independent Torch-aligned v2 on %d attention modules", len(bindings))
    try:
        return _original_server(server, rank, host, port)
    finally:
        restore(bindings)

if __name__ == "__main__":
    if os.environ.get("LINGBOT_ATTN_MODE", "torch") != "torch":
        raise RuntimeError("Construct the canonical Torch model before explicit operator replacement")
    source.run_async_server_mode = run_aligned
    source.init_logger()
    source.main()
