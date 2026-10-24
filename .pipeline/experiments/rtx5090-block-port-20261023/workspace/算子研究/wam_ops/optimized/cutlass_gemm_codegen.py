"""Embed pinned CUTLASS SM80 mainloop in the four compute warps."""
from pathlib import Path


def generate(destination: Path) -> Path:
    root = Path(__file__).resolve().parents[3]
    headers = root / ".pipeline/cache/cutlass-v3.5.1/include"
    source = headers / "cutlass/gemm/threadblock/mma_multistage.h"
    text = source.read_text()
    # Keep the upstream license and arithmetic. Only the class identity and
    # CTA barrier scope change: logical threads 0..127 map to physical 96..223.
    if text.count("__syncthreads();") != 2:
        raise RuntimeError("Pinned CUTLASS mainloop synchronization changed")
    text = text.replace("MmaMultistage", "WamMmaMultistage")
    text = text.replace("__syncthreads();",
                        'asm volatile("bar.sync 2, 128;" ::: "memory");')
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "cutlass_gemm_mainloop.cuh").write_text(text)
    return headers
