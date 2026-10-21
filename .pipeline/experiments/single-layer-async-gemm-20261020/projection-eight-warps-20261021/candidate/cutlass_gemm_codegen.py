"""Embed pinned CUTLASS SM80 mainloop with its actual compute-warp barrier."""
from pathlib import Path


def generate(destination: Path) -> Path:
    root = Path(__file__).resolve().parents[3]
    headers = root / ".pipeline/cache/cutlass-v3.5.1/include"
    source = headers / "cutlass/gemm/threadblock/mma_multistage.h"
    text = source.read_text()
    # Keep the upstream license and arithmetic. Only the class identity and
    # CTA barrier scope change: 4-warp tiles use physical 96..223;
    # the 8-warp projection tile uses all 256 threads. Arithmetic is unchanged.
    if text.count("__syncthreads();") != 2:
        raise RuntimeError("Pinned CUTLASS mainloop synchronization changed")
    text = text.replace("MmaMultistage", "WamMmaMultistage")
    text = text.replace("__syncthreads();",
                        'asm volatile("bar.sync 2, %0;" :: "n"('
                        '32 * Base::WarpCount::kM * Base::WarpCount::kN * '
                        'Base::WarpCount::kK) : "memory");')
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "cutlass_gemm_mainloop.cuh").write_text(text)
    return headers
