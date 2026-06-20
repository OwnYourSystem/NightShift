"""
NightShift Quantization POC — the mechanism, from scratch, in pure NumPy.

No torch, no downloads. This implements real INT8 quantization math so you can
see *exactly* what "compressing a model" does to the numbers, and measure the
two axes that are genuinely measurable on a CPU without special kernels:

    memory footprint  (real 4x reduction: 32-bit floats -> 8-bit ints)
    quality loss      (real output drift vs the full-precision model)

It also demonstrates the single most important quality knob in quantization:
per-tensor vs per-channel scaling.

Honest note on the third axis (speed): pure-NumPy INT8 matmul is NOT faster than
float32 on a CPU, because NumPy upcasts int8 to wider types and there are no
INT8 SIMD kernels in play. The real speed win is a *hardware* property — it shows
up with INT8 tensor cores, or with GGUF/llama.cpp kernels on the target device.
So here we measure memory + quality precisely and are upfront that speed is
device-dependent. (quantize_real.py measures real CPU speedup via PyTorch.)

Run:  python quantize_numpy.py
Output: results_numpy.json + a printed comparison
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RNG = np.random.default_rng(42)


def quantize_int8(W: np.ndarray, per_channel: bool) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric INT8 quantization.

    Returns (int8 weights, scale). Dequantize with  Wq.astype(float32) * scale.
    per_channel=True computes one scale per output row (axis 0), which tracks
    each neuron's own dynamic range and loses far less information.
    """
    if per_channel:
        max_abs = np.abs(W).max(axis=1, keepdims=True)          # one per row
    else:
        max_abs = np.abs(W).max()                                # one for all
    scale = np.where(max_abs == 0, 1.0, max_abs / 127.0)
    Wq = np.clip(np.round(W / scale), -127, 127).astype(np.int8)
    return Wq, scale.astype(np.float32)


def dequantize(Wq: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return Wq.astype(np.float32) * scale


def build_toy_model(n_layers: int = 6, dim: int = 256) -> list[np.ndarray]:
    """A small stack of linear layers with realistic (Gaussian) weights.

    A handful of weights are deliberately large outliers — outliers are exactly
    what makes per-tensor quantization hurt, so this mirrors a real failure mode.
    """
    layers = []
    for _ in range(n_layers):
        W = RNG.standard_normal((dim, dim)).astype(np.float32) * 0.06
        # inject a few fat-tailed outliers, as real trained weights have
        idx = RNG.integers(0, dim, size=(dim // 32, 2))
        W[idx[:, 0], idx[:, 1]] *= 12.0
        layers.append(W)
    return layers


def forward(x: np.ndarray, layers: list[np.ndarray]) -> np.ndarray:
    """Tiny MLP forward pass with a GELU-ish nonlinearity between layers."""
    h = x
    for W in layers:
        h = h @ W.T
        h = h * (h > 0) + 0.01 * h * (h <= 0)   # leaky relu — deterministic
    return h


def bytes_of(layers: list[np.ndarray], dtype_bytes: int, scale_overhead=0) -> int:
    return sum(W.size * dtype_bytes for W in layers) + scale_overhead


def output_drift(ref: np.ndarray, approx: np.ndarray) -> dict:
    """How far the quantized model's output moved from full precision."""
    rel = np.linalg.norm(approx - ref) / np.linalg.norm(ref)
    cos = float(
        (ref.ravel() @ approx.ravel())
        / (np.linalg.norm(ref) * np.linalg.norm(approx))
    )
    return {"relative_error_pct": round(float(rel) * 100, 3),
            "cosine_similarity": round(cos, 6)}


def main() -> int:
    layers = build_toy_model()
    x = RNG.standard_normal((32, layers[0].shape[1])).astype(np.float32)

    ref = forward(x, layers)                                   # FP32 ground truth

    # --- per-tensor quantization (one scale per matrix) ---
    pt_layers, pt_scales = [], 0
    for W in layers:
        Wq, s = quantize_int8(W, per_channel=False)
        pt_layers.append(dequantize(Wq, s))
        pt_scales += s.size * 4
    pt_out = forward(x, pt_layers)

    # --- per-channel quantization (one scale per output row) ---
    pc_layers, pc_scales = [], 0
    for W in layers:
        Wq, s = quantize_int8(W, per_channel=True)
        pc_layers.append(dequantize(Wq, s))
        pc_scales += s.size * 4
    pc_out = forward(x, pc_layers)

    fp32_bytes = bytes_of(layers, 4)
    int8_bytes = bytes_of(layers, 1)   # weights only; scales are tiny (below)

    results = {
        "toy_model": {
            "layers": len(layers),
            "dim": layers[0].shape[0],
            "total_weights": int(sum(W.size for W in layers)),
        },
        "memory": {
            "fp32_kb": round(fp32_bytes / 1024, 1),
            "int8_kb": round(int8_bytes / 1024, 1),
            "reduction_x": round(fp32_bytes / int8_bytes, 2),
            "per_channel_scale_overhead_kb": round(pc_scales / 1024, 2),
        },
        "quality": {
            "per_tensor": output_drift(ref, pt_out),
            "per_channel": output_drift(ref, pc_out),
        },
        "note_on_speed": (
            "Pure-NumPy INT8 matmul is not faster on CPU; the speed win is a "
            "hardware property (INT8 kernels / GGUF on the target device). "
            "See quantize_real.py for measured CPU speedup via PyTorch."
        ),
    }
    (HERE / "results_numpy.json").write_text(json.dumps(results, indent=2))

    m, q = results["memory"], results["quality"]
    print("\n" + "=" * 60)
    print("  NightShift Quantization POC — from scratch (NumPy)")
    print("=" * 60)
    print(f"  toy model: {results['toy_model']['layers']} layers x "
          f"{results['toy_model']['dim']}d = "
          f"{results['toy_model']['total_weights']:,} weights")
    print("  " + "-" * 56)
    print(f"  memory   FP32 {m['fp32_kb']} KB  ->  INT8 {m['int8_kb']} KB"
          f"   ({m['reduction_x']}x smaller)")
    print("  " + "-" * 56)
    print("  quality (output drift vs full precision — lower is better):")
    print(f"    per-tensor    rel.error {q['per_tensor']['relative_error_pct']:>6}%"
          f"   cos {q['per_tensor']['cosine_similarity']}")
    print(f"    per-channel   rel.error {q['per_channel']['relative_error_pct']:>6}%"
          f"   cos {q['per_channel']['cosine_similarity']}")
    print("=" * 60)
    print("  Takeaway: same 4x memory cut, but per-channel scaling keeps")
    print("  far more quality. That choice IS the product.")
    print("  Wrote results_numpy.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
