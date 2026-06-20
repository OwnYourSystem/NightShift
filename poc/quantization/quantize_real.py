"""
NightShift Quantization POC — real model, real numbers.

Loads a small pretrained causal language model, measures its FP32 baseline,
applies INT8 dynamic quantization, and measures the same model again. The point
is to feel the real three-way tradeoff firsthand:

    size on disk   vs   inference speed   vs   quality (perplexity)

We use a GPT-NeoX model (Pythia) on purpose: its weight matrices are
``torch.nn.Linear`` layers, so ``quantize_dynamic`` actually quantizes the
compute-heavy matmuls. (GPT-2 uses a custom ``Conv1D`` that dynamic quant skips,
which would silently produce a no-op "quantized" model — a real gotcha.)

Run:  python quantize_real.py
Output: results_real.json  +  prints a comparison table
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent

# Smallest first. Both are real, downloadable, and use nn.Linear.
CANDIDATE_MODELS = ["EleutherAI/pythia-70m", "EleutherAI/pythia-160m"]

# A fixed, neutral passage so perplexity is comparable run-to-run.
EVAL_TEXT = (
    "The lighthouse had stood at the edge of the harbor for nearly two hundred "
    "years. Every evening the keeper climbed the spiral stairs, lit the great "
    "lamp, and watched the fishing boats find their way home through the fog. "
    "Sailors said the light was steady and kind, never wavering even in the "
    "worst of the winter storms that rolled in off the cold gray sea."
)


def file_size_mb(model: nn.Module) -> float:
    """Serialize the state dict and measure its size on disk in megabytes."""
    tmp = HERE / "_tmp_state.pt"
    torch.save(model.state_dict(), tmp)
    mb = tmp.stat().st_size / 1e6
    tmp.unlink(missing_ok=True)
    return round(mb, 2)


def perplexity(model: nn.Module, tok, text: str) -> float:
    """Lower is better. exp(cross-entropy) of the model on a fixed passage."""
    enc = tok(text, return_tensors="pt")
    input_ids = enc.input_ids
    with torch.no_grad():
        out = model(input_ids, labels=input_ids)
    return round(float(torch.exp(out.loss)), 3)


def tokens_per_sec(model: nn.Module, tok, text: str, n_runs: int = 8) -> float:
    """Higher is better. Forward-pass throughput on CPU."""
    enc = tok(text, return_tensors="pt")
    input_ids = enc.input_ids
    n_tokens = input_ids.shape[1]
    with torch.no_grad():               # warmup
        model(input_ids)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            model(input_ids)
    dt = time.perf_counter() - t0
    return round((n_tokens * n_runs) / dt, 1)


def measure(model, tok) -> dict:
    return {
        "size_mb": file_size_mb(model),
        "perplexity": perplexity(model, tok, EVAL_TEXT),
        "tokens_per_sec": tokens_per_sec(model, tok, EVAL_TEXT),
    }


def main() -> int:
    torch.set_num_threads(os.cpu_count() or 4)

    chosen, tok, model = None, None, None
    last_err = None
    for name in CANDIDATE_MODELS:
        try:
            print(f"Loading {name} ...", flush=True)
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
            model.eval()
            chosen = name
            break
        except Exception as e:  # noqa: BLE001
            print(f"  failed: {e}", flush=True)
            last_err = e
    if model is None:
        print(f"MODEL_LOAD_FAILED: {last_err}", file=sys.stderr)
        return 2

    print("Measuring FP32 baseline ...", flush=True)
    fp32 = measure(model, tok)

    print("Applying INT8 dynamic quantization (nn.Linear -> qint8) ...", flush=True)
    qmodel = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

    print("Measuring INT8 model ...", flush=True)
    int8 = measure(qmodel, tok)

    def pct(new, old):
        return round((new - old) / old * 100, 1)

    results = {
        "model": chosen,
        "method": "PyTorch dynamic quantization, nn.Linear -> qint8 (CPU)",
        "eval_text_chars": len(EVAL_TEXT),
        "cpu_threads": torch.get_num_threads(),
        "fp32": fp32,
        "int8": int8,
        "delta": {
            "size_pct": pct(int8["size_mb"], fp32["size_mb"]),
            "speedup_x": round(int8["tokens_per_sec"] / fp32["tokens_per_sec"], 2),
            "perplexity_pct": pct(int8["perplexity"], fp32["perplexity"]),
        },
    }

    (HERE / "results_real.json").write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 58)
    print(f"  NightShift Quantization POC — {chosen}")
    print("=" * 58)
    print(f"  {'metric':<18}{'FP32':>12}{'INT8':>12}{'change':>14}")
    print("  " + "-" * 54)
    print(f"  {'size (MB)':<18}{fp32['size_mb']:>12}{int8['size_mb']:>12}"
          f"{results['delta']['size_pct']:>13}%")
    print(f"  {'tokens/sec':<18}{fp32['tokens_per_sec']:>12}{int8['tokens_per_sec']:>12}"
          f"{results['delta']['speedup_x']:>12}x")
    print(f"  {'perplexity':<18}{fp32['perplexity']:>12}{int8['perplexity']:>12}"
          f"{results['delta']['perplexity_pct']:>13}%")
    print("=" * 58)
    print("  size down / speed up = good ; perplexity up = quality lost")
    print("  Wrote results_real.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
