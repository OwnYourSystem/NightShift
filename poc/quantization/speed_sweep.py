"""
NightShift Quantization POC — the speed verdict is not a constant.

The headline run showed INT8 *slower* than FP32 on Pythia-70M. That is not a
bug — it's the real lesson. Dynamic quantization adds per-matmul quantize/
dequantize overhead. On tiny matmuls (small model, short sequence) that overhead
can exceed the savings; the win grows with sequence length and model size.

This sweeps the sequence length and reports the FP32 vs INT8 throughput at each,
so the crossover (if any) is *measured* for this exact model + this exact CPU —
which is precisely the thing a customer cannot know without running it.

Run:  python speed_sweep.py
Output: results_speed_sweep.json
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
MODEL = "EleutherAI/pythia-70m"
SEQ_LENS = [8, 32, 128, 512]


def tokens_per_sec(model, input_ids, n_runs=8) -> float:
    with torch.no_grad():
        model(input_ids)                       # warmup
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            model(input_ids)
    dt = time.perf_counter() - t0
    return round((input_ids.shape[1] * n_runs) / dt, 1)


def main() -> int:
    torch.set_num_threads(os.cpu_count() or 4)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    qmodel = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

    base_ids = tok("the lighthouse stood at the edge of the harbor for years ",
                   return_tensors="pt").input_ids[0]

    rows = []
    for L in SEQ_LENS:
        ids = base_ids.repeat((L // base_ids.shape[0]) + 1)[:L].unsqueeze(0)
        fp32 = tokens_per_sec(model, ids)
        int8 = tokens_per_sec(qmodel, ids)
        rows.append({
            "seq_len": L,
            "fp32_tok_s": fp32,
            "int8_tok_s": int8,
            "int8_speedup_x": round(int8 / fp32, 2),
        })

    out = {"model": MODEL, "cpu_threads": torch.get_num_threads(), "sweep": rows}
    (HERE / "results_speed_sweep.json").write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 56)
    print(f"  Speed sweep — {MODEL}")
    print("=" * 56)
    print(f"  {'seq_len':>8}{'FP32 tok/s':>14}{'INT8 tok/s':>14}{'INT8 x':>10}")
    print("  " + "-" * 52)
    for r in rows:
        print(f"  {r['seq_len']:>8}{r['fp32_tok_s']:>14}{r['int8_tok_s']:>14}"
              f"{r['int8_speedup_x']:>9}x")
    print("=" * 56)
    print("  >1.0x means INT8 wins. The verdict moves with sequence length —")
    print("  which is exactly why you measure instead of assume.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
