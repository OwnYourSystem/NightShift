# NightShift POC — Model Quantization (real numbers, honest tradeoffs)

**One-line pitch:** SMEs run unoptimized LLMs on expensive cloud GPUs. Quantization
shrinks a model so it runs on cheap hardware they already own — but doing it blindly
can make the model **slower and dumber**. This POC proves that the valuable thing
isn't the "quantize" button (which is free); it's the **measurement harness** that
tells you whether quantization actually helped *your* model on *your* hardware.

> Built and measured on a Windows CPU box. Every number below is reproducible with
> the scripts in this folder — nothing is simulated.

---

## Who pays, and why

| | |
|---|---|
| **Buyer** | SMEs running open LLMs (privacy / cost reasons) on cloud GPUs |
| **Pain** | A 70B model on a cloud A100 ≈ **$1,500–3,000/mo**; the same workload quantized can run on a **~$1,600 one-time 4090** or on-prem |
| **Trigger** | Cloud bill shock, data-privacy/on-prem mandate, latency |
| **What they actually buy** | Not the conversion (free on Hugging Face) — they buy **"will this still be good enough for my use case, and how do I run it?"** |

---

## What we measured (three findings, all real)

### Finding 1 — Memory is the reliable win; the *scheme* decides quality
From `quantize_numpy.py` (real INT8 math, from scratch, on a 393K-weight toy model):

| scheme | memory | quality loss (output drift) |
|---|---|---|
| FP32 (baseline) | 1536 KB | — |
| INT8 **per-tensor** | 384 KB (**4.0× smaller**) | **12.41%** relative error |
| INT8 **per-channel** | 384 KB (**4.0× smaller**) | **2.40%** relative error |

**Same 4× memory cut — but per-channel scaling loses 5× less quality.** That choice
(per-tensor vs per-channel vs AWQ/GPTQ vs bit-width) *is* the engineering value.

### Finding 2 — Quantization can DESTROY quality on a small model
From `quantize_real.py` (real pretrained **Pythia-70M**, PyTorch dynamic INT8):

| metric | FP32 | INT8 | change |
|---|---|---|---|
| size on disk | 281.7 MB | 147.9 MB | **−47.5%** |
| perplexity (lower=better) | 43.74 | 131.89 | **+201.5%** ⚠️ |

Size dropped ~half (not 4×, because dynamic quant leaves the big **embedding table**
in FP32). But perplexity **tripled** — a naive quantize would have shipped a much
dumber model. Small models have little redundancy to absorb quantization noise.

### Finding 3 — The speed verdict FLIPS with sequence length
From `speed_sweep.py` (same model, INT8 throughput ÷ FP32 throughput):

| sequence length | INT8 speedup |
|---|---|
| 8 tokens | **0.45×** (slower) |
| 32 tokens | **0.27×** (slower) |
| 128 tokens | **0.63×** (slower) |
| 512 tokens | **1.12×** (faster ✅) |

INT8 is *slower* on short inputs (quantize/dequantize overhead beats the savings)
and only wins at long sequences. **You cannot know the crossover without measuring
it for the specific model + hardware.**

---

## The thesis this POC proves

If you had blindly quantized Pythia-70M and shipped it, you'd have delivered a model
that was **slower (at normal sequence lengths) AND 3× worse on quality.** The harness
caught it. That is the product:

> **The conversion is a commodity. The measurement-and-validation harness — quality
> on the customer's real task + latency + memory + cost, across schemes — is the
> defensible value.**

This is the same lesson NightShift is built on: the durable asset is the
closed-loop **measure → compare → report** engine, not the optimizer trick inside it.

---

## How this generalizes (one engine, many markets)

This quantization loop is one `(strategy, verifier)` plug-in on NightShift's engine:

```
upload model → run experiments (quant schemes) → objective verifier
            (size + latency + quality) → ranked report
```

Swap the plug-in and the same engine drives kernel autotuning, hyperparameter
search, or any optimization where the verifier is cheap and objective.

---

## Reproduce it

```bash
# pure-NumPy mechanism demo (no downloads, runs anywhere)
python quantize_numpy.py

# real pretrained model — needs torch + transformers (see requirements.txt)
python quantize_real.py        # FP32 vs INT8: size, speed, perplexity
python speed_sweep.py          # speed vs sequence length
```

Outputs: `results_numpy.json`, `results_real.json`, `results_speed_sweep.json`.

---

## Honest limitations

- **Dynamic INT8 on CPU** is the conservative path. The big "run-at-home" wins come
  from **GGUF/llama.cpp** (quantizes embeddings too, 4-bit) and **AWQ** (GPU,
  outlier-aware) — next steps, not shown here.
- **Pythia-70M is deliberately tiny** so it runs fast on any laptop; quality loss is
  worse than it would be on a 7B model (more redundancy = more robust to quantization).
- **Perplexity is a proxy.** A real engagement evaluates on the **customer's actual
  task**, not a generic passage — that's the multi-metric guardrail against shipping
  a model that scores fine on paper and fails in production.
