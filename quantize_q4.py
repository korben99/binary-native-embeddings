"""
INT4 weight-only quantization of the float embedder.

Tries torchao Int4WeightOnlyConfig first (true Q4), falls back to
PyTorch INT8 dynamic quantization if torchao is unavailable or
the model dimensions are incompatible.

Usage:
  pip install torchao          # optional but recommended for true Q4
  python quantize_q4.py
"""
import copy
import time
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from transformers import BertTokenizer

load_dotenv()

BASE_DIR = Path(__file__).parent
CKPT_DIR = BASE_DIR / "checkpoints"
DATA_DIR = BASE_DIR / "data_cache"


def apply_quantization(model):
    """
    Returns (quantized_model, backend_label).
    Tries torchao INT4 → torchao INT8 → raises.
    Note: torch.quantization.quantize_dynamic is broken on Python 3.13 (no QEngine).
    """
    from torchao.quantization import quantize_, Int4WeightOnlyConfig, Int8WeightOnlyConfig

    # INT4 weight-only (requires mslk on Apple Silicon)
    try:
        m = copy.deepcopy(model)
        quantize_(m, Int4WeightOnlyConfig())
        return m, "torchao INT4 weight-only"
    except Exception as e:
        print(f"  torchao INT4 unavailable ({type(e).__name__}: {e})")

    # INT8 weight-only (universal torchao fallback)
    print("  falling back to torchao INT8 weight-only")
    m = copy.deepcopy(model)
    quantize_(m, Int8WeightOnlyConfig())
    return m, "torchao INT8 weight-only (fallback)"


def bench_latency(model, tokenizer, n_runs=100, batch_size=32):
    texts = ["the quick brown fox jumps over the lazy dog"] * batch_size
    model.eval()
    for _ in range(5):
        model.encode(texts, tokenizer)
    t = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.encode(texts, tokenizer)
        t.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(t))


def eval_stsb_quick(model, tokenizer, use_binary=False):
    from scipy.stats import spearmanr
    from datasets import load_from_disk, load_dataset
    cache = DATA_DIR / "sts_test"
    ds = load_from_disk(str(cache)) if cache.exists() else \
         load_dataset("mteb/stsbenchmark-sts", split="test")
    human = np.array(ds["score"]) / 5.0
    e1 = model.encode(list(ds["sentence1"]), tokenizer).float()
    e2 = model.encode(list(ds["sentence2"]), tokenizer).float()
    e1 = torch.nn.functional.normalize(e1, dim=-1)
    e2 = torch.nn.functional.normalize(e2, dim=-1)
    pred = (e1 * e2).sum(dim=-1).numpy()
    corr, _ = spearmanr(pred, human)
    return float(corr)


def main():
    from models.float_embedder import FloatEmbedder

    tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-mini")

    ckpt = CKPT_DIR / "float_embedder.pt"
    float_model = FloatEmbedder(output_dim=384)
    float_model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    float_model.eval()

    ckpt_mb = ckpt.stat().st_size / 1e6
    print(f"Float checkpoint : {ckpt_mb:.1f} MB")

    print("\nApplying quantization...")
    q4_model, backend = apply_quantization(float_model)
    q4_model.eval()
    print(f"Backend          : {backend}")

    # Model weight size in memory
    def param_mb(m):
        return sum(
            p.nbytes if hasattr(p, "nbytes") else p.numel() * p.element_size()
            for p in m.parameters()
        ) / 1e6

    print(f"\nWeight memory    : float={param_mb(float_model):.1f} MB  q4={param_mb(q4_model):.1f} MB")

    print("\nBenchmarking latency (batch=32, 100 runs)...")
    float_ms = bench_latency(float_model, tokenizer)
    q4_ms    = bench_latency(q4_model, tokenizer)
    speedup  = float_ms / q4_ms
    print(f"  Float32 : {float_ms:.2f} ms")
    print(f"  Q4      : {q4_ms:.2f} ms  ({speedup:.1f}x {'faster' if speedup >= 1 else 'slower'})")

    print("\nSTS-B Spearman (quick check)...")
    float_stsb = eval_stsb_quick(float_model, tokenizer)
    q4_stsb    = eval_stsb_quick(q4_model, tokenizer)
    print(f"  Float32 : {float_stsb:.4f}")
    print(f"  Q4      : {q4_stsb:.4f}  (delta={q4_stsb - float_stsb:+.4f})")

    print(f"\nNote: Q4 output is still float32 384-dim — index memory unchanged (1.46 MB/1k vecs).")
    print(f"Gain is in encoding speed and model weight memory, not index size.")


if __name__ == "__main__":
    main()
