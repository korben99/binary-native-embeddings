"""
Retrieval speed & memory benchmark at scale — multi-dim binary comparison.

Backend selection (automatic):
  - FAISS available (x86_64)  → IndexFlatIP + IndexBinaryFlat (AVX2 + POPCNT)
  - FAISS unavailable (ARM64) → pure NumPy fallback

Usage:
  python benchmark_faiss.py                        # 2048 + 4096 dims
  python benchmark_faiss.py --binary_dims 4096     # single dim
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from transformers import BertTokenizer

load_dotenv()

# Corporate proxy fix: set CURL_CA_BUNDLE= in .env
import os as _os
if _os.environ.get("CURL_CA_BUNDLE", "NOT_SET") == "":
    import ssl as _ssl
    _ssl._create_default_https_context = _ssl._create_unverified_context

BASE_DIR    = Path(__file__).parent
CKPT_DIR    = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"
FLOAT_DIM   = 384

POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

# ── FAISS detection ───────────────────────────────────────────────────────────
try:
    import faiss
    HAVE_FAISS = platform.machine() != "arm64"
except ImportError:
    HAVE_FAISS = False


# ── Packing ───────────────────────────────────────────────────────────────────

def pack_binary(vecs: np.ndarray) -> np.ndarray:
    """Convert {-1,+1} float32 [N, D] -> packed uint8 [N, D//8]."""
    return np.packbits((vecs > 0).astype(np.uint8), axis=1)


# ── Search functions (dim-aware) ──────────────────────────────────────────────

def float_search(queries: np.ndarray, db: np.ndarray, k: int = 10) -> np.ndarray:
    if HAVE_FAISS:
        q, d = queries.copy().astype(np.float32), db.copy().astype(np.float32)
        faiss.normalize_L2(q); faiss.normalize_L2(d)
        idx = faiss.IndexFlatIP(FLOAT_DIM)
        idx.add(d)
        _, I = idx.search(q, k)
        return I
    else:
        q = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-9)
        d = db     / (np.linalg.norm(db,      axis=1, keepdims=True) + 1e-9)
        return np.argpartition(-(q @ d.T), k, axis=1)[:, :k]


def binary_search(q_packed: np.ndarray, db_packed: np.ndarray,
                  binary_dim: int, k: int = 10) -> np.ndarray:
    if HAVE_FAISS:
        idx = faiss.IndexBinaryFlat(binary_dim)
        idx.add(db_packed)
        _, I = idx.search(q_packed, k)
        return I
    else:
        Q, N = len(q_packed), len(db_packed)
        dist = np.empty((Q, N), dtype=np.int32)
        for start in range(0, N, 8_000):
            end = min(start + 8_000, N)
            xor = q_packed[:, None, :] ^ db_packed[None, start:end, :]
            dist[:, start:end] = POPCOUNT[xor].sum(axis=2)
        return np.argpartition(dist, k, axis=1)[:, :k]


# ── Corpus helpers ────────────────────────────────────────────────────────────

def make_float_corpus(seeds: np.ndarray, n: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(seeds), n)
    c = seeds[idx].copy().astype(np.float32)
    c += rng.standard_normal(c.shape).astype(np.float32) * 0.3
    return c


def make_binary_corpus(seeds_packed: np.ndarray, n: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(seeds_packed), n)
    c = seeds_packed[idx].copy()
    c[rng.random(c.shape) < 0.15] ^= np.uint8(0xFF)
    return c


# ── Timing ────────────────────────────────────────────────────────────────────

def bench(fn, *args, n_runs=10, warmup=3) -> float:
    for _ in range(warmup):
        fn(*args)
    t = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(*args)
        t.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(t))


def vs_str(float_ms: float, binary_ms: float) -> str:
    r = float_ms / binary_ms
    return f"{r:.1f}x faster" if r >= 1 else f"{1/r:.1f}x slower"


# ── Main ──────────────────────────────────────────────────────────────────────

def main(binary_dims=(2048, 4096)):
    from models.float_embedder  import FloatEmbedder
    from models.binary_embedder import BinaryEmbedder

    machine = platform.machine()
    backend = "FAISS (AVX2+POPCNT)" if HAVE_FAISS else "NumPy (no hardware POPCNT)"
    plabel  = (f"{platform.node()} | {platform.processor() or machine} "
               f"| Python {platform.python_version()} | {backend}")
    print(f"\nPlatform : {plabel}\nBackend  : {backend}")

    # ── Load models ──
    tokenizer   = BertTokenizer.from_pretrained("prajjwal1/bert-mini")

    float_model = FloatEmbedder(output_dim=FLOAT_DIM)
    ckpt = CKPT_DIR / "float_embedder.pt"
    if ckpt.exists():
        float_model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    float_model.eval()

    binary_models = {}
    for dim in binary_dims:
        m = BinaryEmbedder(binary_dim=dim)
        ckpt = CKPT_DIR / f"binary_embedder_{dim}.pt"
        if not ckpt.exists() and dim == 4096:
            ckpt = CKPT_DIR / "binary_embedder.pt"   # legacy name
        if ckpt.exists():
            m.load_state_dict(torch.load(ckpt, map_location="cpu"))
            print(f"  Loaded binary-{dim}")
        else:
            print(f"  WARNING: checkpoint not found for dim={dim}, using random weights")
        m.eval()
        binary_models[dim] = m

    # ── Encode seed queries ──
    SEED_QUERIES = [
        "what causes alzheimer disease",
        "climate change effects on biodiversity",
        "neural network training optimization",
        "covid-19 vaccine efficacy",
        "quantum computing applications",
        "protein folding structure prediction",
        "machine learning interpretability",
        "antibiotic resistance mechanisms",
        "deep learning natural language processing",
        "solar energy efficiency improvements",
        "cancer immunotherapy treatment",
        "autonomous vehicle safety systems",
        "gene editing CRISPR technology",
        "black hole gravitational waves detection",
        "renewable energy battery storage",
        "microbiome gut health research",
    ]

    print(f"\nEncoding {len(SEED_QUERIES)} seed queries...")
    float_seeds = float_model.encode(SEED_QUERIES, tokenizer).numpy().astype(np.float32)

    bin_seeds = {}   # dim -> (float_seeds, packed_seeds)
    for dim, m in binary_models.items():
        s = m.encode(SEED_QUERIES, tokenizer).numpy().astype(np.float32)
        bin_seeds[dim] = (s, pack_binary(s))

    # ── Benchmark loop ──
    scales  = [10_000, 100_000, 1_000_000]
    results = {"platform": plabel, "backend": backend}

    for n in scales:
        print(f"\n{'='*60}  N={n:,}")

        float_mem = n * FLOAT_DIM * 4 / 1e6
        float_corpus = make_float_corpus(float_seeds, n)
        float_ms = bench(float_search, float_seeds, float_corpus, 10)
        print(f"  float-{FLOAT_DIM:4d}: {float_ms:8.2f} ms  |  {float_mem:6.0f} MB")

        scale_r = {"n_vectors": n,
                   "float_mem_mb": round(float_mem, 1),
                   "float_search_ms": round(float_ms, 2),
                   "binary": {}}

        for dim in binary_dims:
            seeds_f, seeds_p = bin_seeds[dim]
            binary_bytes  = dim // 8
            binary_mem    = n * binary_bytes / 1e6
            binary_corpus = make_binary_corpus(seeds_p, n)
            binary_ms     = bench(binary_search, seeds_p, binary_corpus, dim, 10)
            vs            = vs_str(float_ms, binary_ms)
            mem_ratio     = float_mem / binary_mem
            print(f"  bin-{dim:4d}:  {binary_ms:8.2f} ms  |  {binary_mem:6.0f} MB  =>  {vs}  ({mem_ratio:.0f}x smaller)")
            scale_r["binary"][str(dim)] = {
                "binary_mem_mb":    round(binary_mem, 1),
                "mem_ratio_x":      round(mem_ratio, 1),
                "binary_search_ms": round(binary_ms, 2),
                "vs_float":         vs,
            }

        results[str(n)] = scale_r

    # ── Save ──
    RESULTS_DIR.mkdir(exist_ok=True)
    slug = f"{machine.lower()}_{'faiss' if HAVE_FAISS else 'numpy'}"
    out  = RESULTS_DIR / f"retrieval_benchmark_{slug}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults -> {out}")

    # ── Summary table ──
    W = 13
    sep = "=" * (12 + (W + 2) * (1 + 2 * len(binary_dims)))
    print(f"\n{sep}")
    print(f"  {backend}")
    header = f"{'Scale':>12}"
    header += f"  {'Float (ms)':>{W}}"
    for dim in binary_dims:
        header += f"  {f'Bin-{dim} (ms)':>{W}}"
    for dim in binary_dims:
        header += f"  {f'vs Float ({dim})':>{W}}"
    print(header)
    print("-" * (12 + (W + 2) * (1 + 2 * len(binary_dims))))
    for n_str, r in results.items():
        if not n_str.isdigit():
            continue
        row = f"{r['n_vectors']:>12,}  {r['float_search_ms']:>{W}.2f}"
        for dim in binary_dims:
            row += f"  {r['binary'][str(dim)]['binary_search_ms']:>{W}.2f}"
        for dim in binary_dims:
            row += f"  {r['binary'][str(dim)]['vs_float']:>{W}}"
        print(row)
    print(sep)

    if not HAVE_FAISS:
        print("\n[!] NumPy backend — faiss-cpu on x86/Python ≤3.12 uses AVX2+POPCNT.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary_dims", type=int, nargs="+", default=[2048, 4096])
    args = parser.parse_args()
    main(binary_dims=args.binary_dims)
