"""
Full benchmark: STS-B Spearman, SciFact Recall@10, CPU latency.
Run after training both models:
  python benchmark.py
Results saved to results/benchmark_results.json
"""
import json
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import BertTokenizer

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data_cache"
CKPT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"


# ── Similarity ────────────────────────────────────────────────────────────────

def cosine_sim_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_n = a / (a.norm(dim=-1, keepdim=True) + 1e-9)
    b_n = b / (b.norm(dim=-1, keepdim=True) + 1e-9)
    return torch.mm(a_n, b_n.T)


def hamming_sim_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Similarity for {-1,+1} binary vectors via normalized dot product.
    Equivalent to 1 - 2*hamming_distance/D, range [-1, +1].
    """
    D = a.shape[1]
    return torch.mm(a, b.T) / D


# ── STS-B ─────────────────────────────────────────────────────────────────────

def eval_stsb(model, tokenizer, use_binary=False):
    from datasets import load_from_disk, load_dataset

    cache = DATA_DIR / "sts_test"
    if cache.exists():
        ds = load_from_disk(str(cache))
    else:
        print("  Downloading STS-B...")
        ds = load_dataset("mteb/stsbenchmark-sts", split="test")

    human = np.array(ds["score"]) / 5.0  # normalize to [0,1]
    embs1 = model.encode(list(ds["sentence1"]), tokenizer)
    embs2 = model.encode(list(ds["sentence2"]), tokenizer)

    sim_fn = hamming_sim_matrix if use_binary else cosine_sim_matrix
    pred = sim_fn(embs1, embs2).diag().numpy()

    corr, _ = spearmanr(pred, human)
    return float(corr)


# ── SciFact Recall@10 ─────────────────────────────────────────────────────────

def load_scifact():
    cache = DATA_DIR / "scifact"
    if cache.exists():
        corpus = json.loads((cache / "corpus.json").read_text())
        queries = json.loads((cache / "queries.json").read_text())
        qrels = json.loads((cache / "qrels.json").read_text())
        return corpus, queries, qrels

    try:
        from beir import util
        from beir.datasets.data_loader import GenericDataLoader

        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
        path = util.download_and_unzip(url, str(DATA_DIR / "beir"))
        return GenericDataLoader(data_folder=path).load(split="test")
    except Exception as e:
        print(f"  SciFact unavailable: {e}")
        return None, None, None


def eval_scifact_recall(model, tokenizer, use_binary=False, top_k=10):
    corpus, queries, qrels = load_scifact()
    if corpus is None:
        return None

    # corpus = {doc_id: {"title": ..., "text": ...}}
    doc_ids = list(corpus.keys())
    doc_texts = [f"{corpus[d].get('title','')} {corpus[d].get('text','')}".strip()
                 for d in doc_ids]

    valid_qids = [qid for qid in queries if qid in qrels]
    q_texts = [queries[qid] for qid in valid_qids]

    print(f"  Encoding {len(doc_texts):,} docs...")
    corpus_embs = model.encode(doc_texts, tokenizer)
    print(f"  Encoding {len(q_texts)} queries...")
    query_embs = model.encode(q_texts, tokenizer)

    sim_fn = hamming_sim_matrix if use_binary else cosine_sim_matrix

    recalls = []
    for i, qid in enumerate(valid_qids):
        sims = sim_fn(query_embs[i : i + 1], corpus_embs)[0]
        top_idx = sims.topk(min(top_k, len(doc_ids))).indices.tolist()
        retrieved = {doc_ids[j] for j in top_idx}
        relevant = set(qrels[qid].keys())
        recalls.append(len(retrieved & relevant) / max(len(relevant), 1))

    return float(np.mean(recalls))


# ── Latency ───────────────────────────────────────────────────────────────────

def benchmark_latency(model, tokenizer, n_runs=100, batch_size=32):
    texts = ["the quick brown fox jumps over the lazy dog"] * batch_size
    model.eval()

    for _ in range(5):  # warmup
        model.encode(texts, tokenizer, device="cpu")

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.encode(texts, tokenizer, device="cpu")
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "mean_ms": round(float(np.mean(times)), 2),
        "p50_ms": round(float(np.percentile(times, 50)), 2),
        "p95_ms": round(float(np.percentile(times, 95)), 2),
    }


def memory_per_1k(dim, is_binary):
    bytes_each = dim / 8 if is_binary else dim * 4
    total = bytes_each * 1000
    if total < 1024:
        return f"{total:.0f} B"
    elif total < 1024**2:
        return f"{total/1024:.0f} KB"
    else:
        return f"{total/1024**2:.2f} MB"


# ── Post-hoc binary wrapper ───────────────────────────────────────────────────

class PostHocBinaryWrapper:
    """Wraps a float model, applies sign binarization at inference time."""
    def __init__(self, base):
        self.base = base

    def encode(self, texts, tokenizer, device="cpu", batch_size=64):
        floats = self.base.encode(texts, tokenizer, device=device, batch_size=batch_size)
        return torch.sign(floats).float()  # {-1, +1}

    def eval(self):
        self.base.eval()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(binary_dims=(2048, 4096)):
    from models.float_embedder import FloatEmbedder
    from models.binary_embedder import BinaryEmbedder

    tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-mini")

    print("\n=== Loading models ===")
    float_model = FloatEmbedder(output_dim=384)
    ckpt = CKPT_DIR / "float_embedder.pt"
    if ckpt.exists():
        float_model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        print(f"  float_embedder.pt loaded")
    else:
        print(f"  WARNING: {ckpt} not found — using random weights")
    float_model.eval()

    posthoc_model = PostHocBinaryWrapper(float_model)

    configs = [
        ("float32_384",        float_model,   False, 384,  False),
        ("binary_posthoc_384", posthoc_model, True,  384,  True),
    ]

    for dim in binary_dims:
        binary_model = BinaryEmbedder(binary_dim=dim)
        ckpt = CKPT_DIR / f"binary_embedder_{dim}.pt"
        # fallback to legacy name for 4096
        if not ckpt.exists() and dim == 4096:
            ckpt = CKPT_DIR / "binary_embedder.pt"
        if ckpt.exists():
            binary_model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            print(f"  binary_embedder_{dim}.pt loaded")
        else:
            print(f"  WARNING: {ckpt} not found — using random weights")
        binary_model.eval()
        configs.append((f"binary_native_{dim}", binary_model, True, dim, True))

    results = {}

    for name, model, use_binary, dim, is_binary in configs:
        label = "binary" if is_binary else "float"
        print(f"\n[{label}] Evaluating {name}...")

        print("  STS-B Spearman...")
        stsb = eval_stsb(model, tokenizer, use_binary=use_binary)

        print("  SciFact Recall@10...")
        scifact = eval_scifact_recall(model, tokenizer, use_binary=use_binary)

        print("  CPU latency (batch=32, 100 runs)...")
        lat = benchmark_latency(model, tokenizer)

        results[name] = {
            "dims": dim,
            "dtype": "binary" if is_binary else "float32",
            "stsb_spearman": round(stsb, 4),
            "scifact_recall10": round(scifact, 4) if scifact is not None else None,
            "memory_1k_vecs": memory_per_1k(dim, is_binary),
            "latency_cpu": lat,
        }

        r10 = f"{scifact:.4f}" if scifact is not None else "N/A"
        print(f"  STS-B={stsb:.4f}  R@10={r10}  lat={lat['mean_ms']}ms")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "benchmark_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults -> {out}")

    # Pretty table
    print("\n" + "=" * 90)
    print(f"{'Model':<25} {'Dims':>6} {'Type':>8} {'STS-B':>8} {'R@10':>8} {'Memory':>10} {'Lat (ms)':>10}")
    print("-" * 90)
    for name, r in results.items():
        r10 = f"{r['scifact_recall10']:.4f}" if r["scifact_recall10"] else "   N/A"
        print(
            f"{name:<25} {r['dims']:>6} {r['dtype']:>8} "
            f"{r['stsb_spearman']:>8.4f} {r10:>8} "
            f"{r['memory_1k_vecs']:>10} {r['latency_cpu']['mean_ms']:>9.1f}ms"
        )
    print("=" * 90)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary_dims", type=int, nargs="+", default=[2048, 4096],
                        help="Binary dims to evaluate (e.g. --binary_dims 2048 4096)")
    args = parser.parse_args()
    main(binary_dims=args.binary_dims)
