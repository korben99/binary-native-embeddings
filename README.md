# Binary Native Embeddings

> **Hypothesis:** a transformer trained natively with a 4096-bit binary head and a contrastive binary loss produces better semantic similarity than the same transformer binarized post-hoc — at equal or lower CPU latency.

Tested on a **Mac Mini M4 Pro, CPU only**. No GPU required for inference.

---

## Results

*(Fill in after running `python benchmark.py`)*

| Model | Dims | Type | STS-B Spearman | Recall@10 | Memory / 1k vecs | Latency (CPU) |
|---|---|---|---|---|---|---|
| Float baseline | 384 | float32 | — | — | 1.5 MB | — |
| Post-hoc binary | 384 | binary | — | — | 48 KB | — |
| **Native binary** | **4096** | **binary** | **—** | **—** | **512 KB** | **—** |

The hypothesis is validated if `binary_native_4096 > binary_posthoc_384` on both STS-B and Recall@10.

---

## Architecture

```
Input text
    │
    ▼
bert-mini (4L × 256d, ~11M params)
    │  mean pooling
    ▼
[256-dim float]
    │
    ├── FloatEmbedder:  Linear(256→384)          → 384-dim float32
    │
    └── BinaryEmbedder: Linear(256→4096) + LN    → STE → {0,1}^4096
```

The key ingredient is the **Straight-Through Estimator (STE)**: the forward pass binarizes via `sign()`, while the backward pass passes the gradient unchanged, making the discrete step differentiable.

---

## Quick start

```bash
pip install -r requirements.txt
```

```python
from transformers import AutoTokenizer
from models.binary_embedder import BinaryEmbedder

tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-mini")
model = BinaryEmbedder.from_pretrained("YOUR_HF_REPO")   # after publishing

vecs = model.encode(["binary embeddings are fast on CPU"], tokenizer)
# vecs.shape → (1, 4096), values in {0, 1}
```

---

## Reproduce

### 1 — Environment

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2 — Download datasets

```bash
python data/prepare.py
```

Downloads ~2 GB: NLI triplets (550k pairs), STS-B test set, SciFact corpus + qrels.

### 3 — Smoke test (2 min)

Verifies all shapes, losses, and one gradient step before committing to full training.

```bash
python smoke_test.py
```

### 4 — Train

```bash
# Float baseline  (~1h30 on M4 Pro MPS)
python train.py --mode float --epochs 3 --batch_size 64

# Native binary   (~2h on M4 Pro MPS)
python train.py --mode binary --epochs 3 --batch_size 64
```

Add `--max_samples 5000` for a quick iteration. Add `--no_mps` to force CPU.

Checkpoints saved to `checkpoints/float_embedder.pt` and `checkpoints/binary_embedder.pt`.

### 5 — Benchmark

```bash
python benchmark.py
```

Evaluates all three models (float, post-hoc binary, native binary) on STS-B Spearman correlation, SciFact Recall@10, and CPU latency. Results saved to `results/benchmark_results.json`.

---

## How STE works

Standard `sign()` has zero gradient almost everywhere, making it unusable in training. The STE trick:

```python
class BinarizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return (x > 0).float()          # discrete in forward

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output              # identity in backward
```

This lets gradients flow as if binarization were the identity function, while the forward pass genuinely produces binary outputs.

---

## Why native binary beats post-hoc

Post-hoc binarization forces the model to *accidentally* produce near-binary representations in a 384-dim space it was never trained for. Native training with 4096 binary dimensions gives the model:

1. **10× more dimensions** to distribute information across bits
2. **A loss that explicitly optimizes binary similarity** during training
3. **Redundancy** — similar concepts can be captured by multiple bits, making the representation robust to bit noise

---

## Memory comparison

| Representation | Formula | 1k vectors |
|---|---|---|
| float32 × 384 | 384 × 4B × 1000 | 1.5 MB |
| binary × 384 | 384 / 8B × 1000 | 48 KB |
| binary × 4096 | 4096 / 8B × 1000 | 512 KB |

Native binary at 4096 dims uses **3× more memory than post-hoc binary at 384 dims**, but **3× less than float at 384 dims** — while targeting better accuracy than both.

---

## Project structure

```
binary-native-embeddings/
├── README.md
├── requirements.txt
├── smoke_test.py          ← run first to verify setup
├── train.py               ← --mode float | binary
├── benchmark.py           ← produces results/benchmark_results.json
├── models/
│   ├── ste.py             ← Straight-Through Estimator
│   ├── float_embedder.py  ← baseline + mnrl_loss
│   └── binary_embedder.py ← native binary + binary_contrastive_loss
├── data/
│   └── prepare.py         ← download NLI / STS-B / SciFact
└── results/
    └── benchmark_results.json  ← filled after benchmark
```

---

## License

MIT
