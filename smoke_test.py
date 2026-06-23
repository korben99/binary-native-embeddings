"""
Quick end-to-end smoke test — runs in ~2 min on CPU, verifies everything works
before launching the full training (which takes hours).

Usage: python smoke_test.py
"""
import sys
import torch
from pathlib import Path
from dotenv import load_dotenv
from transformers import BertTokenizer

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from models.float_embedder import FloatEmbedder, mnrl_loss
from models.binary_embedder import BinaryEmbedder, binary_contrastive_loss
from models.ste import binarize


def test_ste():
    x = torch.tensor([-1.0, 0.5, -0.3, 0.9])
    out = binarize(x)
    assert out.tolist() == [-1.0, 1.0, -1.0, 1.0], f"STE forward failed: {out}"

    # gradient must pass through unchanged
    x = torch.randn(4, requires_grad=True)
    binarize(x).sum().backward()
    assert x.grad is not None, "STE gradient is None"
    print("  [OK] STE forward + backward")


def test_float_embedder():
    tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-mini")
    model = FloatEmbedder(output_dim=384)
    texts = ["hello world", "binary embeddings are fast"]
    embs = model.encode(texts, tokenizer)
    assert embs.shape == (2, 384), f"Expected (2,384) got {embs.shape}"
    print(f"  [OK] FloatEmbedder output {embs.shape}")


def test_binary_embedder():
    tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-mini")
    model = BinaryEmbedder(binary_dim=4096)
    texts = ["hello world", "binary embeddings are fast"]
    embs = model.encode(texts, tokenizer)
    assert embs.shape == (2, 4096), f"Expected (2,4096) got {embs.shape}"
    assert set(embs.unique().tolist()).issubset({-1.0, 1.0}), "Output should be binary {-1,+1}"
    print(f"  [OK] BinaryEmbedder output {embs.shape}, values in {{-1,+1}}")


def test_losses():
    B, D_float, D_bin = 8, 384, 4096
    a = torch.randn(B, D_float)
    p = torch.randn(B, D_float)
    loss = mnrl_loss(a, p)
    assert loss.item() > 0, "MNRL loss should be positive"
    print(f"  [OK] mnrl_loss = {loss.item():.4f}")

    a_logits = torch.randn(B, D_bin, requires_grad=True)
    p_logits = torch.randn(B, D_bin, requires_grad=True)
    loss = binary_contrastive_loss(a_logits, p_logits)
    loss.backward()
    assert a_logits.grad is not None, "No gradient for binary loss"
    print(f"  [OK] binary_contrastive_loss = {loss.item():.4f}, gradients OK")


def test_mini_training():
    """One gradient step for each mode."""
    from datasets import load_dataset
    from torch.utils.data import DataLoader, Dataset

    class TinyDataset(Dataset):
        def __init__(self):
            self.a = ["I like dogs"] * 16
            self.p = ["I love dogs"] * 16
        def __len__(self): return 16
        def __getitem__(self, i): return self.a[i], self.p[i]

    tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-mini")
    loader = DataLoader(TinyDataset(), batch_size=8)

    for mode in ("float", "binary"):
        if mode == "float":
            model = FloatEmbedder(output_dim=384)
            loss_fn = mnrl_loss
        else:
            model = BinaryEmbedder(binary_dim=4096)

        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        model.train()

        for anchors, positives in loader:
            enc_a = tokenizer(list(anchors), padding=True, truncation=True,
                              max_length=64, return_tensors="pt")
            enc_p = tokenizer(list(positives), padding=True, truncation=True,
                              max_length=64, return_tensors="pt")
            if mode == "float":
                loss = mnrl_loss(
                    model(enc_a["input_ids"], enc_a["attention_mask"]),
                    model(enc_p["input_ids"], enc_p["attention_mask"]),
                )
            else:
                loss = binary_contrastive_loss(
                    model(enc_a["input_ids"], enc_a["attention_mask"], binarize_output=False),
                    model(enc_p["input_ids"], enc_p["attention_mask"], binarize_output=False),
                )
            opt.zero_grad()
            loss.backward()
            opt.step()
            break  # one step is enough

        print(f"  [OK] mini training step [{mode}] loss={loss.item():.4f}")


if __name__ == "__main__":
    print("Running smoke tests...\n")
    test_ste()
    test_float_embedder()
    test_binary_embedder()
    test_losses()
    test_mini_training()
    print("\nAll tests passed. Ready for full training.")
