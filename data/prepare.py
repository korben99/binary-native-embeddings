"""
Download and cache all datasets needed for training and evaluation.
Run once before training: python data/prepare.py
"""
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from datasets import load_dataset

DATA_DIR = Path(__file__).parent.parent / "data_cache"


def prepare_nli():
    print("Downloading NLI triplets (sentence-transformers/all-nli)...")
    ds = load_dataset("sentence-transformers/all-nli", "triplet", split="train")
    out = DATA_DIR / "nli_train"
    ds.save_to_disk(str(out))
    print(f"  {len(ds):,} triplets  ->  {out}")


def prepare_sts():
    print("Downloading STS-B test set (mteb/stsbenchmark-sts)...")
    ds = load_dataset("mteb/stsbenchmark-sts", split="test")
    out = DATA_DIR / "sts_test"
    ds.save_to_disk(str(out))
    print(f"  {len(ds):,} pairs  ->  {out}")


def prepare_scifact():
    print("Downloading SciFact via BEIR...")
    try:
        from beir import util
        from beir.datasets.data_loader import GenericDataLoader

        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
        data_path = util.download_and_unzip(url, str(DATA_DIR / "beir"))
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

        out = DATA_DIR / "scifact"
        out.mkdir(exist_ok=True)
        (out / "corpus.json").write_text(json.dumps(corpus))
        (out / "queries.json").write_text(json.dumps(queries))
        (out / "qrels.json").write_text(json.dumps(qrels))

        print(f"  {len(corpus):,} docs  |  {len(queries)} queries  |  {len(qrels)} qrels  ->  {out}")

    except ImportError:
        print("  beir not installed, skipping SciFact.")
        print("  Install with: pip install beir")


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    prepare_nli()
    prepare_sts()
    prepare_scifact()
    print("\nAll datasets ready.")
