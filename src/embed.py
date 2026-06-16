"""Sentence-transformer embedding module for the DevOps Log Analyzer.

Provides functions to embed log text using all-MiniLM-L6-v2 and cache results
to avoid redundant computation across pipeline runs.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent.parent

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
PROGRESS_INTERVAL = 50

_model_cache: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Load (or return cached) sentence-transformer model.

    Returns:
        Loaded SentenceTransformer instance.
    """
    global _model_cache
    if _model_cache is None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading model: {MODEL_NAME}")
        _model_cache = SentenceTransformer(MODEL_NAME)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Model loaded. Embedding dim: {EMBEDDING_DIM}")
    return _model_cache


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of strings using all-MiniLM-L6-v2.

    Args:
        texts: List of raw text strings to embed.

    Returns:
        Embedding matrix of shape (n, 384) as float32 numpy array.
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    model = _get_model()
    total = len(texts)
    embeddings_list = []

    for start in range(0, total, PROGRESS_INTERVAL):
        end = min(start + PROGRESS_INTERVAL, total)
        batch = texts[start:end]
        batch_embeddings = model.encode(
            batch,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embeddings_list.append(batch_embeddings)
        pct = (end / total) * 100
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] Embedded {end}/{total} texts ({pct:.0f}%)"
        )

    return np.vstack(embeddings_list).astype(np.float32)


def _csv_mtime(csv_path: Path) -> float:
    """Return mtime of a CSV file, or -1 if not found."""
    try:
        return csv_path.stat().st_mtime
    except FileNotFoundError:
        return -1.0


def _cache_is_valid(csv_path: Path, npy_path: Path) -> bool:
    """Check if the embedding cache is still valid (newer than the source CSV).

    Args:
        csv_path: Source CSV file.
        npy_path: Cached embedding .npy file.

    Returns:
        True if the cache is valid and can be reused.
    """
    if not npy_path.exists():
        return False
    csv_mtime = _csv_mtime(csv_path)
    npy_mtime = npy_path.stat().st_mtime
    return npy_mtime >= csv_mtime


def embed_and_save(input_csv: str, output_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read log_text from a CSV, embed all rows, and save .npy files.

    Caching: if the output .npy exists and is newer than the input CSV, the
    embeddings are loaded from cache instead of recomputed.

    Args:
        input_csv: Path to a CSV file with at minimum columns [log_text, label].
        output_path: Path where the embeddings .npy will be saved. A companion
            *_labels.npy is saved in the same directory.

    Returns:
        Tuple of (embeddings, labels) as numpy arrays.
    """
    csv_path = Path(input_csv)
    npy_path = Path(output_path)
    labels_path = npy_path.parent / (npy_path.stem + "_labels" + npy_path.suffix)

    if _cache_is_valid(csv_path, npy_path) and labels_path.exists():
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Cache hit — loading embeddings from {npy_path}")
        embeddings = np.load(npy_path)
        labels = np.load(labels_path, allow_pickle=True)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Loaded {len(embeddings)} cached embeddings.")
        return embeddings, labels

    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Reading: {csv_path}")
    df = pd.read_csv(csv_path)

    missing = [c for c in ("log_text", "label") if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    texts = df["log_text"].fillna("").tolist()
    labels = df["label"].values

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Embedding {len(texts)} texts...")
    embeddings = embed_texts(texts)

    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, embeddings)
    np.save(labels_path, labels)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Embeddings saved: {npy_path} (shape: {embeddings.shape})")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Labels saved:     {labels_path}")
    return embeddings, labels


def main() -> None:
    """Embed the training dataset and optionally the drift dataset."""
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    train_csv = ROOT / "data" / "raw" / "logs.csv"
    train_emb = ROOT / "data" / "raw" / "embeddings.npy"
    drift_csv = ROOT / "data" / "drift" / "logs_drift.csv"
    drift_emb = ROOT / "data" / "drift" / "embeddings.npy"

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === Embedding Pipeline ===")

    if not train_csv.exists():
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Training data not found, generating...")
        sys.path.insert(0, str(ROOT / "src"))
        from generate_data import main as gen_main
        gen_main()

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] --- Training embeddings ---")
    embeddings, labels = embed_and_save(str(train_csv), str(train_emb))
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Training: {embeddings.shape}, labels: {len(labels)}")

    if drift_csv.exists():
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] --- Drift embeddings ---")
        d_emb, d_labels = embed_and_save(str(drift_csv), str(drift_emb))
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Drift: {d_emb.shape}, labels: {len(d_labels)}")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Done.")


if __name__ == "__main__":
    main()
