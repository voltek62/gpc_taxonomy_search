"""Précalcule les embeddings denses de la taxonomie → embeddings.npy.

Lance-le en local avant de pousser sur GitHub pour un démarrage instantané.
Le texte encodé (feuille — contexte) doit rester identique à celui de app.py.

    python build_index.py
"""

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).parent
TAXO_FILE = BASE / "taxonomy.it-IT.txt"
EMB_FILE = BASE / "embeddings.npy"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def dense_doc(leaf, parent):
    return f"{leaf} — {parent}" if parent else leaf


def load_docs():
    docs = []
    for line in TAXO_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        _id, sep, path = line.partition(" - ")
        if not sep or not path.strip():
            continue
        parts = [p.strip() for p in path.split(">")]
        docs.append(dense_doc(parts[-1], " > ".join(parts[:-1])))
    return docs


def main():
    docs = load_docs()
    print(f"{len(docs)} catégories chargées.")
    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(docs, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    emb = np.asarray(emb, dtype=np.float32)
    np.save(EMB_FILE, emb)
    print(f"Embeddings enregistrés dans {EMB_FILE} · shape={emb.shape}")


if __name__ == "__main__":
    main()
