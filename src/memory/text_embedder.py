# src/memory/text_embedder.py
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-en-v1.5")

def embed_text(text: str):
    emb = model.encode(
        [text],
        batch_size=1,
        normalize_embeddings=False
    )
    return emb[0]
