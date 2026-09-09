import json
import time
from collections.abc import Iterator
from itertools import batched
from pathlib import Path

import typer
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import COLLECTION, DATA_PROCESSED, DENSE, QDRANT_URL, SPARSE, get_device
from src.models import Trial

app = typer.Typer()

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL = "Qdrant/bm25"


def load_trials(path: Path) -> Iterator[Trial]:
    """Stream the JSONL: the file is ~240 MB and is consumed batch by batch."""
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            d["conditions"] = tuple(d["conditions"])
            yield Trial(**d)


def ensure_collection(client: QdrantClient, size: int, recreate: bool) -> None:
    if recreate and client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    if client.collection_exists(COLLECTION):
        return

    client.create_collection(
        COLLECTION,
        vectors_config={DENSE: models.VectorParams(size=size, distance=models.Distance.COSINE)},
        # fastembed's bm25 emits term frequencies only — IDF needs corpus-wide
        # statistics it cannot see. Modifier.IDF makes Qdrant apply it at query
        # time from the collection's own document frequencies. Without it the
        # ranking silently degrades to raw TF.
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )


@app.command()
def main(
    source: str = "trials_judged.jsonl",
    batch_size: int = 256,
    encode_batch: int = 64,
    detailed: bool = False,
    recreate: bool = False,
    device: str | None = None,
) -> None:
    device = device or get_device()
    print(f"device: {device}   dense: {DENSE_MODEL}   detailed: {detailed}")

    dense_model = SentenceTransformer(DENSE_MODEL, device=device)
    sparse_model = SparseTextEmbedding(SPARSE_MODEL)

    client = QdrantClient(url=QDRANT_URL)
    ensure_collection(client, dense_model.get_embedding_dimension(), recreate)

    indexed = empty = 0
    trials = load_trials(DATA_PROCESSED / source)

    for batch in tqdm(batched(trials, batch_size), unit="batch"):
        pairs = [(t, doc) for t in batch if (doc := t.to_document(detailed))]
        empty += len(batch) - len(pairs)
        if not pairs:
            continue
        chunk, docs = zip(*pairs)

        # normalize_embeddings must match Distance.COSINE here and in search.py:
        # normalized vectors make cosine equal to the dot product Qdrant computes.
        dense_vecs = dense_model.encode(
            list(docs),
            batch_size=encode_batch,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        sparse_vecs = sparse_model.embed(docs)

        points = [
            models.PointStruct(
                # Deterministic: re-running overwrites instead of duplicating.
                id=trial.point_id,
                vector={
                    DENSE: dense.tolist(),
                    SPARSE: models.SparseVector(
                        indices=sparse.indices.tolist(), values=sparse.values.tolist()
                    ),
                },
                # Minimal on purpose: the run file only needs the NCT id, and the
                # text is already in the JSONL.
                payload={"nct_id": trial.nct_id, "title": trial.title},
            )
            for trial, dense, sparse in zip(chunk, dense_vecs, sparse_vecs)
        ]
        client.upsert(COLLECTION, points=points, wait=False)
        indexed += len(points)

    # Upserts were sent with wait=False, so Qdrant is still indexing: block
    # until it settles, otherwise the count below reports a partial number.
    while client.get_collection(COLLECTION).status != models.CollectionStatus.GREEN:
        time.sleep(1)

    print(f"\nindexed: {indexed}   empty document: {empty}")
    print(f"collection {COLLECTION}: {client.count(COLLECTION).count} points")


if __name__ == "__main__":
    app()
