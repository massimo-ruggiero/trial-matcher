import json
from collections.abc import Iterator
from itertools import batched
from pathlib import Path

import typer
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import (
    DATA_PROCESSED,
    DEFAULT_ENCODER,
    DENSE,
    ENCODERS,
    QDRANT_PATH,
    SPARSE,
    collection_for,
    get_device,
)
from src.models import Trial

app = typer.Typer()

SPARSE_MODEL = "Qdrant/bm25"


def load_trials(path: Path) -> Iterator[Trial]:
    """Stream the JSONL: the file is ~240 MB and is consumed batch by batch."""
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            d["conditions"] = tuple(d["conditions"])
            yield Trial(**d)


def ensure_collection(client: QdrantClient, collection: str, size: int, recreate: bool) -> None:
    if recreate and client.collection_exists(collection):
        client.delete_collection(collection)
    if client.collection_exists(collection):
        return

    client.create_collection(
        collection,
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
    encoder: str = DEFAULT_ENCODER,
) -> None:
    device = device or get_device()
    dense_model_name = ENCODERS[encoder].model
    collection = collection_for(encoder)
    print(f"device: {device}   encoder: {encoder} ({dense_model_name})   detailed: {detailed}")

    dense_model = SentenceTransformer(dense_model_name, device=device)
    sparse_model = SparseTextEmbedding(SPARSE_MODEL)

    client = QdrantClient(path=str(QDRANT_PATH))
    ensure_collection(client, collection, dense_model.get_embedding_dimension(), recreate)

    indexed = empty = 0
    trials = load_trials(DATA_PROCESSED / source)

    for batch in tqdm(batched(trials, batch_size), unit="batch"):
        pairs = [(t, doc) for t in batch if (doc := t.to_document(detailed))]
        empty += len(batch) - len(pairs)
        if not pairs:
            continue
        chunk, docs = zip(*pairs)

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
                id=trial.point_id,
                vector={
                    DENSE: dense.tolist(),
                    SPARSE: models.SparseVector(
                        indices=sparse.indices.tolist(), values=sparse.values.tolist()
                    ),
                },
                payload={"nct_id": trial.nct_id, "title": trial.title},
            )
            for trial, dense, sparse in zip(chunk, dense_vecs, sparse_vecs)
        ]
        client.upsert(collection, points=points)
        indexed += len(points)

    print(f"\nindexed: {indexed}   empty document: {empty}")
    print(f"collection {collection}: {client.count(collection).count} points")


if __name__ == "__main__":
    app()
