from dataclasses import dataclass
from enum import StrEnum

import typer
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from src.config import COLLECTION, DENSE, QDRANT_PATH, SPARSE, get_device

app = typer.Typer()

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL = "Qdrant/bm25"

# bge-*-v1.5 are asymmetric: the query gets this prefix, the document gets none.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Qdrant's server-side fusion hardcodes 1/(1+rank). That decay is far steeper
# than the customary k=60, so a strong ranking and a weak one end up nearly
# equally weighted; on this collection it costs ~8% nDCG@10 against fusing here.
RRF_K = 60


class Mode(StrEnum):
    """Which ranking to produce. The single-signal modes are the baselines the
    hybrid one has to beat."""

    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"


def rrf(rankings: list[list[str]], k: int) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: sum 1/(k+rank) across rankings, ignoring scores.

    The two rankings score on incomparable scales (cosine in [-1,1] against
    unnormalized BM25), so only positions can be combined. A larger k flattens
    the decay, which is what rewards documents both rankings agree on.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, 1):
            scores[doc] = scores.get(doc, 0.0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda item: -item[1])


@dataclass(frozen=True, slots=True)
class Searcher:
    """Holds the loaded models: opening one per query would dominate the runtime
    of an evaluation sweep (75 topics x 3 modes)."""

    client: QdrantClient
    dense_model: SentenceTransformer
    sparse_model: SparseTextEmbedding

    @classmethod
    def open(cls, device: str | None = None) -> "Searcher":
        return cls(
            client=QdrantClient(path=str(QDRANT_PATH)),
            dense_model=SentenceTransformer(DENSE_MODEL, device=device or get_device()),
            sparse_model=SparseTextEmbedding(SPARSE_MODEL),
        )

    def dense_query(self, text: str) -> list[float]:
        # normalize_embeddings must match build.py, or cosine means nothing.
        vec = self.dense_model.encode(
            QUERY_PREFIX + text, normalize_embeddings=True, show_progress_bar=False
        )
        return vec.tolist()

    def sparse_query(self, text: str) -> models.SparseVector:
        # query_embed, not embed: BM25 puts term saturation and length
        # normalization on the document side, the query only carries its terms.
        emb = next(iter(self.sparse_model.query_embed(text)))
        return models.SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())

    def _points(self, query, using: str, limit: int) -> list[tuple[str, float]]:
        result = self.client.query_points(
            COLLECTION, query=query, using=using, limit=limit, with_payload=["nct_id"]
        )
        return [(point.payload["nct_id"], point.score) for point in result.points]

    def search(
        self,
        text: str,
        mode: Mode = Mode.HYBRID,
        limit: int = 1000,
        prefetch: int | None = None,
        rrf_k: int | None = RRF_K,
    ) -> list[tuple[str, float]]:
        """Ranked (nct_id, score) pairs. Scores are comparable within a mode
        only: hybrid returns RRF scores, not similarities.

        rrf_k=None fuses server-side in a single round trip, at Qdrant's fixed
        k=1; any integer fuses here instead, which costs one extra query.
        """
        # Fusion depth is independent of how many results you want to see: with a
        # shallow prefetch most documents reach RRF from one list only, and the
        # fusion degenerates into whichever ranking happened to find them.
        prefetch = prefetch or max(limit, 500)

        if mode is Mode.DENSE:
            return self._points(self.dense_query(text), DENSE, limit)
        if mode is Mode.BM25:
            return self._points(self.sparse_query(text), SPARSE, limit)

        if rrf_k is None:
            result = self.client.query_points(
                COLLECTION,
                prefetch=[
                    models.Prefetch(query=self.dense_query(text), using=DENSE, limit=prefetch),
                    models.Prefetch(query=self.sparse_query(text), using=SPARSE, limit=prefetch),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=["nct_id"],
            )
            return [(point.payload["nct_id"], point.score) for point in result.points]

        rankings = [
            [nct_id for nct_id, _ in self._points(self.dense_query(text), DENSE, prefetch)],
            [nct_id for nct_id, _ in self._points(self.sparse_query(text), SPARSE, prefetch)],
        ]
        return rrf(rankings, rrf_k)[:limit]


@app.command()
def main(query: str, mode: Mode = Mode.HYBRID, limit: int = 10) -> None:
    searcher = Searcher.open()
    for rank, (nct_id, score) in enumerate(searcher.search(query, mode, limit), 1):
        print(f"{rank:>3}  {score:.4f}  {nct_id}")


if __name__ == "__main__":
    app()
