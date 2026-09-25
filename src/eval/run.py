import json
from pathlib import Path

import pytrec_eval
import typer
from tqdm import tqdm

from src.config import DATA_PROCESSED, DATA_RAW, DEFAULT_ENCODER, RUNS
from src.models import Mode

app = typer.Typer()

MEASURES = {"ndcg_cut_10", "recall_1000", "P_10"}


def load_topics(year: int) -> list[dict]:
    with open(DATA_PROCESSED / f"topics{year}.jsonl") as f:
        return [json.loads(line) for line in f]


def load_qrels(year: int) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    for line in (DATA_RAW / "trec-ct" / f"qrels{year}.txt").read_text().splitlines():
        if not line.strip():
            continue
        topic, _, doc, judgement = line.split()
        qrels.setdefault(topic, {})[doc] = int(judgement)
    return qrels


def eligible_only(qrels: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    return {t: {d: int(j == 2) for d, j in docs.items()} for t, docs in qrels.items()}


def write_run(searcher, topics: list[dict], mode: Mode, limit: int, name: str) -> Path:
    RUNS.mkdir(exist_ok=True)
    path = RUNS / f"{name}.txt"
    with open(path, "w") as f:
        for topic in tqdm(topics, unit="topic", desc=name):
            hits = searcher.search(topic["text"], mode, limit=limit)
            for rank, (nct_id, _) in enumerate(hits, 1):
                # trec_eval ignores the rank column and sorts by score, breaking
                # ties by document id. RRF ties on more than half of its results,
                # so a synthetic decreasing score is written instead: it preserves
                # the order the retriever produced. Only the order is ever used.
                f.write(f"{topic['topic_id']} Q0 {nct_id} {rank} {limit - rank + 1} {name}\n")
    return path


def evaluate(run: dict, qrels: dict, measures: set[str]) -> dict[str, float]:
    per_topic = pytrec_eval.RelevanceEvaluator(qrels, measures).evaluate(run)
    names = next(iter(per_topic.values())).keys()
    return {m: sum(t[m] for t in per_topic.values()) / len(per_topic) for m in names}


def excluded_in_top_k(run: dict, qrels: dict[str, dict[str, int]], k: int = 10) -> float:
    """Average number of trials the patient is barred from, in the top k. No
    standard measure captures this, and it is what the second pass must reduce."""
    counts = []
    for topic, docs in run.items():
        top = sorted(docs, key=docs.get, reverse=True)[:k]
        counts.append(sum(1 for d in top if qrels.get(topic, {}).get(d) == 1))
    return sum(counts) / len(counts)


@app.command()
def main(
    year: int = 2021,
    limit: int = 1000,
    encoder: str = DEFAULT_ENCODER,
    prefix: bool = True,
) -> None:
    # Imported here: everything above this function reads files, and only
    # producing a run needs the encoder.
    from src.retrieve.search import Searcher

    topics = load_topics(year)
    qrels = load_qrels(year)
    binary = eligible_only(qrels)
    searcher = Searcher.open(encoder, prefix=None if prefix else "")
    # Runs of different encoders must not overwrite each other.
    tag = "" if encoder == DEFAULT_ENCODER else f"_{encoder}"
    tag += "" if prefix else "_noprefix"

    rows = []
    for mode in Mode:
        path = write_run(searcher, topics, mode, limit, f"{mode.value}{tag}{year}")
        with open(path) as f:
            run = pytrec_eval.parse_run(f)
        rows.append(
            (
                mode,
                evaluate(run, qrels, MEASURES),
                evaluate(run, binary, MEASURES),
                excluded_in_top_k(run, qrels),
            )
        )

    print(f"\n=== TREC CT {year}, judged subset, top-{limit}, encoder {encoder} ===")
    head = (
        f"{'mode':<8}{'nDCG@10':>9}{'nDCG@10*':>10}"
        f"{'P@10':>8}{'P@10*':>8}{'R@1000':>9}{'excl@10':>9}"
    )
    print(head)
    print("-" * len(head))
    for mode, graded, strict, excl in rows:
        print(
            f"{mode.value:<8}{graded['ndcg_cut_10']:>9.4f}{strict['ndcg_cut_10']:>10.4f}"
            f"{graded['P_10']:>8.4f}{strict['P_10']:>8.4f}"
            f"{graded['recall_1000']:>9.4f}{excl:>9.2f}"
        )
    print("\n* = eligible only (excluded counts as non relevant)")


if __name__ == "__main__":
    app()
