import json
from collections import Counter
from enum import IntEnum
from pathlib import Path

import typer

from src.assess.judge import MODEL, verdicts_path
from src.assess.prompt import PROMPT_VERSION
from src.eval.run import MEASURES, RUNS, eligible_only, evaluate, excluded_in_top_k, load_qrels

app = typer.Typer()


SMOOTHED, FRACTION, COUNT, RETRIEVAL = "smoothed", "fraction", "count", "retrieval"
# Additive (Laplace) smoothing: a trial with two criteria met out of two has
# thinner evidence than one with nine out of ten, and should not outrank it.
ALPHA = 1.0


class Label(IntEnum):
    """Predicted class. The value is the tier in the reranked list."""

    ELIGIBLE = 0
    UNJUDGED = 1
    INELIGIBLE = 2
    EXCLUDED = 3


def effective(row: dict, grounded_only: bool) -> str:
    """A decisive verdict only counts if a quote from the note backs it."""
    verdict = row["verdict"]
    if verdict is None:
        return "unclear"
    if grounded_only and verdict != "unclear" and not row["grounded"]:
        return "unclear"
    return verdict


def score_of(inclusion: list[str], unclear: float, scoring: str) -> float:
    """How eligible trials are ordered among themselves. "retrieval" scores them
    all alike, which leaves them in the order the search produced."""
    if scoring == RETRIEVAL:
        return 0.0
    met = sum(1.0 if v == "yes" else unclear for v in inclusion)
    if scoring == COUNT:
        return met
    if scoring == FRACTION:
        return met / len(inclusion) if inclusion else unclear
    return (met + ALPHA) / (len(inclusion) + 2 * ALPHA)


def classify(
    record: dict | None,
    unclear: float = 0.5,
    grounded_only: bool = False,
    scoring: str = SMOOTHED,
) -> tuple[Label, float]:
    if record is None or record["error"]:
        return Label.UNJUDGED, 0.0
    rows = [(r["kind"], effective(r, grounded_only)) for r in record["criteria"]]
    if any(kind == "exclusion" and v == "yes" for kind, v in rows):
        return Label.EXCLUDED, 0.0
    if any(kind == "inclusion" and v == "no" for kind, v in rows):
        return Label.INELIGIBLE, 0.0
    inclusion = [v for kind, v in rows if kind == "inclusion"]
    return Label.ELIGIBLE, score_of(inclusion, unclear, scoring)


def rerank(
    ranked: list[str],
    records: dict[str, dict],
    depth: int,
    unclear: float = 0.5,
    grounded_only: bool = False,
    scoring: str = SMOOTHED,
) -> list[str]:
    """Reorder the head by predicted class, then score, then original rank."""

    def key(item: tuple[int, str]) -> tuple[Label, float, int]:
        position, nct_id = item
        label, score = classify(records.get(nct_id), unclear, grounded_only, scoring)
        return label, -score, position

    head = [nct_id for _, nct_id in sorted(enumerate(ranked[:depth]), key=key)]
    return head + ranked[depth:]


def load_run(path: Path) -> dict[str, list[str]]:
    run: dict[str, list[str]] = {}
    with open(path) as f:
        for line in f:
            topic, _, nct_id, *_ = line.split()
            run.setdefault(topic, []).append(nct_id)
    return run


def load_records(year: int, model: str, prompt: str) -> dict[tuple[str, str], dict]:
    with open(verdicts_path(year)) as f:
        return {
            (r["topic_id"], r["nct_id"]): r
            for r in map(json.loads, f)
            if r["model"] == model and r["prompt_version"] == prompt
        }


def as_run(ranked: dict[str, list[str]]) -> dict[str, dict[str, float]]:
    return {t: {n: float(len(docs) - i) for i, n in enumerate(docs)} for t, docs in ranked.items()}


def oracle_p10(ranked: dict[str, list[str]], qrels: dict, depth: int) -> float:
    hits = [sum(qrels[t].get(n) == 2 for n in docs[:depth]) for t, docs in ranked.items()]
    return sum(min(10, h) / 10 for h in hits) / len(hits)


@app.command()
def main(
    year: int = 2021,
    depth: int = 50,
    run: str = "dense",
    model: str = MODEL,
    prompt: str = PROMPT_VERSION,
    unclear: float = 0.5,
    grounded_only: bool = False,
    scoring: str = SMOOTHED,
) -> None:
    qrels = load_qrels(year)
    binary = eligible_only(qrels)
    records = load_records(year, model, prompt)
    topics = sorted({t for t, _ in records}, key=int)
    if not topics:
        print(f"no verdicts for {model} / prompt {prompt} in {verdicts_path(year).name}")
        return

    base = {t: docs for t, docs in load_run(RUNS / f"{run}{year}.txt").items() if t in topics}
    by_topic = {t: {n: r for (tt, n), r in records.items() if tt == t} for t in topics}
    reranked = {
        t: rerank(base[t], by_topic[t], depth, unclear, grounded_only, scoring) for t in topics
    }

    name = f"rerank_{run}{year}"
    with open(RUNS / f"{name}.txt", "w") as f:
        for t, docs in reranked.items():
            for rank, nct_id in enumerate(docs, 1):
                f.write(f"{t} Q0 {nct_id} {rank} {len(docs) - rank + 1} {name}\n")

    shortlist = sum(len(base[t][:depth]) for t in topics)
    print(f"{len(topics)} topics, {len(records)} judged of {shortlist} shortlisted trials")
    print(
        f"model {model}  prompt {prompt}  unclear={unclear}  "
        f"grounded_only={grounded_only}  scoring={scoring}\n"
    )

    head = f"{'':<10}{'nDCG@10':>9}{'nDCG@10*':>10}{'P@10*':>8}{'excl@10':>9}"
    print(head)
    print("-" * len(head))
    for label, ranked in ((run, base), ("reranked", reranked)):
        r = as_run(ranked)
        graded, strict = evaluate(r, qrels, MEASURES), evaluate(r, binary, MEASURES)
        print(
            f"{label:<10}{graded['ndcg_cut_10']:>9.4f}{strict['ndcg_cut_10']:>10.4f}"
            f"{strict['P_10']:>8.4f}{excluded_in_top_k(r, qrels):>9.2f}"
        )
    print(f"{'oracle':<10}{'':>9}{'':>10}{oracle_p10(base, qrels, depth):>8.4f}")
    print("* = eligible only\n")

    confusion: Counter[tuple[Label, int | None]] = Counter()
    for (t, n), record in records.items():
        label, _ = classify(record, unclear, grounded_only, scoring)
        confusion[label, qrels.get(t, {}).get(n)] += 1

    gold = (2, 1, 0, None)
    print(
        f"{'predicted':<12}"
        + "".join(f"{str(g):>10}" for g in ("eligible", "excluded", "not rel", "unjudged"))
    )
    for label in Label:
        print(f"{label.name.lower():<12}" + "".join(f"{confusion[label, g]:>10}" for g in gold))

    tp = confusion[Label.EXCLUDED, 1]
    predicted = sum(confusion[Label.EXCLUDED, g] for g in gold)
    actual = sum(confusion[label, 1] for label in Label)
    print(
        f"\nexcluded detection: precision {tp / predicted if predicted else 0:.2f}"
        f"  recall {tp / actual if actual else 0:.2f}  ({tp}/{actual} found)"
    )


if __name__ == "__main__":
    app()
