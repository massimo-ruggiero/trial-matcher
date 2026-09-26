import json
import time
from collections import Counter
from pathlib import Path

import requests
import typer
from tqdm import tqdm

from src.assess.prompt import PROMPT_VERSION, build_messages
from src.assess.schema import parse, schema
from src.config import DATA_OUT, DATA_PROCESSED, RUNS
from src.eval.run import load_topics

app = typer.Typer()

OLLAMA = "http://localhost:11434/api/chat"
MODEL = "gemma4:12b"
# Ollama silently truncates prompts beyond its context window, so it is set
# explicitly and every call is checked against it.
NUM_CTX = 16384


def verdicts_path(year: int) -> Path:
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    return DATA_OUT / f"verdicts{year}.jsonl"


def expand(spec: str) -> set[str]:
    """ "1-3,11" -> {"1", "2", "3", "11"}"""
    chosen: set[str] = set()
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            chosen.update(str(i) for i in range(int(lo), int(hi) + 1))
        elif part.strip():
            chosen.add(part.strip())
    return chosen


def load_shortlist(path: Path, depth: int) -> dict[str, list[str]]:
    shortlist: dict[str, list[str]] = {}
    with open(path) as f:
        for line in f:
            topic, _, nct_id, rank, _, _ = line.split()
            if int(rank) <= depth:
                shortlist.setdefault(topic, []).append(nct_id)
    return shortlist


def load_blocks(nct_ids: set[str]) -> dict[str, str]:
    with open(DATA_PROCESSED / "trials_judged.jsonl") as f:
        return {
            r["nct_id"]: r["criteria_text"] for r in map(json.loads, f) if r["nct_id"] in nct_ids
        }


def is_grounded(evidence: list[str], note: str) -> bool:
    """Every quote must appear in the note, whitespace and case aside."""

    def normalize(text: str) -> str:
        return " ".join(text.split()).casefold()

    haystack = normalize(note)
    quotes = [q for q in evidence if q.strip()]
    return bool(quotes) and all(normalize(q) in haystack for q in quotes)


def call(messages: list[dict], model: str) -> dict:
    response = requests.post(
        OLLAMA,
        timeout=1800,
        json={
            "model": model,
            "messages": messages,
            "format": schema(),
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "num_ctx": NUM_CTX},
        },
    )
    response.raise_for_status()
    return response.json()


def judge(note: str, block: str, nct_id: str, model: str) -> dict:
    if not block.strip():
        return {"criteria": [], "error": "no criteria"}

    started = time.time()
    reply = call(build_messages(note, block), model)
    usage = {
        "prompt_tokens": reply.get("prompt_eval_count", 0),
        "output_tokens": reply.get("eval_count", 0),
        "seconds": round(time.time() - started, 1),
    }
    try:
        rows = parse(reply["message"]["content"], nct_id)
    except json.JSONDecodeError:
        return {"criteria": [], "error": "invalid json", **usage}

    for row in rows:
        row["grounded"] = is_grounded(row["evidence"], note)
    return {"criteria": rows, "error": None, **usage}


def done_keys(path: Path) -> set[tuple[str, str, str, str]]:
    if not path.exists():
        return set()
    with open(path) as f:
        return {
            (r["topic_id"], r["nct_id"], r["model"], r["prompt_version"])
            for r in map(json.loads, f)
        }


@app.command()
def main(
    year: int = 2021,
    depth: int = 50,
    topics: str = "",
    model: str = MODEL,
    run: str = "dense",
) -> None:
    """Judge the shortlist of the given topics, e.g. --topics 1-3 or --topics 11,12."""
    notes = {str(t["topic_id"]): t["text"] for t in load_topics(year)}
    shortlist = load_shortlist(RUNS / f"{run}{year}.txt", depth)
    chosen = sorted(expand(topics) & set(shortlist) if topics else set(shortlist), key=int)

    out = verdicts_path(year)
    done = done_keys(out)
    planned = [(t, n) for t in chosen for n in shortlist[t]]
    # Cached work is dropped here rather than skipped inside the loop: a progress
    # bar that counts trials already paid for reports a finishing time that is
    # wrong exactly when it matters, which is against a session time limit.
    todo = [(t, n) for t, n in planned if (t, n, model, PROMPT_VERSION) not in done]
    cached = len(planned) - len(todo)
    blocks = load_blocks({n for _, n in todo})

    where = sorted({t for t, _ in todo}, key=int)
    print(f"{model}  prompt {PROMPT_VERSION}  topics {topics or 'all'} x top-{depth}")
    print(f"{len(planned)} shortlisted, {cached} already judged, {len(todo)} to go", end="")
    print(f", from topic {where[0]}" if where else "")

    stats: Counter[str] = Counter()
    with open(out, "a") as f:
        for topic, nct_id in tqdm(todo, unit="trial"):
            record = judge(notes[topic], blocks.get(nct_id, ""), nct_id, model)
            f.write(
                json.dumps(
                    {
                        "topic_id": topic,
                        "nct_id": nct_id,
                        "model": model,
                        "prompt_version": PROMPT_VERSION,
                        **record,
                    }
                )
                + "\n"
            )
            f.flush()

            stats["judged"] += 1
            if record["error"]:
                stats[f"error: {record['error']}"] += 1
            stats["seconds"] += record.get("seconds", 0)
            stats["criteria"] += len(record["criteria"])
            if record.get("prompt_tokens", 0) + record.get("output_tokens", 0) >= NUM_CTX:
                stats["context full"] += 1
            for row in record["criteria"]:
                stats[f"verdict: {row['verdict']}"] += 1
                stats[f"kind: {row['kind']}"] += 1
                if row["verdict"] in ("yes", "no") and not row["grounded"]:
                    stats["decisive but ungrounded"] += 1

    judged = stats["judged"]
    print(f"\njudged: {judged}   cached: {cached}   -> {out.name}")
    if judged:
        print(f"mean seconds per trial: {stats['seconds'] / judged:.1f}")
        print(f"criteria per trial:     {stats['criteria'] / judged:.1f}")
        for key in sorted(k for k in stats if k.startswith(("verdict", "kind", "error"))):
            print(f"  {key:<28} {stats[key]:>6}")
        for key in ("decisive but ungrounded", "context full"):
            print(f"  {key:<28} {stats[key]:>6}")


if __name__ == "__main__":
    app()
