import json
import re
from collections import Counter

import typer
from tqdm import tqdm

from src.config import DATA_PROCESSED
from src.models import Criterion, Kind, Source

app = typer.Typer()

MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
HEADER = re.compile(r"^\W*(?:\w+\s+){0,2}(inclusion|exclusion)\s+criteria\b", re.I)
SENTENCE = re.compile(r"(?<=[.;])\s+(?=[A-Z0-9])")
EXCLUSION_CUE = re.compile(
    r"\b(?:exclusion|excluded|exclude|ineligible|not\s+be\s+eligible"
    r"|must\s+not|may\s+not|will\s+not|cannot)\b",
    re.I,
)

# Shorter than this and it is a stray bullet or a section number, not a criterion.
MIN_LENGTH = 8


def section(line: str) -> str | None:
    """The section a line declares, if it is a header at all.

    Headers carry qualifiers ("Key Exclusion Criteria:") and sometimes a bullet
    of their own ("-  Exclusion Criteria:"), so the marker is stripped first. A
    combined header ("Inclusion and Exclusion Criteria:") declares no polarity
    and returns "both" rather than silently picking the first word it matches.
    """
    body = MARKER.sub("", line).strip()
    if not body or len(body) > 60:
        return None
    match = HEADER.match(body)
    if not match:
        return None
    lowered = body.lower()
    if "inclusion" in lowered and "exclusion" in lowered:
        return "both"
    return match.group(1).lower()


def _from_markers(lines: list[str]) -> list[tuple[Kind, str, Source]]:
    """A criterion runs from its marker to the next marker or header.

    Continuation lines are indented under their bullet and carry no marker of
    their own, so splitting on newlines would cut a single criterion into five.
    Text sitting between a header and the first bullet is a lead-in sentence
    ("Patients may be eligible for this study if they:") and is dropped.
    """
    out: list[tuple[Kind, str, Source]] = []
    kind, source = Kind.INCLUSION, Source.DEFAULT
    current: list[str] = []
    in_item = False

    def flush() -> None:
        if current:
            body = " ".join(current).strip()
            if body:
                out.append((kind, body, source))
            current.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        declared = section(line)
        if declared:
            flush()
            in_item = False
            if declared != "both":
                kind = Kind.INCLUSION if declared == "inclusion" else Kind.EXCLUSION
                source = Source.HEADER
            continue

        marker = MARKER.match(line)
        if marker:
            flush()
            current.append(line[marker.end() :].strip())
            in_item = True
        elif in_item:
            current.append(stripped)

    flush()
    return out


def _from_prose(text: str) -> list[tuple[Kind, str, Source]]:
    """No markers anywhere: fall back to sentences and infer polarity.

    In prose the switch happens mid-sentence ("Exclusion criteria include...")
    rather than on a header line, and everything after it stays excluded, which
    is the order these blocks are written in.
    """
    out: list[tuple[Kind, str, Source]] = []
    kind, source = Kind.INCLUSION, Source.DEFAULT
    for sentence in SENTENCE.split(" ".join(text.split())):
        sentence = sentence.strip()
        if not sentence:
            continue
        if EXCLUSION_CUE.search(sentence):
            kind, source = Kind.EXCLUSION, Source.INFERRED
        out.append((kind, sentence, source))
    return out


def split(text: str) -> list[tuple[Kind, str, Source]]:
    """Criteria block to (kind, text, source) triples. Pure: no I/O."""
    if not text.strip():
        return []
    lines = text.splitlines()
    if any(MARKER.match(line) for line in lines):
        return _from_markers(lines)
    return _from_prose(text)


def parse_criteria(nct_id: str, text: str) -> tuple[list[Criterion], int]:
    """Criteria for one trial, plus how many fragments were too short to keep."""
    criteria, dropped = [], 0
    index: Counter[Kind] = Counter()
    for kind, body, source in split(text):
        if len(body) < MIN_LENGTH:
            dropped += 1
            continue
        index[kind] += 1
        criteria.append(
            Criterion(nct_id=nct_id, kind=kind, text=body, index=index[kind], source=source)
        )
    return criteria, dropped


@app.command()
def main(source: str = "trials_judged.jsonl", out: str = "criteria.jsonl") -> None:
    src = DATA_PROCESSED / source
    dst = DATA_PROCESSED / out

    trials = written = dropped = no_criteria = 0
    by_source: Counter[str] = Counter()
    by_kind: Counter[str] = Counter()

    with open(src) as f, open(dst, "w") as g:
        for line in tqdm(f, unit="trial"):
            trial = json.loads(line)
            trials += 1
            criteria, short = parse_criteria(trial["nct_id"], trial["criteria_text"])
            dropped += short
            if not criteria:
                no_criteria += 1
                continue
            for criterion in criteria:
                g.write(json.dumps(criterion.to_dict()) + "\n")
                by_source[criterion.source.value] += 1
                by_kind[criterion.kind.value] += 1
                written += 1

    print(f"\ntrials read:      {trials}")
    print(f"criteria written: {written}   ({written / trials:.1f} per trial)")
    print(f"trials yielding nothing: {no_criteria} ({no_criteria / trials:.1%})")
    print(f"fragments under {MIN_LENGTH} chars, discarded: {dropped}")
    print("\nkind:")
    for kind, n in by_kind.most_common():
        print(f"  {kind:<12} {n:>8}  ({n / written:5.1%})")
    print("\nhow the kind was decided:")
    for name, n in by_source.most_common():
        print(f"  {name:<12} {n:>8}  ({n / written:5.1%})")
    print(f"\n-> {dst.name}")


if __name__ == "__main__":
    app()
