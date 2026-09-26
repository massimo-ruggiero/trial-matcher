"""Build the input pack a judge run needs on a machine that has neither the
corpus nor the index: the topics, the shortlist, and the criteria blocks of the
trials in it. Everything else stays here."""

import json
import shutil
from collections import Counter
from pathlib import Path

import typer

from src.assess.judge import expand, load_shortlist, verdicts_path
from src.assess.prompt import PROMPT_VERSION
from src.config import DATA_PROCESSED, ROOT, RUNS

app = typer.Typer()


def megabytes(path: Path) -> str:
    return f"{path.stat().st_size / 1e6:6.1f} MB  {path.relative_to(path.parents[1])}"


@app.command()
def main(
    out: Path = Path("kaggle/pack"),
    year: int = 2021,
    run: str = "dense_medembed-small",
    # Topic 1 is a tuning topic and rides along for smoke tests: a run that gets
    # inspected line by line must not be one we later report numbers on.
    topics: str = "1,12-43",
    depth: int = 50,
    archive: bool = True,
) -> None:
    """Write kaggle/pack, to be uploaded as a Kaggle dataset.

    File names are the ones the code already expects, so a judge run needs no
    change beyond pointing DATA_PROCESSED and RUNS at the pack.
    """
    shortlist = load_shortlist(RUNS / f"{run}{year}.txt", depth)
    chosen = sorted(expand(topics) & set(shortlist), key=int)
    if not chosen:
        raise typer.BadParameter(f"no topic of {topics} is in {run}{year}.txt")
    wanted = {n for t in chosen for n in shortlist[t]}

    processed, runs = out / "processed", out / "runs"
    processed.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)

    # The shortlist, cut to the topics and the depth that will be judged: the
    # pack must not carry a deeper list than the run it is built for.
    with open(RUNS / f"{run}{year}.txt") as f, open(runs / f"{run}{year}.txt", "w") as g:
        kept = 0
        for line in f:
            topic, _, _, rank, _, _ = line.split()
            if topic in set(chosen) and int(rank) <= depth:
                g.write(line)
                kept += 1

    # Only the criteria blocks of the shortlisted trials: the full file is 239 MB
    # and 47,000 of its trials would never be read.
    with (
        open(DATA_PROCESSED / "trials_judged.jsonl") as f,
        open(processed / "trials_judged.jsonl", "w") as g,
    ):
        found = 0
        for line in f:
            row = json.loads(line)
            if row["nct_id"] in wanted:
                g.write(line)
                found += 1

    # The code travels with the data: this repository is private, and a notebook
    # has no credential to clone it with. 74 KB of Python.
    shutil.copytree(
        ROOT / "src",
        out / "src",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    topics_file = processed / f"topics{year}.jsonl"
    topics_file.write_text((DATA_PROCESSED / f"topics{year}.jsonl").read_text())

    # The verdicts travel too, so the pack is the whole state of the experiment
    # and not just its inputs: a machine that receives it resumes from what has
    # already been judged, wherever it was judged, instead of paying twice.
    verdicts = verdicts_path(year)
    if verdicts.exists():
        shutil.copy(verdicts, processed / verdicts.name)

    print(f"topics: {len(chosen)} ({chosen[0]}-{chosen[-1]})   depth: {depth}")
    print(f"shortlisted trials: {len(wanted)}   blocks found: {found}")
    if found < len(wanted):
        print(f"MISSING {len(wanted) - found} blocks: those trials will be judged empty")
    print(f"judge calls per model: {kept}")

    if verdicts.exists():
        rows = [json.loads(line) for line in open(verdicts)]
        current = [r for r in rows if r["prompt_version"] == PROMPT_VERSION]
        per_topic: dict[str, Counter] = {}
        for r in current:
            per_topic.setdefault(r["model"], Counter())[r["topic_id"]] += 1
        print(f"\nverdicts carried: {len(current)} on prompt {PROMPT_VERSION}")
        for model, counts in sorted(per_topic.items()):
            whole = sorted((t for t, n in counts.items() if n >= depth), key=int)
            done = f"{len(whole)} topics complete" if whole else "no topic complete"
            print(f"  {model:<16} {sum(counts.values()):>5} trials, {done}")
    print()
    for path in sorted(out.rglob("*.jsonl")) + sorted(out.rglob("*.txt")):
        print(megabytes(path))
    print(f"{sum(1 for _ in (out / 'src').rglob('*.py')):>9} .py  src/")

    if archive:
        # One file to drag into Kaggle's dataset form, which unzips it and
        # keeps the folders.
        zipped = Path(shutil.make_archive(str(out.parent / out.name), "zip", root_dir=out))
        print(f"\n-> {zipped}  ({zipped.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"\n-> {out}")


if __name__ == "__main__":
    app()
