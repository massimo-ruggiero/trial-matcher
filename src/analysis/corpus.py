import random
import re
import statistics
from collections import Counter

import typer
from lxml import etree
from tqdm import tqdm

from src.config import DATA_RAW

app = typer.Typer()
TREC = DATA_RAW / "trec-ct"

INCL = re.compile(r"inclusion\s+criteria", re.I)
EXCL = re.compile(r"exclusion\s+criteria", re.I)


@app.command()
def main(sample: int | None = None) -> None:
    for year in (2021, 2022):
        # ---- topics --------------------------------------------------------
        tree = etree.parse(str(TREC / f"topics{year}.xml"))
        topics = tree.xpath("//topic")
        lengths = [len(t.text.strip()) for t in topics if t.text]

        print(f"=== topics {year} ===")
        print(f"count:  {len(topics)}")
        print(
            f"length: median {statistics.median(lengths):.0f} chars, "
            f"min {min(lengths)}, max {max(lengths)}"
        )

    # ---- trials ------------------------------------------------------------
    files = list((TREC / "corpus").rglob("*.xml"))
    print(f"\n=== corpus ===\nfiles: {len(files)}")
    if not files:
        print("no XML found — check the extraction path")
        return

    if sample is None:
        picked = files
    else:
        picked = random.sample(files, min(sample, len(files)))

    has_criteria = 0
    both_sections = 0
    only_inclusion = 0
    only_exclusion = 0
    criteria_lengths = []
    bullet_style: Counter = Counter()

    for path in tqdm(picked, unit="file"):
        root = etree.parse(str(path)).getroot()

        node = root.find(".//eligibility/criteria/textblock")
        if node is None or not node.text:
            continue
        has_criteria += 1
        text = node.text
        criteria_lengths.append(len(text))

        if INCL.search(text) and EXCL.search(text):
            both_sections += 1
        elif INCL.search(text):
            only_inclusion += 1
        elif EXCL.search(text):
            only_exclusion += 1

        # Only the FIRST marker in the document is counted, so a stray "1997."
        # or a protocol code at the start of a line outranks the 40 dashes that
        # follow it. The tail of this distribution is that noise, not real styles.
        markers = re.findall(r"^\s*([-*•]|\d+[.)])\s", text, re.M)
        bullet_style[markers[0] if markers else "none"] += 1

    print("\n--- bullet markers (first one seen per trial) ---")
    top = bullet_style.most_common(6)
    tail = sum(n for _, n in bullet_style.most_common()[6:])
    total = sum(bullet_style.values())
    for marker, n in top:
        print(f"  {marker:<8} {n:>7}  ({n / total:5.1%})")
    rest = max(len(bullet_style) - 6, 0)
    print(f"  {'other':<8} {tail:>7}  ({tail / total:5.1%})  [{rest} distinct]")

    print("\n--- eligibility criteria ---")
    n = len(picked)
    print(f"with a criteria textblock:   {has_criteria}/{n} ({has_criteria / n:.1%})")
    print(f"both inclusion + exclusion:  {both_sections} ({both_sections / has_criteria:.1%})")
    print(f"inclusion only:              {only_inclusion} ({only_inclusion / has_criteria:.1%})")
    print(f"exclusion only:              {only_exclusion} ({only_exclusion / has_criteria:.1%})")
    print(f"length: median {statistics.median(criteria_lengths):.0f} chars")


if __name__ == "__main__":
    app()
