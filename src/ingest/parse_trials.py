import json
from pathlib import Path

import typer
from lxml import etree
from tqdm import tqdm

from src.config import DATA_PROCESSED, DATA_RAW
from src.models import Trial

app = typer.Typer()

TREC = DATA_RAW / "trec-ct"
CORPUS = TREC / "corpus"


def text_at(root, path: str) -> str:
    value = root.findtext(path)
    return " ".join(value.split()) if value else ""


def parse_trial(path: Path) -> Trial | None:
    try:
        root = etree.parse(str(path)).getroot()
    except etree.XMLSyntaxError:
        return None

    nct_id = text_at(root, ".//id_info/nct_id")
    if not nct_id:
        return None

    criteria = root.findtext(".//eligibility/criteria/textblock") or ""

    return Trial(
        nct_id=nct_id,
        title=text_at(root, ".//brief_title") or text_at(root, ".//official_title"),
        summary=text_at(root, ".//brief_summary/textblock"),
        detailed=text_at(root, ".//detailed_description/textblock"),
        conditions=tuple(" ".join(c.text.split()) for c in root.iter("condition") if c.text),
        criteria_text=criteria.strip("\n"),
        min_age=text_at(root, ".//eligibility/minimum_age"),
        max_age=text_at(root, ".//eligibility/maximum_age"),
        gender=text_at(root, ".//eligibility/gender"),
    )


def judged_ids() -> set[str]:
    ids = set()
    for path in TREC.glob("qrels*.txt"):
        for line in path.read_text().splitlines():
            if line.strip():
                ids.add(line.split()[2])
    return ids


@app.command()
def main(only_judged: bool = True) -> None:
    files = sorted(CORPUS.rglob("*.xml"))

    keep = judged_ids() if only_judged else None
    if keep:
        print(f"keeping only the {len(keep)} judged trials")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = DATA_PROCESSED / ("trials_judged.jsonl" if only_judged else "trials.jsonl")

    written = unparseable = 0
    with open(out, "w") as f:
        for path in tqdm(files, unit="file"):
            trial = parse_trial(path)
            if trial is None:
                unparseable += 1
                continue
            if keep is not None and trial.nct_id not in keep:
                continue
            f.write(json.dumps(trial.to_dict()) + "\n")
            written += 1

    print(f"\nwritten: {written}   unparseable: {unparseable}   -> {out.name}")


if __name__ == "__main__":
    app()
