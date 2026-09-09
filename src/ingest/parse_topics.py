import json
from pathlib import Path

from lxml import etree

from src.config import DATA_PROCESSED, DATA_RAW
from src.models import Topic


def parse(path: Path) -> list[Topic]:
    root = etree.parse(str(path)).getroot()
    return [
        Topic(topic_id=int(node.get("number")), text=" ".join(node.text.split()))
        for node in root.iter("topic")
        if node.text and node.text.strip()
    ]


def main() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    for year in (2021, 2022):
        topics = parse(DATA_RAW / "trec-ct" / f"topics{year}.xml")
        out = DATA_PROCESSED / f"topics{year}.jsonl"

        with open(out, "w") as f:
            for t in topics:
                f.write(json.dumps(t.to_dict()) + "\n")

        print(f"{len(topics)} topics -> {out.name}")


if __name__ == "__main__":
    main()
