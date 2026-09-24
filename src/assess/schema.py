import json
from enum import StrEnum


class Verdict(StrEnum):
    YES = "yes"
    NO = "no"
    UNCLEAR = "unclear"


def schema() -> dict:
    """The model reads the criteria block and finds the criteria itself, so
    neither their number nor their polarity can be imposed here. Evidence is a
    list because one criterion can be settled by two separate passages."""
    return {
        "type": "object",
        "properties": {
            "criteria": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "criterion": {"type": "string"},
                        "kind": {"type": "string", "enum": ["inclusion", "exclusion"]},
                        "evidence": {"type": "array", "items": {"type": "string"}},
                        "verdict": {"type": "string", "enum": [v.value for v in Verdict]},
                    },
                    "required": ["criterion", "kind", "evidence", "verdict"],
                },
            }
        },
        "required": ["criteria"],
    }


def clean(evidence: list[str]) -> list[str]:
    return [q.strip() for q in evidence if q.strip()]


def parse(raw: str, nct_id: str) -> list[dict]:
    return [
        {
            "ref": f"{nct_id}|{i}",
            "text": item["criterion"],
            "kind": item["kind"],
            "verdict": Verdict(item["verdict"]).value,
            "evidence": clean(item["evidence"]),
        }
        for i, item in enumerate(json.loads(raw)["criteria"], 1)
    ]
