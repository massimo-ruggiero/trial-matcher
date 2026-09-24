import json

from src.assess.schema import clean, parse, schema


def answer(*items: tuple[str, str, list[str], str]) -> str:
    return json.dumps(
        {
            "criteria": [
                {"criterion": c, "kind": k, "evidence": e, "verdict": v} for c, k, e, v in items
            ]
        }
    )


def test_the_model_assigns_the_polarity():
    rows = parse(
        answer(
            ("Age 18 or older", "inclusion", ["45-year-old"], "yes"),
            ("History of cancer", "exclusion", [], "unclear"),
        ),
        "NCT1",
    )
    assert [r["kind"] for r in rows] == ["inclusion", "exclusion"]
    assert [r["ref"] for r in rows] == ["NCT1|1", "NCT1|2"]
    assert rows[0]["text"] == "Age 18 or older"


def test_evidence_is_generated_before_the_verdict():
    # Generation follows field order: quoting first makes the model decide
    # after looking for evidence, not justify a verdict it already gave.
    fields = list(schema()["properties"]["criteria"]["items"]["properties"])
    assert fields.index("evidence") < fields.index("verdict")


def test_evidence_is_a_list_of_passages():
    fields = schema()["properties"]["criteria"]["items"]["properties"]
    assert fields["evidence"]["type"] == "array"


def test_only_three_verdicts_and_two_kinds_are_allowed():
    fields = schema()["properties"]["criteria"]["items"]["properties"]
    assert set(fields["verdict"]["enum"]) == {"yes", "no", "unclear"}
    assert set(fields["kind"]["enum"]) == {"inclusion", "exclusion"}


def test_the_number_of_criteria_is_not_imposed():
    # The model finds them itself, so coverage is measured afterwards against
    # the block, never forced here.
    assert "minItems" not in schema()["properties"]["criteria"]


def test_quotes_are_trimmed_and_empty_ones_dropped():
    assert clean(["  45-year-old man ", "   ", ""]) == ["45-year-old man"]


def test_reasoning_is_generated_after_the_quotes():
    # Same reason as the verdict: the model looks for a passage to copy before
    # it is allowed to write a sentence of its own.
    fields = list(schema()["properties"]["criteria"]["items"]["properties"])
    assert fields.index("evidence") < fields.index("rationale") < fields.index("verdict")


def test_the_rationale_is_carried_and_trimmed():
    raw = json.dumps(
        {
            "criteria": [
                {
                    "criterion": "Age 18 or older",
                    "kind": "inclusion",
                    "evidence": [],
                    "rationale": "  45 is over 18.  ",
                    "verdict": "yes",
                }
            ]
        }
    )
    assert parse(raw, "NCT1")[0]["rationale"] == "45 is over 18."


def test_a_missing_rationale_is_empty_not_absent():
    # Verdicts judged before the field existed are still readable.
    assert (
        parse(answer(("Age 18 or older", "inclusion", [], "unclear")), "NCT1")[0]["rationale"] == ""
    )
