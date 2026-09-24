from src.assess.rerank import Label, classify, rerank


def row(kind: str, verdict: str | None, grounded: bool = True) -> dict:
    return {"kind": kind, "verdict": verdict, "grounded": grounded}


def record(*rows: dict) -> dict:
    return {"criteria": list(rows), "error": None}


def test_an_exclusion_that_holds_excludes():
    assert classify(record(row("inclusion", "yes"), row("exclusion", "yes")))[0] is Label.EXCLUDED


def test_an_inclusion_that_fails_makes_ineligible():
    label, _ = classify(record(row("inclusion", "no"), row("exclusion", "no")))
    assert label is Label.INELIGIBLE


def test_exclusion_wins_over_failed_inclusion():
    assert classify(record(row("inclusion", "no"), row("exclusion", "yes")))[0] is Label.EXCLUDED


def test_ungrounded_verdict_counts_as_unclear_only_when_asked():
    # Seen in calibration: "no" to an inclusion criterion with an empty quote.
    # By default it is trusted; --grounded-only discards it instead.
    r = record(row("inclusion", "no", grounded=False), row("inclusion", "yes"))
    assert classify(r)[0] is Label.INELIGIBLE
    assert classify(r, grounded_only=True)[0] is Label.ELIGIBLE


def test_missing_verdict_counts_as_unclear():
    assert classify(record(row("inclusion", None, grounded=False)))[0] is Label.ELIGIBLE


def test_unclear_weight_sets_the_score():
    r = record(row("inclusion", "yes"), row("inclusion", "unclear"))
    assert classify(r, unclear=0.5, scoring="fraction")[1] == 0.75
    assert classify(r, unclear=0.0, scoring="fraction")[1] == 0.5


def test_failed_or_missing_record_is_unjudged():
    assert classify(None)[0] is Label.UNJUDGED
    assert classify({"criteria": [], "error": "invalid json"})[0] is Label.UNJUDGED


def test_rerank_orders_by_tier_then_score_then_original_rank():
    records = {
        "a": record(row("exclusion", "yes")),
        "b": record(row("inclusion", "unclear")),
        "c": record(row("inclusion", "yes")),
        "d": record(row("inclusion", "no")),
        "f": record(row("inclusion", "unclear")),
    }
    ranked = ["a", "b", "c", "d", "e", "f", "tail"]
    assert rerank(ranked, records, depth=6) == ["c", "b", "f", "e", "d", "a", "tail"]


def test_scoring_modes_order_eligible_trials_differently():
    few = record(row("inclusion", "yes"), row("inclusion", "yes"))
    many = record(*([row("inclusion", "yes")] * 8), *([row("inclusion", "unclear")] * 2))
    # fraction favours the small trial, count favours the one with more proof
    assert classify(few, scoring="fraction")[1] > classify(many, scoring="fraction")[1]
    assert classify(many, scoring="count")[1] > classify(few, scoring="count")[1]


def test_retrieval_scoring_leaves_the_search_order_alone():
    records = {
        "a": record(row("inclusion", "unclear")),
        "b": record(row("inclusion", "yes")),
        "c": record(row("exclusion", "yes")),
    }
    # the excluded trial still goes down; the eligible ones keep their order
    assert rerank(["c", "b", "a"], records, depth=3, scoring="retrieval") == ["b", "a", "c"]


def test_smoothed_score_stays_between_zero_and_one():
    none_met = record(*([row("inclusion", "unclear")] * 4))
    all_met = record(*([row("inclusion", "yes")] * 4))
    assert 0.0 < classify(none_met)[1] < classify(all_met)[1] < 1.0


def test_smoothing_stops_a_tiny_trial_from_outranking_a_thorough_one():
    # 2 criteria met out of 2 is thinner evidence than 9 out of 10.
    tiny = record(*([row("inclusion", "yes")] * 2))
    thorough = record(*([row("inclusion", "yes")] * 9), row("inclusion", "unclear"))
    assert classify(thorough)[1] > classify(tiny)[1]
    assert classify(tiny, scoring="fraction")[1] > classify(thorough, scoring="fraction")[1]


def test_defaults_are_the_tuned_ones():
    # Chosen on the tuning topics; changing them changes every reported number.
    import inspect

    defaults = {name: p.default for name, p in inspect.signature(classify).parameters.items()}
    assert defaults["scoring"] == "smoothed"
    assert defaults["grounded_only"] is False
    assert defaults["unclear"] == 0.5
