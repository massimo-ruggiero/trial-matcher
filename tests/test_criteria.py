from src.ingest.criteria import parse_criteria, split
from src.models import Kind, Source

DASH = """
        Inclusion Criteria:

        Patients may be eligible for this study if they:

          -  Are at least 21 years old.

          -  Have a diagnosis of unilateral posterior semicircular canal BPPV according to
             established clinical test criteria.

        Key Exclusion Criteria:

          -  Have a history of prior ear surgery.
"""

NUMBERED = """
        1. Men and women > 45 years.

          2. Women of child-bearing potential must agree to use two forms of
             contraception, and undergo monthly pregnancy testing.
"""

PROSE = (
    "        Children between the ages of 6 and 12 years with myopia are eligible. "
    "Exclusion criteria include visual acuity greater than 20/25, strabismus, and "
    "use of contact lenses."
)


def test_continuation_lines_stay_in_one_criterion():
    incl = [c for c in split(DASH) if c[0] is Kind.INCLUSION]
    assert len(incl) == 2
    # the second bullet wraps onto a second line and must not become two criteria
    assert incl[1][1].endswith("established clinical test criteria.")


def test_lead_in_sentence_is_not_a_criterion():
    assert not any("may be eligible for this study" in text for _, text, _ in split(DASH))


def test_qualified_header_switches_polarity():
    # "Key Exclusion Criteria:" is a header, not a bullet
    excl = [c for c in split(DASH) if c[0] is Kind.EXCLUSION]
    assert len(excl) == 1
    assert excl[0][2] is Source.HEADER


def test_numbered_list_without_headers_defaults_and_says_so():
    parsed = split(NUMBERED)
    assert len(parsed) == 2
    assert all(kind is Kind.INCLUSION and src is Source.DEFAULT for kind, _, src in parsed)


def test_prose_infers_polarity_mid_sentence():
    parsed = split(PROSE)
    assert parsed[0][0] is Kind.INCLUSION
    assert parsed[-1][0] is Kind.EXCLUSION
    assert parsed[-1][2] is Source.INFERRED


def test_index_restarts_per_kind():
    criteria, _ = parse_criteria("NCT1", DASH)
    assert [c.ref for c in criteria] == ["NCT1|incl1", "NCT1|incl2", "NCT1|excl1"]


def test_empty_block_yields_nothing():
    assert split("   \n  ") == []
