"""Streamlit front end: write a note, see the trials, verify one on demand.

Launched from gui.py at the project root: uv run streamlit run gui.py
"""

import html
import json
import re

import requests
import streamlit as st
from qdrant_client import models

from src.assess.judge import MODEL, OLLAMA, judge
from src.assess.rerank import Label, classify
from src.config import DATA_PROCESSED, DEFAULT_ENCODER, DENSE, ENCODERS, SPARSE, collection_for
from src.eval.run import load_topics
from src.models import Trial
from src.retrieve.search import Mode, Searcher

BADGE = {
    Label.ELIGIBLE: ("ELIGIBLE", "#2ea043"),
    Label.EXCLUDED: ("EXCLUDED", "#da3633"),
    Label.INELIGIBLE: ("NOT ELIGIBLE", "#d29922"),
    Label.UNJUDGED: ("not checked", "#6e7681"),
}
MARK = {"yes": "✓", "no": "✗", "unclear": "?"}
GOOD, BAD, UNKNOWN = "#2ea043", "#da3633", "#6e7681"


def consequence(kind: str, verdict: str) -> tuple[str, str]:
    """Colour by what the verdict does to the patient, not by its truth value:
    for an exclusion criterion, true is the bad news."""
    if verdict == "unclear" or verdict is None:
        return UNKNOWN, ""
    disqualifies = (kind == "exclusion") == (verdict == "yes")
    return (BAD, "excludes" if kind == "exclusion" else "not met") if disqualifies else (GOOD, "")


@st.cache_resource(show_spinner="Loading the encoder and the index...")
def get_searcher() -> Searcher:
    return Searcher.open()


@st.cache_resource(show_spinner="Loading the trials...")
def get_trials() -> dict[str, dict]:
    """Title and criteria block per trial: the payload in Qdrant is minimal."""
    with open(DATA_PROCESSED / "trials_judged.jsonl") as f:
        return {
            r["nct_id"]: {"title": r["title"], "criteria": r["criteria_text"]}
            for r in map(json.loads, f)
        }


@st.cache_data(show_spinner=False)
def get_topics(year: int) -> dict[str, str]:
    return {str(t["topic_id"]): t["text"] for t in load_topics(year)}


def ollama_is_up() -> bool:
    try:
        return requests.get(OLLAMA.replace("/api/chat", "/api/tags"), timeout=2).ok
    except requests.RequestException:
        return False


def badge(label: Label, extra: str = "") -> str:
    text, colour = BADGE[label]
    return (
        f"<span style='background:{colour};color:#0d1117;padding:2px 10px;"
        f"border-radius:10px;font-size:0.75rem;font-weight:600'>{text}</span>"
        f"<span style='color:#8b949e;font-size:0.8rem'> {extra}</span>"
    )


def highlight(note: str, quotes: list[str]) -> str:
    """The note with every verified quote marked, so the evidence is visible
    where it was found instead of only in the table."""
    out = html.escape(note)
    for quote in sorted({q for q in quotes if q.strip()}, key=len, reverse=True):
        pattern = re.escape(html.escape(quote.strip()))
        out = re.sub(
            pattern,
            lambda m: f"<mark style='background:#f2cc60;color:#0d1117'>{m.group(0)}</mark>",
            out,
            flags=re.IGNORECASE,
        )
    return out


def render_detail(record: dict, note: str) -> None:
    quotes = [q for row in record["criteria"] if row["grounded"] for q in row["evidence"]]
    st.markdown(
        f"<div style='background:#161b22;padding:12px;border-radius:8px;"
        f"font-family:monospace;font-size:0.85rem;line-height:1.6'>{highlight(note, quotes)}</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    for row in record["criteria"]:
        colour, effect = consequence(row["kind"], row["verdict"])
        mark = MARK.get(row["verdict"], "?")
        kind = "EXCL" if row["kind"] == "exclusion" else "incl"
        quote = " · ".join(row["evidence"]) if row["evidence"] else "—"
        tag = (
            f"<span style='color:{BAD};font-size:0.7rem;font-weight:600'>{effect}</span>"
            if effect
            else ""
        )
        st.markdown(
            f"<div style='display:flex;gap:10px;padding:3px 0;font-size:0.85rem'>"
            f"<span style='color:{colour};font-weight:700'>{mark}</span>"
            f"<span style='color:#8b949e;font-family:monospace;min-width:42px'>{kind}</span>"
            f"<span style='flex:1;color:{colour if effect else '#e6edf3'}'>"
            f"{html.escape(row['text'])} {tag}</span>"
            f"<span style='flex:1;color:#8b949e;font-style:italic'>{html.escape(quote)}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    unclear = sum(1 for r in record["criteria"] if r["verdict"] == "unclear")
    st.caption(
        f"{len(record['criteria'])} criteria · {unclear} not addressed by the note "
        f"· {record.get('seconds', 0)}s"
    )


def patient_tab(searcher: Searcher, trials: dict, settings: dict) -> None:
    topics = get_topics(2021)
    # A form so the note and the button commit together: outside one, the first
    # click only confirms the text and a second is needed to search.
    with st.form("search", border=False):
        col_note, col_pick = st.columns([4, 1])
        with col_pick:
            chosen = st.selectbox("load a topic", ["—", *topics], label_visibility="collapsed")
        with col_note:
            note = st.text_area(
                "Admission note",
                value=topics.get(chosen, ""),
                height=140,
                placeholder="Paste or write the patient's note...",
            )
        searched = st.form_submit_button("Search trials", type="primary")

    if searched and note.strip():
        with st.spinner("searching..."):
            st.session_state.hits = searcher.search(note, Mode.DENSE, limit=settings["depth"])
        st.session_state.note = note
        st.session_state.verdicts = {}

    hits = st.session_state.get("hits", [])
    if not hits:
        return
    st.caption(f"{len(hits)} trials · encoder {DEFAULT_ENCODER}")

    for rank, (nct_id, score) in enumerate(hits, 1):
        trial = trials.get(nct_id, {"title": "(not in the corpus)", "criteria": ""})
        record = st.session_state.verdicts.get(nct_id)
        label, points = classify(record)
        extra = f"{points:.2f}" if record and label is Label.ELIGIBLE else ""

        with st.container(border=True):
            head, action = st.columns([5, 1])
            with head:
                st.markdown(
                    f"<span style='color:#8b949e'>{rank}</span> "
                    f"<code>{nct_id}</code> &nbsp; {badge(label, extra)}<br>"
                    f"<span style='font-size:0.95rem'>{html.escape(trial['title'])}</span>",
                    unsafe_allow_html=True,
                )
            with action:
                if record is None:
                    if st.button("verify", key=f"go{nct_id}", disabled=not settings["ollama"]):
                        with st.spinner("the model is reading the criteria..."):
                            st.session_state.verdicts[nct_id] = judge(
                                st.session_state.note, trial["criteria"], nct_id, MODEL
                            )
                        st.rerun()
            if record is not None:
                if record["error"]:
                    st.warning(record["error"])
                else:
                    with st.expander("criteria", expanded=True):
                        render_detail(record, st.session_state.note)


def new_trial_tab(searcher: Searcher, trials: dict) -> None:
    with st.form("new_trial"):
        nct_id = st.text_input("NCT id", value="NCT99999999")
        title = st.text_input("Title")
        conditions = st.text_input("Conditions (comma separated)")
        summary = st.text_area("Summary", height=100)
        criteria = st.text_area(
            "Eligibility criteria",
            height=200,
            placeholder="Inclusion Criteria:\n  - ...\n\nExclusion Criteria:\n  - ...",
        )
        submitted = st.form_submit_button("Save and index", type="primary")

    if not submitted:
        return
    trial = Trial(
        nct_id=nct_id,
        title=title,
        summary=summary,
        conditions=tuple(c.strip() for c in conditions.split(",") if c.strip()),
        criteria_text=criteria,
    )
    document = trial.to_document()
    if not document.strip():
        st.error("Title, conditions or summary are needed: that is what gets indexed.")
        return

    with st.spinner("embedding and indexing..."):
        dense = searcher.dense_model.encode(document, normalize_embeddings=True)
        sparse = next(iter(searcher.sparse_model.embed([document])))
        searcher.client.upsert(
            searcher.collection,
            points=[
                models.PointStruct(
                    id=trial.point_id,
                    vector={
                        DENSE: dense.tolist(),
                        SPARSE: models.SparseVector(
                            indices=sparse.indices.tolist(), values=sparse.values.tolist()
                        ),
                    },
                    payload={"nct_id": trial.nct_id, "title": trial.title},
                )
            ],
        )
    trials[nct_id] = {"title": title, "criteria": criteria}
    st.success(f"{nct_id} indexed. Indexed text: {document[:120]}...")


def main() -> None:
    st.set_page_config(page_title="Trial Matcher", layout="wide")
    searcher, trials = get_searcher(), get_trials()
    online = ollama_is_up()

    with st.sidebar:
        st.markdown("### Trial Matcher")
        st.caption(f"Encoder · `{ENCODERS[DEFAULT_ENCODER].model.split('/')[-1]}`")
        dot = "#2ea043" if online else "#da3633"
        st.markdown(
            f"<span style='color:#8b949e'>Judge · <code>{MODEL}</code></span> "
            f"<span style='color:{dot}'>●</span>",
            unsafe_allow_html=True,
        )
        st.caption(f"Index · {searcher.client.count(collection_for(DEFAULT_ENCODER)).count} trials")
        st.divider()
        st.caption("Results")
        settings = {
            "depth": st.segmented_control(
                "Results", [5, 10, 20], default=10, label_visibility="collapsed"
            )
            or 10,
            "ollama": online,
        }
        if not online:
            st.warning("Ollama is not responding: verification is disabled.")

    patient, new_trial = st.tabs(["Patient", "New trial"])
    with patient:
        patient_tab(searcher, trials, settings)
    with new_trial:
        new_trial_tab(searcher, trials)


if __name__ == "__main__":
    main()
