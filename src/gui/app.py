"""Streamlit front end: write a note, see the trials, verify one on demand.

Launched from gui.py at the project root: uv run streamlit run gui.py
"""

import html
import json
import re

import requests
import streamlit as st
from qdrant_client import models

from src.assess.judge import MODEL, OLLAMA, is_grounded, judge
from src.assess.rerank import Label, classify
from src.config import DATA_PROCESSED, DEFAULT_ENCODER, DENSE, ENCODERS, SPARSE
from src.eval.run import load_topics
from src.ingest.criteria import parse_criteria
from src.models import Mode, Trial
from src.retrieve.search import Searcher

BADGE = {
    Label.ELIGIBLE: ("ELIGIBLE", "#2ea043"),
    Label.EXCLUDED: ("EXCLUDED", "#da3633"),
    Label.INELIGIBLE: ("NOT ELIGIBLE", "#d29922"),
    Label.UNJUDGED: ("not checked", "#6e7681"),
}
MARK = {"yes": "✓", "no": "✗", "unclear": "?"}
DEPTH = 10
GOOD, BAD, UNKNOWN = "#2ea043", "#da3633", "#6e7681"
AMBER = "#d29922"


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
        f"<span style='color:#8b949e;font-size:0.8rem;margin-left:10px'>{extra}</span>"
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


def panel(content: str) -> None:
    """The one text panel of the app: the note with its highlights after a
    verification, the raw criteria block before it. Same box either way."""
    st.markdown(
        f"<div style='background:#161b22;padding:12px;border-radius:8px;white-space:pre-wrap;"
        f"font-family:monospace;font-size:0.85rem;line-height:1.6'>"
        f"{content.replace(chr(10), '<br>')}</div>",
        unsafe_allow_html=True,
    )


def preview(nct_id: str, criteria_text: str) -> dict:
    """The criteria block split into its criteria, shaped like a verdict record
    but with nothing decided: the expander then looks the same before and after
    the verification. The split is our own, so the list can differ by a line or
    two from the one the model will produce."""
    criteria, _ = parse_criteria(nct_id, criteria_text)
    return {
        "criteria": [
            {
                "text": c.text,
                "kind": c.kind.value,
                "verdict": None,
                "evidence": [],
                "rationale": "",
            }
            for c in criteria
        ]
    }


def support(row: dict, note: str) -> str:
    """What backs the verdict, quotation kept apart from reasoning: a passage
    the note really contains, a passage it does not, and the model's own
    sentence are three different things for whoever has to trust the answer."""
    parts = []
    for quote in row["evidence"]:
        missing = (
            ""
            if is_grounded([quote], note)
            else f"<span style='color:{AMBER}'> (not in the note)</span>"
        )
        parts.append(
            f"<span style='color:#8b949e;font-style:italic'>{html.escape(quote)}</span>{missing}"
        )
    if row.get("rationale"):
        parts.append(
            f"<span style='color:{AMBER};font-size:0.7rem;font-weight:600'>WHY </span>"
            f"<span style='color:#8b949e'>{html.escape(row['rationale'])}</span>"
        )
    if parts:
        return "<br>".join(parts)
    if row["verdict"] is None:
        return "—"
    return f"<span style='color:{UNKNOWN}'>the note does not settle it</span>"


def render_detail(record: dict, note: str) -> None:
    quotes = [q for row in record["criteria"] for q in row["evidence"]]
    panel(highlight(note, quotes))
    st.write("")
    # Grouped under one heading per kind: the criteria of a trial come in two
    # blocks anyway, and reading "incl" on every line says nothing new.
    for heading, kind, colour in (
        ("INCLUSION", "inclusion", GOOD),
        ("EXCLUSION", "exclusion", BAD),
    ):
        rows = [row for row in record["criteria"] if row["kind"] == kind]
        if not rows:
            continue
        st.markdown(
            f"<div style='color:{colour};font-weight:700;font-size:0.75rem;"
            f"letter-spacing:0.08em;margin:10px 0 2px'>{heading}</div>",
            unsafe_allow_html=True,
        )
        for row in rows:
            row_colour, effect = consequence(row["kind"], row["verdict"])
            mark = MARK.get(row["verdict"], "-")
            quote = support(row, note)
            tag = (
                f"<span style='color:{BAD};font-size:0.7rem;font-weight:600'>{effect}</span>"
                if effect
                else ""
            )
            st.markdown(
                f"<div style='display:flex;gap:10px;padding:3px 0 3px 10px;font-size:0.85rem'>"
                f"<span style='color:{row_colour};font-weight:700'>{mark}</span>"
                f"<span style='flex:1;color:{row_colour if effect else '#e6edf3'}'>"
                f"{html.escape(row['text'])} {tag}</span>"
                f"<span style='flex:1;font-size:0.8rem'>{quote}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
    total = len(record["criteria"])
    if not any(row["verdict"] for row in record["criteria"]):
        st.caption(f"{total} criteria")
        return
    unclear = sum(1 for r in record["criteria"] if r["verdict"] == "unclear")
    st.caption(
        f"{total} criteria · {unclear} not addressed by the note · {record.get('seconds', 0)}s"
    )


def judge_one(nct_id: str, trial: dict) -> None:
    st.session_state.verdicts[nct_id] = judge(
        st.session_state.note, trial["criteria"], nct_id, MODEL
    )


def patient_tab(searcher: Searcher, trials: dict, online: bool) -> None:
    topics = get_topics(2021)
    # The batch is a queue judged one trial per script run, not a loop: a loop
    # would hold the page on the frame it started from, hiding every card until
    # the last trial came back.
    queue = st.session_state.get("queue", [])
    busy = bool(queue)

    # A form so the note and the button commit together: outside one, the first
    # click only confirms the text and a second is needed to search.
    with st.form("search", border=False):
        col_note, col_pick = st.columns([4, 1])
        with col_pick:
            # hidden, not collapsed: the empty label keeps its line, so the box
            # starts at the same height as the note beside it.
            chosen = st.selectbox("load a topic", ["—", *topics], label_visibility="hidden")
        with col_note:
            note = st.text_area(
                "Admission note",
                value=topics.get(chosen, ""),
                height=140,
                placeholder="Paste or write the patient's note...",
            )
        # A form hands its text over only on submit, so this button cannot grey
        # itself out as the note is typed: what it can do is refuse the work.
        searched = st.form_submit_button("Search trials", type="primary", disabled=busy)

    if searched and not note.strip():
        st.warning("Write a note first.")
    # Searching the same note again would return the same trials and throw away
    # every verdict already paid for.
    elif searched and note.strip() != st.session_state.get("note", "").strip():
        with st.spinner("searching..."):
            st.session_state.hits = searcher.search(note, Mode.DENSE, limit=DEPTH)
        st.session_state.note = note
        st.session_state.verdicts = {}

    hits = st.session_state.get("hits", [])
    if not hits:
        return

    pending = [nct_id for nct_id, _ in hits if nct_id not in st.session_state.verdicts]
    if pending and st.button(f"Verify all ({len(pending)})", disabled=busy or not online):
        st.session_state.queue = pending
        st.session_state.batch = len(pending)
        st.rerun()
    if busy:
        done = st.session_state.batch - len(queue)
        st.progress(done / st.session_state.batch, f"{done} of {st.session_state.batch} verified")

    running = None
    for rank, (nct_id, score) in enumerate(hits, 1):
        trial = trials.get(nct_id, {"title": "(not in the corpus)", "criteria": ""})
        record = st.session_state.verdicts.get(nct_id)
        label, points = classify(record)
        extra = f"{points:.2f}" if record and label is Label.ELIGIBLE else ""

        with st.container(border=True):
            head, action = st.columns([6, 1])
            with head:
                st.markdown(
                    f"<span style='color:#8b949e'>{rank}</span> "
                    f"<code>{nct_id}</code> &nbsp; {badge(label, extra)}<br>"
                    f"<span style='font-size:0.95rem'>{html.escape(trial['title'])}</span>",
                    unsafe_allow_html=True,
                )
            with action:
                if record is not None:
                    pass
                elif busy and nct_id == queue[0]:
                    # Kept empty for now and filled with the spinner once the
                    # whole list is on screen.
                    running = st.empty()
                else:
                    # One slot: the spinner takes the button's place instead of
                    # pushing a second widget into a narrow column.
                    slot = st.empty()
                    if slot.button("Verify", key=f"go{nct_id}", disabled=busy or not online):
                        with slot.container(), st.spinner(""):
                            judge_one(nct_id, trial)
                        st.rerun()
            if record is None:
                # The criteria as they are written, so they can be read by hand
                # before spending a minute of model time on them.
                with st.expander("criteria"):
                    render_detail(preview(nct_id, trial["criteria"]), st.session_state.note)
            elif record["error"]:
                st.warning(record["error"])
            else:
                with st.expander("criteria", expanded=True):
                    render_detail(record, st.session_state.note)

    if busy:
        nct_id = queue[0]
        with (running or st.empty()).container(), st.spinner(""):
            judge_one(nct_id, trials.get(nct_id, {"criteria": ""}))
        st.session_state.queue = queue[1:]
        st.rerun()


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
        if not online:
            st.warning("Ollama is not responding: verification is disabled.")

    patient, new_trial = st.tabs(["Patient", "New trial"])
    with patient:
        patient_tab(searcher, trials, online)
    with new_trial:
        new_trial_tab(searcher, trials)


if __name__ == "__main__":
    main()
