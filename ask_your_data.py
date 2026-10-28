"""Streamlit UI for the "Ask Your Data" tab.

Understanding and calculation happen in ``askyourdata.question_router`` and,
for follow-up questions, ``askyourdata.conversation``; this file only draws
the conversation, the question box, example buttons and answers.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from askyourdata import exports, ui_state
from askyourdata.conversation import MAX_TURNS, Conversation, suggest_followups
from askyourdata.question_router import (
    ANSWER,
    CLARIFY,
    ERROR,
    MAX_QUESTION_LENGTH,
    UNSUPPORTED,
    Answer,
    suggest_questions,
)

QUESTION_KEY = "ask_question"
PENDING_KEY = "ask_pending"
HISTORY_KEY = "ask_history"  # newest first: [{"id", "question", "answer", "follow_up"}]
DATASET_KEY = "ask_dataset"
COUNTER_KEY = "ask_counter"
CONVERSATION_KEY = "ask_conversation"
MAX_HISTORY = MAX_TURNS
VISIBLE_EXAMPLES = 6


def _ask_example(question: str) -> None:
    # Button callback: the question is answered on the next redraw; the question box
    # stays empty, ready for a follow-up.
    st.session_state[PENDING_KEY] = question


def _submit_question() -> None:
    # "Ask" callback: hand the typed question over and empty the box before the redraw,
    # so it is ready for a follow-up.
    st.session_state[PENDING_KEY] = st.session_state.get(QUESTION_KEY, "")
    st.session_state[QUESTION_KEY] = ""


def _clear_history() -> None:
    st.session_state[HISTORY_KEY] = []
    conversation = st.session_state.get(CONVERSATION_KEY)
    if conversation is not None:
        conversation.clear()


def _conversation(fingerprint: str | None) -> Conversation:
    conversation = st.session_state.get(CONVERSATION_KEY)
    if not isinstance(conversation, Conversation):
        conversation = st.session_state[CONVERSATION_KEY] = Conversation(fingerprint)
    if conversation.reset_if_dataset_changed(fingerprint):
        st.session_state[HISTORY_KEY] = []
    return conversation


def _ask(df: pd.DataFrame, conversation: Conversation, question: str) -> None:
    try:
        turn = conversation.ask(df, question)
        answer, follow_up = turn.answer, turn.follow_up
    except Exception:  # the conversation never raises, but never show a traceback
        answer = Answer(ERROR, "Something went wrong while answering. Please try rephrasing.")
        follow_up = None
    st.session_state[COUNTER_KEY] = st.session_state.get(COUNTER_KEY, 0) + 1
    history = st.session_state.setdefault(HISTORY_KEY, [])
    history.insert(0, {
        "id": st.session_state[COUNTER_KEY], "question": question, "answer": answer, "follow_up": follow_up,
    })  # fmt: skip
    del history[MAX_HISTORY:]


def show_downloads(entry: dict, answer: Answer, has_table: bool) -> None:
    """Download buttons for exactly the table and chart shown for this answer."""
    left, right = st.columns(2)
    if has_table:
        left.download_button(
            "⬇️ Download table (CSV)",
            exports.table_to_csv(answer.table),
            file_name=exports.export_filename(entry["question"], "csv"),
            mime="text/csv",
            key=f"ask_csv_{entry['id']}",
        )
    if answer.figure is not None:
        right.download_button(
            "⬇️ Download chart (interactive HTML)",
            exports.figure_to_html(answer.figure),
            file_name=exports.export_filename(entry["question"], "html"),
            mime="text/html",
            key=f"ask_html_{entry['id']}",
            help="Opens in any browser (it loads Plotly from the internet).",
        )
        st.caption("📷 For a PNG image, click the camera icon in the chart's toolbar (top right).")


_STEP_ICONS = {"done": "✅", "failed": "❌", "skipped": "⏭️"}
_PLAN_SOURCES = {
    "planner": "Multi-step request: each step was validated, then run in this order.",
    "follow-up": "Several follow-ups on the previous answer, applied one after another.",
    "router": "Steps performed for this single question.",
}


def show_plan(report) -> None:
    """The interpreted request and the numbered steps with what actually happened."""
    expanded = report.source != "router" or report.status == "failed"
    with st.expander(f"🧭 Analysis plan ({len(report.outcomes)} steps, {report.status})", expanded=expanded):
        st.caption(_PLAN_SOURCES.get(report.source, ""))
        if report.plan.interpretation:
            st.markdown(f"**Interpreted request:** {report.plan.interpretation}")
        lines = []
        for number, outcome in enumerate(report.outcomes, 1):
            line = f"{number}. {_STEP_ICONS.get(outcome.status, '•')} **{outcome.step.description}**"
            show_message = outcome.status != "done" or report.source == "planner"
            if outcome.message and show_message:  # what actually happened at this step
                line += f": {outcome.message}"
            lines.append(line)
        st.markdown("\n".join(lines))


def show_answer(entry: dict) -> None:
    """One exchange: the user's question, then the assistant's answer."""
    answer: Answer = entry["answer"]
    with st.chat_message("user"):
        st.markdown(entry["question"])
    with st.chat_message("assistant"):
        kind = entry.get("follow_up")
        if kind == "plan":  # a new multi-step request, not a follow-up
            st.caption("🧭 Multi-step request")
        elif kind:
            st.caption(f"↪ Follow-up ({kind}) to the previous answer")
        if answer.status == ANSWER:
            st.markdown(answer.text)
        elif answer.status == CLARIFY:
            st.info(f"🤔 {answer.text}")
            if answer.rephrasings:
                st.markdown("Ask instead:")
                for i, rephrased in enumerate(answer.rephrasings):
                    st.button(
                        rephrased, key=f"ask_rephrase_{entry['id']}_{i}",
                        on_click=_ask_example, args=(rephrased,), width="stretch",
                    )  # fmt: skip
        elif answer.status == UNSUPPORTED:
            st.warning(answer.text)
            if answer.suggestions:
                st.markdown("Questions I can answer here:")
                st.markdown("\n".join(f"- {s}" for s in answer.suggestions))
        elif answer.status == ERROR:
            st.error(answer.text)
        else:
            st.info(answer.text)
        if answer.plan is not None:
            show_plan(answer.plan)

        has_table = answer.table is not None and not answer.table.empty
        if answer.figure is not None:
            png_name = exports.export_filename(entry["question"], "png").removesuffix(".png")
            st.plotly_chart(
                answer.figure,
                key=f"ask_figure_{entry['id']}",
                config={
                    "displaylogo": False,
                    "toImageButtonOptions": {"format": "png", "filename": png_name, "scale": 2},
                },
            )
        if has_table:
            with st.expander("Result table", expanded=answer.figure is None):
                st.dataframe(answer.table, hide_index=True)
        if has_table or answer.figure is not None:
            show_downloads(entry, answer, has_table)
        if answer.details:
            with st.expander("How this was calculated"):
                st.markdown("\n".join(f"- {d}" for d in answer.details))


def _show_examples(examples: list[str]) -> None:
    visible = st.container()  # created first, so it is drawn above "More examples"
    more = st.expander("More examples") if len(examples) > VISIBLE_EXAMPLES else None
    for i, example in enumerate(examples):
        container = visible if i < VISIBLE_EXAMPLES else more
        container.button(
            example, key=f"ask_example_{i}", on_click=_ask_example, args=(example,), width="stretch",
        )  # fmt: skip


def _question_form(examples: list[str], name: str, df: pd.DataFrame, follow_up_hint: bool) -> None:
    with st.form("ask_form"):
        st.text_input(
            "Your question" if not follow_up_hint else "Ask a new question or a follow-up",
            key=QUESTION_KEY,
            max_chars=MAX_QUESTION_LENGTH,
            placeholder=(
                "e.g. Sort it from lowest to highest" if follow_up_hint
                else (f"e.g. {examples[0]}" if examples else "")
            ),
        )  # fmt: skip
        st.form_submit_button("Ask", type="primary", on_click=_submit_question)
    st.caption(
        f"Asking about **{name}** ({len(df):,} rows × {df.shape[1]} columns). Filter in the "
        "question (*'… for Electronics'*, *'… in March 2026'*), or follow up on the last "
        "answer (*'that city'*, *'sort it'*, *'show only the top 3'*, *'plot it as a pie chart'*)."
    )


def show_ask_your_data(dataset: tuple[str, pd.DataFrame] | None) -> None:
    st.markdown(
        "Ask a question about the selected dataset in plain English, then follow up on the "
        "answer. Answers are calculated with pandas by the app's own rule-based engine: no AI "
        "service, internet connection or API key is used, and nothing you type is ever run "
        "as code or SQL."
    )
    if dataset is None:
        st.info(
            "👈 Select a dataset first: upload a file, pick a sample in the sidebar, or send "
            "a query result from the **Database Explorer** tab."
        )
        return

    name, df = dataset
    # Answers and conversation context belong to one dataset. The fingerprint changes when
    # the values change, even if the name, size and columns stay the same.
    identity = ui_state.dataset_fingerprint(df)
    if st.session_state.get(DATASET_KEY) != identity:
        st.session_state[DATASET_KEY] = identity
        st.session_state[HISTORY_KEY] = []
    conversation = _conversation(identity)
    examples = suggest_questions(df)

    # Questions from the Ask button and from example / suggestion / rephrasing buttons are
    # answered before anything is drawn, so the new exchange appears right away.
    pending = st.session_state.pop(PENDING_KEY, None)
    if pending is not None:
        _ask(df, conversation, pending)

    history = st.session_state.get(HISTORY_KEY, [])
    if not history:  # start of a conversation: question box beside the examples
        ask_column, examples_column = st.columns([3, 2], gap="large")
        with ask_column:
            _question_form(examples, name, df, follow_up_hint=False)
        with examples_column:
            if examples:
                st.markdown("**Example questions for this dataset**")
                _show_examples(examples)
        st.divider()
        st.caption("No answers yet. Ask a question above or click an example.")
        return

    title, clear = st.columns([4, 1])
    title.markdown(f"**Conversation** (last {MAX_HISTORY} questions are kept, oldest first)")
    clear.button("Clear conversation", on_click=_clear_history, width="stretch")
    for entry in reversed(history):
        show_answer(entry)

    followups = suggest_followups(conversation.last, df)
    if followups:
        st.markdown("**Suggested follow-ups**")
        for i, followup in enumerate(followups):
            st.button(
                followup, key=f"ask_followup_{i}", on_click=_ask_example, args=(followup,),
                width="stretch",
            )  # fmt: skip

    _question_form(examples, name, df, follow_up_hint=conversation.last is not None)
    if examples:
        with st.expander("Example questions for this dataset"):
            _show_examples(examples)
