"""End-to-end checks of the Streamlit app with Streamlit's own test runner.

These run app.py headlessly (no browser): they click widgets and inspect what
would be shown. They are slower than the unit tests (a few seconds each).
"""

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parent.parent / "app.py")
TIMEOUT = 120


@pytest.fixture
def app():
    return AppTest.from_file(APP, default_timeout=TIMEOUT).run()


def choose(at, source):
    at.sidebar.radio(key="data_source").set_value(source).run()
    return at


def ask(at, question):
    at.text_input(key="ask_question").set_value(question)
    next(b for b in at.button if b.label == "Ask").click().run()
    return at.session_state["ask_history"][0]["answer"]


def no_errors(at):
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_starts_with_onboarding_and_all_tabs(app):
    no_errors(app)
    assert [t.label for t in app.tabs][:6] == [
        "📋 Summary", "💡 Insights", "🧭 Explore", "🔎 Analysis tools", "💬 Ask Your Data",
        "🗄️ Database Explorer",
    ]  # fmt: skip
    assert any("Welcome" in m.value for m in app.markdown)
    assert any("No dataset is selected yet" in i.value for i in app.info)


def test_sample_button_loads_a_dataset(app):
    app.button(key="welcome_ecommerce").click().run()
    no_errors(app)
    assert app.sidebar.radio(key="data_source").value == "Sample e-commerce dataset"
    assert not any("Welcome" in m.value for m in app.markdown)


@pytest.mark.parametrize("source", ["Sample dataset", "Sample e-commerce dataset"])
def test_every_tab_renders_for_the_samples(app, source):
    choose(app, source)
    no_errors(app)
    labels = [m.label for m in app.metric]
    assert "Rows" in labels and "Complete cells" in labels
    assert any(m.label.startswith("Total ") and "(calculated)" in m.label for m in app.metric)
    assert any("calculated by the app, not a column" in i.value for i in app.info)
    assert len(app.get("plotly_chart")) >= 2  # summary trend and comparison
    for tool in ["Filter & sort", "Group & summarise", "Top N", "Trends over time", "Correlations", "Distribution"]:
        app.radio(key="analysis_tool").set_value(tool).run()
        no_errors(app)


def test_switching_datasets_clears_answers_and_settings(app):
    choose(app, "Sample dataset")
    ask(app, "How many rows are in this dataset?")
    app.radio(key="analysis_tool").set_value("Group & summarise").run()
    app.multiselect(key="grp_by").set_value(["category"]).run()
    assert len(app.session_state["ask_history"]) == 1

    choose(app, "Sample e-commerce dataset")
    no_errors(app)
    assert app.session_state["ask_history"] == []
    assert not any("30 rows" in m.value for m in app.markdown)  # no stale answer on screen
    assert app.multiselect(key="grp_by").value == ["City"]  # the new dataset's default


def test_new_database_result_with_same_shape_clears_answers(app):
    first = pd.DataFrame({"city": ["Pune", "Delhi"], "n": [1, 2]})
    second = pd.DataFrame({"city": ["Pune", "Delhi"], "n": [100, 200]})
    app.session_state["db_dataset"] = ("Custom SQL query", first)
    choose(app, "Database query result")
    assert "**3**" in ask(app, "total n").text
    app.session_state["db_dataset"] = ("Custom SQL query", second)  # same name, shape and columns
    app.run()
    no_errors(app)
    assert app.session_state["ask_history"] == []
    assert "**300**" in ask(app, "total n").text


def test_answers_survive_reruns_for_the_same_dataset(app):
    choose(app, "Sample e-commerce dataset")
    ask(app, "Show net revenue by city")
    app.radio(key="analysis_tool").set_value("Top N").run()  # any other interaction
    assert len(app.session_state["ask_history"]) == 1
    assert any(d.label.startswith("⬇️ Download table") for d in app.get("download_button"))


def test_clarification_offers_rephrasing_buttons(app):
    choose(app, "Sample e-commerce dataset")
    answer = ask(app, "Compare revenue across categories")
    assert answer.status == "clarify"
    buttons = [b for b in app.button if b.key and b.key.startswith("ask_rephrase_")]
    assert [b.label for b in buttons] == list(answer.rephrasings)
    buttons[0].click().run()
    assert app.session_state["ask_history"][0]["answer"].ok


def test_database_explorer_stays_read_only(app):
    app.radio(key="db_mode").set_value("Demo database (no setup needed)").run()
    next(b for b in app.button if b.label == "Open demo database").click().run()
    app.radio(key="db_section").set_value("Custom SQL").run()
    for sql in ["DROP TABLE sales", "DELETE FROM sales", "SELECT * FROM staff_salaries"]:
        app.text_area(key="db_sql_text").set_value(sql).run()
        app.button(key="db_sql_run").click().run()
        assert app.warning, sql
    app.text_area(key="db_sql_text").set_value("SELECT COUNT(*) AS n FROM sales").run()
    app.button(key="db_sql_run").click().run()
    no_errors(app)
    assert app.session_state["db_result_sql"][1].data["n"].tolist() == [30]  # data unchanged


@pytest.mark.parametrize(
    ("df", "summary_text"),
    [
        (pd.DataFrame({"city": pd.Series([], dtype=str), "n": pd.Series([], dtype=float)}), "no rows"),
        (pd.DataFrame({"note": ["a", "b", "c"]}), "3 rows and 1 column;"),  # nothing to calculate
    ],
)
def test_empty_and_unsupported_datasets_render_without_errors(app, df, summary_text):
    app.session_state["db_dataset"] = ("Odd query result", df)
    choose(app, "Database query result")
    no_errors(app)
    assert any(summary_text in m.value for m in app.markdown)
    for tool in ["Filter & sort", "Group & summarise", "Top N", "Trends over time", "Correlations", "Distribution"]:
        app.radio(key="analysis_tool").set_value(tool).run()
        assert not app.exception, tool  # warnings/info are fine, tracebacks are not
    answer = ask(app, "What is the average n?")
    assert answer.status in ("error", "clarify")


# --- conversation (follow-up questions) ---------------------------------------------------------


def test_follow_up_through_a_suggested_button(app):
    choose(app, "Sample e-commerce dataset")
    first = ask(app, "Which city has the most orders?")
    assert first.ok
    buttons = [b for b in app.button if b.key and b.key.startswith("ask_followup_")]
    reference = next(b for b in buttons if "in that City" in b.label)
    reference.click().run()
    no_errors(app)
    latest = app.session_state["ask_history"][0]
    assert latest["follow_up"] == "reference"
    assert latest["answer"].ok and "`City` is Bengaluru" in latest["answer"].text
    roles = [m.name for m in app.chat_message]
    assert roles.count("user") == 2 and roles.count("assistant") == 2


def test_typed_follow_ups_chain(app):
    choose(app, "Sample e-commerce dataset")
    ask(app, "Show total quantity by category")
    assert ask(app, "Sort it from lowest to highest").ok
    pie = ask(app, "Plot it as a pie chart")
    assert pie.ok and pie.figure.data[0].type == "pie"
    assert [h["follow_up"] for h in app.session_state["ask_history"]] == ["chart", "sort", None]


def test_clear_conversation_forgets_the_context(app):
    choose(app, "Sample e-commerce dataset")
    ask(app, "Show orders from Pune")
    next(b for b in app.button if b.label == "Clear conversation").click().run()
    assert app.session_state["ask_history"] == []
    assert app.session_state["ask_conversation"].last is None
    answer = ask(app, "Now group them by category")
    assert answer.status == "clarify" and "no earlier answer" in answer.text


def test_switching_datasets_clears_the_conversation_context(app):
    choose(app, "Sample e-commerce dataset")
    ask(app, "Which city has the most orders?")
    choose(app, "Sample dataset")
    no_errors(app)
    answer = ask(app, "Show the top 5 products in that city")
    assert answer.status == "clarify" and "no earlier answer" in answer.text


def test_question_box_is_empty_after_asking(app):
    choose(app, "Sample e-commerce dataset")
    ask(app, "Show total quantity by category")
    assert app.text_input(key="ask_question").value == ""
    next(b for b in app.button if b.key and b.key.startswith("ask_followup_")).click().run()
    assert app.text_input(key="ask_question").value == ""  # suggestion buttons don't fill it either
    assert len(app.session_state["ask_history"]) == 2


# --- multi-step plans ------------------------------------------------------------------------------


def test_multi_step_plan_in_the_app(app):
    choose(app, "Sample e-commerce dataset")
    answer = ask(app, "Find the top 5 cities by net revenue for Electronics and show a bar chart.")
    no_errors(app)
    assert answer.ok and answer.plan.status == "completed"
    assert app.session_state["ask_history"][0]["follow_up"] == "plan"
    assert any(e.label.startswith("🧭 Analysis plan (8 steps, completed)") for e in app.expander)
    assert any("Interpreted request:" in m.value for m in app.markdown)
    assert any("✅ **Draw a bar chart**" in m.value for m in app.markdown)
    assert any(d.label.startswith("⬇️ Download table") for d in app.get("download_button"))


def test_failed_plan_shows_no_result_in_the_app(app):
    choose(app, "Sample e-commerce dataset")
    ask(app, "Show total quantity by category")
    answer = ask(app, "sort it descending and plot it as a line chart")
    assert not app.exception  # a refused step is a warning, never a traceback
    assert answer.status == "unsupported" and answer.table is None and answer.figure is None
    assert any("❌" in m.value for m in app.markdown)
    assert any(w.value.startswith("Step 2") for w in app.warning)


def test_plan_context_is_cleared_on_dataset_switch(app):
    choose(app, "Sample e-commerce dataset")
    ask(app, "Filter to Pune, group quantity by category, and sort descending.")
    choose(app, "Sample dataset")
    answer = ask(app, "Sort it from lowest to highest and plot it as a pie chart")
    assert answer.status == "clarify" and "no earlier answer" in answer.text


def test_a_new_plan_is_not_labelled_as_a_follow_up(app):
    choose(app, "Sample e-commerce dataset")
    ask(app, "Show the top 3 products by quantity and display a chart.")
    captions = [c.value for c in app.caption]
    assert "🧭 Multi-step request" in captions
    assert not any(c.startswith("↪ Follow-up") for c in captions)


# --- automatic insights -------------------------------------------------------------------------------


def insight_titles(at):
    insights_tab = at.tabs[1]  # only the cards in the 💡 Insights tab, not the rest of the page
    assert insights_tab.label == "💡 Insights"
    return [m.value for m in insights_tab.markdown if m.value.startswith("**") and m.value.endswith("**")]


def test_insights_tab_shows_findings(app):
    choose(app, "Sample e-commerce dataset")
    no_errors(app)
    assert any(m.value.startswith("#### ") and "insights found" in m.value for m in app.markdown)
    assert [s.value for s in app.subheader][:3] == ["📈 Performance", "📅 Trends", "🧹 Data quality"]
    titles = insight_titles(app)
    assert "**Total Net revenue (calculated): 159,020.80**" in titles
    assert "**Bengaluru has the highest total Net revenue (calculated) by City**" in titles


def test_insights_follow_the_dataset_without_stale_results(app):
    choose(app, "Sample e-commerce dataset")
    ecommerce = insight_titles(app)
    choose(app, "Sample dataset")
    no_errors(app)
    sales = insight_titles(app)
    assert "**Total Gross amount (calculated): 3,965.64**" in sales
    assert not set(ecommerce) & set(sales)  # nothing from the previous dataset is shown
    choose(app, "Sample e-commerce dataset")
    assert insight_titles(app) == ecommerce  # same data, same insights


def test_regenerate_insights(app):
    choose(app, "Sample e-commerce dataset")
    before = insight_titles(app)
    app.button(key="insights_regenerate").click().run()
    no_errors(app)
    assert app.session_state["insights_nonce"] == 1
    assert insight_titles(app) == before  # deterministic: the same findings again


def test_insights_empty_states(app):
    app.session_state["db_dataset"] = ("Notes", pd.DataFrame({"note": [f"text {i}" for i in range(30)]}))
    choose(app, "Database query result")
    no_errors(app)
    infos = [i.value for i in app.info]
    assert any("No text column with 2–500 different values" in i for i in infos)
    assert any("No date column was found" in i for i in infos)


def test_every_cache_is_bounded():
    """Uploaded data and results must not stay in memory forever (privacy and memory use)."""
    import ast

    tree = ast.parse(Path(APP).read_text(encoding="utf-8"))
    caches = [
        d for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        for d in node.decorator_list if isinstance(d, ast.Call) and ast.unparse(d.func) == "st.cache_data"
    ]  # fmt: skip
    assert len(caches) == 3
    for decorator in caches:
        keywords = {k.arg for k in decorator.keywords}
        assert {"max_entries", "ttl"} <= keywords, ast.unparse(decorator)
