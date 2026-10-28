"""The example questions printed in README.md must behave as the README says."""

from pathlib import Path

import pandas as pd
import pytest

from askyourdata.question_router import CLARIFY, answer_question

ROOT = Path(__file__).parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def ecommerce():
    return pd.read_csv(ROOT / "data" / "sample_ecommerce.csv")


ANSWERED = [
    "Show net revenue by city.",
    "Which product has the highest net revenue?",
    "Show total quantity for Electronics products.",
    "Find the top 5 products in Pune.",
    "Compare payment methods for orders above quantity 2.",
    "Show orders between 2026-02-01 and 2026-02-28.",
    "What is the order trend by month?",
    "Are there any missing values?",
]
CLARIFIED = {
    "Compare revenue across categories.": "net revenue",
    "Show orders from Pnue": "Did you mean Pune?",
    "orders on 03/04/2026": "could mean 3 April or March 4",
    "orders above 2": "Which column should be greater than 2",
}


CONVERSATION = [
    ("Which city has the most orders?", "Bengaluru"),
    ("Show the top 5 products in that city.", "`City` is Bengaluru"),
    ("Sort it from lowest to highest.", "sorted"),
    ("Plot it as a pie chart.", "pie chart"),
    ("Now group them by category.", "`City` is Bengaluru"),
    ("What about Pune?", "`City` is Pune"),
]


PLANS = [
    "Find the top 5 cities by net revenue for Electronics and show a bar chart.",
    "Filter to Pune, group quantity by category, and sort descending.",
    "Show the top 3 products by quantity and display a chart.",
    "Calculate total quantity by category and explain the result.",
]


@pytest.mark.parametrize("question", PLANS)
def test_readme_multi_step_examples(ecommerce, question):
    from askyourdata.planner import COMPLETED, answer_plan

    assert question in README
    answer = answer_plan(ecommerce, question)
    assert answer.ok and answer.plan.status == COMPLETED, answer.text


def test_readme_conversation_example(ecommerce):
    from askyourdata.conversation import Conversation

    conversation = Conversation("readme")
    for question, expected in CONVERSATION:
        assert question in README
        answer = conversation.ask(ecommerce, question).answer
        assert answer.ok, f"{question!r} -> {answer.text}"
        assert expected in answer.text, f"{question!r} -> {answer.text}"


@pytest.mark.parametrize("question", ANSWERED)
def test_readme_example_is_answered(ecommerce, question):
    assert question in README
    answer = answer_question(ecommerce, question)
    assert answer.ok, answer.text


@pytest.mark.parametrize(("question", "expected"), CLARIFIED.items())
def test_readme_example_asks_for_clarification(ecommerce, question, expected):
    assert question in README
    answer = answer_question(ecommerce, question)
    assert answer.status == CLARIFY
    assert expected in answer.text
