"""Streamlit UI for the Insights tab (the findings come from askyourdata.insights)."""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from askyourdata import exports
from askyourdata.insights import CATEGORIES, PERFORMANCE, QUALITY, TREND, Insight, InsightReport

SECTION_TITLES = {PERFORMANCE: "📈 Performance", TREND: "📅 Trends", QUALITY: "🧹 Data quality"}
CARDS_PER_ROW = 2


def show_insight(insight: Insight) -> None:
    with st.container(border=True):
        st.markdown(f"**{insight.title}**")
        st.markdown(insight.explanation)
        st.markdown("  \n".join(f"**{label}:** {value}" for label, value in insight.values))
        with st.expander("How this was found"):
            st.markdown(insight.evidence)
            if insight.table is not None and not insight.table.empty:
                st.dataframe(exports.display_table(insight.table).head(20), hide_index=True)


def show_insights(report: InsightReport, on_regenerate: Callable[[], None]) -> None:
    counts = {category: len(report.by_category(category)) for category in CATEGORIES}
    total = sum(counts.values())
    header, button = st.columns([4, 1])
    header.markdown(f"#### {total} insight{'s' if total != 1 else ''} found")
    header.caption(
        f"{counts[PERFORMANCE]} performance · {counts[TREND]} trend · {counts[QUALITY]} data quality. "
        f"Calculated locally with pandas from the current dataset ({report.rows:,} rows); no AI "
        "service is used and every number below comes from the data."
    )
    button.button("🔄 Regenerate", on_click=on_regenerate, key="insights_regenerate", width="stretch",
                  help="Recalculate the insights for the current dataset.")  # fmt: skip

    for category in CATEGORIES:
        st.subheader(SECTION_TITLES[category])
        items = report.by_category(category)
        if not items:
            st.info(report.notes.get(category, "Nothing reliable to show for this dataset."))
            continue
        if category in report.notes:  # some parts could not be analysed
            st.caption(f"ℹ️ {report.notes[category]}")
        for start in range(0, len(items), CARDS_PER_ROW):
            for column, insight in zip(st.columns(CARDS_PER_ROW), items[start : start + CARDS_PER_ROW]):
                with column:
                    show_insight(insight)

    st.caption(
        "Insights describe this dataset only. Differences between groups are not tested for "
        "statistical significance, no causes are claimed, and potential issues are heuristic "
        "checks, not confirmed errors."
    )
