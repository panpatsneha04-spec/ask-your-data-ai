# Portfolio texts

Every statement below describes a feature that is implemented and covered by the
test suite (749 passing tests in the last run). Wording deliberately avoids
claims such as "production-ready", "enterprise-grade" or "LLM-powered": the app
is a local, rule-based analyst.

## Resume entry

**AskYourData AI: data-analyst web app** · Python, Streamlit, pandas, Plotly, SQLAlchemy, pytest

- Built a Streamlit app that profiles CSV/Excel data, produces an executive summary
  and automatic insights, and answers plain-English questions with charts and CSV exports.
- Designed a deterministic, rule-based question router and multi-step planner.
  Requests become validated plans of eight allowlisted operations (filter, metric,
  aggregate, sort, limit, chart…) that run with pandas: no code generation, no LLM.
- Implemented conversational follow-ups ("that city", "sort it", "plot it as a pie
  chart") that resolve references only from actual previous results and ask for
  clarification instead of guessing.
- Added a read-only database explorer for MySQL/SQLite: SQL parsing with sqlglot,
  a table allowlist, row limits, timeouts and bound parameters. Credentials are kept
  out of errors and logs.
- Wrote 749 automated tests (pytest + Streamlit AppTest). They include hand-checked
  calculations, unsafe-input refusal and stale-state reset on dataset changes.

## Short portfolio description

AskYourData AI is a local data-analyst app built with Python and Streamlit.
Upload a CSV or Excel file (or read from a database in read-only mode) and it
summarises the data, finds patterns such as top groups, changes over time and
data-quality issues, and answers questions like *"Find the top 5 cities by net
revenue for Electronics and show a bar chart."*

Every answer shows how it was calculated: filters, formulas and a numbered plan
of the steps that ran. The "intelligence" is deterministic and rule-based, not an
LLM, so every number is calculated by pandas and can be traced. When a question
is unclear, the app asks instead of guessing. The project includes 749 automated tests.

## Talking points for interviews

- **Why rule-based instead of an LLM:** answers are reproducible, every number is
  calculated rather than generated, and user input can never become executed code or SQL.
  The trade-off is a narrower range of understood questions; the README lists the limits.
- **Safety layering:** database permissions first, then SQL parsing, the allowlist,
  row limits and a read-only session, and the app-level checks are documented as
  "not a substitute for database permissions".
- **Bugs found by testing:**
  - a metric input matcher once multiplied `units × units`;
  - weekly trends on sparse data showed "−100%";
  - a "that city" reference could silently fall back to another value.

  Each one now has a regression test.

## Screenshots available

Real captures of the running app (synthetic sample data only), in `docs/screenshots/`:
`welcome.jpg`, `summary.jpg`, `insights.jpg`, `ask-plan.jpg`, `database.jpg`.
See [screenshots/README.md](screenshots/README.md) for how to retake them at a
higher resolution.
