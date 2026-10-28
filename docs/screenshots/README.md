# Screenshots

These are real captures of the running app, made with the bundled **synthetic**
e-commerce sample during the Milestone 8 audit. The README uses them.

| File | Shows | Status |
|---|---|---|
| `welcome.jpg` | Start page with no dataset selected (welcome panel, sample buttons) | ✅ captured |
| `summary.jpg` | 📋 Summary for the e-commerce sample (KPIs incl. calculated net revenue) | ✅ captured |
| `insights.jpg` | 💡 Insights: performance cards with values and explanations | ✅ captured |
| `ask-plan.jpg` | 💬 Ask Your Data: "Find the top 5 cities by net revenue for Electronics and show a bar chart." with its 🧭 Analysis plan | ✅ captured |
| `database.jpg` | 🗄️ Database Explorer (demo database) refusing `DROP TABLE sales` | ✅ captured |

## Retaking them at a higher resolution (optional)

The captures are scaled-down screenshots (800 × 500 px). For a sharper portfolio:

1. Start the app with `.\.venv\Scripts\python.exe -m streamlit run app.py`.
2. Use a browser window about 1400 px wide.
3. Recreate each screen from the table above (use the 🛒 E-commerce orders button).
4. Save each screenshot under the same file name, so the README needs no changes.

Optional extra shots:
- `conversation.png`: a follow-up such as "Show the top 5 products in that city"
  after "Which city has the most orders?";
- `trend.png`: the 📅 Trends section of 💡 Insights.

Only use the synthetic sample data in screenshots, never real customer data.
