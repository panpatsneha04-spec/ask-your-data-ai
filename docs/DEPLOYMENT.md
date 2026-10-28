# Deployment readiness assessment

**Status: not deployed.** No account, repository or hosting service has been set
up. This document assesses what a deployment would involve. Hosting terms and
free-tier limits change often, so check the provider's current documentation
before deciding; the points below are deliberately general.

## Can it be deployed without changing the architecture?

Yes, for a demo. The app is a single Streamlit process:
- **No server-side storage:** uploads are read into memory, the demo database is
  an in-memory SQLite database, and exports are generated in memory. A source scan
  found no code that writes files to disk.
- **No outbound calls:** the app makes no network calls except to a MySQL server
  you configure yourself.
- **Hosting:** any host that runs `streamlit run app.py` with Python 3.10+ and the
  packages in `requirements.txt` can serve it.

## Points to address first

| Topic | Current state | Before a public deployment |
|---|---|---|
| **Upload size** | Streamlit's default limit (200 MB per file); all data is held in memory | Lower it with `server.maxUploadSize` in `.streamlit/config.toml` (e.g. 25 MB), because free hosts have little memory |
| **Memory / caching** | Caches are limited to 20 entries and 1 hour each (added in Milestone 8) | Keep; consider lower limits on a small host |
| **Temporary files** | None written | Nothing to clean up. Note: data in memory is lost when the host restarts or the app sleeps |
| **Authentication** | None: anyone with the URL can use the app | For a public demo, show only the sample datasets, or put the app behind the host's access control |
| **Privacy** | Uploaded business data is processed on whichever machine runs the app | On a hosted service, uploads leave your computer and are processed on third-party servers. Don't upload confidential or personal data to a public demo; add a visible notice saying so |
| **Secrets** | `.env` locally (git-ignored); settings are read from environment variables | Use the host's secrets settings, never commit `.env`. The app reads `DB_*` environment variables, so map secrets to those names |
| **MySQL** | Tested against SQLite and error cases, not a live MySQL server | A hosted app needs a network-reachable MySQL with a **SELECT-only** user, encryption (`DB_SSL_CA`) and network restrictions. For a public demo, the built-in SQLite demo is safer; consider `DB_ALLOW_CUSTOM_SQL=false` |
| **Reproducible install** | `requirements.txt` (minimum versions) and `requirements-lock.txt` (exact versions that passed the tests) | Deploy with the lock file, or pin the same versions |
| **Python version** | Developed and tested with Python 3.12 | Choose 3.12 on the host |

## SQLite demo vs real MySQL

- **SQLite demo:** created in memory from `data/sample_sales.csv` when you click
  "Open demo database". It is read-only (`PRAGMA query_only`) and disappears when
  the session ends. Safe for a public demo.
- **Real MySQL:** needs credentials stored as host secrets, a read-only account,
  and a database reachable from the host's network. Exposing a business database
  to a public app is a real risk, even with the app's read-only checks; database
  permissions are the primary protection.

## Hosting options (low cost)

| Option | Fit | Things to know |
|---|---|---|
| **Streamlit Community Cloud** | Best fit: made for Streamlit apps, free tier, deploys from a GitHub repository, has a secrets manager | Requires a GitHub repository (public for the free tier, check current rules); apps sleep after inactivity and have modest memory; data is processed on Streamlit's servers |
| Hugging Face Spaces (Docker) | Possible: free CPU tier | Needs a small Dockerfile; Spaces may sleep when idle; public by default |
| Render / Railway / Fly.io | Possible | Free tiers are limited or time-bounded and spin down when idle; paid plans cost a few dollars per month |
| Your own small VM | Full control | You maintain updates, HTTPS and security yourself |

## Recommendation

For a **portfolio demo**: **Streamlit Community Cloud**, showing only the
bundled **synthetic** sample datasets and the SQLite demo database (no MySQL
credentials), with a short "don't upload confidential data" notice. This needs
no architectural change and should cost nothing within the free tier.

**Prerequisites (each needs your decision; nothing has been done):**
1. Initialise Git and review what will be published (the `.gitignore` already
   excludes `.env`, `.venv/` and local tool settings).
2. Create a GitHub repository and push the code.
3. Add `.streamlit/config.toml` with a lower `server.maxUploadSize`, and optionally
   a privacy notice in the app for the hosted version.
4. Create the Streamlit Community Cloud app from the repository, choose
   Python 3.12, and leave the `DB_*` secrets empty for a demo.
5. After deploying, re-run the manual checks from the README on the hosted URL.
