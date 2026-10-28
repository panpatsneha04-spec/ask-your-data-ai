"""Session-state helpers that keep results tied to the dataset they came from.

Streamlit keeps widget values and stored results in ``st.session_state``
across reruns. When the active dataset changes, every dataset-specific value
(analysis settings, Ask Your Data answers, ...) must be cleared, or results
from the previous dataset could be shown for the new one. These helpers work
on any dict-like object, so they can be tested without Streamlit.
"""

from __future__ import annotations

import hashlib
from collections.abc import MutableMapping

import pandas as pd

ACTIVE_DATASET_KEY = "active_dataset_fingerprint"

# Session keys starting with these belong to one dataset. Database connection
# and SQL settings ("db_...") and the data-source choice are kept.
DATASET_SCOPED_PREFIXES = (
    "flt_", "srt_", "grp_", "top_", "trd_", "cor_", "hist_", "ask_", "preview_", "summary_", "insights_",
)  # fmt: skip


def dataset_fingerprint(df: pd.DataFrame | None) -> str | None:
    """A short hash of a dataset's columns, types and values.

    Two datasets with the same name and shape but different values get
    different fingerprints, unlike a check on the name or column list alone.
    """
    if df is None:
        return None
    digest = hashlib.sha256()
    digest.update(repr((df.shape, [str(c) for c in df.columns], [str(t) for t in df.dtypes])).encode())
    try:
        digest.update(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
    except TypeError:  # cells pandas can't hash (e.g. lists): fall back to the text form
        digest.update(df.to_csv().encode("utf-8", "replace"))
    return digest.hexdigest()[:16]


def reset_if_dataset_changed(
    state: MutableMapping, fingerprint: str | None, prefixes: tuple[str, ...] = DATASET_SCOPED_PREFIXES
) -> bool:
    """Clear dataset-specific keys when the active dataset changes.

    Returns True when the dataset changed (including the first dataset of a
    session). Must run before any dataset-specific widget is drawn.
    """
    if ACTIVE_DATASET_KEY in state and state[ACTIVE_DATASET_KEY] == fingerprint:
        return False
    for key in [k for k in list(state.keys()) if isinstance(k, str) and k.startswith(prefixes)]:
        del state[key]
    state[ACTIVE_DATASET_KEY] = fingerprint
    return True
