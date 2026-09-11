"""When a tool result is worth looking at rather than reading out.

Leti already had one way to put something on screen: a tool returns a "visual"
key in its output and core/orchestrator.py pushes it to whatever surface can
render it. Four tools do that - image search, the sketch tool, the chart in
data analysis and the social feeds - and everything else, however tabular or
however long, arrived as prose for the model to summarise.

This closes that gap WITHOUT adding a second decision-maker. There is no extra
model call and no classifier: the rules below read the SHAPE of a result that
already exists.

  - a list of at least two dictionaries that share their keys, whose values are
    scalars, is a table;
  - a list of pages with a url and a title is a set of research sources;
  - one long piece of prose under a document-ish key is a document.

A result that does not clearly match stays text, which is the important half of
the deal: the failure mode to avoid is a panel opening over the interface for
something that reads perfectly well in a sentence.

Shape, not tool name, on purpose. Keying off names would be a list to maintain
beside the tools, and it would be wrong the moment a tool changed what it
returns; the shape IS what it returns.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("leti.artifacts")

# A table of one row is a sentence, and a table of two hundred is a file. Both
# ends are bounded; what is shown is a readable panel, not the whole result.
TABLE_MIN_ROWS = 2
TABLE_MAX_ROWS = 60
TABLE_MIN_COLUMNS = 2
TABLE_MAX_COLUMNS = 8
CELL_MAX_CHARS = 160

# Under this, prose belongs in the reply, not in a window of its own.
DOCUMENT_MIN_CHARS = 400
DOCUMENT_MAX_CHARS = 20000

RESEARCH_MIN_ITEMS = 2
RESEARCH_MAX_ITEMS = 12
SNIPPET_MAX_CHARS = 320

# Keys whose value is a whole document when it is a long enough string.
DOCUMENT_KEYS = ("report", "document", "article", "summary", "markdown",
                 "content", "answer", "text")

# Lists that are sources rather than rows. Checked before the table rules, because
# a list of pages would otherwise render as a two-column table of urls.
RESEARCH_KEYS = ("sources_read", "sources", "results", "other_results_not_read")

# Lists that are already handled by the "visual" key their own tool sets, or that
# are not content at all.
IGNORED_KEYS = ("visual", "items", "queries", "failed_queries", "steps",
                "history", "tools", "column_details")

_SCALARS = (str, int, float, bool)


def derive(output: Any) -> Optional[Dict[str, Any]]:
    """The visual payload this result deserves, or None for most results.

    Called by the orchestrator only where a tool did not provide a "visual" of its
    own, so a tool that knows better always wins.
    """
    if not isinstance(output, dict) or not output:
        return None
    try:
        return (_as_research(output) or _as_table(output) or _as_document(output))
    except Exception:
        # A malformed result should reach the model as text, not take the turn down.
        logger.debug("Couldn't derive an artifact from a tool result.", exc_info=True)
        return None


# --------------------------------------------------------------------------- #
# Research
# --------------------------------------------------------------------------- #

def _as_research(output: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for key in RESEARCH_KEYS:
        rows = output.get(key)
        if not isinstance(rows, list) or len(rows) < RESEARCH_MIN_ITEMS:
            continue
        pages = [r for r in rows if isinstance(r, dict) and r.get("url") and r.get("title")]
        if len(pages) < RESEARCH_MIN_ITEMS:
            continue
        items = []
        for page in pages[:RESEARCH_MAX_ITEMS]:
            snippet = page.get("snippet") or page.get("text") or ""
            items.append({
                "title": _clip(str(page.get("title", "")), CELL_MAX_CHARS),
                "url": str(page.get("url", "")),
                "domain": str(page.get("domain", "")),
                "snippet": _clip(" ".join(str(snippet).split()), SNIPPET_MAX_CHARS),
            })
        return {"type": "research",
                "title": _clip(str(output.get("topic") or _first_query(output) or "Research"), 80),
                "items": items,
                "more": max(0, len(pages) - len(items))}
    return None


def _first_query(output: Dict[str, Any]) -> str:
    queries = output.get("queries") or output.get("query")
    if isinstance(queries, list) and queries:
        return str(queries[0])
    return str(queries or "")


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #

def _as_table(output: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for key, value in output.items():
        if key in IGNORED_KEYS or key in RESEARCH_KEYS:
            continue
        table = _table_from(value, key)
        if table:
            return table
    return None


def _table_from(value: Any, key: str) -> Optional[Dict[str, Any]]:
    if not isinstance(value, list) or len(value) < TABLE_MIN_ROWS:
        return None
    rows = [r for r in value if isinstance(r, dict)]
    if len(rows) != len(value):
        return None            # a mixed list is not a table

    # Only columns every row has, and only ones whose values are all scalar: a
    # column that is a nested object in row three is not a column.
    columns = [c for c in rows[0]
               if all((c in r and isinstance(r[c], _SCALARS)) or r.get(c) is None
                      for r in rows)]
    columns = [c for c in columns if any(r.get(c) is not None for r in rows)]
    if not (TABLE_MIN_COLUMNS <= len(columns) <= TABLE_MAX_COLUMNS):
        return None

    shown = rows[:TABLE_MAX_ROWS]
    return {
        "type": "table",
        "title": _label(key),
        "columns": [_label(c) for c in columns],
        "rows": [[_cell(r.get(c)) for c in columns] for r in shown],
        "more": max(0, len(rows) - len(shown)),
    }


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:,.4g}"
    if isinstance(value, int):
        return f"{value:,}"
    return _clip(" ".join(str(value).split()), CELL_MAX_CHARS)


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #

def _as_document(output: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for key in DOCUMENT_KEYS:
        value = output.get(key)
        if isinstance(value, str) and len(value) >= DOCUMENT_MIN_CHARS:
            return {"type": "document",
                    "title": _label(str(output.get("title") or key)),
                    "text": value[:DOCUMENT_MAX_CHARS],
                    "truncated": len(value) > DOCUMENT_MAX_CHARS}
    return None


def _label(key: str) -> str:
    return str(key).replace("_", " ").strip().capitalize() or str(key)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def titles() -> List[str]:
    """The artifact kinds this module can produce, for tests and documentation."""
    return ["research", "table", "document"]
