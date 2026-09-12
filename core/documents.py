"""Reading what is in a file without putting the file in the prompt.

Leti could already read a text file and load a dataset. What it could not do was
the ordinary thing a person means by "read these PDFs and compare the offers":
look at a folder, work out which files the question is actually about, open them,
and quote the part that answers it - with somewhere to point afterwards.

The whole design is about NOT loading everything. A folder of ten contracts is
several hundred thousand tokens; the answer to "which one is cheaper" is two
numbers. So a request goes through four stages, each one cheaper than reading:

    find_relevant   which files could this be about, by name and kind and where
                    they live. No contents are opened at all.
    describe        what a file is: type, size, how many pages or sheets or rows.
                    Structure, not text.
    outline         the shape of one file - its pages, sheets or headings, with a
                    line from each. Enough to choose where to read.
    extract         the parts that answer the question, and only those, each one
                    carrying the label it came from.

That last part is what makes a citation possible. Every piece of text this module
returns is attached to a SECTION - "Page 7", "Sheet 'Pricing'", "Rows 51-100",
"## Payment terms" - because a quotation nobody can locate is worth less than no
quotation, and a located one Leti can be held to. Nothing here invents a label:
a section's name comes from the file's own structure.

Selection is deterministic: term overlap against the file's name and its sections,
scored the way core/tool_router.py scores tools. There is no second model deciding
what is relevant, and no embedding index to keep in step with the disk.

What it cannot read, it says so about. A corrupt PDF, a format with no reader
installed, a file that is not there - each comes back as a plain refusal with a
reason, never as an empty extraction that reads like an empty document.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import math
import re
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("leti.documents")

# What a section is allowed to carry, and how much of a file can come back from
# one extraction. Both are budgets on the PROMPT, not on the file: a 400-page PDF
# is fine, it just cannot arrive all at once.
MAX_SECTION_CHARS = 4_000
DEFAULT_EXTRACT_CHARS = 6_000
MAX_EXTRACT_CHARS = 24_000
MAX_SECTIONS_RETURNED = 12
OUTLINE_PREVIEW_CHARS = 140
MAX_FILE_BYTES = 80 * 1024 * 1024      # past this, reading is a disk problem

# Extensions Leti can open, and what it does with each.
KINDS: Dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "text", ".log": "text", ".rst": "text",
    ".md": "markdown", ".markdown": "markdown",
    ".csv": "csv", ".tsv": "csv",
    ".xlsx": "xlsx", ".xlsm": "xlsx",
    ".json": "json",
    ".png": "image", ".jpg": "image", ".jpeg": "image",
    ".gif": "image", ".bmp": "image", ".webp": "image",
}

_WORD = re.compile(r"[a-z0-9]+")
# Words that match everything and therefore rank nothing.
_STOPWORDS = frozenset("""
a an and are as at be been but by can could do does for from has have how in into
is it its me my of on or our so than that the their them then there these they
this those to us was were what when where which who why will with you your
file files document documents please tell show find read give get compare
""".split())


# --------------------------------------------------------------------------- #
# What a file is
# --------------------------------------------------------------------------- #

def kind_of(path: Path) -> Optional[str]:
    return KINDS.get(path.suffix.lower())


def is_supported(path: Path) -> bool:
    return kind_of(path) is not None


def describe(path: Any) -> Dict[str, Any]:
    """Type, size and shape. Opens the file only far enough to count its parts."""
    p = Path(path)
    if not p.exists():
        return _problem(p, "there is no file at that path")
    if not p.is_file():
        return _problem(p, "that is a folder, not a file")

    stat = p.stat()
    info: Dict[str, Any] = {
        "ok": True,
        "path": str(p),
        "name": p.name,
        "kind": kind_of(p) or "unsupported",
        "bytes": stat.st_size,
        "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)),
    }
    if info["kind"] == "unsupported":
        info["ok"] = False
        info["error"] = (f"Leti has no reader for '{p.suffix or 'a file with no extension'}'. "
                         f"Readable kinds: {', '.join(sorted(set(KINDS.values())))}.")
        return info
    if stat.st_size > MAX_FILE_BYTES:
        info["ok"] = False
        info["error"] = f"{stat.st_size:,} bytes is past the {MAX_FILE_BYTES:,} byte limit."
        return info

    try:
        info.update(_shape(p, info["kind"]))
    except Exception as e:
        info["ok"] = False
        info["error"] = f"Couldn't read the structure of {p.name}: {e}"
    return info


def _problem(path: Path, why: str) -> Dict[str, Any]:
    return {"ok": False, "path": str(path), "name": path.name, "error": why}


def _shape(path: Path, kind: str) -> Dict[str, Any]:
    """How many parts the file has, without reading all of them."""
    if kind == "pdf":
        reader = _pdf_reader(path)
        return {"pages": len(reader.pages)}
    if kind == "docx":
        paragraphs, tables = _docx_body(path)
        return {"paragraphs": len(paragraphs), "tables": tables}
    if kind == "xlsx":
        from openpyxl import load_workbook

        book = load_workbook(path, read_only=True, data_only=True)
        try:
            return {"sheets": [{"name": s.title, "rows": s.max_row, "columns": s.max_column}
                               for s in book.worksheets]}
        finally:
            book.close()
    if kind == "csv":
        with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.reader(fh, delimiter=_csv_delimiter(path))
            header = next(reader, [])
            rows = sum(1 for _ in reader)
        return {"columns": header, "rows": rows}
    if kind == "json":
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        if isinstance(data, dict):
            return {"top_level_keys": sorted(data)[:60]}
        if isinstance(data, list):
            return {"items": len(data)}
        return {"json_type": type(data).__name__}
    if kind == "image":
        try:
            from PIL import Image

            with Image.open(path) as im:
                return {"width": im.width, "height": im.height, "format": im.format,
                        "note": "An image. Use look_at_image to have it described."}
        except Exception as e:
            return {"note": f"An image Leti could not open ({e})."}
    text = path.read_text(encoding="utf-8", errors="replace")
    return {"characters": len(text), "lines": text.count("\n") + 1}


# --------------------------------------------------------------------------- #
# Sections: the unit of citation
# --------------------------------------------------------------------------- #

def sections(path: Any) -> List[Dict[str, str]]:
    """The file in labelled pieces. Labels come from the file, never invented."""
    p = Path(path)
    kind = kind_of(p)
    if kind is None:
        raise ValueError(f"No reader for '{p.suffix}'.")
    reader = {
        "pdf": _pdf_sections, "docx": _docx_sections, "xlsx": _xlsx_sections,
        "csv": _csv_sections, "json": _json_sections, "markdown": _markdown_sections,
        "text": _text_sections, "image": _image_sections,
    }[kind]
    out = []
    for label, text in reader(p):
        text = (text or "").strip()
        if not text:
            continue
        out.append({"label": label, "text": text[:MAX_SECTION_CHARS]})
    return out


def _pdf_reader(path: Path):
    try:
        from pypdf import PdfReader
    except ImportError as e:      # pragma: no cover - depends on the install
        raise RuntimeError(
            "Reading PDFs needs the 'pypdf' package (pip install pypdf). Leti will not "
            "guess at a PDF's contents without it."
        ) from e
    return PdfReader(str(path))


def _pdf_sections(path: Path) -> Iterable[Tuple[str, str]]:
    reader = _pdf_reader(path)
    for number, page in enumerate(reader.pages, start=1):
        try:
            yield f"Page {number}", page.extract_text() or ""
        except Exception as e:
            logger.debug(f"{path.name} page {number} would not extract: {e}")
            yield f"Page {number}", ""


# --- DOCX. A .docx is a zip of XML, so this needs no package: the alternative
#     was a dependency to read a format the standard library can already open.
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _docx_body(path: Path) -> Tuple[List[Tuple[str, str]], int]:
    """Every paragraph as (style, text), and how many tables there are."""
    import xml.etree.ElementTree as ET

    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml")
    root = ET.fromstring(xml)
    body = root.find(f"{_W}body")
    if body is None:
        return [], 0

    paragraphs: List[Tuple[str, str]] = []
    tables = 0
    for node in body.iter():
        if node.tag == f"{_W}p":
            style_node = node.find(f"{_W}pPr/{_W}pStyle")
            style = style_node.get(f"{_W}val", "") if style_node is not None else ""
            text = "".join(t.text or "" for t in node.iter(f"{_W}t"))
            if text.strip():
                paragraphs.append((style, text))
        elif node.tag == f"{_W}tbl":
            tables += 1
    return paragraphs, tables


def _docx_sections(path: Path) -> Iterable[Tuple[str, str]]:
    paragraphs, _ = _docx_body(path)
    label, buffer, number = None, [], 0
    for style, text in paragraphs:
        if style.lower().startswith("heading"):
            if buffer:
                number += 1
                yield label or f"Section {number}", "\n".join(buffer)
                buffer = []
            label = text.strip()[:80]
            continue
        buffer.append(text)
        if sum(len(b) for b in buffer) > MAX_SECTION_CHARS:
            number += 1
            yield label or f"Section {number}", "\n".join(buffer)
            label, buffer = (f"{label} (continued)" if label else None), []
    if buffer:
        number += 1
        yield label or f"Section {number}", "\n".join(buffer)


def _xlsx_sections(path: Path) -> Iterable[Tuple[str, str]]:
    from openpyxl import load_workbook

    book = load_workbook(path, read_only=True, data_only=True)
    try:
        for sheet in book.worksheets:
            lines, header = [], None
            for row in sheet.iter_rows(values_only=True):
                cells = ["" if c is None else str(c) for c in row]
                if header is None:
                    header = cells
                lines.append("\t".join(cells))
                if sum(len(line) for line in lines) > MAX_SECTION_CHARS:
                    break
            yield f"Sheet '{sheet.title}'", "\n".join(lines)
    finally:
        book.close()


def _csv_delimiter(path: Path) -> str:
    return "\t" if path.suffix.lower() == ".tsv" else ","


def _csv_sections(path: Path) -> Iterable[Tuple[str, str]]:
    """Row ranges, with the header repeated so a cited chunk still reads."""
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        rows = list(csv.reader(fh, delimiter=_csv_delimiter(path)))
    if not rows:
        return
    header, body = rows[0], rows[1:]
    header_line = "\t".join(header)
    chunk, start = [], 1
    for index, row in enumerate(body, start=1):
        chunk.append("\t".join(str(c) for c in row))
        if sum(len(c) for c in chunk) > MAX_SECTION_CHARS:
            yield f"Rows {start}-{index}", header_line + "\n" + "\n".join(chunk)
            chunk, start = [], index + 1
    if chunk:
        yield f"Rows {start}-{len(body)}", header_line + "\n" + "\n".join(chunk)


def _json_sections(path: Path) -> Iterable[Tuple[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if isinstance(data, dict):
        for key in sorted(data):
            yield f"key '{key}'", json.dumps(data[key], indent=1, default=str)
        return
    if isinstance(data, list):
        for start in range(0, len(data), 25):
            block = data[start:start + 25]
            yield (f"items {start + 1}-{start + len(block)}",
                   json.dumps(block, indent=1, default=str))
        return
    yield "value", json.dumps(data, default=str)


def _markdown_sections(path: Path) -> Iterable[Tuple[str, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    label, buffer = None, []
    for line in text.splitlines():
        if line.startswith("#"):
            if buffer:
                yield label or "Before the first heading", "\n".join(buffer)
                buffer = []
            label = line.strip()[:80]
            continue
        buffer.append(line)
    if buffer:
        yield label or "Whole file", "\n".join(buffer)


def _text_sections(path: Path) -> Iterable[Tuple[str, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    chunk, start = [], 1
    for index, line in enumerate(lines, start=1):
        chunk.append(line)
        if sum(len(c) for c in chunk) > MAX_SECTION_CHARS:
            yield f"Lines {start}-{index}", "\n".join(chunk)
            chunk, start = [], index + 1
    if chunk:
        yield f"Lines {start}-{len(lines)}", "\n".join(chunk)


def _image_sections(path: Path) -> Iterable[Tuple[str, str]]:
    shape = describe(path)
    yield "image", (f"{path.name}: {shape.get('width', '?')}x{shape.get('height', '?')} "
                    f"{shape.get('format', '')}. Leti does not read text out of images "
                    "here - use look_at_image to have the vision model describe it.")


# --------------------------------------------------------------------------- #
# Outline and extraction
# --------------------------------------------------------------------------- #

def outline(path: Any) -> Dict[str, Any]:
    """The shape of a file: its parts, and a line from each. Enough to choose."""
    p = Path(path)
    info = describe(p)
    if not info.get("ok"):
        return info
    try:
        parts = sections(p)
    except Exception as e:
        return {**info, "ok": False, "error": f"Couldn't open {p.name}: {e}"}
    info["sections_total"] = len(parts)
    info["sections"] = [{"label": s["label"],
                         "preview": " ".join(s["text"].split())[:OUTLINE_PREVIEW_CHARS]}
                        for s in parts[:60]]
    return info


def _stem(word: str) -> str:
    """Just enough to make "offers" match Offer_Company_A.pdf.

    Deliberately crude and deliberately visible: a real stemmer would be a
    dependency and a source of surprises, and the whole job here is matching a
    plural in a question against a singular in a filename.
    """
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("ses", "xes", "zes", "ches", "shes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _terms(text: str) -> List[str]:
    return [_stem(w) for w in _WORD.findall(str(text or "").lower())
            if len(w) > 2 and w not in _STOPWORDS]


def _score_sections(parts: List[Dict[str, str]], query: str) -> List[Tuple[float, int]]:
    """Rank sections against the question. Rare words count for more.

    The same shape of scoring core/tool_router.py uses, for the same reason: a
    word that appears in every section separates nothing.
    """
    wanted = set(_terms(query))
    if not wanted:
        return [(0.0, i) for i in range(len(parts))]
    haystacks = [set(_terms(p["label"] + " " + p["text"])) for p in parts]
    total = len(parts) or 1
    scored = []
    for index, words in enumerate(haystacks):
        score = 0.0
        for term in wanted & words:
            appearing = sum(1 for h in haystacks if term in h) or 1
            score += math.log(1 + total / appearing)
        scored.append((score, index))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return scored


def extract(path: Any, query: str = "", max_chars: int = DEFAULT_EXTRACT_CHARS,
            section: str = "") -> Dict[str, Any]:
    """The parts of one file that answer the question, each carrying its label.

    With no query this is the beginning of the file rather than all of it, which
    is the honest answer to "what is in here" - and `sections_total` says how much
    was left, so asking for more is a decision rather than a guess.
    """
    p = Path(path)
    info = describe(p)
    if not info.get("ok"):
        return info
    try:
        parts = sections(p)
    except Exception as e:
        return {**info, "ok": False, "error": f"Couldn't read {p.name}: {e}"}
    if not parts:
        return {**info, "ok": False,
                "error": f"{p.name} opened, but no readable text came out of it. It may be "
                         "a scan rather than text, or empty."}

    budget = max(500, min(int(max_chars or DEFAULT_EXTRACT_CHARS), MAX_EXTRACT_CHARS))

    if section:
        wanted = [i for i, s in enumerate(parts) if section.lower() in s["label"].lower()]
        if not wanted:
            return {**info, "ok": False,
                    "error": f"{p.name} has no section matching '{section}'.",
                    "available": [s["label"] for s in parts[:40]]}
        chosen = wanted
    else:
        ranked = _score_sections(parts, query)
        chosen = []
        used = 0
        for score, index in ranked:
            if len(chosen) >= MAX_SECTIONS_RETURNED:
                break
            if query and score <= 0 and chosen:
                break            # nothing more in this file speaks to the question
            length = len(parts[index]["text"])
            if used and used + length > budget:
                continue
            chosen.append(index)
            used += length
        if not chosen:
            chosen = [ranked[0][1]]

    chosen.sort()
    returned, used = [], 0
    for index in chosen:
        part = parts[index]
        room = budget - used
        if room <= 0:
            break
        text = part["text"][:room]
        returned.append({"label": part["label"], "text": text,
                         "truncated": len(text) < len(part["text"])})
        used += len(text)

    return {
        **info,
        "source": p.name,
        "query": query or None,
        "sections": returned,
        "sections_total": len(parts),
        "sections_returned": len(returned),
        "characters": used,
        "more_available": len(returned) < len(parts),
        "how_to_cite": (f"Attribute anything quoted here to {p.name} and the section "
                        "label it came from. Do not cite a label that is not listed."),
    }


# --------------------------------------------------------------------------- #
# Which files a request is about
# --------------------------------------------------------------------------- #

def candidate_files(folder: Any, recursive: bool = True) -> List[Path]:
    root = Path(folder)
    if not root.is_dir():
        return []
    walk = root.rglob("*") if recursive else root.glob("*")
    return [p for p in walk if p.is_file() and is_supported(p)]


def find_relevant(request: str, files: Iterable[Any], limit: int = 12,
                  look_inside: int = 0) -> List[Dict[str, Any]]:
    """Rank files against a request by what can be known cheaply.

    Names, folders and kinds first, because they cost nothing and are usually
    enough - "Offer_Company_A.pdf" answers "compare the offers" on its name
    alone. `look_inside` opens at most that many of the best remaining candidates
    to check, which is the expensive path and therefore never the first one.
    """
    wanted = set(_terms(request))
    paths = [Path(f) for f in files]
    scored: List[Dict[str, Any]] = []
    for path in paths:
        name_words = set(_terms(str(path)))
        hits = sorted(wanted & name_words)
        scored.append({
            "path": str(path), "name": path.name, "kind": kind_of(path) or "unsupported",
            "bytes": path.stat().st_size if path.exists() else 0,
            "score": float(len(hits)),
            "matched": hits,
            "why": (f"the name matches {', '.join(hits)}" if hits else "same folder"),
        })

    scored.sort(key=lambda f: (-f["score"], f["name"]))
    if look_inside > 0 and wanted:
        for entry in scored[:look_inside]:
            if entry["score"] > 0:
                continue
            try:
                inside = _score_sections(sections(entry["path"]), request)
            except Exception:
                continue
            best = inside[0][0] if inside else 0.0
            if best > 0:
                entry["score"] = round(best, 3)
                entry["why"] = "the question's words appear inside it"
        scored.sort(key=lambda f: (-f["score"], f["name"]))

    # When nothing matched by name there is no relevance to report, only a folder
    # listing - which is the honest thing to hand back rather than a ranking.
    return scored[:limit]
