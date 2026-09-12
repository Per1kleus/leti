"""Working with the files a request is actually about.

Four of these five tools read nothing until they are told which file, and none of
them ever hands a whole file to the model. That is the point: a folder of
contracts is several hundred thousand tokens and the answer to "which is cheaper"
is two numbers, so the work is finding the two numbers.

The intended order, and each step is cheaper than the one after it:

    find_documents      which files this could be about - names, kinds, sizes
    inspect_document    what one file is, and what parts it has
    read_document       the parts that answer the question, with their labels
    compare_documents   the same question across several files at once

Everything that comes back carries the label it came from - "Page 7", "Sheet
'Pricing'", "Rows 51-100" - so a claim can be attributed to a place in a file
rather than to a file in general. core/documents.py builds those labels from the
file's own structure; nothing here invents one.

These are ordinary registered tools. They go through the router and the guard
like every other tool, they read from disk and nowhere else, and a file they
cannot open comes back as a refusal with a reason rather than as an empty
document.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from core import documents
from core.config_loader import resolve_path
from tools.base import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger("leti.tools.documents")

MAX_COMPARED = 8          # a comparison, not a corpus scan


def _where_to_look(folder: str = "", project: str = "") -> Dict[str, Any]:
    """The folder a request means, and whether a project chose it.

    A task that belongs to a project looks in that project first, because that is
    what belonging to a project is for - searching the whole disk when the user
    has already said which workspace they are in is the expensive wrong answer.
    """
    from tools.projects import project_dir, project_exists, resolve_project

    from tools.projects import own_files

    if folder:
        directory = resolve_path(folder)
        return {"directory": directory, "project": None,
                "files": documents.candidate_files(directory),
                "scope": f"the folder {folder}"}
    name = resolve_project(project or None)
    if name and project_exists(name):
        directory = project_dir(name)
        # own_files is the project's own answer to "what is in me": it already
        # leaves out the manifest and anything belonging to a nested project, so
        # asking it beats walking the folder and rediscovering both rules.
        return {"directory": directory, "project": name,
                "files": [f for f in own_files(directory) if documents.is_supported(f)],
                "scope": f"the '{name}' project"}
    if project:
        return {"directory": None, "project": None,
                "error": f"There is no project called '{project}'."}
    return {"directory": None, "project": None,
            "error": ("No folder was given and no project is open. Say which folder to "
                      "look in, or open a project first.")}


class FindDocumentsTool(BaseTool):
    name = "find_documents"
    description = (
        "Which files a request is about, before opening any of them. Ranks the readable "
        "files in the open project (or a folder you name) by how well they match, and "
        "returns names, kinds and sizes - never contents. Use this first for 'compare "
        "these offers', 'summarise this folder', 'find the invoice', 'write a report "
        "based on these files'. Sees a PDF, a Word document (docx), an Excel spreadsheet "
        "(xlsx), a CSV, JSON, Markdown or text - a contract, an invoice, a report."
    )
    parameters = [
        ToolParameter(name="request", type="string",
                      description="What the user asked for, in their words."),
        ToolParameter(name="folder", type="string", required=False,
                      description="Where to look. Defaults to the open project's folder."),
        ToolParameter(name="project", type="string", required=False,
                      description="A project name, if not the one that is open."),
        ToolParameter(name="look_inside", type="boolean", required=False,
                      description="Also open the best few candidates to check their text. "
                                  "Slower; only when the names alone are not enough."),
    ]

    async def run(self, request: str, folder: str = "", project: str = "",
                  look_inside: bool = False, **kwargs) -> ToolResult:
        where = _where_to_look(folder, project)
        if where.get("error"):
            return ToolResult(success=False, error=where["error"])
        directory = where["directory"]
        if not directory.is_dir():
            return ToolResult(success=False, error=f"Not a folder: {directory}")

        files = where["files"]
        if not files:
            return ToolResult(success=True, output={
                "scope": where["scope"], "directory": str(directory),
                "files": [], "count": 0,
                "note": "There are no files Leti can read in there.",
            })
        ranked = documents.find_relevant(request, files,
                                         look_inside=4 if look_inside else 0)
        matched = [f for f in ranked if f["score"] > 0]
        return ToolResult(success=True, output={
            "scope": where["scope"],
            "project": where["project"],
            "directory": str(directory),
            "count": len(files),
            "files": ranked,
            "note": ("Ranked by how well each name matches the request."
                     if matched else
                     "Nothing matched the request by name, so this is the folder's "
                     "readable files, most relevant first by kind only. Open one with "
                     "inspect_document before assuming what is in it."),
            "next": "inspect_document to see a file's structure, read_document to read part of it.",
        })


class InspectDocumentTool(BaseTool):
    name = "inspect_document"
    description = (
        "What a file IS and what parts it has, without reading it: type, size, and its "
        "pages, sheets, rows or headings with one line from each. Use it to decide where "
        "to read, and to check a file before quoting it. Says plainly if the file is "
        "missing, corrupt or of a kind Leti cannot read."
    )
    parameters = [
        ToolParameter(name="path", type="string", description="Path to the file."),
    ]

    async def run(self, path: str, **kwargs) -> ToolResult:
        info = documents.outline(resolve_path(path))
        if not info.get("ok"):
            return ToolResult(success=False, error=info.get("error", "Couldn't read that file."),
                              output=info)
        return ToolResult(success=True, output=info)


class ReadDocumentTool(BaseTool):
    name = "read_document"
    description = (
        "Read the parts of one file that answer a question - not the whole file. Returns "
        "the relevant sections, each labelled with where it came from ('Page 7', \"Sheet "
        "'Pricing'\", 'Rows 51-100', 'Table 2'), so you can attribute what you quote. Reads "
        "PDF, Word (docx), Excel (xlsx), CSV, JSON, Markdown and text: contracts, invoices, "
        "offers, reports, notes. Name a section or a range ('Page 7', 'Page 3-7') to read "
        "just that part. Ask again with another question when it says more is available, "
        "and never say what is in a file without reading it here first."
    )
    parameters = [
        ToolParameter(name="path", type="string", description="Path to the file."),
        ToolParameter(name="question", type="string", required=False,
                      description="What you are looking for. Leave blank to read from the start."),
        ToolParameter(name="section", type="string", required=False,
                      description="A specific section label, e.g. 'Page 7'. Overrides the question."),
        ToolParameter(name="max_characters", type="number", required=False,
                      description=f"How much to return at most (default "
                                  f"{documents.DEFAULT_EXTRACT_CHARS:,})."),
    ]

    async def run(self, path: str, question: str = "", section: str = "",
                  max_characters: Any = None, **kwargs) -> ToolResult:
        try:
            budget = int(max_characters) if max_characters else documents.DEFAULT_EXTRACT_CHARS
        except (TypeError, ValueError):
            budget = documents.DEFAULT_EXTRACT_CHARS
        result = documents.extract(resolve_path(path), query=question or "",
                                   max_chars=budget, section=section or "")
        if not result.get("ok"):
            return ToolResult(success=False, error=result.get("error", "Couldn't read that file."),
                              output=result)
        return ToolResult(success=True, output=result)


class CompareDocumentsTool(BaseTool):
    name = "compare_documents"
    description = (
        "Ask the same question of several files at once, each answer labelled by file and "
        "by the place in it. For 'compare these offers', 'compare these Excel "
        "spreadsheets and say which is better', 'which of these PDFs is cheaper'. Reads "
        "only the relevant part of each. A file it cannot read is reported as such and "
        "the rest are still compared."
    )
    parameters = [
        ToolParameter(name="paths", type="array",
                      description="The files to compare, by path."),
        ToolParameter(name="question", type="string",
                      description="The question to ask of each file."),
        ToolParameter(name="max_characters_each", type="number", required=False,
                      description="How much to take from each file (default 2,500)."),
    ]

    async def run(self, paths: Any, question: str, max_characters_each: Any = None,
                  **kwargs) -> ToolResult:
        wanted = [str(p) for p in (paths or []) if str(p).strip()]
        if len(wanted) < 2:
            return ToolResult(success=False,
                              error="Give at least two files to compare.")
        try:
            budget = int(max_characters_each) if max_characters_each else 2_500
        except (TypeError, ValueError):
            budget = 2_500

        # Per-file budgets multiply: eight files at 2,500 characters is 20,000, and
        # the point of comparing is the few lines that differ. So there is a total
        # as well, and a file that would take it past the total is shortened
        # rather than the comparison being cut off without saying so.
        compared, unreadable, spent = [], [], 0
        for path in wanted[:MAX_COMPARED]:
            room = documents.MAX_COMPARE_CHARS - spent
            if room < 400:
                unreadable.append({"file": Path(path).name,
                                   "problem": "not read - the comparison was already at its "
                                              "size limit. Ask about this file separately."})
                continue
            result = documents.extract(resolve_path(path), query=question,
                                       max_chars=min(budget, room))
            if not result.get("ok"):
                unreadable.append({"file": Path(path).name,
                                   "problem": result.get("error", "could not be read")})
                continue
            spent += result.get("characters", 0)
            compared.append({
                "file": result["source"],
                "path": result["path"],
                "found": [{"label": s["label"], "text": s["text"]} for s in result["sections"]],
                "more_available": result["more_available"],
                "note": result.get("note"),
            })

        if not compared:
            return ToolResult(success=False,
                              error="None of those files could be read.",
                              output={"unreadable": unreadable})
        return ToolResult(success=True, output={
            "question": question,
            "compared": compared,
            "unreadable": unreadable,
            "skipped": wanted[MAX_COMPARED:],
            "characters": spent,
            "how_to_cite": ("Attribute every figure to its file and the label beside it. "
                            "If a file is in 'unreadable', say so rather than guessing "
                            "what it contained."),
        })


class LookAtImageTool(BaseTool):
    name = "look_at_image"
    description = (
        "Have the vision model describe an image FILE on disk - a saved screenshot, a "
        "photo, a scanned page - and answer a question about it. Same model read_screen "
        "uses, pointed at a file, so it takes a few seconds. Only when the picture itself "
        "is the question; for text files use read_document."
    )
    parameters = [
        ToolParameter(name="path", type="string", description="Path to the image file."),
        ToolParameter(name="question", type="string",
                      description="What to look for, e.g. 'what does the invoice total say'."),
    ]

    def __init__(self, llm_client):
        self.llm_client = llm_client

    async def run(self, path: str, question: str, **kwargs) -> ToolResult:
        import base64

        target = resolve_path(path)
        info = documents.describe(target)
        if not info.get("ok"):
            return ToolResult(success=False, error=info.get("error", "Couldn't open that file."))
        if info.get("kind") != "image":
            return ToolResult(success=False,
                              error=f"{target.name} is a {info.get('kind')} file, not an image. "
                                    "Use read_document for it.")
        try:
            encoded = base64.b64encode(target.read_bytes()).decode("utf-8")
            answer = await self.llm_client.analyze_image(question, encoded)
        except Exception as e:
            return ToolResult(success=False, error=f"The vision model couldn't read {target.name}: {e}")
        asked_about_text = bool(re.search(
            r"\b(text|read|says?|written|wording|number|serial|invoice number|caption|"
            r"label|transcri\w+|ocr)\b", question, re.I))
        return ToolResult(success=True, output={
            "source": target.name,
            "question": question,
            "answer": answer,
            "kind": "image understanding",
            "text_extraction": "unavailable (Leti has no OCR)",
            "how_to_cite": (
                f"This is the vision model looking at {target.name} and saying what it "
                "sees. It is not text extracted from the file."
                + (" You asked about text: report anything it read as the model's reading "
                   "of a picture, which can be wrong, and never as the file's contents."
                   if asked_about_text else "")),
        })
