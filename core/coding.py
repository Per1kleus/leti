"""What Coding Mode knows that Default Mode does not.

Three things live here, and all three are deliberately small and on-demand. There
is no index, no background scan, no watcher and no cache that outlives a task:
"fix the authentication bug" walks the repository once, reads the handful of files
it decides matter, and forgets. A permanent index of somebody's source tree is a
thing that goes stale, costs memory while Leti is idle, and would have to be
invalidated by exactly the filesystem watcher this project has spent its life
not having.

  Symbols. Which files a request is about, what is defined in them, and what
  refers to what. Python gets a real parse (ast); everything else gets patterns
  per language. The difference is stated in the output rather than papered over -
  a regex is a good guess and a parse is a fact, and a caller deserves to know
  which it has.

  Test selection. Which tests relate to the files that changed, so a one-file
  change runs the tests for that file rather than the whole suite. The mapping is
  by name and by import, and when it cannot tell, it says so and the whole suite
  is the honest answer.

  Verification. The coding-shaped property checks - syntax, tests, scope, secrets,
  git state. The vocabulary they report in (VERIFIED, NOT VERIFIED, FAILED, NOT
  APPLICABLE) and the summarise() that combines them are core/verification.py's,
  shared with every other domain rather than restated here. Nothing is reported
  as checked that was not run.
"""
from __future__ import annotations

import ast
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from core import verification

logger = logging.getLogger("leti.coding")

MAX_FILES_SCANNED = 4_000
MAX_FILE_BYTES = 400_000
MAX_SYMBOLS_PER_FILE = 60
MAX_RESULTS = 12
MAX_TRACE = 20

SKIP = {".git", "__pycache__", "node_modules", ".venv", "venv", "env", "dist", "build",
        ".next", ".nuxt", "target", ".tox", ".mypy_cache", ".pytest_cache", "vendor",
        ".idea", ".vscode", "coverage", ".gradle", "Pods", "DerivedData"}

CODE_SUFFIXES = {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".rb",
                 ".php", ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".swift", ".kt",
                 ".m", ".scala", ".sh", ".sql", ".lua", ".r", ".jl"}

# Definitions, per language family. Python is parsed properly below; these are for
# everything else, and what they produce is labelled "pattern" rather than "parsed".
_PATTERNS = {
    "class": re.compile(r"^\s*(?:export\s+)?(?:abstract\s+)?class\s+([A-Za-z_]\w*)", re.M),
    "function": re.compile(
        r"^\s*(?:export\s+)?(?:async\s+)?(?:function|func|fn|def|sub)\s+([A-Za-z_]\w*)", re.M),
    "method": re.compile(r"^\s{2,}(?:public|private|protected|static|async)?\s*"
                         r"([A-Za-z_]\w*)\s*\([^)]*\)\s*\{", re.M),
    "const": re.compile(r"^\s*(?:export\s+)?(?:const|let|var|static final)\s+([A-Z_][A-Z0-9_]{2,})",
                        re.M),
    "arrow": re.compile(r"^\s*(?:export\s+)?(?:const|let)\s+([A-Za-z_]\w*)\s*=\s*"
                        r"(?:async\s*)?\([^)]*\)\s*=>", re.M),
}

_IMPORT_PATTERNS = (
    re.compile(r"""^\s*(?:import|from)\s+['"]?([\w./@-]+)""", re.M),
    re.compile(r"""require\(['"]([\w./@-]+)['"]\)""", re.M),
    re.compile(r"""^\s*use\s+([\w:]+)""", re.M),
)

ENTRY_POINTS = {"main.py", "app.py", "__main__.py", "manage.py", "index.js", "index.ts",
                "main.go", "main.rs", "Program.cs", "server.js", "app.js", "wsgi.py",
                "asgi.py", "cli.py"}
ROUTE_HINT = re.compile(r"""@(?:app|router|blueprint)\.(?:route|get|post|put|patch|delete)\s*\(\s*['"]([^'"]+)|"""
                        r"""(?:app|router)\.(?:get|post|put|patch|delete)\s*\(\s*['"]([^'"]+)""")

STOPWORDS = {"the", "a", "an", "and", "or", "of", "in", "on", "for", "to", "is", "are",
             "why", "how", "what", "when", "fix", "add", "make", "this", "that", "it",
             "does", "do", "not", "with", "from", "into", "code", "file", "files"}


# --------------------------------------------------------------------------- #
# The workspace: what the current coding task is about
# --------------------------------------------------------------------------- #

@dataclass
class Workspace:
    """Where Coding Mode is working, and what it has learnt this task.

    Released when the mode is left (core/modes.py leave()), so a coding session
    costs nothing once it is over.
    """
    root: Optional[str] = None
    repository: str = ""            # owner/name on GitHub, if one is selected
    branch: str = ""
    opened_at: float = field(default_factory=time.time)
    notes: Dict[str, Any] = field(default_factory=dict)


_workspace: Optional[Workspace] = None


def workspace() -> Optional[Workspace]:
    return _workspace


def open_workspace(root: str = "", repository: str = "", branch: str = "") -> Workspace:
    global _workspace

    if _workspace is None:
        _workspace = Workspace()
    if root:
        _workspace.root = str(root)
    if repository:
        _workspace.repository = repository
    if branch:
        _workspace.branch = branch
    return _workspace


def release() -> None:
    """Leaving Coding Mode. Nothing coding-specific stays in memory."""
    global _workspace

    _workspace = None
    try:
        from core import git_ops

        git_ops.clear_checkpoints()
    except Exception:
        logger.debug("Couldn't clear git checkpoints on leaving coding mode.")


def status() -> Dict[str, Any]:
    from core import github_client

    space = _workspace
    out: Dict[str, Any] = {
        "project_root": space.root if space else None,
        "repository": (space.repository if space else "") or github_client.default_repository(),
        "branch": space.branch if space else "",
        "github": "connected" if github_client.is_connected() else "not connected",
    }
    if space and space.root:
        from core import git_ops

        mark = git_ops.last_checkpoint(Path(space.root))
        out["checkpoint"] = mark["id"] if mark else None
    return out


# --------------------------------------------------------------------------- #
# Finding the files a request is about
# --------------------------------------------------------------------------- #

def _terms(text: str) -> Set[str]:
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", str(text or "").lower())
    out: Set[str] = set()
    for word in words:
        if word in STOPWORDS:
            continue
        out.add(word)
        # authService -> auth, service; user_repository -> user, repository
        out.update(p for p in re.split(r"_", word) if len(p) > 2)
    return out


def walk(root: Path) -> List[Path]:
    """Every code file under root, bounded, skipping what nobody means."""
    found: List[Path] = []
    for current, dirnames, filenames in __import__("os").walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP and not d.startswith(".")]
        for name in filenames:
            if Path(name).suffix.lower() in CODE_SUFFIXES:
                found.append(Path(current) / name)
                if len(found) >= MAX_FILES_SCANNED:
                    return found
    return found


def is_test_file(path: Path) -> bool:
    name = path.name.lower()
    return (name.startswith("test_") or name.endswith("_test.py")
            or ".test." in name or ".spec." in name
            or "test" in {p.lower() for p in path.parts[:-1]}
            or "tests" in {p.lower() for p in path.parts[:-1]})


MIN_STEM = 4        # "auth" may stand for "authentication"; "id" may not


def _related(wanted: Set[str], present: Set[str]) -> Set[str]:
    """Which request words this name answers to, allowing for abbreviation.

    Exact matching was the first version and it could not connect "fix the
    authentication bug" to auth.py, which is the single most obvious thing this
    function has to do. So one word counts for another when either is a prefix of
    the other and the shorter is at least MIN_STEM long: auth/authentication yes,
    id/identity no.
    """
    hits: Set[str] = set()
    for term in wanted:
        for name in present:
            if term == name:
                hits.add(term)
            elif len(min(term, name, key=len)) >= MIN_STEM and (
                    term.startswith(name) or name.startswith(term)):
                hits.add(term)
    return hits


def find_files(root: Path, request: str, limit: int = MAX_RESULTS) -> List[Dict[str, Any]]:
    """The files a request is most likely about, by name and by content.

    Names are weighted far above contents on purpose: a file called auth.py is
    about authentication, and a file that mentions the word once is usually not.
    """
    wanted = _terms(request)
    if not wanted:
        return []
    scored: List[Tuple[float, Dict[str, Any]]] = []

    for path in walk(root):
        relative = path.relative_to(root)
        name_terms = _terms(str(relative))
        name_hits = _related(wanted, name_terms)
        score = 4.0 * len(name_hits)
        body_hits: Set[str] = set()
        try:
            if path.stat().st_size <= MAX_FILE_BYTES:
                text = path.read_text(errors="replace")
                lowered = text.lower()
                body_hits = {w for w in wanted if w in lowered or
                             (len(w) >= MIN_STEM and w[:MIN_STEM] in lowered)}
                score += min(len(body_hits), 6) * 0.75
        except OSError:
            continue
        if score <= 0:
            continue
        scored.append((score, {
            "path": str(relative),
            "score": round(score, 2),
            "matched_in_name": sorted(name_hits),
            "matched_in_body": sorted(body_hits)[:6],
            "is_test": is_test_file(relative),
        }))

    scored.sort(key=lambda s: -s[0])
    return [entry for _, entry in scored[:limit]]


# --------------------------------------------------------------------------- #
# Symbols
# --------------------------------------------------------------------------- #

def symbols_in(path: Path) -> Dict[str, Any]:
    """What is defined in one file, and what it imports.

    Python is parsed. Everything else is matched with patterns, and says so - a
    caller deciding what to read next deserves to know whether it is looking at a
    fact or a good guess.
    """
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return {"path": str(path), "error": "too large to analyse in one go",
                    "symbols": [], "imports": []}
        text = path.read_text(errors="replace")
    except OSError as e:
        return {"path": str(path), "error": str(e), "symbols": [], "imports": []}

    if path.suffix == ".py":
        return _python_symbols(path, text)
    return _pattern_symbols(path, text)


def _python_symbols(path: Path, text: str) -> Dict[str, Any]:
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        return {"path": str(path), "how": "parsed", "symbols": [], "imports": [],
                "error": f"syntax error at line {e.lineno}: {e.msg}"}

    symbols, imports = [], []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.extend(_import_names(node))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = [b.name for b in node.body
                       if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))]
            symbols.append({"kind": "class", "name": node.name, "line": node.lineno,
                            "methods": methods[:MAX_SYMBOLS_PER_FILE]})
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append({"kind": "function", "name": node.name, "line": node.lineno,
                            "args": [a.arg for a in node.args.args][:10]})
        elif isinstance(node, ast.Assign) and isinstance(getattr(node, "parent", None), type(None)):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    symbols.append({"kind": "constant", "name": target.id, "line": node.lineno})

    return {"path": str(path), "how": "parsed", "language": "python",
            "symbols": symbols[:MAX_SYMBOLS_PER_FILE],
            "imports": sorted(set(imports))[:40],
            "entry_point": path.name in ENTRY_POINTS,
            "routes": _routes(text)}


def _import_names(node: Any) -> List[str]:
    if isinstance(node, ast.Import):
        return [a.name for a in node.names]
    return [f"{node.module}" for _ in [0]] if getattr(node, "module", None) else []


def _pattern_symbols(path: Path, text: str) -> Dict[str, Any]:
    symbols = []
    for kind, pattern in _PATTERNS.items():
        for match in pattern.finditer(text):
            name = match.group(1)
            line = text.count("\n", 0, match.start()) + 1
            symbols.append({"kind": "function" if kind == "arrow" else kind,
                            "name": name, "line": line})
            if len(symbols) >= MAX_SYMBOLS_PER_FILE:
                break
    imports: Set[str] = set()
    for pattern in _IMPORT_PATTERNS:
        imports.update(m.group(1) for m in pattern.finditer(text))
    return {"path": str(path), "how": "pattern",
            "language": path.suffix.lstrip("."),
            "note": "Matched with patterns rather than parsed - treat as a guide, "
                    "and read the file before relying on it.",
            "symbols": symbols, "imports": sorted(imports)[:40],
            "entry_point": path.name in ENTRY_POINTS, "routes": _routes(text)}


def _routes(text: str) -> List[str]:
    found = []
    for match in ROUTE_HINT.finditer(text):
        route = match.group(1) or match.group(2)
        if route and route not in found:
            found.append(route)
    return found[:12]


def references_to(root: Path, name: str, limit: int = MAX_TRACE) -> List[Dict[str, Any]]:
    """Where a symbol is used. A whole-word search, not a substring one."""
    clean = str(name or "").strip()
    if not re.fullmatch(r"[A-Za-z_]\w*", clean):
        return []
    pattern = re.compile(rf"\b{re.escape(clean)}\b")
    out: List[Dict[str, Any]] = []
    for path in walk(root):
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                out.append({"path": str(path.relative_to(root)), "line": number,
                            "text": line.strip()[:160],
                            "is_test": is_test_file(path.relative_to(root))})
                if len(out) >= limit:
                    return out
    return out


# --------------------------------------------------------------------------- #
# Which tests to run
# --------------------------------------------------------------------------- #

def tests_for(root: Path, changed: Iterable[str], limit: int = 12) -> Dict[str, Any]:
    """The tests that relate to the files that changed.

    Two signals, both cheap: a test whose NAME echoes the changed file
    (auth.py -> test_auth.py), and a test whose CONTENTS mention the changed
    module or one of its symbols. When neither finds anything, that is said
    plainly and the whole suite is the honest answer rather than a guess.
    """
    changed_paths = [Path(str(c)) for c in changed if str(c).strip()]
    if not changed_paths:
        return {"tests": [], "whole_suite": True,
                "why": "nothing was named as changed, so there is nothing to narrow to"}

    stems = {p.stem for p in changed_paths}
    stems |= {p.stem.replace("_", "") for p in changed_paths}
    modules = {p.stem for p in changed_paths}
    symbol_names: Set[str] = set()
    for path in changed_paths:
        full = root / path
        if full.suffix == ".py" and full.is_file():
            for symbol in symbols_in(full).get("symbols", []):
                symbol_names.add(symbol["name"])

    matched: List[Dict[str, Any]] = []
    for path in walk(root):
        relative = path.relative_to(root)
        if not is_test_file(relative):
            continue
        reasons = []
        test_stem = relative.stem.lower().replace("test_", "").replace("_test", "")
        if test_stem in {s.lower() for s in stems}:
            reasons.append(f"named after {test_stem}")
        try:
            text = path.read_text(errors="replace") if path.stat().st_size <= MAX_FILE_BYTES else ""
        except OSError:
            text = ""
        if text:
            hit_modules = sorted({m for m in modules if re.search(rf"\b{re.escape(m)}\b", text)})
            hit_symbols = sorted({s for s in symbol_names
                                  if re.search(rf"\b{re.escape(s)}\b", text)})[:4]
            if hit_modules:
                reasons.append("imports " + ", ".join(hit_modules[:3]))
            if hit_symbols:
                reasons.append("exercises " + ", ".join(hit_symbols))
        if reasons:
            matched.append({"path": str(relative), "why": "; ".join(reasons)})
        if len(matched) >= limit:
            break

    if not matched:
        return {"tests": [], "whole_suite": True,
                "why": ("No test file mentions any of those files or their symbols. "
                        "Running the whole suite is the honest answer - a narrower run "
                        "would be a guess dressed up as a decision.")}
    return {"tests": matched, "whole_suite": False,
            "why": f"{len(matched)} test file(s) relate to what changed.",
            "next": "Run these first; widen to the whole suite before reporting done."}


# --------------------------------------------------------------------------- #
# Classifying a failure
# --------------------------------------------------------------------------- #

ENVIRONMENT = re.compile(
    r"\b(ModuleNotFoundError|ImportError: lib|No such file or directory: '/usr|"
    r"command not found|connection refused|Address already in use|permission denied|"
    r"could not connect|DNS|SSLError|socket\.gaierror|OSError: \[Errno 28)", re.I)
DEPENDENCY = re.compile(
    r"\b(ModuleNotFoundError: No module named|ImportError: cannot import name|"
    r"Cannot find module|package .* is not installed|pip install|npm ERR!)", re.I)
SYNTAX = re.compile(r"\b(SyntaxError|IndentationError|TabError|Unexpected token|"
                    r"expected .* but found)", re.I)


def classify_failure(output: str, changed: Iterable[str] = (),
                     failed_before: Iterable[str] = ()) -> Dict[str, Any]:
    """What kind of failure this is - which decides what to do about it.

    Editing code until a test passes is how a suite becomes decorative. The most
    useful answer here is often "this was already failing" or "this is your
    environment", because neither is a reason to change the code.
    """
    text = str(output or "")
    changed_names = {Path(str(c)).name for c in changed if str(c).strip()}
    already = {str(f) for f in failed_before if str(f).strip()}

    mentioned = sorted({n for n in changed_names if n and n in text})
    named_tests = sorted(set(re.findall(r"(?:FAILED|ERROR)\s+([^\s:]+(?:::[^\s]+)?)", text)))
    pre_existing = sorted({t for t in named_tests if t in already})

    if SYNTAX.search(text):
        kind, what = "syntax", "the code does not parse - fix that before anything else"
    elif DEPENDENCY.search(text):
        kind, what = "dependency", ("something is not installed. Install it or say so - "
                                    "do not work around a missing package by changing code")
    elif ENVIRONMENT.search(text):
        kind, what = "environment", ("this is the machine, not the change - a port, a path, "
                                     "a permission or a network call")
    elif pre_existing and len(pre_existing) == len(named_tests):
        kind, what = "pre_existing", ("every failing test here was already failing before "
                                      "this change. Say so; do not adopt them")
    elif mentioned:
        kind, what = "caused_by_change", ("the failure names a file this task changed, so "
                                          "start there")
    elif named_tests:
        kind, what = "unrelated", ("nothing in the failure names a changed file. Check "
                                   "whether it failed before touching it")
    else:
        kind, what = "ambiguous", ("the output does not say what failed. Re-run the single "
                                   "test with more detail rather than guessing")

    return {"kind": kind, "what_to_do": what, "failing_tests": named_tests[:20],
            "pre_existing": pre_existing[:20], "mentions_changed_files": mentioned[:10],
            "never": "Do not edit code you have not read to make a test go green."}


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #

# The vocabulary and the summary live in core/verification.py, which every other
# domain now uses as well. They are imported rather than restated so there is one
# definition of what VERIFIED means, not a coding-shaped copy of it - see the note
# at the top of that module.
VERIFIED = verification.VERIFIED
NOT_VERIFIED = verification.NOT_VERIFIED
FAILED = verification.FAILED
NOT_APPLICABLE = verification.NOT_APPLICABLE
summarise = verification.summarise

SECRET_PATTERNS = (
    (re.compile(r"\b(gh[pousr]_[A-Za-z0-9]{16,}|github_pat_[A-Za-z0-9_]{20,})"), "a GitHub token"),
    (re.compile(r"\b(sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})"), "an API key"),
    (re.compile(r"""(?i)\b(password|passwd|secret|api[_-]?key|token)\s*[:=]\s*["'][^"'\s]{8,}"""),
     "a hard-coded credential"),
    (re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"), "a private key"),
)

DEBUG_LEFTOVERS = (
    (re.compile(r"^\s*(?:import pdb|pdb\.set_trace\(\)|breakpoint\(\))", re.M), "a debugger call"),
    (re.compile(r"^\s*console\.log\(", re.M), "a console.log"),
    (re.compile(r"\bTODO: remove\b|\bXXX\b|\bHACK\b", re.M), "a leftover marker"),
)


def check_syntax(root: Path, paths: Iterable[str]) -> Dict[str, Any]:
    """Python files parse. Anything else is NOT APPLICABLE rather than assumed fine."""
    checked, problems, skipped = [], [], []
    for name in paths:
        path = root / str(name)
        if path.suffix != ".py" or not path.is_file():
            skipped.append(str(name))
            continue
        try:
            ast.parse(path.read_text(errors="replace"))
            checked.append(str(name))
        except SyntaxError as e:
            problems.append({"path": str(name), "line": e.lineno, "problem": e.msg})
        except OSError as e:
            problems.append({"path": str(name), "problem": str(e)})
    if problems:
        return {"property": "syntax", "result": FAILED, "problems": problems,
                "detail": f"{len(problems)} file(s) do not parse."}
    if not checked:
        return {"property": "syntax", "result": NOT_APPLICABLE,
                "detail": "No Python files among the changes; Leti cannot parse the rest.",
                "not_checked": skipped[:20]}
    return {"property": "syntax", "result": VERIFIED,
            "detail": f"{len(checked)} Python file(s) parse.",
            "not_checked": skipped[:20]}


def check_secrets(root: Path, paths: Iterable[str]) -> Dict[str, Any]:
    found = []
    for name in paths:
        path = root / str(name)
        try:
            if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
                continue
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for pattern, what in SECRET_PATTERNS:
            match = pattern.search(text)
            if match:
                found.append({"path": str(name), "looks_like": what,
                              "line": text.count("\n", 0, match.start()) + 1})
    if found:
        return {"property": "secrets", "result": FAILED, "found": found,
                "detail": "Something that looks like a credential is in the changes."}
    return {"property": "secrets", "result": VERIFIED,
            "detail": "No token, key or hard-coded credential pattern in the changed files.",
            "limit": "Pattern matching finds the common shapes, not every possible secret."}


def check_leftovers(root: Path, paths: Iterable[str]) -> Dict[str, Any]:
    found = []
    for name in paths:
        path = root / str(name)
        try:
            if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
                continue
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for pattern, what in DEBUG_LEFTOVERS:
            if pattern.search(text):
                found.append({"path": str(name), "found": what})
    if found:
        return {"property": "leftovers", "result": NOT_VERIFIED, "found": found,
                "detail": "Debug leftovers in the changed files - remove them or say why "
                          "they stay."}
    return {"property": "leftovers", "result": VERIFIED,
            "detail": "No debugger calls, console.log or leftover markers."}


def check_scope(intended: Iterable[str], actually_changed: Iterable[str],
                user_changed: Iterable[str] = ()) -> Dict[str, Any]:
    """Did the task change what it meant to, and nothing else?"""
    meant, did = {str(i) for i in intended}, {str(a) for a in actually_changed}
    theirs = {str(u) for u in user_changed}
    unexpected = sorted(did - meant - theirs)
    missing = sorted(meant - did)
    if not meant:
        return {"property": "scope", "result": NOT_APPLICABLE,
                "detail": "No intended file list was given, so scope cannot be checked.",
                "changed": sorted(did)}
    if unexpected or missing:
        return {"property": "scope", "result": NOT_VERIFIED,
                "unexpected": unexpected, "not_changed": missing,
                "detail": "The diff does not match what the task said it would touch."}
    return {"property": "scope", "result": VERIFIED,
            "detail": f"{len(did)} file(s) changed, all of them intended.",
            "untouched_user_work": sorted(theirs)}


