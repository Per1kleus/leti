"""Running, testing and inspecting code.

Scope note, because most of "a coding system" already exists here. Reading and
writing source files is read_file/write_file/list_files. Arbitrary commands,
git, and deployment are run_shell_command. Neither is reimplemented. What this
module adds is the part those can't do well:

  - run_code executes a snippet or file in a named language, handling the
    per-language mechanics (which interpreter, how to pass a snippet, MATLAB's
    -batch versus Octave's --eval) instead of making the model construct a shell
    line per language and get the quoting right.
  - run_tests finds how a project is tested rather than being told.
  - install_dependency knows pip from npm and asks before installing.
  - inspect_project reads a codebase's shape in one call, which is otherwise a
    dozen list_files round-trips before any work can start.

The workflow the tools are shaped for is PLAN -> WRITE/MODIFY -> RUN -> TEST ->
FIX -> VERIFY -> REPORT. Nothing enforces that ordering: it lives in the system
prompt, because it's a habit for the model, not a state machine. What these
tools do enforce is that "it works" has to come from output that actually
happened - every result carries the real exit code, stdout and stderr.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config_loader import resolve_path
from tools.base import BaseTool, ToolParameter, ToolResult
from tools.command_runner import run_command
from tools.projects import project_dir, resolve_project

DEFAULT_TIMEOUT = 120

# How to run each language: the interpreters to look for in order, and how to
# hand them code. Ordered by preference - python3 before python, node before
# nodejs, real MATLAB before Octave.
LANGUAGES: Dict[str, Dict[str, Any]] = {
    "python": {
        "interpreters": ["python3", "python"],
        "extension": ".py",
        "file_args": lambda exe, path: [exe, path],
    },
    "javascript": {
        "interpreters": ["node", "nodejs"],
        "extension": ".js",
        "file_args": lambda exe, path: [exe, path],
    },
    "typescript": {
        # tsx/ts-node run TypeScript directly; deno and bun understand it natively.
        "interpreters": ["tsx", "ts-node", "deno", "bun"],
        "extension": ".ts",
        "file_args": lambda exe, path: (
            [exe, "run", "--allow-all", path] if exe.endswith("deno") else [exe, path]
        ),
    },
    "bash": {
        "interpreters": ["bash", "sh"],
        "extension": ".sh",
        "file_args": lambda exe, path: [exe, path],
    },
    "matlab": {
        # MATLAB's -batch runs a script and exits non-zero on error, which is what
        # makes failures visible. Octave is the usual stand-in when MATLAB isn't
        # installed; most teaching-level .m code runs on it unchanged.
        "interpreters": ["matlab", "octave", "octave-cli"],
        "extension": ".m",
        "file_args": lambda exe, path: (
            [exe, "-batch", f"run('{path}')"] if exe.endswith("matlab")
            else [exe, "--no-gui", "--quiet", path]
        ),
    },
    "sql": {
        # SQL needs a database to run against, so this is sqlite-only by design;
        # anything else belongs in run_shell_command with that engine's client.
        "interpreters": ["sqlite3"],
        "extension": ".sql",
        "file_args": lambda exe, path: [exe],
    },
}

LANGUAGE_ALIASES = {
    "py": "python", "python3": "python",
    "js": "javascript", "node": "javascript",
    "ts": "typescript",
    "sh": "bash", "shell": "bash",
    "m": "matlab", "octave": "matlab",
    "sqlite": "sql", "sqlite3": "sql",
}


def normalize_language(language: str) -> str:
    key = (language or "").strip().lower()
    return LANGUAGE_ALIASES.get(key, key)


def find_interpreter(language: str) -> Optional[str]:
    spec = LANGUAGES.get(normalize_language(language))
    if not spec:
        return None
    for name in spec["interpreters"]:
        found = shutil.which(name)
        if found:
            return found
    return None


def working_directory(project: Optional[str], working_dir: Optional[str]) -> Path:
    """Where code runs: an explicit directory, else the active project's folder,
    else the user's home. Defaulting to the project is what lets "run the tests"
    work without repeating where the code is."""
    if working_dir:
        return resolve_path(working_dir)
    target = resolve_project(project)
    if target:
        try:
            directory = project_dir(target)
            if directory.is_dir():
                return directory
        except (ValueError, KeyError):
            pass
    return Path.home()


class RunCodeTool(BaseTool):
    name = "run_code"
    description = (
        "Run code and see what it actually does. Give `code` to run a snippet, or `file` "
        "to run something already on disk. Supports python, javascript, typescript, bash, "
        "matlab (falls back to Octave) and sql (sqlite).\n"
        "Run the code you write - a snippet that executed and printed the right answer is "
        "evidence; one that looks correct is not. The result carries the real exit code, "
        "stdout and stderr, so report failures from those rather than guessing at causes."
    )
    parameters = [
        ToolParameter(
            name="language", type="string",
            description="python, javascript, typescript, bash, matlab, or sql.",
        ),
        ToolParameter(name="code", type="string", required=False, description="Source to run. Omit if using `file`."),
        ToolParameter(name="file", type="string", required=False, description="Path to a file to run instead."),
        ToolParameter(
            name="arguments", type="array", items_type="string", required=False,
            description="Command-line arguments to pass to the program.",
        ),
        ToolParameter(name="project", type="string", required=False,
                      description="Project to run in. Defaults to the active project."),
        ToolParameter(name="working_dir", type="string", required=False,
                      description="Directory to run in. Overrides the project's folder."),
        ToolParameter(name="timeout_seconds", type="number", required=False,
                      description=f"Kill the program after this long (default {DEFAULT_TIMEOUT})."),
        ToolParameter(name="stdin", type="string", required=False,
                      description="Text to feed the program on standard input."),
    ]

    async def run(self, language: str, code: str = "", file: str = "",
                  arguments: Optional[List[str]] = None, project: str = "",
                  working_dir: str = "", timeout_seconds: float = DEFAULT_TIMEOUT,
                  stdin: str = "", **kwargs) -> ToolResult:
        canonical = normalize_language(language)
        spec = LANGUAGES.get(canonical)
        if not spec:
            return ToolResult(success=False, error=(
                f"Unsupported language '{language}'. Supported: {', '.join(sorted(LANGUAGES))}. "
                f"For anything else, run_shell_command can invoke its interpreter directly."
            ))
        if not code and not file:
            return ToolResult(success=False, error="Give either `code` to run or a `file` to run.")

        interpreter = find_interpreter(canonical)
        if not interpreter:
            wanted = ", ".join(spec["interpreters"])
            return ToolResult(success=False, error=(
                f"No interpreter for {canonical} on this machine (looked for: {wanted}). "
                f"Tell the user what to install rather than working around it."
            ))

        cwd = working_directory(project, working_dir)
        cwd.mkdir(parents=True, exist_ok=True)

        temp_path: Optional[Path] = None
        if file:
            target = resolve_path(file) if os.path.isabs(file) or file.startswith((".", "~")) else (cwd / file)
            if not target.is_file():
                return ToolResult(success=False, error=f"No such file: {target}")
            script = target
        else:
            # Written into the working directory, not /tmp: relative imports and
            # data files next to the code are the normal case, and a snippet run
            # from elsewhere would fail on them for reasons that look like bugs.
            handle, name = tempfile.mkstemp(suffix=spec["extension"], prefix="leti_run_", dir=str(cwd))
            with os.fdopen(handle, "w", encoding="utf-8") as f:
                f.write(code)
            script = temp_path = Path(name)

        try:
            if canonical == "sql":
                # sqlite3 takes SQL on stdin; the "file" case pipes its contents.
                sql_text = code or script.read_text()
                argv = [interpreter] + [str(a) for a in (arguments or [])]
                result = await _run_with_stdin(argv, sql_text, cwd, timeout_seconds)
            else:
                argv = spec["file_args"](interpreter, str(script)) + [str(a) for a in (arguments or [])]
                result = (await _run_with_stdin(argv, stdin, cwd, timeout_seconds)
                          if stdin else await run_command(argv, timeout=float(timeout_seconds), cwd=str(cwd)))
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

        output = {
            "language": canonical,
            "interpreter": interpreter,
            "working_dir": str(cwd),
            "exit_code": result.returncode,
            "stdout": result.stdout[-20000:],
            "stderr": result.stderr[-20000:],
        }
        if result.ok:
            return ToolResult(success=True, output=output)
        output["error"] = result.failure_reason()
        # success=False so a failing run can never be mistaken for a passing one,
        # with the full output attached so the fix comes from the real error.
        return ToolResult(success=False, output=output, error=(
            f"{canonical} exited with {result.returncode}: {result.failure_reason()}"
        ))


async def _run_with_stdin(argv: List[str], text: str, cwd: Path, timeout: float):
    """run_command doesn't write to stdin, and SQL and interactive scripts need it."""
    from tools.command_runner import CommandResult

    try:
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=str(cwd),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return CommandResult(None, "", "", error=f"not-found: '{argv[0]}' is not installed")
    except OSError as e:
        return CommandResult(None, "", "", error=f"could not run '{argv[0]}': {e}")

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate((text or "").encode()), timeout=timeout
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return CommandResult(None, "", "", error=f"timed out after {timeout}s")

    return CommandResult(proc.returncode, stdout.decode(errors="replace"), stderr.decode(errors="replace"))


# --- Tests ---------------------------------------------------------------------

# How a project is tested, in the order they're checked. First match wins, so a
# repo with both a package.json and a pytest layout is tested the way its
# manifest says rather than by guess.
TEST_RUNNERS = [
    ("package.json", "npm", ["npm", "test", "--silent"], '"test"'),
    ("pyproject.toml", "pytest", ["pytest", "-q"], None),
    ("pytest.ini", "pytest", ["pytest", "-q"], None),
    ("tox.ini", "pytest", ["pytest", "-q"], None),
    ("setup.py", "pytest", ["pytest", "-q"], None),
    ("Cargo.toml", "cargo", ["cargo", "test"], None),
    ("go.mod", "go", ["go", "test", "./..."], None),
    ("Makefile", "make", ["make", "test"], "test:"),
]


def detect_test_command(directory: Path) -> Optional[Dict[str, Any]]:
    """Work out how this project runs its tests, or None if it can't be told."""
    for marker, runner, command, required_text in TEST_RUNNERS:
        manifest = directory / marker
        if not manifest.is_file():
            continue
        if required_text:
            # package.json without a "test" script, or a Makefile without a test
            # target, would otherwise "run tests" and report a meaningless failure.
            try:
                if required_text not in manifest.read_text(errors="replace"):
                    continue
            except OSError:
                continue
        return {"runner": runner, "command": command, "detected_from": marker}

    if (directory / "tests").is_dir() or list(directory.glob("test_*.py")):
        return {"runner": "pytest", "command": ["pytest", "-q"], "detected_from": "tests/ directory"}
    return None


class RunTestsTool(BaseTool):
    name = "run_tests"
    description = (
        "Run a project's test suite and return what actually happened. Works out how the "
        "project is tested (pytest, npm test, cargo, go, make) from its files, or takes an "
        "explicit command. Run this after changing code - a change that hasn't been tested "
        "isn't finished, and the failures here are what to fix rather than what you expect "
        "them to be."
    )
    parameters = [
        ToolParameter(name="project", type="string", required=False,
                      description="Project to test. Defaults to the active project."),
        ToolParameter(name="working_dir", type="string", required=False,
                      description="Directory to test in. Overrides the project's folder."),
        ToolParameter(name="command", type="array", items_type="string", required=False,
                      description="Explicit test command, e.g. ['pytest','-k','parser']."),
        ToolParameter(name="timeout_seconds", type="number", required=False,
                      description="Default 600 - suites are slower than snippets."),
    ]

    async def run(self, project: str = "", working_dir: str = "",
                  command: Optional[List[str]] = None, timeout_seconds: float = 600,
                  **kwargs) -> ToolResult:
        cwd = working_directory(project, working_dir)
        if not cwd.is_dir():
            return ToolResult(success=False, error=f"No such directory: {cwd}")

        if command:
            argv, detected = [str(c) for c in command], "given explicitly"
        else:
            found = detect_test_command(cwd)
            if not found:
                return ToolResult(success=False, error=(
                    f"Couldn't tell how {cwd} is tested - no pyproject/pytest.ini/package.json "
                    f"with a test script/Cargo.toml/go.mod/Makefile test target, and no tests/ "
                    f"directory. Pass `command` explicitly."
                ))
            argv, detected = found["command"], found["detected_from"]

        if not shutil.which(argv[0]):
            return ToolResult(success=False, error=(
                f"'{argv[0]}' isn't installed on this machine, so the tests can't run."
            ))

        result = await run_command(argv, timeout=float(timeout_seconds), cwd=str(cwd))
        output = {
            "command": argv,
            "detected_from": detected,
            "working_dir": str(cwd),
            "exit_code": result.returncode,
            "stdout": result.stdout[-30000:],
            "stderr": result.stderr[-30000:],
            "passed": result.ok,
        }
        if result.ok:
            return ToolResult(success=True, output=output)
        return ToolResult(success=False, output=output,
                          error=f"Tests failed ({result.failure_reason()}).")


# --- Dependencies ---------------------------------------------------------------

PACKAGE_MANAGERS = {
    "pip": {"command": lambda pkgs: ["pip", "install", *pkgs], "probe": "pip"},
    "npm": {"command": lambda pkgs: ["npm", "install", *pkgs], "probe": "npm"},
}


class InstallDependencyTool(BaseTool):
    name = "install_dependency"
    description = (
        "Install packages a project needs (pip or npm). Requires the user's confirmation - "
        "this downloads and runs third-party code. Prefer installing into a project that has "
        "its own environment; say what you're installing and why before calling this."
    )
    parameters = [
        ToolParameter(name="manager", type="string", enum=["pip", "npm"], description="Which package manager."),
        ToolParameter(name="packages", type="array", items_type="string",
                      description="Package names, e.g. ['requests','pandas==2.2.0']."),
        ToolParameter(name="project", type="string", required=False,
                      description="Project to install into. Defaults to the active project."),
        ToolParameter(name="working_dir", type="string", required=False, description="Directory to install in."),
    ]

    async def run(self, manager: str, packages: List[str], project: str = "",
                  working_dir: str = "", **kwargs) -> ToolResult:
        spec = PACKAGE_MANAGERS.get((manager or "").lower())
        if not spec:
            return ToolResult(success=False, error=f"Unknown package manager '{manager}'. Use pip or npm.")
        names = [str(p).strip() for p in (packages or []) if str(p).strip()]
        if not names:
            return ToolResult(success=False, error="No packages given.")
        # A package name is about to become an argv entry to a downloader; a flag
        # smuggled in as a "name" (--index-url, -e) changes what gets installed
        # and from where.
        bad = [n for n in names if n.startswith("-")]
        if bad:
            return ToolResult(success=False, error=f"Package names can't start with '-': {bad}")
        if not shutil.which(spec["probe"]):
            return ToolResult(success=False, error=f"'{manager}' isn't installed on this machine.")

        cwd = working_directory(project, working_dir)
        result = await run_command(spec["command"](names), timeout=900, cwd=str(cwd))
        if not result.ok:
            return ToolResult(success=False, error=(
                f"Installing {names} failed: {result.failure_reason()}"
            ), output={"stdout": result.stdout[-8000:], "stderr": result.stderr[-8000:]})
        return ToolResult(success=True, output={
            "installed": names, "manager": manager, "working_dir": str(cwd),
            "stdout": result.stdout[-4000:],
        })


# --- Reading a codebase ----------------------------------------------------------

# Never worth walking into: they're large, generated, and tell you nothing about
# the code someone wrote.
SKIP_DIRECTORIES = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", "env",
    ".tox", ".pytest_cache", ".mypy_cache", "dist", "build", ".next", "target",
    ".idea", ".vscode", "site-packages", ".ruff_cache",
}
MANIFEST_FILES = [
    "README.md", "README.rst", "pyproject.toml", "setup.py", "requirements.txt",
    "package.json", "tsconfig.json", "Cargo.toml", "go.mod", "Makefile",
    "Dockerfile", "docker-compose.yml", ".env.example",
]


class InspectProjectTool(BaseTool):
    name = "inspect_project"
    description = (
        "Read the shape of a codebase in one call: its files by language, entry points, "
        "manifests, and how it's tested. Use this before changing code you haven't seen - "
        "it's what read_file is for afterwards, once you know which files matter."
    )
    parameters = [
        ToolParameter(name="project", type="string", required=False,
                      description="Project to inspect. Defaults to the active project."),
        ToolParameter(name="path", type="string", required=False,
                      description="Directory to inspect. Overrides the project's folder."),
        ToolParameter(name="max_files", type="number", required=False,
                      description="Cap on files listed (default 200)."),
    ]

    async def run(self, project: str = "", path: str = "", max_files: int = 200, **kwargs) -> ToolResult:
        root = resolve_path(path) if path else working_directory(project, "")
        if not root.is_dir():
            return ToolResult(success=False, error=f"Not a directory: {root}")

        max_files = max(10, min(int(max_files), 1000))
        by_extension: Dict[str, int] = {}
        files: List[str] = []
        total = 0

        for current, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRECTORIES and not d.startswith(".")]
            for filename in filenames:
                if filename.startswith("."):
                    continue
                full = Path(current) / filename
                total += 1
                by_extension[full.suffix or "(none)"] = by_extension.get(full.suffix or "(none)", 0) + 1
                if len(files) < max_files:
                    files.append(str(full.relative_to(root)))

        manifests = {}
        for name in MANIFEST_FILES:
            candidate = root / name
            if candidate.is_file():
                try:
                    text = candidate.read_text(errors="replace")
                except OSError:
                    continue
                manifests[name] = text[:4000]

        tests = detect_test_command(root)
        return ToolResult(success=True, output={
            "root": str(root),
            "file_count": total,
            "files": sorted(files),
            "truncated": total > len(files),
            "by_extension": dict(sorted(by_extension.items(), key=lambda kv: -kv[1])),
            "manifests": manifests,
            "test_command": tests,
            "skipped_directories": sorted(SKIP_DIRECTORIES),
        })
