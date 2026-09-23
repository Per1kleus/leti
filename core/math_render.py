"""Mathematics you can look at, built from mathematics the model wrote.

core/speech.py turns an equation into words, because a voice cannot say markup.
This is the other half: the same equation as something to SEE, because words
cannot carry a matrix and "the integral from zero to L of f of x d x" is not
what anyone means by showing somebody an integral.

WHAT IT PRODUCES, AND WHY THAT

The output is a tree of MathML elements, as plain dictionaries. Not HTML, not a
string of markup, and not a picture:

  - MathML is what a browser already knows how to lay out. Every surface Leti
    has - the desktop window, a browser, a phone - renders it natively, so
    nothing new is installed, downloaded or served. LaTeX is not renderable by
    any of them; that is the whole reason a conversion exists.

  - A TREE rather than a string is the security answer. gui/hud.html builds
    each node with createElementNS and sets text with textContent, so there is
    no point at which a string of markup is handed to a parser. An expression
    cannot contain a tag, because a tag is not representable: the renderer
    accepts an element name from a fixed list and text, and nothing else. No
    innerHTML, no <script>, no <foreignObject>, no href.

  - Nothing is rasterised. matplotlib is already here and could draw an
    equation, but that is a figure to render, a file to write and a picture
    that cannot be selected, searched or scaled.

sympy is already a dependency and already emits LaTeX - tools/engineering.py
returns sympy.latex(result) for every symbolic answer. Its MathML printer is
therefore the natural route for anything sympy produced, and is used where an
expression parses. But sympy cannot READ LaTeX without antlr4, which is not
installed and is not worth installing, so the reader below is written here: a
small recursive descent over the subset of LaTeX that appears in an answer.

WHAT IT REFUSES

Detection is conservative on purpose (see `detect`). "The answer is 42" is not
mathematics, and a panel that opens for it is worse than no panel. An
expression this cannot parse returns None rather than a half-rendered guess,
and the text stays text.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("leti.math_render")

# The only element names the renderer will ever be asked for. gui/hud.html holds
# the same list and drops anything not in it, so this is a belt and braces pair:
# neither side trusts the other to have checked.
ELEMENTS = frozenset({
    "math", "mrow", "mi", "mn", "mo", "mtext", "msup", "msub", "msubsup",
    "mfrac", "msqrt", "mroot", "munderover", "munder", "mover",
    "mtable", "mtr", "mtd", "mfenced", "mspace", "mstyle",
})

# Bounds. An expression past these is not an equation anybody is reading; it is
# a model that has run away, and rendering it would hang the page rather than
# inform it.
MAX_EXPRESSION_CHARS = 2000
MAX_NODES = 600
MAX_DEPTH = 24
MAX_EXPRESSIONS = 6          # per answer: a panel, not a textbook

_GREEK = (
    "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta",
    "theta", "vartheta", "iota", "kappa", "lambda", "mu", "nu", "xi", "pi",
    "rho", "sigma", "tau", "upsilon", "phi", "varphi", "chi", "psi", "omega",
    "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon",
    "Phi", "Psi", "Omega",
)
_GREEK_CHARS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "varepsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ", "vartheta": "ϑ",
    "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ",
    "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ",
    "phi": "φ", "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ",
    "Pi": "Π", "Sigma": "Σ", "Upsilon": "Υ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
}

_SYMBOLS = {
    "times": "×", "cdot": "⋅", "div": "÷", "pm": "±", "mp": "∓",
    "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥", "neq": "≠", "ne": "≠",
    "approx": "≈", "equiv": "≡", "propto": "∝", "infty": "∞",
    "partial": "∂", "nabla": "∇", "int": "∫", "iint": "∬", "oint": "∮",
    "sum": "∑", "prod": "∏", "lim": "lim", "to": "→", "rightarrow": "→",
    "leftarrow": "←", "Rightarrow": "⇒", "in": "∈", "notin": "∉",
    "subset": "⊂", "cup": "∪", "cap": "∩", "forall": "∀", "exists": "∃",
    "ldots": "…", "cdots": "⋯", "dots": "…", "degree": "°", "circ": "∘",
    "angle": "∠", "perp": "⊥", "parallel": "∥", "sqrt": "√",
}

# Functions written as words, set upright rather than italic.
_FUNCTIONS = ("sin", "cos", "tan", "sec", "csc", "cot", "arcsin", "arccos",
              "arctan", "sinh", "cosh", "tanh", "log", "ln", "exp", "det",
              "min", "max", "gcd", "lcm", "lim", "deg", "dim", "ker")

# The big operators that take limits above and below rather than beside.
_BIG = {"sum": "∑", "prod": "∏", "int": "∫", "iint": "∬", "oint": "∮", "lim": "lim"}


# --------------------------------------------------------------------------- #
# Finding mathematics in an answer
# --------------------------------------------------------------------------- #

# Delimited spans, in the forms models emit. Display forms first: $$..$$ must be
# tried before $..$ or it matches the empty string between two dollars.
_DELIMITED = (
    (re.compile(r"\$\$(.+?)\$\$", re.S), True),
    (re.compile(r"\\\[(.+?)\\\]", re.S), True),
    (re.compile(r"\\begin\{equation\*?\}(.+?)\\end\{equation\*?\}", re.S), True),
    (re.compile(r"\\\((.+?)\\\)", re.S), False),
    (re.compile(r"(?<!\$)\$([^$\n]+)\$(?!\$)", re.S), False),
)

# An undelimited line that is nevertheless plainly an equation: a relation with
# a power, a fraction, a Greek letter or a named operator in it. Deliberately
# demanding - see the module docstring on why "the answer is 42" must not match.
_BARE_EQUATION = re.compile(
    r"^[^=\n]{1,40}=[^=\n]{1,80}$")
_LOOKS_MATHEMATICAL = re.compile(
    r"\^|_\{|\\frac|\\sqrt|\\sum|\\int|\\[A-Za-z]{2,}|[∫∑√πθαβω≤≥≠∞]")


def detect(text: str) -> List[Dict[str, Any]]:
    """The mathematical expressions in `text`, in the order they appear.

    Each is {"latex": str, "display": bool}. An answer with no mathematics -
    which is nearly all of them - comes back empty, and that is the common path
    this is written to be fast on.
    """
    if not isinstance(text, str) or not text or "=" not in text and "\\" not in text \
            and "$" not in text and "^" not in text:
        return []

    found: List[Tuple[int, str, bool]] = []
    remaining = text
    for pattern, display in _DELIMITED:
        for match in pattern.finditer(remaining):
            body = match.group(1).strip()
            if body:
                found.append((match.start(), body, display))
        # Take the matched spans out of the running so $..$ does not re-find
        # the inside of a $$..$$ that already matched.
        remaining = pattern.sub(lambda m: " " * len(m.group(0)), remaining)

    if not found:
        for line in text.splitlines():
            stripped = line.strip()
            if (_BARE_EQUATION.match(stripped) and _LOOKS_MATHEMATICAL.search(stripped)):
                found.append((text.find(stripped), stripped, True))

    found.sort(key=lambda row: row[0])
    seen, out = set(), []
    for _, body, display in found:
        if body in seen:
            continue
        seen.add(body)
        out.append({"latex": body[:MAX_EXPRESSION_CHARS], "display": display})
        if len(out) >= MAX_EXPRESSIONS:
            break
    return out


# --------------------------------------------------------------------------- #
# Reading LaTeX
#
# A recursive descent over a token stream. It handles what appears in an answer
# and nothing more; anything it does not know becomes an <mi> with the literal
# text, which renders as itself rather than as a hole.
# --------------------------------------------------------------------------- #

# Stripped from any literal text an expression carries - see _literal.
_ANGLE = re.compile(r"[<>]")

_TOKEN = re.compile(
    r"\\begin\{[a-zA-Z*]+\}|\\end\{[a-zA-Z*]+\}|\\[a-zA-Z]+|\\\\|\\[,;:!\s]|"
    r"\d+\.\d+|\d+|[a-zA-Z]|[+\-*/=<>(),.|\[\]]|[\^_{}&]|[ \t]+|\S")


class _Over(Exception):
    """A bound was hit. The caller returns None: half an equation is not one."""


class _Reader:
    def __init__(self, latex: str):
        self.tokens = _TOKEN.findall(latex)
        self.at = 0
        self.nodes = 0

    def peek(self, ahead: int = 0) -> str:
        index = self.at + ahead
        return self.tokens[index] if index < len(self.tokens) else ""

    def next(self) -> str:
        token = self.peek()
        self.at += 1
        return token

    def node(self, name: str, children=None, text: str = "") -> Dict[str, Any]:
        self.nodes += 1
        if self.nodes > MAX_NODES:
            raise _Over()
        if name not in ELEMENTS:
            raise _Over()
        out: Dict[str, Any] = {"e": name}
        if text:
            out["t"] = text
        if children:
            out["c"] = children
        return out

    # -- groups ---------------------------------------------------------- #

    def group(self, depth: int) -> Dict[str, Any]:
        """The next {..} as one node, or the next single atom."""
        if depth > MAX_DEPTH:
            raise _Over()
        if self.peek() == "{":
            self.next()
            children = self.sequence(depth + 1, stop="}")
            if self.peek() == "}":
                self.next()
            return self.node("mrow", children)
        # _base, not atom: an unbraced argument is a single symbol, and letting
        # it take the NEXT script as its own turns x_i^2 into x subscript (i
        # squared) rather than x sub i, squared.
        while self.peek() and not self.peek().strip():
            self.next()
        return self._base(self.next(), depth + 1) or self.node("mrow", [])

    def sequence(self, depth: int, stop: str = "") -> List[Dict[str, Any]]:
        if depth > MAX_DEPTH:
            raise _Over()
        out: List[Dict[str, Any]] = []
        while self.peek() and self.peek() != stop:
            if stop == "}" and self.peek() == "}":
                break
            before = self.at
            node = self.atom(depth)
            if node is not None:
                out.append(node)
            if self.at == before:        # nothing consumed: stop rather than spin
                self.next()
        return out

    # -- atoms ----------------------------------------------------------- #

    def atom(self, depth: int) -> Optional[Dict[str, Any]]:
        if depth > MAX_DEPTH:
            raise _Over()
        token = self.next()
        if not token:
            return None

        base = self._base(token, depth)
        if base is None:
            return None
        return self._scripts(base, depth)

    def _base(self, token: str, depth: int) -> Optional[Dict[str, Any]]:
        if token in ("}", "&", "\\\\"):
            return None
        if not token.strip():
            return None                   # spacing: meaningful only inside \text
        if token.startswith("\\,") or token.startswith("\\;") or token.startswith("\\!") \
                or token in ("\\ ", "\\:"):
            return self.node("mspace")

        if token.startswith("\\begin{"):
            return self._matrix(token[7:-1], depth)
        if token.startswith("\\end{"):
            return None

        if token.startswith("\\"):
            return self._command(token[1:], depth)

        if token == "{":
            children = self.sequence(depth + 1, stop="}")
            if self.peek() == "}":
                self.next()
            return self.node("mrow", children)

        if re.fullmatch(r"\d+(?:\.\d+)?", token):
            return self.node("mn", text=token)
        if re.fullmatch(r"[a-zA-Z]", token):
            return self.node("mi", text=token)
        if token in "+-*/=<>(),.|[]":
            return self.node("mo", text={"*": "⋅", "-": "−"}.get(token, token))
        return self.node("mi", text=token)

    def _command(self, name: str, depth: int) -> Optional[Dict[str, Any]]:
        if name in ("frac", "dfrac", "tfrac"):
            top = self.group(depth + 1)
            bottom = self.group(depth + 1)
            return self.node("mfrac", [top, bottom])

        if name == "sqrt":
            if self.peek() == "[":
                self.next()
                index = self.sequence(depth + 1, stop="]")
                if self.peek() == "]":
                    self.next()
                return self.node("mroot", [self.group(depth + 1),
                                           self.node("mrow", index)])
            return self.node("msqrt", [self.group(depth + 1)])

        if name in _BIG:
            operator = self.node("mo", text=_BIG[name])
            lower = upper = None
            if self.peek() == "_":
                self.next()
                lower = self.group(depth + 1)
            if self.peek() == "^":
                self.next()
                upper = self.group(depth + 1)
            if lower is not None and upper is not None:
                return self.node("munderover", [operator, lower, upper])
            if lower is not None:
                return self.node("munder", [operator, lower])
            if upper is not None:
                return self.node("mover", [operator, upper])
            return operator

        if name in ("mathrm", "text", "textrm", "mathbf", "mathit", "operatorname"):
            return self.node("mtext", text=self._literal(depth))

        if name in _GREEK:
            return self.node("mi", text=_GREEK_CHARS.get(name, name))
        if name in _FUNCTIONS:
            return self.node("mi", text=name)
        if name in _SYMBOLS:
            return self.node("mo", text=_SYMBOLS[name])
        if name in ("left", "right", "displaystyle", "limits", "quad", "qquad"):
            return self.node("mspace") if name.endswith("quad") else None
        # An unknown command renders as its own name rather than disappearing:
        # a reader can see what was meant and what was not understood.
        return self.node("mi", text=name)

    def _literal(self, depth: int) -> str:
        """The raw text of the next {..}, for \\text and \\mathrm."""
        if self.peek() != "{":
            return self.next()
        self.next()
        parts: List[str] = []
        while self.peek() and self.peek() != "}":
            parts.append(self.next())
            if len(parts) > 120:
                raise _Over()
        if self.peek() == "}":
            self.next()
        # Joined with nothing: the tokeniser emits one letter at a time and
        # emits the spaces between words, so the pieces already carry the
        # spacing. Joining with spaces would give "m / s" and "h e l l o".
        literal = "".join(parts).strip()
        # Angle brackets carry no meaning in \text or \mathrm, and dropping them
        # means the payload cannot contain something that LOOKS like a tag even
        # as text. The page sets this with textContent and would not parse it
        # either way; this is the second lock on the same door, so that a future
        # renderer written in a hurry still cannot be made to build markup.
        return _ANGLE.sub("", literal)[:200]

    def _scripts(self, base: Dict[str, Any], depth: int) -> Dict[str, Any]:
        sub = sup = None
        for _ in range(2):
            if self.peek() == "_" and sub is None:
                self.next()
                sub = self.group(depth + 1)
            elif self.peek() == "^" and sup is None:
                self.next()
                sup = self.group(depth + 1)
            else:
                break
        if sub is not None and sup is not None:
            return self.node("msubsup", [base, sub, sup])
        if sub is not None:
            return self.node("msub", [base, sub])
        if sup is not None:
            return self.node("msup", [base, sup])
        return base

    def _matrix(self, environment: str, depth: int) -> Dict[str, Any]:
        """A matrix environment as an mtable, with its own brackets if it has any."""
        rows: List[List[Dict[str, Any]]] = [[]]
        cell: List[Dict[str, Any]] = []
        while self.peek() and not self.peek().startswith("\\end{"):
            token = self.peek()
            if token == "&":
                self.next()
                rows[-1].append(self.node("mtd", cell))
                cell = []
                continue
            if token == "\\\\":
                self.next()
                rows[-1].append(self.node("mtd", cell))
                cell = []
                rows.append([])
                continue
            before = self.at
            node = self.atom(depth + 1)
            if node is not None:
                cell.append(node)
            if self.at == before:
                self.next()
        if self.peek().startswith("\\end{"):
            self.next()
        rows[-1].append(self.node("mtd", cell))

        table = self.node("mtable", [self.node("mtr", row) for row in rows if row])
        opener, closer = {
            "bmatrix": ("[", "]"), "Bmatrix": ("{", "}"), "pmatrix": ("(", ")"),
            "vmatrix": ("|", "|"), "Vmatrix": ("‖", "‖"),
        }.get(environment, ("", ""))
        if not opener:
            return table
        return self.node("mrow", [self.node("mo", text=opener), table,
                                  self.node("mo", text=closer)])


# Structures whose children are not optional. \frac with nothing on top is not a
# fraction, it is a fraction that was cut off - and rendering it puts an empty
# box on screen where an equation was meant to be.
_NEEDS_CHILDREN = ("mfrac", "msqrt", "mroot", "msup", "msub", "msubsup",
                   "munder", "mover", "munderover")


def _has_hole(node: Any) -> bool:
    """Is there a structural element here with an empty slot?"""
    if not isinstance(node, dict):
        return False
    children = node.get("c") or []
    if node.get("e") in _NEEDS_CHILDREN:
        if not children or any(not _carries_content(child) for child in children):
            return True
    return any(_has_hole(child) for child in children)


def _carries_content(node: Any) -> bool:
    if not isinstance(node, dict):
        return False
    if node.get("t"):
        return True
    return any(_carries_content(child) for child in node.get("c") or [])


def to_mathml(latex: str, display: bool = True) -> Optional[Dict[str, Any]]:
    """`latex` as a MathML tree, or None if it could not be read.

    None is the honest answer for anything malformed: a half-rendered equation
    is worse than the LaTeX it came from, because the LaTeX at least says what
    was meant.
    """
    if not isinstance(latex, str):
        return None
    latex = latex.strip()
    if not latex or len(latex) > MAX_EXPRESSION_CHARS:
        return None
    if latex.count("{") != latex.count("}"):
        return None                       # unbalanced: something was cut off
    try:
        reader = _Reader(latex)
        children = reader.sequence(0)
        if not children:
            return None
        tree = {"e": "math", "display": "block" if display else "inline",
                "c": [{"e": "mrow", "c": children}]}
        if _has_hole(tree):
            return None
        return tree
    except _Over:
        logger.debug("Expression too large or too deep to render.")
        return None
    except Exception:
        logger.debug("Couldn't read an expression as LaTeX.", exc_info=True)
        return None


# --------------------------------------------------------------------------- #
# The visual payload
# --------------------------------------------------------------------------- #

def visual(text: str, title: str = "") -> Optional[Dict[str, Any]]:
    """A "math" artifact for the expressions in `text`, or None.

    The shape core/orchestrator.py already pushes through visual_callback and
    gui/hud.html already dispatches on: {"type": ..., ...}. Nothing new carries
    it and nothing new renders the panel it lands in.
    """
    expressions = detect(text)
    if not expressions:
        return None
    rendered = []
    for expression in expressions:
        tree = to_mathml(expression["latex"], expression["display"])
        if tree is not None:
            # The LaTeX is echoed only so a surface with no MathML support can
            # show something rather than an empty box, and it is displayed as
            # text. Angle brackets go for the same reason they go from \text:
            # this payload should not carry anything that reads as a tag, so
            # that embedding it somewhere careless cannot end badly either.
            rendered.append({"mathml": tree,
                             "latex": _ANGLE.sub("", expression["latex"])})
    if not rendered:
        return None
    return {"type": "math", "title": title or "Mathematics", "items": rendered}


def element_names(tree: Any) -> List[str]:
    """Every element name in a tree. For the test that proves only ELEMENTS
    can ever come out of here."""
    out: List[str] = []
    if isinstance(tree, dict):
        name = tree.get("e")
        if isinstance(name, str):
            out.append(name)
        for child in tree.get("c") or []:
            out.extend(element_names(child))
    elif isinstance(tree, list):
        for child in tree:
            out.extend(element_names(child))
    return out
