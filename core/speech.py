"""Turning an answer into something a voice can say, in pieces a voice can say well.

Leti speaks its answers. That was already true - core/orchestrator.py hands the
finished text to a speak_callback and audio/tts.py says it. What was missing is
everything between those two: the text a model writes is written to be READ.

Two problems come from that, and this module is both halves of the fix.

  The first is markup. A model asked about kinetic energy writes \\(E_k =
  \\frac{1}{2}mv^2\\), and a speech engine handed that says "backslash left
  paren E sub k equals backslash frac". Nobody wants to hear that. So `say()`
  rewrites mathematical notation into the words a person would use - "one half
  m v squared" - and leaves the original alone for the visual layer, which
  wants exactly the markup speech cannot use.

  The second is pacing. Speaking a whole answer as one utterance means nothing
  can be interrupted until it ends, and speaking it token by token produces the
  stutter that makes synthetic speech sound synthetic. So `chunks()` cuts at
  the places a person pauses - the end of a sentence, a strong break - and
  refuses to cut anywhere else. "3.14", "Dr. Adams", "/etc/hosts",
  "example.com/a.b" and "v = 9.8 m/s" all contain a full stop and none of them
  is the end of anything.

No model call, no network, no state. Both functions are pure, take a string and
return strings, and run in tens of microseconds - which is what lets them sit in
front of every spoken answer without being felt.
"""
from __future__ import annotations

import logging
import re
from typing import List

logger = logging.getLogger("leti.speech")

# The longest one utterance may get before it is cut at a weaker boundary. Past
# this a listener has waited too long for a pause and interruption has too little
# to grab. Well above a normal sentence, so ordinary prose is never touched.
MAX_CHUNK_CHARS = 320

# The shortest a chunk may be on its own. Under this it is joined to the next
# one instead: "Yes." spoken as its own utterance is the stutter this exists to
# avoid, and an engine re-primed for two syllables sounds like a hiccup.
MIN_CHUNK_CHARS = 24

# How much text is ever spoken in one call. A model that runs away does not get
# to hold the speaker for ten minutes with no way in.
MAX_SPOKEN_CHARS = 6000


# --------------------------------------------------------------------------- #
# What a full stop is not the end of
#
# Every pattern here protects a full stop, not a sentence. They are applied by
# masking the matched span before the split and restoring it after, so the
# splitter never sees a boundary that was not one.
# --------------------------------------------------------------------------- #

_PROTECTED = (
    # A url, with or without a scheme. Longest first so the scheme form wins.
    re.compile(r"\bhttps?://\S+", re.I),
    re.compile(r"\b(?:www\.)?[\w-]+\.(?:com|org|net|io|dev|ai|co\.uk|edu|gov)\b(?:/\S*)?", re.I),
    # An absolute or relative file path, posix or windows.
    re.compile(r"(?:[A-Za-z]:\\|\\\\|/|\./|~/)[\w./\\-]+"),
    # A bare filename with a known extension - "report.md", "main.py".
    re.compile(r"\b[\w-]+\.(?:py|md|txt|json|ya?ml|csv|pdf|docx?|xlsx?|html?|js|ts|sh|toml|ini|cfg|log)\b", re.I),
    # A number with a decimal point, a thousands separator, or a version.
    re.compile(r"\b\d+(?:\.\d+)+\b"),
    re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b"),
    # An initialism written with stops - "U.S.", "e.g.", "a.m.".
    re.compile(r"\b(?:[A-Za-z]\.){2,}"),
    # Code between backticks, and an inline unit like "9.8 m/s" or "25 kg".
    re.compile(r"`[^`\n]+`"),
    re.compile(r"\b\d+(?:\.\d+)?\s?(?:[munkMGT]?(?:m|s|g|N|Pa|J|W|V|A|K|Hz|B)|m/s|km/h|m/s2|kg|mm|cm|km|ms|kW|MPa|GPa)\b"),
)

# Abbreviations whose full stop is part of the word. Matched case-sensitively
# where case carries the meaning: "Dr." is a title, "dr" is not.
_ABBREVIATIONS = (
    "Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr", "St", "Mt", "Rev", "Hon",
    "Inc", "Ltd", "Co", "Corp", "Dept", "Univ", "Est",
    "vs", "etc", "eg", "ie", "al", "approx", "fig", "Fig", "No", "cf",
    "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Sept", "Oct", "Nov", "Dec",
    "Mon", "Tue", "Tues", "Wed", "Thu", "Thur", "Thurs", "Fri", "Sat", "Sun",
)
_ABBREVIATION = re.compile(
    r"\b(?:" + "|".join(re.escape(a) for a in _ABBREVIATIONS) + r")\.")

# A sentence ends at . ? ! or … followed by whitespace, once the protections
# above have taken their spans out of the running.
_SENTENCE_END = re.compile(r"(?<=[.!?…])[\"'”’)\]]*\s+")

# Weaker places to break a sentence that has gone on too long. Semicolons and
# colons first, then a comma before a conjunction - all places a person breathes.
_CLAUSE_BREAK = re.compile(r"(?<=[;:])\s+|(?<=,)\s+(?=(?:and|but|or|so|then|which|while|because|although)\b)", re.I)

# The private-use character used to hold a protected span's place. Outside any
# alphabet, so it cannot collide with text a model produced.
_MASK = "\ue000"


def _mask(text: str):
    """Replace every protected span with a placeholder, keeping the originals."""
    kept: List[str] = []

    def take(match: re.Match) -> str:
        kept.append(match.group(0))
        # A placeholder of the same length keeps offsets honest for anything
        # measuring the result, and contains no character the splitter looks at.
        return _MASK + str(len(kept) - 1) + _MASK

    masked = text
    for pattern in _PROTECTED:
        masked = pattern.sub(take, masked)
    masked = _ABBREVIATION.sub(take, masked)
    return masked, kept


def _unmask(text: str, kept: List[str]) -> str:
    def give(match: re.Match) -> str:
        index = int(match.group(1))
        return kept[index] if 0 <= index < len(kept) else match.group(0)

    return re.sub(_MASK + r"(\d+)" + _MASK, give, text)


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #

def chunks(text: str) -> List[str]:
    """The utterances this text should be spoken as, in order.

    One per sentence, except that a very short sentence is joined to the one
    after it and a very long one is cut at a clause. Never one per word and
    never one per token: every boundary here is somewhere a person would pause.
    """
    if not isinstance(text, str):
        return []
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    cleaned = cleaned[:MAX_SPOKEN_CHARS]

    masked, kept = _mask(cleaned)
    # Unmasked as soon as the sentence split is done, because every decision
    # after this one is about LENGTH - and a placeholder standing in for
    # "example.com/a.b" is a quarter of its width, which would make a full
    # sentence look like a fragment and get it glued to its neighbour.
    pieces = [_unmask(p, kept).strip() for p in _SENTENCE_END.split(masked) if p.strip()]

    # A sentence longer than the ceiling is cut again at a clause, and if it has
    # no clause boundary either it is left whole - a wall of text with no pause
    # in it is better said badly than chopped mid-word.
    split_long: List[str] = []
    for piece in pieces:
        if len(piece) <= MAX_CHUNK_CHARS:
            split_long.append(piece)
            continue
        split_long.extend(_split_clause(piece))

    return [p.strip() for p in _join_short(split_long) if p.strip()]


def _split_clause(piece: str) -> List[str]:
    parts = [p for p in _CLAUSE_BREAK.split(piece) if p and p.strip()]
    if len(parts) < 2:
        return [piece]
    # Re-join anything still over the ceiling with what follows rather than
    # cutting mid-clause; the ceiling is a preference, a word boundary is not.
    out: List[str] = []
    for part in parts:
        if out and len(out[-1]) + len(part) + 1 <= MAX_CHUNK_CHARS:
            out[-1] = out[-1] + " " + part
        else:
            out.append(part)
    return out


def _join_short(pieces: List[str]) -> List[str]:
    """Fold a too-short piece into its neighbour, so nothing is said in a gasp."""
    out: List[str] = []
    for piece in pieces:
        if out and len(out[-1]) < MIN_CHUNK_CHARS and len(out[-1]) + len(piece) + 1 <= MAX_CHUNK_CHARS:
            out[-1] = out[-1] + " " + piece
        else:
            out.append(piece)
    # A trailing fragment has nothing after it to join, so it goes backwards.
    if len(out) > 1 and len(out[-1]) < MIN_CHUNK_CHARS and \
            len(out[-2]) + len(out[-1]) + 1 <= MAX_CHUNK_CHARS:
        tail = out.pop()
        out[-1] = out[-1] + " " + tail
    return out


# --------------------------------------------------------------------------- #
# Saying mathematics
#
# The rewrites below are deliberately shallow. A full LaTeX reader belongs in
# core/math_render.py, where the result is looked at rather than heard; here the
# only job is that nothing reaches the engine as markup. An expression this does
# not recognise is reduced to its letters and numbers, which reads as clumsy
# English - and clumsy English is the correct failure, because the alternative
# is "backslash left brace".
# --------------------------------------------------------------------------- #

# Display and inline math delimiters, in the forms models actually emit.
_MATH_SPANS = (
    re.compile(r"\$\$(.+?)\$\$", re.S),
    re.compile(r"\\\[(.+?)\\\]", re.S),
    re.compile(r"\\\((.+?)\\\)", re.S),
    re.compile(r"(?<!\$)\$([^$\n]+)\$(?!\$)"),
)

_GREEK = {
    "alpha": "alpha", "beta": "beta", "gamma": "gamma", "delta": "delta",
    "epsilon": "epsilon", "zeta": "zeta", "eta": "eta", "theta": "theta",
    "iota": "iota", "kappa": "kappa", "lambda": "lambda", "mu": "mu",
    "nu": "nu", "xi": "xi", "pi": "pi", "rho": "rho", "sigma": "sigma",
    "tau": "tau", "upsilon": "upsilon", "phi": "phi", "chi": "chi",
    "psi": "psi", "omega": "omega",
}

# Powers people have a word for. Anything else becomes "to the power n".
_POWER_WORDS = {"2": "squared", "3": "cubed"}

_OPERATORS = (
    (re.compile(r"\\(?:times|cdot)\b"), " times "),
    (re.compile(r"\\div\b"), " divided by "),
    (re.compile(r"\\pm\b"), " plus or minus "),
    (re.compile(r"\\(?:leq|le)\b"), " is less than or equal to "),
    (re.compile(r"\\(?:geq|ge)\b"), " is greater than or equal to "),
    (re.compile(r"\\neq\b"), " is not equal to "),
    (re.compile(r"\\approx\b"), " is approximately "),
    (re.compile(r"\\infty\b"), " infinity "),
    (re.compile(r"\\partial\b"), " partial "),
    (re.compile(r"\\pi\b"), " pi "),
)


def _fractions(expression: str) -> str:
    """\\frac{a}{b} -> "a over b", innermost first so nesting reads correctly."""
    pattern = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
    for _ in range(6):                      # bounded: nesting deeper than this is not speech
        replaced = pattern.sub(lambda m: f" {m.group(1)} over {m.group(2)} ", expression)
        if replaced == expression:
            break
        expression = replaced
    # The two fractions everybody says as a word rather than a ratio.
    expression = re.sub(r"\b1 over 2\b", "one half", expression)
    expression = re.sub(r"\b1 over 3\b", "one third", expression)
    return expression


def _say_expression(expression: str) -> str:
    """One mathematical expression, as words. Never returns markup."""
    said = expression

    said = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r" the square root of \1 ", said)
    said = _fractions(said)
    said = re.sub(r"\\sum\s*(?:_\{[^{}]*\})?\s*(?:\^\{[^{}]*\})?", " the sum of ", said)
    said = re.sub(r"\\int\s*(?:_\{[^{}]*\}|_\d)?\s*(?:\^\{[^{}]*\}|\^\w)?", " the integral of ", said)
    said = re.sub(r"\\begin\{[a-z]*matrix\}.*?\\end\{[a-z]*matrix\}", " a matrix ", said, flags=re.S)
    said = re.sub(r"\\(?:mathrm|text|mathbf|mathit|operatorname)\s*\{([^{}]*)\}", r" \1 ", said)

    for pattern, word in _OPERATORS:
        said = pattern.sub(word, said)

    # Greek letters, capitalised or not.
    def greek(match: re.Match) -> str:
        name = match.group(1)
        word = _GREEK.get(name.lower())
        return f" {word} " if word else " "

    said = re.sub(r"\\([A-Za-z]+)", greek, said)

    # Sub- and superscripts, braced or bare.
    said = re.sub(r"\^\s*\{([^{}]*)\}", lambda m: _power(m.group(1)), said)
    said = re.sub(r"\^\s*(\w)", lambda m: _power(m.group(1)), said)
    said = re.sub(r"_\s*\{([^{}]*)\}", r" sub \1 ", said)
    said = re.sub(r"_\s*(\w)", r" sub \1 ", said)

    said = said.replace("=", " equals ").replace("+", " plus ")
    said = re.sub(r"(?<=[\w\s)])-(?=[\w\s(])", " minus ", said)
    said = re.sub(r"(?<=\w)/(?=\w)", " over ", said)

    # Whatever is left of the markup goes, rather than being pronounced.
    said = re.sub(r"[\\{}$&]", " ", said)
    said = said.replace("\\\\", " ")
    return " ".join(said.split())


def _power(exponent: str) -> str:
    exponent = exponent.strip()
    word = _POWER_WORDS.get(exponent)
    return f" {word} " if word else f" to the power {exponent} "


def say(text: str) -> str:
    """`text` with every mathematical expression replaced by how it is said.

    The input is left untouched apart from those spans, so an answer with no
    mathematics in it comes back as itself.
    """
    if not isinstance(text, str) or not text:
        return text if isinstance(text, str) else ""
    spoken = text
    for pattern in _MATH_SPANS:
        spoken = pattern.sub(lambda m: " " + _say_expression(m.group(1)) + " ", spoken)
    # Markdown emphasis and headings are punctuation to a reader and noise to a
    # listener. Stripped here rather than in chunks() so the visual layer keeps
    # them.
    spoken = re.sub(r"(?m)^#{1,6}\s*", "", spoken)
    spoken = re.sub(r"\*\*([^*]+)\*\*", r"\1", spoken)
    spoken = re.sub(r"(?<!\w)\*([^*\n]+)\*(?!\w)", r"\1", spoken)
    spoken = " ".join(spoken.split())
    # Rewriting an expression leaves a space where its delimiter was, so a
    # sentence can end " ." - which some engines read as a pause and a beat.
    return re.sub(r"\s+([.,;:!?])", r"\1", spoken)


def utterances(text: str) -> List[str]:
    """What the speech engine is actually given: math as words, cut at pauses.

    The one function the voice path needs. `say` then `chunks`, in that order,
    so a rewritten expression is chunked as the words it became rather than as
    the markup it was.
    """
    return chunks(say(text))


def contains_markup(text: str) -> bool:
    """Is there still raw notation in here? Used by the tests that guarantee none
    of it reaches the engine, and cheap enough to assert with."""
    if not isinstance(text, str):
        return False
    return bool(re.search(r"\\[A-Za-z]+|\\[\[\]()]|\$\$?|\{|\}|\^|(?<!\w)_(?!\w)", text))
