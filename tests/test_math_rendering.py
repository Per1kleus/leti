"""Mathematics you can look at, and the things it must never become.

Two halves. The equations in §11 of the brief all have to render - a fraction, an
integral, a derivative, a sum, Greek letters, a matrix, engineering units - and
nothing that is not mathematics may open a panel, because a window over the
interface for "the answer is 42" is worse than no window.

The third thing, which is not about mathematics at all: the renderer must not be
a way to put markup on the page. Every test at the bottom of this file is about
that, because a math panel that accepts a string of HTML is an arbitrary-markup
path into an interface that otherwise has none.
"""
from __future__ import annotations

import pathlib
import time

import pytest

from core import math_render as mr


def _names(latex, display=True):
    tree = mr.to_mathml(latex, display)
    assert tree is not None, f"{latex!r} did not render"
    return set(mr.element_names(tree))


def _texts(tree):
    out = []
    if isinstance(tree, dict):
        if tree.get("t"):
            out.append(tree["t"])
        for child in tree.get("c") or []:
            out.extend(_texts(child))
    return out


# --- The notation the brief asks for ---------------------------------------------------

def test_an_inline_equation_renders():
    tree = mr.to_mathml("F = ma", display=False)
    assert tree["display"] == "inline"
    assert "".join(_texts(tree)).replace(" ", "") == "F=ma"


def test_a_display_equation_renders():
    assert mr.to_mathml("F = ma", display=True)["display"] == "block"


def test_a_fraction_renders_as_a_fraction():
    assert "mfrac" in _names(r"a = \frac{F}{m}")


def test_an_integral_renders_with_its_limits():
    names = _names(r"\int_0^L f(x)\,dx")
    assert "munder" in names or "munderover" in names
    assert "∫" in _texts(mr.to_mathml(r"\int_0^L f(x)\,dx"))


def test_a_derivative_renders_as_a_fraction():
    assert "mfrac" in _names(r"\frac{dy}{dx}")


def test_a_summation_renders_with_both_limits():
    assert "munderover" in _names(r"\sum_{i=1}^{n} x_i")
    assert "∑" in _texts(mr.to_mathml(r"\sum_{i=1}^{n} x_i"))


@pytest.mark.parametrize("name,character", [
    (r"\alpha", "α"), (r"\beta", "β"), (r"\theta", "θ"), (r"\omega", "ω"),
    (r"\sigma", "σ"), (r"\Omega", "Ω"), (r"\pi", "π"),
])
def test_greek_symbols_render_as_symbols(name, character):
    assert character in _texts(mr.to_mathml(name))


def test_a_matrix_renders_as_a_table_with_its_brackets():
    names = _names(r"\begin{bmatrix} a & b \\ c & d \end{bmatrix}")
    assert {"mtable", "mtr", "mtd"} <= names
    texts = _texts(mr.to_mathml(r"\begin{bmatrix} a & b \\ c & d \end{bmatrix}"))
    assert "[" in texts and "]" in texts


def test_a_matrix_keeps_its_shape():
    tree = mr.to_mathml(r"\begin{bmatrix} a & b \\ c & d \end{bmatrix}")
    rows = _rows(tree)
    assert len(rows) == 2, f"a two-row matrix rendered as {len(rows)}"
    assert all(len(row) == 2 for row in rows), "a column went missing"


def _rows(node):
    if isinstance(node, dict):
        if node.get("e") == "mtable":
            return [[c for c in (row.get("c") or [])]
                    for row in (node.get("c") or [])]
        for child in node.get("c") or []:
            found = _rows(child)
            if found:
                return found
    return []


def test_engineering_units_render_upright():
    assert "mtext" in _names(r"25 \, \mathrm{m/s}")
    assert "m/s" in " ".join(_texts(mr.to_mathml(r"25 \, \mathrm{m/s}")))


def test_a_ratio_of_greek_letters_renders():
    names = _names(r"E = \frac{\sigma}{\epsilon}")
    assert "mfrac" in names


def test_a_square_root_renders():
    assert "msqrt" in _names(r"\sqrt{x^2 + y^2}")


def test_a_power_renders_as_a_superscript():
    assert "msup" in _names("E = mc^2")


def test_a_subscript_and_a_superscript_together():
    assert "msubsup" in _names("x_i^2")


# --- What must not render ---------------------------------------------------------------

@pytest.mark.parametrize("broken", [
    r"\frac{", r"\frac{a}{", r"x^", r"\sqrt{}", r"{{{{", r"\frac{}{}",
])
def test_malformed_mathematics_is_refused_rather_than_half_drawn(broken):
    assert mr.to_mathml(broken) is None, \
        "half an equation on screen is worse than the LaTeX it came from"


def test_nothing_at_all_renders_to_nothing():
    assert mr.to_mathml("") is None
    assert mr.to_mathml(None) is None
    assert mr.to_mathml("   ") is None


def test_a_runaway_expression_is_refused():
    assert mr.to_mathml("x" * (mr.MAX_EXPRESSION_CHARS + 1)) is None
    assert mr.to_mathml(r"\frac{a}{b}" * 400) is None


def test_deep_nesting_is_refused_rather_than_recursed():
    assert mr.to_mathml(r"\frac{" * 40 + "a" + r"}{b}" * 40) is None


# --- Detection is conservative ----------------------------------------------------------

@pytest.mark.parametrize("prose", [
    "The answer is 42.",
    "There are 3 files in that folder and 2 of them are empty.",
    "I sent the email to chris@example.com this morning.",
    "The meeting is at 2 and it runs until 4.",
    "Revenue was up 12% year on year.",
    "Use a = b to assign it.",
])
def test_ordinary_prose_is_not_mathematics(prose):
    assert mr.detect(prose) == [], f"a panel would have opened for {prose!r}"
    assert mr.visual(prose) is None


@pytest.mark.parametrize("answer,expected", [
    (r"We know \(E = mc^2\) well.", "E = mc^2"),
    (r"Thus $$F = ma$$ holds.", "F = ma"),
    (r"The ratio $\frac{a}{b}$ is fixed.", r"\frac{a}{b}"),
    ("Written as \\[x = \\frac{-b}{2a}\\] here.", r"x = \frac{-b}{2a}"),
])
def test_delimited_mathematics_is_found(answer, expected):
    found = mr.detect(answer)
    assert found and found[0]["latex"] == expected


def test_display_and_inline_are_told_apart():
    assert mr.detect(r"$$F = ma$$")[0]["display"] is True
    assert mr.detect(r"\(F = ma\)")[0]["display"] is False


def test_the_inside_of_a_display_block_is_not_found_twice():
    found = mr.detect(r"$$E = mc^2$$")
    assert len(found) == 1, f"one equation was counted {len(found)} times"


def test_only_a_handful_of_expressions_leave():
    answer = " ".join(rf"\(x_{i} = {i}^2\)" for i in range(30))
    assert len(mr.detect(answer)) <= mr.MAX_EXPRESSIONS


def test_a_visual_is_only_produced_where_something_rendered():
    assert mr.visual(r"Consider \(\frac{\)") is None       # unbalanced: nothing renders
    good = mr.visual(r"We have \(E = mc^2\).")
    assert good["type"] == "math" and len(good["items"]) == 1
    assert good["items"][0]["latex"] == "E = mc^2"


# --- The renderer is presentation only ---------------------------------------------------

_HOSTILE = [
    r"<script>alert(1)</script>",
    r"\text{<img src=x onerror=alert(1)>}",
    r"\mathrm{</math><script>alert(1)</script>}",
    r"E = mc^2 <iframe src=javascript:alert(1)>",
    r"\href{javascript:alert(1)}{click}",
    r"\begin{bmatrix}<b>a</b> & b \\ c & d\end{bmatrix}",
]


@pytest.mark.parametrize("hostile", _HOSTILE)
def test_no_element_outside_the_allowed_list_can_ever_come_out(hostile):
    """The whole security argument in one assertion.

    The payload is a tree of NAMED elements, not a string of markup. A name that
    is not in ELEMENTS cannot be produced, so there is no expression - however
    written - that becomes a tag, a handler or a script. The page checks the
    same list again on its side.
    """
    tree = mr.to_mathml(hostile)
    if tree is None:
        return                       # refusing outright is also a correct answer
    assert set(mr.element_names(tree)) <= mr.ELEMENTS


@pytest.mark.parametrize("hostile", _HOSTILE)
def test_hostile_input_never_produces_an_attribute_or_a_handler(hostile):
    tree = mr.to_mathml(hostile)
    if tree is None:
        return
    assert _keys(tree) <= {"e", "t", "c", "display"}, \
        "a payload key other than element/text/children/display could become an attribute"


def _keys(node):
    out = set()
    if isinstance(node, dict):
        out |= set(node.keys())
        for child in node.get("c") or []:
            out |= _keys(child)
    return out


@pytest.mark.parametrize("hostile", _HOSTILE)
def test_no_text_an_expression_carries_can_open_a_tag(hostile):
    """The property that makes markup impossible, rather than a list of words.

    Every string in the payload is set with textContent, which never parses -
    so even a literal "<script>" would be shown as those characters. On top of
    that, angle brackets are stripped from any text an expression carries, so
    the payload cannot contain something that even LOOKS like a tag. With no
    '<' there is no tag, and therefore no attribute and no handler, whatever
    words are left over.
    """
    tree = mr.to_mathml(hostile)
    if tree is None:
        return
    for text in _texts(tree):
        assert isinstance(text, str)
        if "<" in text or ">" in text:
            # Less-than and greater-than are real operators, so they are allowed
            # - but only ALONE, as the whole of one element's text. A tag needs a
            # name next to its bracket, and that cannot happen when the bracket
            # is the entire string and the string is set with textContent.
            assert text in ("<", ">"), f"a tag could be opened: {text!r}"


def test_the_payload_is_data_with_no_markup_string_in_it():
    visual = mr.visual(r"We have \[E_k = \frac{1}{2}mv^2\] here.")
    import json

    blob = json.dumps(visual)
    for banned in ("<", ">", "javascript:", "onerror", "<script"):
        assert banned not in blob, f"the payload carries {banned!r}"


def test_the_page_and_python_agree_on_which_elements_exist():
    """Two lists, deliberately - core/math_render.py's and gui/hud.html's. This
    is what keeps them the same list."""
    page = pathlib.Path("gui/hud.html").read_text()
    start = page.index("const MATHML_ELEMENTS = new Set([")
    end = page.index("]);", start)
    listed = {name.strip().strip("',\"")
              for name in page[start:end].split("[")[1].replace("\n", "").split(",")
              if name.strip().strip("',\"")}
    assert listed == set(mr.ELEMENTS), (
        f"the page allows {listed ^ set(mr.ELEMENTS)} that Python does not, or vice versa")


def test_the_page_builds_nodes_rather_than_parsing_markup():
    page = pathlib.Path("gui/hud.html").read_text()
    start = page.index("function buildMathML")
    body = page[start:page.index("function artifactMath")]
    for forbidden in ("innerHTML", "outerHTML", "insertAdjacentHTML", "eval(",
                      "document.write", "createContextualFragment"):
        assert forbidden not in body, f"the math renderer uses {forbidden}"
    assert "createElementNS" in body and "textContent" in body


# --- Cost ---------------------------------------------------------------------------------

def test_detection_is_cheap_on_the_answers_that_have_no_mathematics():
    """The common path. Every answer is looked at; almost none of them match."""
    prose = "The meeting is at three and Chris will bring the figures. " * 20
    mr.detect(prose)
    started = time.perf_counter()
    for _ in range(500):
        mr.detect(prose)
    each = (time.perf_counter() - started) / 500
    assert each < 0.002, f"{each * 1e6:.0f} us on every answer that has no mathematics"


def test_rendering_is_fast_enough_to_do_when_it_is_asked_for():
    latex = r"E_k = \frac{1}{2}mv^2 + \sum_{i=1}^{n} \frac{x_i^2}{2}"
    mr.to_mathml(latex)
    started = time.perf_counter()
    for _ in range(200):
        mr.to_mathml(latex)
    each = (time.perf_counter() - started) / 200
    assert each < 0.01, f"{each * 1000:.1f} ms per expression"


def test_the_module_reaches_nothing_outside_this_machine():
    import ast

    tree = ast.parse(pathlib.Path("core/math_render.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for forbidden in ("httpx", "requests", "urllib", "socket", "subprocess",
                      "threading", "matplotlib"):
        assert forbidden not in imported, f"core/math_render.py imports {forbidden}"
