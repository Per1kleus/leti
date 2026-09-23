"""What Leti says, and in what pieces.

Two guarantees, and they are the whole point of core/speech.py. Raw notation
never reaches the engine - a voice that reads out "backslash frac" is a voice
nobody will use. And nothing is ever cut where a person would not pause: a full
stop inside "3.14", "Dr. Adams" or "/etc/hosts" is not the end of a sentence,
and one utterance per word is the stutter this exists to prevent.
"""
from __future__ import annotations

import time

import pytest

from core import speech


# --- Nothing is said as markup --------------------------------------------------------

@pytest.mark.parametrize("answer", [
    r"The kinetic energy is \[E_k = \frac{1}{2}mv^2\] as we said.",
    r"We know that \(F = ma\) holds here.",
    r"Consider $$\int_0^L f(x)\,dx$$ over the beam.",
    r"The ratio $\frac{\sigma}{\epsilon}$ is Young's modulus.",
    r"Summing, \(\sum_{i=1}^{n} x_i\), gives the total.",
    r"The matrix \[\begin{bmatrix} a & b \\ c & d \end{bmatrix}\] is singular.",
    r"Here $\alpha$, $\beta$ and $\theta$ appear.",
    r"Speed is \(25 \, \mathrm{m/s}\) downstream.",
])
def test_no_latex_survives_into_speech(answer):
    said = speech.say(answer)
    assert not speech.contains_markup(said), f"markup reached the voice: {said!r}"
    for banned in ("\\frac", "\\int", "\\sum", "\\begin", "backslash", "$"):
        assert banned not in said


def test_every_utterance_is_clean_not_just_the_whole():
    """Chunking must not put markup back. Checked per utterance, because the
    engine is handed one at a time and one bad piece is one bad noise."""
    answer = (r"First, \(E = mc^2\). Then the fraction \(\frac{a}{b}\) matters. "
              r"Finally $$\sum_{i=1}^{n} i^2$$ closes it.")
    spoken = speech.utterances(answer)
    assert spoken
    for utterance in spoken:
        assert not speech.contains_markup(utterance), utterance


def test_an_equation_becomes_the_words_a_person_uses():
    said = speech.say(r"The energy is \(E_k = \frac{1}{2}mv^2\).")
    assert "one half" in said
    assert "squared" in said
    assert "equals" in said


def test_a_fraction_is_read_as_over():
    assert "over" in speech.say(r"\(a = \frac{F}{m}\)")


def test_an_integral_and_a_sum_are_named():
    assert "integral" in speech.say(r"\(\int_0^L f(x)dx\)")
    assert "sum" in speech.say(r"\(\sum_{i=1}^{n} x_i\)")


def test_a_matrix_is_not_read_out_cell_by_cell():
    said = speech.say(r"\[\begin{bmatrix} a & b \\ c & d \end{bmatrix}\]")
    assert "matrix" in said
    assert "bmatrix" not in said


def test_greek_letters_are_named_not_spelled():
    said = speech.say(r"\(\alpha + \beta = \omega\)")
    assert "alpha" in said and "beta" in said and "omega" in said


def test_prose_with_no_mathematics_is_left_alone():
    plain = "The meeting is at three and Chris will bring the figures."
    assert speech.say(plain) == plain


# --- Where a sentence ends, and where it does not --------------------------------------

@pytest.mark.parametrize("text,inside", [
    ("Pi is 3.14159 and that is that.", "3.14159"),
    ("Dr. Adams signed it off this afternoon without any trouble.", "Dr. Adams"),
    ("The file is at /etc/hosts on that machine, as you expected.", "/etc/hosts"),
    ("Look at example.com/a.b for the full write-up of the results.", "example.com/a.b"),
    ("Open report.md when you get a chance this afternoon please.", "report.md"),
    ("It travels at 9.8 m/s on the way down, which is standard.", "9.8 m/s"),
    ("That is the U.S. figure rather than the European one entirely.", "U.S."),
    ("Use `pip install .` to put it on the path for now.", "pip install ."),
])
def test_a_full_stop_inside_something_is_not_a_boundary(text, inside):
    spoken = speech.chunks(text)
    assert any(inside in utterance for utterance in spoken), \
        f"{inside!r} was split across utterances: {spoken}"


def test_real_sentences_are_separated():
    """Two full sentences, each comfortably over the joining floor, stay two -
    so a long answer has somewhere to be interrupted."""
    spoken = speech.chunks("The first measurement finished cleanly. "
                           "The second one is still running now.")
    assert len(spoken) == 2
    assert spoken[0].endswith(".")


def test_a_question_and_an_exclamation_also_end_a_sentence():
    spoken = speech.chunks("Did that work for you at all? It certainly looked like it did!")
    assert len(spoken) == 2


def test_nothing_is_ever_said_one_word_at_a_time():
    answer = ("The acceleration of the object is equal to the force divided by the mass, "
              "which is Newton's second law.")
    spoken = speech.chunks(answer)
    assert len(spoken) == 1, f"a single sentence was cut up: {spoken}"


def test_a_very_short_sentence_is_joined_rather_than_gasped():
    spoken = speech.chunks("Yes. That is exactly what the second measurement showed us.")
    assert len(spoken) == 1
    assert spoken[0].startswith("Yes.")


def test_a_very_long_sentence_is_broken_at_a_clause():
    long_one = ("The system reads the plan and then it reads the arguments; "
                + "it compares the two carefully against each other, "
                * 8 + "and finally it reports.")
    spoken = speech.chunks(long_one)
    assert len(spoken) > 1, "a sentence far past the ceiling was never broken"
    assert all(len(u) <= speech.MAX_CHUNK_CHARS * 2 for u in spoken)


def test_a_sentence_with_no_break_at_all_is_left_whole_rather_than_chopped():
    """Better said badly than cut mid-word: there is nowhere to pause here."""
    wall = "word " * 200
    spoken = speech.chunks(wall)
    assert len(spoken) == 1


def test_no_utterance_is_empty_or_only_punctuation():
    for answer in ["Hello. . . Done.", "  ", "One.  Two.   Three."]:
        for utterance in speech.chunks(answer):
            assert utterance.strip(), "an empty utterance would prime the engine for nothing"


def test_rewriting_does_not_leave_a_space_before_a_full_stop():
    said = speech.say(r"The rate is \(\frac{dy}{dx}\).")
    assert " ." not in said


# --- Bounds and cost ---------------------------------------------------------------------

def test_a_runaway_answer_does_not_hold_the_speaker_forever():
    spoken = speech.utterances("This is a sentence. " * 2000)
    assert sum(len(u) for u in spoken) <= speech.MAX_SPOKEN_CHARS + 64


def test_nothing_is_returned_for_nothing():
    assert speech.chunks("") == []
    assert speech.chunks(None) == []
    assert speech.utterances("") == []
    assert speech.say(None) == ""


def test_preparing_speech_is_cheap_enough_to_do_on_every_answer():
    answer = ("The acceleration is force over mass. " * 12 +
              r"We write that as \(a = \frac{F}{m}\).")
    speech.utterances(answer)
    started = time.perf_counter()
    for _ in range(200):
        speech.utterances(answer)
    each = (time.perf_counter() - started) / 200
    assert each < 0.01, f"{each * 1000:.1f} ms in front of every spoken answer"


def test_the_module_starts_nothing_and_calls_nothing():
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path("core/speech.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for forbidden in ("threading", "asyncio", "subprocess", "httpx", "requests",
                      "socket", "multiprocessing"):
        assert forbidden not in imported, f"core/speech.py imports {forbidden}"
