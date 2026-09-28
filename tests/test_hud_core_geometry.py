"""The core in the middle of the HUD has to be a closed, smooth ring.

It was not. It was drawn as a Catmull-Rom curve through BINS points whose radii
came from four different per-bin sources, and every one of those sources had a
step in it between the last point and the first - because a closed curve needs a
periodic input and none of them was periodic. The worst of them, the live
microphone, put the loudest part of a voice next to the quietest at twelve
o'clock and tore the ring open there on every frame.

Measured, as the sharpest turn anywhere on the densely sampled outline. A perfect
circle measures 0.32 degrees at this sampling:

                              before      after
    idle                        1.25       0.43   worst corner was at 99.9% round
    listening, no browser mic   2.56       0.43   ... at 98.0% round
    speaking                    6.58       1.38
    listening, real mic        12-15  1.01-1.32   ... at 0-4% round

The seam between the last point and the first went from 16.7 pixels of radius,
where neighbours are 7.1 pixels apart, to 0.000.

"99.9% round" and "0-4% round" are the seam. That the worst corner sat there, in
three of the four states, is what identified the cause.

These tests are structural and read gui/hud.html, the way the other GUI tests in
this directory do: there is no DOM here to run it in, and the properties that
matter - periodic inputs, a mirrored spectrum, smoothing that wraps - are visible
in the source and cannot be satisfied by accident. The numeric check that produced
the table above lives beside them and runs when node is available.
"""
from __future__ import annotations

import pathlib
import re
import shutil
import subprocess

import pytest

HUD = pathlib.Path(__file__).resolve().parent.parent / "gui" / "hud.html"


@pytest.fixture(scope="module")
def hud() -> str:
    return HUD.read_text(encoding="utf-8")


def _number(hud: str, name: str) -> int:
    match = re.search(rf"\b{name}\s*=\s*(\d+)", hud)
    assert match, f"{name} is gone from the HUD"
    return int(match.group(1))


# --- Every per-bin wave has to come back to where it started ---------------------

def test_the_ring_harmonic_helper_exists_and_is_in_whole_turns(hud):
    """`i * k` only closes when k * BINS is a whole number of turns. The helper is
    what makes that impossible to get wrong by hand."""
    assert "const ringHarmonic = (turns) => (2 * Math.PI * turns) / BINS;" in hud


@pytest.mark.parametrize("name", ["IDLE_WAVES", "ALERT_WAVES",
                                  "SPEAK_WAVES", "SPEAK_RIPPLE"])
def test_each_wave_count_is_a_whole_number_of_turns(hud, name):
    turns = _number(hud, name)
    assert turns >= 1, f"{name} must complete at least one turn round the ring"
    assert float(turns).is_integer()


def test_no_animation_uses_a_raw_per_bin_phase_any_more(hud):
    """The four literals that did not close: 0.55, 0.42, 0.75 and 0.33. A new one
    would reopen the seam, so none of them may come back as `i*<number>`."""
    offenders = re.findall(r"i\s*\*\s*0\.\d+", hud)
    assert not offenders, (
        "a per-bin phase went back to a raw multiplier instead of ringHarmonic: "
        f"{offenders}")


def test_every_target_wave_goes_through_the_harmonic_helper(hud):
    body = hud[hud.index("function updateTargets("):]
    body = body[:body.index("\n  function tickFrame(")]
    phases = re.findall(r"i\s*\*\s*([A-Za-z0-9_.()]+)", body)
    assert phases, "updateTargets no longer has a per-bin phase at all"
    for phase in phases:
        assert phase.startswith("ringHarmonic("), f"`i * {phase}` does not close the ring"


# --- The live spectrum ------------------------------------------------------------

def test_the_spectrum_is_mirrored_round_the_ring(hud):
    """A spectrum is not periodic and a ring is. Reading it in frequency order put
    the loud low end next to the silent top end at twelve o'clock. Mirrored, low
    meets low at the seam and high meets high opposite it."""
    body = hud[hud.index("function updateTargets("):]
    assert "target[b] = mean;" in body
    assert "target[BINS - b - 1] = mean;" in body


def test_each_band_is_an_average_not_a_single_sample(hud):
    """It read one FFT bin per point and threw the rest away, so neighbouring
    points got unrelated samples of a noisy spectrum."""
    body = hud[hud.index("function updateTargets("):]
    assert "for(let k=lo; k<hi; k++) sum += dataArray[k];" in body
    assert "const mean = sum / (hi - lo) / 255;" in body
    assert "dataArray[i*step]" not in hud, "the point-sampling came back"


def test_the_smoothing_wraps_round_the_ring(hud):
    """Smoothing that stopped at the ends of the array would leave the seam exactly
    where it was, and put a second discontinuity at the other end."""
    assert "function smoothRing(" in hud
    assert "ringScratch[(i - 1 + BINS) % BINS]" in hud
    assert "ringScratch[(i + 1) % BINS]" in hud
    assert _number(hud, "RING_SMOOTHING_PASSES") >= 1


# --- The speaking wobble ----------------------------------------------------------

def test_the_noise_field_is_built_from_anchors_not_per_bin_randomness(hud):
    """It was one independent random value per bin: smooth through time, white
    around the ring. Neighbours 7 pixels apart were handed radii up to 19 pixels
    apart, which is what made the speaking core look crinkled."""
    assert "function fillRingNoise(" in hud
    assert _number(hud, "NOISE_ANCHORS") >= 4
    assert "noiseAnchors[(a + 1) % NOISE_ANCHORS]" in hud, (
        "the last anchor must interpolate back to the first, or the field is not "
        "periodic and the seam is still there")
    assert "noiseTo[i] = Math.random()" not in hud, "the white-noise field came back"


def test_the_noise_field_is_still_refreshed_over_time(hud):
    """Smooth around the ring must not have cost it its movement."""
    assert "noiseFrom.set(noiseTo);" in hud
    assert "fillRingNoise(noiseTo);" in hud
    assert "noisePhase += dt * ticksPerSecond;" in hud


def test_the_wobble_was_not_flattened_to_achieve_any_of_this(hud):
    """The brief's rule: not fixed by turning the animation off. The amplitudes the
    speaking wave is built from are unchanged, and so is the noise's weight."""
    body = hud[hud.index("if(mode === 'speaking'){"):]
    body = body[:body.index("return;")]
    for piece in ("0.34", "0.20*Math.sin", "0.13*Math.sin", "noiseAt(i) * 0.30"):
        assert piece in body, f"the speaking wave lost {piece}"


def test_the_animation_still_runs_in_every_state(hud):
    """Nothing here may be achieved by not drawing. Each state still has a target."""
    body = hud[hud.index("function updateTargets("):]
    body = body[:body.index("\n  function tickFrame(")]
    assert "mode === 'listening' && analyser" in body
    assert "mode === 'speaking'" in body
    assert "mode === 'listening'" in body
    assert "target[i] = 0.10 + 0.055*Math.sin" in body, "the idle breath is gone"


def test_the_states_the_core_moves_between_are_all_still_there(hud):
    """Idle -> Listening -> Processing -> Speaking -> Idle, as the brief names them.
    "Processing" is thinking and executing, which share the idle animation on
    purpose - see the comment on STATES - so this checks the state table, not the
    animation count."""
    for state in ("idle:", "listening:", "thinking:", "executing:", "speaking:"):
        assert state in hud, f"the {state} state is gone from STATES"
    assert "const BUSY_STATES = ['listening', 'thinking', 'executing', 'speaking'];" in hud


# --- The numeric check that produced the table in the docstring ------------------

RING_MEASURE = pathlib.Path(__file__).resolve().parent / "hud_ring_measure.js"


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not installed; the structural tests above still run")
def test_the_drawn_curve_has_no_corner_sharper_than_a_gentle_one():
    """Runs the real functions out of gui/hud.html and measures the curve they
    produce, in every state, including against a synthetic speech spectrum.

    The threshold is 2 degrees: a perfect circle measures 0.32 at this sampling,
    the worst state now measures 1.38, and the live microphone measured 12 to 15
    before any of this. Anything approaching 2 means a seam has reopened.

    Both regressions were reintroduced deliberately to check this is not vacuous.
    Putting the frequency-order point sampling back measures 20.46; putting the
    per-bin random noise back measures 8.23. Both fail here.
    """
    result = subprocess.run(["node", str(RING_MEASURE), str(HUD)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    worst = float(re.search(r"WORST ([\d.]+)", result.stdout).group(1))
    assert worst < 2.0, f"the core has a {worst:.2f} degree corner in it:\n{result.stdout}"
