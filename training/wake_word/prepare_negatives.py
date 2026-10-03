"""Build the negative side of the training data, locally.

WHY THIS IS NOT A DOWNLOAD

openWakeWord's own recipe fills its negative half from large precomputed feature
sets - ACAV100M-derived speech, AudioSet noise, the MIT impulse-response survey -
hosted on Hugging Face. None of those hosts is reachable from the environment
this model was trained in (see README.md, "What was not reachable"), and the
licence position of AudioSet-derived material is in any case less clear than
anything produced here. So the negative half is built from three sources that are
local, auditable and provenance-clean:

  negative speech   English sentences, spoken by the same multi-speaker
                    checkpoint as the positives but drawn from a wholly
                    different set of texts. Three kinds of text, because a
                    wake-word model's false positives come from three different
                    places: ordinary prose (what a person actually says near the
                    machine), phonetic neighbours of the target phrase (what
                    sounds like it), and word sequences with no sentence
                    structure at all (which stop the model learning that
                    "something sentence-shaped" means "not the wake word").

  background noise  Synthesised: coloured noise, mains hum, impulsive clatter,
                    and babble built by overlapping several speech clips. A room
                    with four people talking at once is the single most common
                    source of wake-word false positives, and babble is the part
                    of this that can be made faithfully without a corpus.

  impulse responses Reverberation. piper-sample-generator ships eight, under
                    MIT; the rest are synthesised as exponentially decaying
                    noise with a direct path, which is what a small room's
                    impulse response looks like to a model this size.

The honest consequence is written down rather than hidden: a false-positive rate
measured against synthetic negative audio is optimistic relative to a real room,
because synthesised speech varies less than people do. README.md says so, and the
threshold chosen in validate.py leaves margin for it.
"""
from __future__ import annotations

import argparse
import logging
import random
import re
import shutil
import wave
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("hey_leti.negatives")

SAMPLE_RATE = 16000

# Prose is read from the project's own writing. It is real English, it is here
# with no network and no licence question, and it is the register Leti is
# actually spoken to in - it talks about microphones, models, settings and files.
PROSE_SOURCES = ("README.md", "LETi_USER_GUIDE.txt")

# And the docstrings, which are where most of this project's English actually is.
# Ordinary explanatory prose about microphones, models, settings and files - the
# subject matter somebody is most likely to be talking about in the room where
# Leti is listening, which makes it the most useful kind of negative there is.
PROSE_CODE_DIRS = ("audio", "core", "gui", "launcher", "tools")


def _sentences(text: str) -> List[str]:
    """Split prose into speakable sentences, discarding what is not prose."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)        # code blocks
    text = re.sub(r"`[^`]*`", " ", text)                      # inline code
    text = re.sub(r"https?://\S+", " ", text)                 # urls
    text = re.sub(r"[#*_>|\[\]()]", " ", text)                # markdown furniture
    text = re.sub(r"\s+", " ", text)
    out = []
    for raw in re.split(r"(?<=[.!?])\s+", text):
        candidate = raw.strip()
        if not (20 <= len(candidate) <= 160):
            continue
        letters = sum(ch.isalpha() or ch.isspace() for ch in candidate)
        if letters / len(candidate) < 0.9:                    # tables, paths, numbers
            continue
        if len(candidate.split()) < 4:
            continue
        out.append(candidate)
    return out


def _docstrings(source: str) -> List[str]:
    """Every docstring in a Python file, as text."""
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node)
            if doc:
                out.append(doc)
    return out


def prose_texts(root: Path) -> List[str]:
    found: List[str] = []
    for name in PROSE_SOURCES:
        path = root / name
        if path.is_file():
            found.extend(_sentences(path.read_text(encoding="utf-8", errors="ignore")))
    for folder in PROSE_CODE_DIRS:
        for path in sorted((root / folder).rglob("*.py")):
            source = path.read_text(encoding="utf-8", errors="ignore")
            for doc in _docstrings(source):
                found.extend(_sentences(doc))
    # Said twice is still said once: the same sentence recorded many times would
    # weight the corpus towards whatever this project repeats most.
    seen = set()
    unique = []
    for sentence in found:
        key = sentence.lower()
        if key not in seen:
            seen.add(key)
            unique.append(sentence)
    return unique


def word_salad(rng: random.Random, count: int,
               min_words: int = 2, max_words: int = 7) -> List[str]:
    """Sequences of real words in no order, for phonetic breadth.

    Drawn from CMUdict, which is what `pronouncing` already carries, so this
    needs nothing downloaded. Sentence structure is deliberately absent.
    """
    import cmudict

    vocabulary = [w for w in cmudict.dict().keys()
                  if w.isalpha() and 3 <= len(w) <= 12]
    vocabulary.sort()                                   # a seeded run is repeatable
    return [" ".join(rng.choice(vocabulary)
                     for _ in range(rng.randint(min_words, max_words)))
            for _ in range(count)]


def phonetic_neighbours(seed_phrase: str, count: int,
                        rng_seed: int = 0) -> List[str]:
    """Phrases that sound like the wake word but are not it.

    openWakeWord's own `generate_adversarial_texts` does this, by searching
    CMUdict for words whose phonemes are one or two substitutions away. It is the
    single most valuable kind of negative: a model that has never heard "hay
    letty" will answer to it.

    `seed_phrase` is spelled so that CMUdict knows every word, because the
    fallback for an unknown word is a neural phonemiser downloaded from a host
    that is not reachable here. "letty" has the pronunciation of "Leti"
    (L EH1 T IY0), which is what the search actually works from - the spelling
    never reaches the model.
    """
    import numpy as np_local
    from openwakeword.data import generate_adversarial_texts

    np_local.random.seed(rng_seed)
    texts = generate_adversarial_texts(
        input_text=seed_phrase, N=count,
        include_partial_phrase=1.0, include_input_words=0.2)
    return [t for t in texts if t.strip()]


def write_texts(path: Path, texts: Sequence[str], target_phrase: str) -> int:
    """Write the negative texts, with the wake word itself removed.

    The filter is the point. Prose from this project mentions the wake word, and
    a negative clip that actually says it would teach the model that its own name
    is not its name.
    """
    needle = re.sub(r"[^a-z ]", " ", target_phrase.lower())
    needle = re.sub(r"\s+", " ", needle).strip()
    kept = []
    for text in texts:
        flat = re.sub(r"[^a-z ]", " ", text.lower())
        flat = re.sub(r"\s+", " ", flat).strip()
        if needle and needle in flat:
            continue
        kept.append(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return len(kept)



# --------------------------------------------------------------------------- #
# The positive side, and keeping the two apart
# --------------------------------------------------------------------------- #

STRESS_MARKS = {"\u02c8", "\u02cc", "\u0361", "\u0320"}


def spoken_form(text: str, voice: str = "en-us") -> str:
    """The phonemes a text is actually spoken as, ignoring stress and spacing.

    Two texts with the same spoken form are the same sound, whatever they look
    like, and one of them cannot be a negative example for the other.
    """
    import unicodedata

    from piper_phonemize import phonemize_espeak

    out = []
    for sentence in phonemize_espeak(text, voice):
        for phoneme in sentence:
            if (phoneme in STRESS_MARKS or phoneme.isspace()
                    or unicodedata.category(phoneme).startswith("P")):
                continue
            out.append(phoneme)
    return "".join(out)


def positive_lines(weights: Dict[str, int]) -> List[str]:
    """The positive corpus, as one line per draw.

    The generator picks a line uniformly, so repeating a phrase is how it gets
    weighted. Writing it out rather than carrying weights through the code keeps
    the corpus a file somebody can read.
    """
    lines: List[str] = []
    for text, weight in weights.items():
        lines.extend([text] * int(weight))
    return lines


def drop_homophones(adversarial: Sequence[str],
                    positives: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Remove adversarial phrases that sound exactly like a positive one.

    The phonetic-neighbour search excludes homophones of its seed, but the seed is
    one spelling and the positive corpus has several pronunciations in it. A
    phrase that is a homophone of any of them would be labelled negative while
    sounding positive, which teaches the model to hesitate on its own name.
    """
    wanted = {spoken_form(text) for text in positives}
    kept, dropped = [], []
    for text in adversarial:
        (dropped if spoken_form(text) in wanted else kept).append(text)
    return kept, dropped


# --------------------------------------------------------------------------- #
# Noise
# --------------------------------------------------------------------------- #

def _coloured(rng: np.random.Generator, n: int, exponent: float) -> np.ndarray:
    """Noise with a 1/f**exponent spectrum. 0 is white, 1 pink, 2 brown."""
    spectrum = rng.normal(size=n // 2 + 1) + 1j * rng.normal(size=n // 2 + 1)
    freqs = np.arange(len(spectrum))
    freqs[0] = 1
    spectrum /= freqs ** (exponent / 2.0)
    out = np.fft.irfft(spectrum, n=n)
    return out / (np.abs(out).max() or 1.0)


def _hum(rng: np.random.Generator, n: int) -> np.ndarray:
    """Mains hum with a few harmonics - what a cheap microphone adds by itself."""
    t = np.arange(n) / SAMPLE_RATE
    base = float(rng.choice([50.0, 60.0]))
    out = np.zeros(n)
    for harmonic in range(1, 6):
        out += (1.0 / harmonic) * np.sin(2 * np.pi * base * harmonic * t
                                         + rng.uniform(0, 2 * np.pi))
    out += 0.02 * _coloured(rng, n, 1.0)
    return out / (np.abs(out).max() or 1.0)


def _clatter(rng: np.random.Generator, n: int) -> np.ndarray:
    """Sparse impulsive noise: keys, cutlery, a keyboard, a door."""
    out = np.zeros(n)
    for _ in range(int(rng.integers(5, 40))):
        at = int(rng.integers(0, max(1, n - 1600)))
        length = int(rng.integers(200, 1600))
        burst = _coloured(rng, length, float(rng.uniform(0.0, 1.5)))
        envelope = np.exp(-np.linspace(0, rng.uniform(4, 12), length))
        out[at:at + length] += burst * envelope * rng.uniform(0.2, 1.0)
    return out / (np.abs(out).max() or 1.0)


def _tones(rng: np.random.Generator, n: int) -> np.ndarray:
    """Musical-ish material: a few held notes. Stands in for a radio."""
    t = np.arange(n) / SAMPLE_RATE
    out = np.zeros(n)
    for _ in range(int(rng.integers(2, 5))):
        semitone = int(rng.integers(-24, 25))
        freq = 220.0 * (2 ** (semitone / 12.0))
        start = int(rng.integers(0, max(1, n // 2)))
        length = int(rng.integers(n // 4, n - start)) if n - start > n // 4 else n - start
        seg = np.sin(2 * np.pi * freq * t[:length]) * np.hanning(length)
        out[start:start + length] += seg * rng.uniform(0.3, 1.0)
    out += 0.05 * _coloured(rng, n, 1.0)
    return out / (np.abs(out).max() or 1.0)


def _babble(rng: np.random.Generator, n: int, speech: Sequence[Path]) -> np.ndarray:
    """Several people talking at once.

    Overlapping speech is the hardest background for a wake-word model, and the
    one kind of realistic noise that can be built from clips rather than a
    corpus.
    """
    out = np.zeros(n)
    for _ in range(int(rng.integers(3, 9))):
        clip = _read_wav(speech[int(rng.integers(0, len(speech)))])
        if clip.size == 0:
            continue
        at = int(rng.integers(0, max(1, n)))
        chunk = clip[: max(0, n - at)]
        out[at:at + len(chunk)] += chunk * rng.uniform(0.3, 1.0)
    peak = np.abs(out).max()
    return out / peak if peak else out


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        raw = wav.readframes(wav.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _write_wav(path: Path, samples: np.ndarray) -> None:
    clipped = np.clip(samples, -1.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setframerate(SAMPLE_RATE)
        wav.setsampwidth(2)
        wav.setnchannels(1)
        wav.writeframes((clipped * 32767).astype(np.int16).tobytes())


def make_backgrounds(out_dir: Path, count: int, seconds: float = 10.0,
                     seed: int = 0, speech_dir: Optional[Path] = None) -> int:
    """Write `count` background clips, a mix of all the kinds above."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SAMPLE_RATE)
    speech = sorted(speech_dir.glob("*.wav")) if speech_dir else []
    kinds: List[str] = ["white", "pink", "brown", "hum", "clatter", "tones"]
    if speech:
        kinds += ["babble", "babble", "babble"]      # weighted: the realistic one
    else:
        logger.warning("No speech clips given, so no babble will be made.")
    made = 0
    for i in range(count):
        kind = kinds[i % len(kinds)]
        if kind == "white":
            samples = _coloured(rng, n, 0.0)
        elif kind == "pink":
            samples = _coloured(rng, n, 1.0)
        elif kind == "brown":
            samples = _coloured(rng, n, 2.0)
        elif kind == "hum":
            samples = _hum(rng, n)
        elif kind == "clatter":
            samples = _clatter(rng, n)
        elif kind == "tones":
            samples = _tones(rng, n)
        else:
            samples = _babble(rng, n, speech)
        _write_wav(out_dir / f"{kind}_{i:04d}.wav", samples * rng.uniform(0.2, 1.0))
        made += 1
    return made


def make_impulse_responses(out_dir: Path, count: int, seed: int = 0,
                           bundled: Optional[Path] = None) -> int:
    """Reverberation. The bundled MIT-licensed ones, plus synthesised rooms."""
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    if bundled and bundled.is_dir():
        for path in sorted(bundled.glob("*.wav")):
            shutil.copy(path, out_dir / f"piper_{path.name.replace(' ', '_')}")
            copied += 1
    rng = np.random.default_rng(seed)
    for i in range(count):
        # A direct path, then noise decaying exponentially: the shape of a small
        # room's response, with the decay time as the room's size.
        rt60 = float(rng.uniform(0.15, 0.9))
        n = int(rt60 * 1.5 * SAMPLE_RATE)
        tail = _coloured(rng, n, float(rng.uniform(0.0, 1.0)))
        decay = np.exp(-np.arange(n) / (rt60 * SAMPLE_RATE / 6.9))
        response = tail * decay
        delay = int(rng.integers(0, 120))
        response[:delay] = 0.0
        response[delay] = 1.0
        _write_wav(out_dir / f"synthetic_room_{i:03d}.wav",
                   response / (np.abs(response).max() or 1.0))
    return copied + count


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="what", required=True)

    texts = sub.add_parser("texts", help="write the negative text corpus")
    texts.add_argument("--out", required=True)
    texts.add_argument("--project-root", default=".")
    texts.add_argument("--target-phrase", default="hey leti")
    texts.add_argument("--neighbour-seed-phrase", default="hey letty")
    texts.add_argument("--neighbours", type=int, default=6000)
    texts.add_argument("--salad", type=int, default=6000)
    texts.add_argument("--seed", type=int, default=0)
    texts.add_argument("--max-chars", type=int, default=160,
                       help="drop texts longer than this. Clips are truncated to "
                            "the training window anyway, so long sentences cost "
                            "generation time and buy nothing")
    texts.add_argument("--salad-words", default="2,7",
                       help="min,max words in a CMUdict word sequence")
    texts.add_argument("--no-prose", action="store_true",
                       help="leave out the project's own prose, for a corpus of "
                            "nothing but phonetic neighbours")

    noise = sub.add_parser("backgrounds", help="write synthetic background noise")
    noise.add_argument("--out", required=True)
    noise.add_argument("--count", type=int, default=120)
    noise.add_argument("--seconds", type=float, default=10.0)
    noise.add_argument("--speech-dir", default=None)
    noise.add_argument("--seed", type=int, default=0)

    pos = sub.add_parser("positives",
                         help="write the positive corpus from the config's weights")
    pos.add_argument("--config", required=True)
    pos.add_argument("--out", required=True)

    dec = sub.add_parser("deconflict",
                         help="drop adversarial phrases that sound like a positive")
    dec.add_argument("--positives", required=True)
    dec.add_argument("--adversarial", required=True)

    rir = sub.add_parser("impulses", help="write impulse responses")
    rir.add_argument("--out", required=True)
    rir.add_argument("--count", type=int, default=40)
    rir.add_argument("--bundled", default=None,
                     help="piper-sample-generator's impulses/ directory")
    rir.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.what == "texts":
        rng = random.Random(args.seed)
        prose = [] if args.no_prose else prose_texts(Path(args.project_root))
        print(f"  prose sentences from the project: {len(prose)}")
        low, high = (int(v) for v in args.salad_words.split(","))
        salad = word_salad(rng, args.salad, min_words=low, max_words=high)
        print(f"  word sequences from CMUdict: {len(salad)}")
        neighbours = phonetic_neighbours(args.neighbour_seed_phrase, args.neighbours,
                                         rng_seed=args.seed)
        print(f"  phonetic neighbours of '{args.neighbour_seed_phrase}': {len(neighbours)}")
        everything = [t for t in prose + salad + neighbours
                      if len(t) <= args.max_chars]
        kept = write_texts(Path(args.out), everything, args.target_phrase)
        print(f"  wrote {kept} texts to {args.out} "
              f"(anything containing '{args.target_phrase}' removed)")
        return 0

    if args.what == "positives":
        import yaml

        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        lines = positive_lines(config["positive_phrases"])
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  wrote {len(lines)} lines ({len(config['positive_phrases'])} "
              f"distinct phrases) to {args.out}")
        return 0

    if args.what == "deconflict":
        positives = [line.strip() for line
                     in Path(args.positives).read_text(encoding="utf-8").splitlines()
                     if line.strip()]
        adversarial = [line.strip() for line
                       in Path(args.adversarial).read_text(encoding="utf-8").splitlines()
                       if line.strip()]
        kept, dropped = drop_homophones(adversarial, positives)
        Path(args.adversarial).write_text("\n".join(kept) + "\n", encoding="utf-8")
        print(f"  kept {len(kept)}, dropped {len(dropped)} homophones of a positive")
        for text in sorted(set(dropped)):
            print(f"    dropped: {text}")
        return 0

    if args.what == "backgrounds":
        made = make_backgrounds(Path(args.out), args.count, seconds=args.seconds,
                                seed=args.seed,
                                speech_dir=Path(args.speech_dir) if args.speech_dir else None)
        print(f"  wrote {made} background clips to {args.out}")
        return 0

    made = make_impulse_responses(Path(args.out), args.count, seed=args.seed,
                                  bundled=Path(args.bundled) if args.bundled else None)
    print(f"  wrote {made} impulse responses to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
