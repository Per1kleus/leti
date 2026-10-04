"""Generate the synthetic speech the 'hey leti' wake-word model is trained on.

WHY THIS FILE EXISTS INSTEAD OF `openwakeword.train --generate_clips`

The training, augmentation and feature computation in this recipe are all
openWakeWord's own code. Only the clip generation is driven from here, for three
reasons that openWakeWord's own generation step cannot express:

1. SPEAKER HOLDOUT. piper-sample-generator walks its speaker pairs with
   `itertools.cycle(itertools.product(range(n), range(n)))`, which means the
   validation clips are drawn from the same speakers as the training clips -
   measured recall would then be recall on voices the model was fitted to, which
   is not a number worth having. Here the usable speakers are split into two
   disjoint sets before anything is generated, and the validation set is never
   trained on.

2. SAMPLE DIVERSITY. That same cycle advances the speaker pair in lockstep with
   a short cycle of speaking speeds, so speed and speaker stay correlated across
   the whole corpus. Here every clip draws its speaker pair, blend weight,
   speaking speed and the two prosody-noise parameters independently from a
   seeded generator, so 20,000 clips are 20,000 different voices rather than
   20,000 samples of a few dozen.

3. ACCENT. The checkpoint phonemises with espeak, and espeak has more than one
   English. Passing a different espeak voice changes the phoneme sequence rather
   than the speaker, which is what an accent largely is. Only voices whose
   phonemes the checkpoint actually maps are used - see `usable_voices`.

Everything it produces is 16 kHz mono 16-bit WAV, which is what openWakeWord's
augmentation step expects.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import wave
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger("hey_leti.generate")

# The speakers at the end of LibriTTS-R have very little audio behind them and
# piper-sample-generator's own README says using them causes artefacts. Nothing
# above this index is used, for training or for validation, so that a validation
# clip being hard to hear is never the reason recall looks bad.
USABLE_SPEAKERS = 800

# One speaker in five is held out. Interleaved rather than a tail slice, so the
# two sets are drawn from the same quality of audio - a tail slice would have put
# every artefact-prone speaker in the validation set.
HOLDOUT_EVERY = 5


def speaker_split(usable: int = USABLE_SPEAKERS,
                  every: int = HOLDOUT_EVERY) -> Tuple[List[int], List[int]]:
    """(training speakers, validation speakers) - disjoint, by construction."""
    train = [i for i in range(usable) if i % every != 0]
    validation = [i for i in range(usable) if i % every == 0]
    return train, validation


def _release_free_heap() -> None:
    """Hand glibc's free lists back to the kernel.

    Not a leak, measured: across six rounds of forty clips the live Python object
    count stays at 137,423 and the live tensor count at 749, while resident memory
    climbs from 588 MB to 1,977 MB. Every batch allocates tensors whose size
    depends on the text it is speaking and the audio that comes out, so the heap
    fragments and glibc keeps the arenas. malloc_trim(0) returned 654 MB of that
    immediately.

    It matters because four of these run at once. Unchecked, the kernel OOM-killed
    three of four shards 9,700 clips into a 24,000-clip corpus, with no error in
    any log - which is what an OOM kill looks like from inside.

    Best effort: a platform without glibc simply does not have this function, and
    the generation does not depend on it.
    """
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _add_psg(psg_path: str) -> None:
    """Put piper-sample-generator v2 on the import path.

    Its functions are used as they are: this file chooses what to synthesise and
    with which voice, and piper does the synthesising.
    """
    path = os.path.abspath(psg_path)
    if path not in sys.path:
        sys.path.insert(0, path)


def usable_voices(config: Dict, candidates: Sequence[str],
                  probe: str = "hey leti") -> List[str]:
    """The espeak voices whose phonemes this checkpoint can actually say.

    A phoneme the checkpoint has no id for is silently dropped by piper's
    `get_phonemes`, so an unsupported accent does not fail - it quietly produces
    speech with sounds missing. Each candidate is phonemised and kept only if
    every phoneme it produces is in the checkpoint's map.
    """
    from piper_phonemize import phonemize_espeak

    id_map = config["phoneme_id_map"]
    kept = []
    for voice in candidates:
        try:
            phonemes = [p for sentence in phonemize_espeak(probe, voice) for p in sentence]
        except Exception as e:
            logger.info(f"espeak has no voice '{voice}' ({e}).")
            continue
        missing = sorted({p for p in phonemes if p not in id_map})
        if missing:
            logger.info(f"Skipping accent '{voice}': the checkpoint cannot say {missing}.")
            continue
        kept.append(voice)
    return kept


class ClipGenerator:
    """A loaded piper generator, plus the seeded sampling this recipe needs."""

    def __init__(self, model_path: str, psg_path: str, seed: int = 0,
                 speakers: Optional[Sequence[int]] = None,
                 voices: Sequence[str] = ("en-us",),
                 length_scale_range: Tuple[float, float] = (0.65, 1.45),
                 noise_scale_range: Tuple[float, float] = (0.55, 1.05),
                 noise_scale_w_range: Tuple[float, float] = (0.55, 1.05),
                 slerp_range: Tuple[float, float] = (0.0, 1.0)):
        _add_psg(psg_path)
        import generate_samples as psg

        self._psg = psg
        self.model = torch.load(model_path, map_location="cpu")
        self.model.eval()
        with open(f"{model_path}.json", encoding="utf-8") as f:
            self.config = json.load(f)
        self.voices = list(voices)
        self.rng = np.random.default_rng(seed)
        train, _ = speaker_split()
        self.speakers = list(speakers if speakers is not None else train)
        self.length_scale_range = length_scale_range
        self.noise_scale_range = noise_scale_range
        self.noise_scale_w_range = noise_scale_w_range
        self.slerp_range = slerp_range
        # piper generates at 22.05 kHz; openWakeWord works at 16 kHz.
        self.resampler = __import__("torchaudio").transforms.Resample(
            22050, 16000, lowpass_filter_width=64,
            rolloff=0.9475937167399596, resampling_method="kaiser_window",
            beta=14.769656459379492)

    def _pick(self, upper: int, count: int) -> List[int]:
        """`count` random indices below `upper`, from the seeded generator."""
        return [int(i) for i in self.rng.integers(0, upper, size=count)]

    def _phoneme_ids(self, text: str, voice: str) -> List[int]:
        return self._psg.get_phonemes(voice, self.config, text, False)

    def generate(self, texts: Sequence[str], count: int, out_dir: Path,
                 batch_size: int = 16, prefix: str = "",
                 on_batch=None) -> List[Dict]:
        """Write `count` clips built from `texts`, and return what each one was.

        The record is returned rather than only written, because "which speaker,
        which speed, which accent" is the difference between a corpus and a pile
        of wav files - the validation step needs it to report recall by accent
        and by voice pitch.

        `prefix` goes in front of every file name so that several processes can
        fill one directory at once. Generation is the slow part of this recipe
        and it shards almost perfectly - four single-threaded processes finish
        512 clips in 16 seconds where one four-threaded process takes 30 - so
        running it in shards is the difference between an hour and two.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        written: List[Dict] = []
        made = 0
        while made < count:
            n = min(batch_size, count - made)
            # Every clip in the batch is an independent draw. Only the phoneme
            # padding is shared, which is what makes a batch worth doing at all.
            # Indices, not Generator.choice(a_list_of_strings). That call converts
            # the whole list to a numpy array of fixed-width unicode EVERY TIME,
            # and the negative corpus is 15,000 sentences - 4 MB per draw, eight
            # draws a batch, thousands of batches. Measured: it took a generation
            # process to 7.9 GB resident and the kernel OOM-killed three of four
            # shards. With indices the same work is flat.
            chosen_texts = [texts[i] for i in self._pick(len(texts), n)]
            voices = [self.voices[i] for i in self._pick(len(self.voices), n)]
            s1 = [self.speakers[i] for i in self._pick(len(self.speakers), n)]
            s2 = [self.speakers[i] for i in self._pick(len(self.speakers), n)]
            slerp_w = float(self.rng.uniform(*self.slerp_range))
            length = float(self.rng.uniform(*self.length_scale_range))
            noise = float(self.rng.uniform(*self.noise_scale_range))
            noise_w = float(self.rng.uniform(*self.noise_scale_w_range))

            ids = [self._phoneme_ids(t, v) for t, v in zip(chosen_texts, voices)]
            width = max(len(i) for i in ids)
            ids = [i + [1] * (width - len(i)) for i in ids]

            with torch.no_grad():
                audio = self._psg.generate_audio(
                    self.model, torch.LongTensor(s1), torch.LongTensor(s2), ids,
                    slerp_w, noise, noise_w, length, None)
                audio = self.resampler(audio.cpu()).numpy()
            pcm = self._psg.audio_float_to_int16(audio)

            for i in range(pcm.shape[0]):
                samples = self._psg.remove_silence(pcm[i].flatten())
                name = f"{prefix}{made:06d}.wav"
                _write_wav(out_dir / name, samples)
                written.append({
                    "file": name, "text": chosen_texts[i], "voice": voices[i],
                    "speaker_1": s1[i], "speaker_2": s2[i],
                    "slerp_weight": slerp_w, "length_scale": length,
                    "noise_scale": noise, "noise_scale_w": noise_w,
                    "samples": int(len(samples)),
                })
                made += 1
                if made >= count:
                    break
            _release_free_heap()
            if on_batch:
                on_batch(made, count)
        return written


def _write_wav(path: Path, samples: np.ndarray, rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setframerate(rate)
        wav.setsampwidth(2)
        wav.setnchannels(1)
        wav.writeframes(samples.tobytes() if hasattr(samples, "tobytes") else samples)


def read_texts(spec: str) -> List[str]:
    """A phrase, or a file of one phrase per line."""
    path = Path(spec)
    if path.is_file():
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    return [spec]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="the piper generator .pt")
    parser.add_argument("--psg", required=True,
                        help="a piper-sample-generator v2.0.0 checkout")
    parser.add_argument("--texts", required=True,
                        help="a phrase, or a file with one phrase per line")
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--split", choices=("train", "validation"), default="train",
                        help="which half of the speaker split to draw from")
    parser.add_argument("--voices", default="en-us",
                        help="comma-separated espeak voices to try")
    parser.add_argument("--prefix", default="",
                        help="put this in front of every file name, so several "
                             "shards can write into one directory")
    parser.add_argument("--manifest", default=None,
                        help="where to write what each clip was (JSON lines)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train, validation = speaker_split()
    speakers = train if args.split == "train" else validation

    _add_psg(args.psg)
    with open(f"{args.model}.json", encoding="utf-8") as f:
        config = json.load(f)
    voices = usable_voices(config, [v.strip() for v in args.voices.split(",") if v.strip()])
    if not voices:
        print("None of the requested accents are usable with this checkpoint.",
              file=sys.stderr)
        return 1
    print(f"  accents: {', '.join(voices)}")
    print(f"  speakers: {len(speakers)} ({args.split} half of the split)")

    generator = ClipGenerator(args.model, args.psg, seed=args.seed,
                              speakers=speakers, voices=voices)
    texts = read_texts(args.texts)
    print(f"  phrases: {len(texts)}")

    def progress(done: int, total: int) -> None:
        if done % 500 < args.batch_size or done == total:
            print(f"  {done}/{total}", flush=True)

    written = generator.generate(texts, args.count, Path(args.out),
                                 batch_size=args.batch_size, prefix=args.prefix,
                                 on_batch=progress)
    if args.manifest:
        with open(args.manifest, "w", encoding="utf-8") as f:
            for row in written:
                f.write(json.dumps(row) + "\n")
    print(f"  wrote {len(written)} clips to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
