# Training the "hey leti" wake-word model

`config/settings.yaml` ships `app.wake_word: "hey_leti"`. That is not one of
openWakeWord's six built-in models, so it only works if a model file for it
exists at `data/wake_words/hey_leti.onnx`. This directory is how that file was
made, and how to make it again.

Nothing here runs when Leti runs. `requirements.txt` in the project root still
has no PyTorch in it, and it must not gain one: the runtime needs
`onnxruntime` to **use** a wake-word model and PyTorch only to **make** one.

---

## Reproducing the training environment

A separate interpreter, outside the one Leti uses, so that nothing here can
change what the application depends on:

```sh
python3 -m venv trainenv
trainenv/bin/python -m pip install --upgrade pip 'setuptools<82'
trainenv/bin/python -m pip install -r training/wake_word/requirements.txt
```

`setuptools<82` is not decoration: `webrtcvad` imports `pkg_resources`, which
newer setuptools no longer installs.

openWakeWord's shared melspectrogram and speech-embedding models have to be in
*that* interpreter's copy of the package, because openWakeWord looks for them
beside its own `__init__.py`:

```sh
trainenv/bin/python -c "import openwakeword.utils as u; u.download_models()"
```

And piper-sample-generator, which is the text-to-speech this recipe speaks with.
Version 2.0.0 specifically, because that is the layout openWakeWord 0.6.0's own
training code expects (`generate_samples.py` at the top level), and because the
multi-speaker generator checkpoint is published against that tag:

```sh
git clone https://github.com/rhasspy/piper-sample-generator
git -C piper-sample-generator fetch --depth 1 origin tag v2.0.0
git -C piper-sample-generator worktree add ../psg_v2 v2.0.0

mkdir -p psg_models
curl -L -o psg_models/en_US-libritts_r-medium.pt \
  https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt
cp psg_v2/models/en_US-libritts_r-medium.pt.json psg_models/
```

The `.pt.json` must sit beside the `.pt`: piper reads the phoneme map and the
speaker count from it.

### Versions this model was built with

Python 3.11.15, 4 CPU cores, no GPU. Every package version is pinned in
`requirements.txt`; the ones that decide the result are openwakeword 0.6.0,
torch 2.2.2, numpy 1.26.4 and piper-phonemize 1.1.0.

---

## Licences

Everything used, and why it is fine to ship a model built from it.

| What | Licence | Used for |
| --- | --- | --- |
| openWakeWord 0.6.0 | Apache-2.0 | the augmentation, the features, the network, the training loop, the ONNX export |
| piper-sample-generator 2.0.0 | MIT | turning text into speech; also its eight bundled impulse responses |
| `en_US-libritts_r-medium.pt` | the generator is published by the MIT-licensed piper-sample-generator; it was trained on **LibriTTS-R**, which is CC BY 4.0, itself derived from public-domain LibriVox recordings | every voice in the corpus |
| CMUdict, via `pronouncing`/`cmudict` | BSD-2-Clause | the phonetic-neighbour search, and the word sequences |
| This project's own prose | the project's | the ordinary-speech negatives |

Attribution for the voices, which CC BY 4.0 asks for: *LibriTTS-R: A Restored
Multi-Speaker Text-to-Speech Corpus*, Koizumi et al., 2023, derived from LibriTTS
(Zen et al., 2019) and LibriVox.

**Nothing from Hugging Face, AudioSet or ACAV100M is used.** That is partly
because they are not reachable (below) and partly a relief: the provenance of
AudioSet-derived audio is a chain of YouTube uploads, and a model trained on it
carries that chain with it. Everything above is either permissively licensed or
written by this project.

---

## What was not reachable

The environment this was trained in reaches `pypi.org`, `github.com` (git and
release assets) and nothing else that matters here. Measured, not assumed - the
egress proxy answers 403 to the rest:

| Host | What openWakeWord's own recipe wants from it |
| --- | --- |
| `huggingface.co` | the precomputed negative feature sets (ACAV100M speech, AudioSet noise), the false-positive validation set, and the piper voice catalogue |
| `openslr.org`, `us.openslr.org`, `openslr.elda.org` | LibriSpeech, MUSAN |
| `zenodo.org`, `archive.org`, `datashare.ed.ac.uk`, `dl.fbaipublicfiles.com`, `freesound.org`, `mcdermottlab.mit.edu` | noise and impulse-response corpora |
| `public-asai-dl-models.s3.eu-central-1.amazonaws.com` | DeepPhonemizer, which openWakeWord downloads to pronounce words CMUdict does not know |
| `download.pytorch.org` | the CPU-only PyTorch wheels |

So three substitutions were made, each of them local and each of them written
down rather than quietly assumed:

1. **The negative corpora are generated, not downloaded.** See
   `prepare_negatives.py`. Three channels - phonetic neighbours, ordinary prose,
   and word sequences with no grammar - plus synthesised background noise and
   impulse responses.

2. **The false-positive validation set is generated too**, as two and a half
   hours of continuous speech from the held-out voices.
   `openwakeword.train.Model.auto_train` hard-codes `val_set_hrs = 11.3` for
   openWakeWord's own published set, so `train_model.py` writes out auto_train's
   three sequences with the real duration instead of calling it. Nothing else
   about the schedule changes; the docstring in that file explains why this
   matters rather than being tidiness.

3. **"leti" is not in CMUdict**, and the neural phonemiser openWakeWord falls
   back to is on a blocked host. So the phonetic-neighbour search is seeded with
   **"hey letty"**, which CMUdict pronounces `HH EY1 . L EH1 T IY0` - the
   pronunciation of "Leti". The spelling is only the key into the dictionary; no
   clip is ever generated from it. Any neighbour that turned out to be a
   homophone of a positive was then removed (eight were: "hay letty", "haye
   letty", "hey lettie").

The PyTorch pin comes from PyPI, which carries CUDA libraries this machine has
no use for. They are inert and they are only in the training environment.

---

## Running it

From the project root, with `trainenv`, `psg_v2` and `psg_models` beside it and
`$WORK` somewhere with about 20 GB free.

### 1. The texts

```sh
V=trainenv/bin/python
PYTHONPATH=. $V -m training.wake_word.prepare_negatives texts \
  --out $WORK/data/negative_short.txt --project-root . \
  --max-chars 70 --salad 14000 --salad-words 2,5 --neighbours 0 --seed 11
PYTHONPATH=. $V -m training.wake_word.prepare_negatives texts \
  --out $WORK/data/negative_long.txt --project-root . \
  --max-chars 200 --salad 0 --neighbours 0 --seed 12
PYTHONPATH=. $V -m training.wake_word.prepare_negatives texts \
  --out $WORK/data/adversarial_texts.txt --no-prose \
  --neighbours 20000 --salad 0 --seed 11
```

Then the positive corpus, written out from the weights in `hey_leti.yaml` so that
the file somebody reads is the file the generator draws from, and the removal of
any adversarial phrase that turns out to be a homophone of one of them:

```sh
PYTHONPATH=. $V -m training.wake_word.prepare_negatives positives \
  --config training/wake_word/hey_leti.yaml --out $WORK/data/positive_texts.txt
PYTHONPATH=. $V -m training.wake_word.prepare_negatives deconflict \
  --positives $WORK/data/positive_texts.txt \
  --adversarial $WORK/data/adversarial_texts.txt
```

### 2. The clips

Seven corpora, 73,500 clips, about two and a half hours on four cores:

```sh
PYTHONPATH=. $V -m training.wake_word.generate_all \
  --config training/wake_word/hey_leti.yaml --work $WORK \
  --model psg_models/en_US-libritts_r-medium.pt --psg psg_v2 --concurrency 3
```

**Run it again if it stops.** It is resumable, and it is resumable because it had
to be: a generation process's resident memory climbs about 3 MB per clip through
heap fragmentation - not a leak, the live object and tensor counts are flat - and
four unbounded ones at once got three of them OOM-killed 9,728 clips into the
negative corpus, with no error in any log. So each process now makes
`clips_per_process` clips and exits, which holds it at about 1 GB, and a chunk
already on disk is skipped. A chunk that stopped part way is redone rather than
topped up: the seeded draw is a sequence, and restarting it half way would not
continue it.

`generate_clips.py` is the single corpus underneath, and can be run directly:

```sh
PYTHONPATH=. OMP_NUM_THREADS=1 $V -m training.wake_word.generate_clips \
  --model psg_models/en_US-libritts_r-medium.pt --psg psg_v2 \
  --texts $WORK/data/positive_texts.txt --out $WORK/clips/positive_train \
  --count 400 --batch-size 8 --seed 100 --split train --prefix c000_ \
  --voices en-us,en-gb-x-rp,en-gb-scotland,en-029,en-gb-x-gbclan,en-gb-x-gbcwmd,en-us-nyc,en
```

Every count, seed and chunk size is in `hey_leti.yaml`, including
`clips_per_process`: it is part of the corpus definition rather than a tuning knob,
because each chunk is separately seeded and changing it changes what is generated.

### 3. The backgrounds and the rooms

Babble is made out of the negative speech, so this comes after step 2:

```sh
PYTHONPATH=. $V -m training.wake_word.prepare_negatives backgrounds \
  --out $WORK/backgrounds --count 160 --seconds 10 \
  --speech-dir $WORK/clips/negative_speech --seed 800
PYTHONPATH=. $V -m training.wake_word.prepare_negatives impulses \
  --out $WORK/impulses --count 48 --bundled psg_v2/impulses --seed 900
```

`--bundled` resamples piper's eight impulse responses to 16 kHz mono rather than
copying them. They are 44.1 kHz stereo, and openWakeWord's `augment_clips` neither
resamples nor mixes down an impulse response - it also rebinds its own `sr`
parameter from the file it just loaded, so one 44.1 kHz response makes the next
batch raise `ValueError: Error! Clip does not have the correct sample rate!`. See
`prepare_negatives._to_16k_mono` for the whole of it.

### 4. Training

```sh
PYTHONPATH=. $V -m training.wake_word.train_model \
  --config training/wake_word/hey_leti.yaml --work $WORK \
  --out data/wake_words --report $WORK/training_report.json --ncpu 4
```

That augments every corpus with openWakeWord's `augment_clips`, computes
openWakeWord features, runs the three training sequences, merges the checkpoints
above the 90th percentile, and writes `data/wake_words/hey_leti.onnx`.

**There is no `.tflite`.** openWakeWord's `convert_onnx_to_tflite` needs
`tensorflow-cpu==2.8.1`, `onnx-tf==1.10.0` and `tensorflow-probability==0.16.0`,
and TensorFlow 2.8.1 has no wheel for any Python newer than 3.10. The format is
not missed: `tflite-runtime` is declared by openWakeWord only for Linux, so on
Windows - where the reports that started this work came from - there is no tflite
runtime at all and ONNX is the only format that can run. `audio/wake_word.py`
loads whichever of the two is present and says which it used.

### 5. Validation

```sh
PYTHONPATH=. $V -m training.wake_word.validate \
  --model data/wake_words/hey_leti.onnx \
  --generator psg_models/en_US-libritts_r-medium.pt --psg psg_v2 \
  --config training/wake_word/hey_leti.yaml --work $WORK \
  --per-condition 120 --report training/wake_word/validation.json \
  --fixtures tests/fixtures/wake_word
```

`validation.json` is committed. It is the evidence for the threshold in
`config/settings.yaml` and for every number quoted about this model.

`--fixtures` writes fifteen of the clips it just measured into
`tests/fixtures/wake_word/`, which is how `tests/test_hey_leti_model.py` and
`tests/test_voice_end_to_end.py` can assert that the shipped model hears the wake
word on a machine with no microphone. They are samples of the measured conditions,
spoken by held-out voices, and they are committed for the same reason the model is:
no installation can regenerate them.

---

---

## What it measures

Everything below is from `validation.json`, at the chosen threshold of **0.95**,
on speech from 160 synthetic voices that no training clip used.

### It wakes

| condition | wakes | clips |
| --- | --- | --- |
| normal speech | 100% | 120 |
| spoken quickly | 98% | 120 |
| spoken slowly | 100% | 120 |
| spoken quietly | 100% | 120 |
| spoken loudly | 100% | 120 |
| deeper voices | 100% | 40 |
| higher voices | 100% | 40 |
| a pause between the words | 99% | 120 |
| over background noise | 100% | 120 |
| over babble | 70% | 120 |
| inside a sentence | 5% | 120 |
| each of 8 English accents | 97%–100% | 240 |

### It stays quiet

| condition | wakes | clips |
| --- | --- | --- |
| silence | 0% | 10 |
| background noise alone | 0% | 40 |
| babble alone | 0% | 40 |
| unrelated conversation | 3% | 240 |
| similar-sounding phrases | 9% | 240 |
| 'hey' on its own | 8% | 120 |
| the name inside a sentence | 20% | 120 |
| the name on its own | 45% | 120 |
| a steady sound starting suddenly | 35% | 40 |

### And the rest

- **Recall 0.8900** overall across the waking conditions.
- **1.582 false positives an hour** of continuous speech (5 firings in 3.16 hours, so the measurement resolves no finer than 0.316 an hour; the 95% interval reaches 3.327).
- **90.48 ms** to load, once, at startup.
- **2.2685 ms** to score an 80 ms frame (35.3x realtime, about 2.8% of one core), p99 3.546 ms.
- Wakes **155 ms before the phrase ends** (median; 120/120 detected).
- **2,113,155 bytes** of ONNX. No `.tflite` - see above.
- No network, one thread, and resident memory flat over 10,000 frames (13 minutes of listening): 355.7 MB to 355.8 MB.

### The four limitations, in one place

| what | rate | why |
| --- | --- | --- |
| saying just "Leti" wakes it | 45% | it is most of the phrase. Inside a sentence it is 20% |
| a sustained sound starting abruptly | 35% | an onset transient: one wake per onset. The same sound while playing is 0% |
| the wake word under babble | 70% | babble cannot be mixed under positives without mislabelling them |
| the phrase mid-sentence | 5% | training always places it at the window's end with silence before it |

## What this does and does not prove

It proves that the model answers to "hey leti" spoken by 160 synthetic voices it
was never trained on, across eight English accents, at speaking speeds from 0.6 to
1.55 times normal, quietly and loudly, with a pause in the middle and with
background noise over it; and that it stays quiet through phonetic neighbours,
through either half of the phrase on its own, through the name used in an ordinary
sentence, through unrelated conversation, through silence and through noise.

It does not prove anything about a real microphone in a real room. Every clip is
synthesised. Synthetic speech varies less than people do, and synthetic noise
less than a kitchen, so the false-positive rate measured here is a floor and the
recall is optimistic. The threshold chosen in `validate.py` deliberately sits one
notch above the one the measurements alone would justify, for exactly that
reason.

The honest test is a person saying "Hey Leti" into their own microphone. That has
not been done here: this environment has no sound card.
