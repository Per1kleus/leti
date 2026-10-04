"""Train the 'hey leti' wake-word model from the generated clips.

WHAT IS OPENWAKEWORD'S AND WHAT IS THIS RECIPE'S

Everything that decides what the model is comes from openWakeWord: the
augmentation (`openwakeword.data.augment_clips`), the features
(`openwakeword.utils.compute_features_from_generator`, which is the melspectrogram
and speech-embedding models the runtime also uses), the network
(`openwakeword.train.Model`), the training loop (`Model.train_model`), the
checkpoint merge (`Model.average_models`) and the ONNX export
(`Model.export_model`). This file supplies the data and the order, and corrects
exactly one thing.

THE ONE THING

`Model.auto_train` is openWakeWord's three-sequence training schedule, and it
contains

    val_set_hrs = 11.3

because it was written for openWakeWord's own published validation set, which is
11.3 hours long. The false-positive rate it optimises towards is
`false_positives / val_set_hrs`, so running it against a validation set of some
other length makes that rate wrong by the ratio of the two - and the rate is what
decides when to stop pushing the weight on negative examples. Against a
two-and-a-half-hour set, auto_train would think it had four times fewer false
positives per hour than it really had, and would ship a model four times more
trigger-happy than asked for.

So the three sequences are written out here, identically, with the real duration
measured from the real audio. Nothing else about them is changed: same learning
rates, same warmup and hold fractions, same negative-weight ramp, same doubling
of that weight when a sequence misses the target, same 90th-percentile merge.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import wave
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import yaml

logger = logging.getLogger("hey_leti.train")

SAMPLE_RATE = 16000


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)


def total_length_for(clip_dir: Path, sample: int = 200,
                     seed: int = 0) -> int:
    """How long every training example is padded to, in samples.

    openWakeWord's own rule, from train.py: the median clip length rounded to the
    nearest thousand samples, plus 12,000 of room, with 32,000 (two seconds) as
    both the floor and a magnet - a window that is not two seconds long for no
    reason is a window the published models' hyper-parameters were not chosen for.
    """
    clips = sorted(clip_dir.glob("*.wav"))
    if not clips:
        raise FileNotFoundError(f"No clips in {clip_dir}.")
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(clips), size=min(sample, len(clips)), replace=False)
    lengths = [len(_read_wav(clips[int(i)])) for i in picks]
    total = int(round(float(np.median(lengths)) / 1000) * 1000) + 12000
    if total < 32000:
        return 32000
    if abs(total - 32000) <= 4000:
        return 32000
    return total


def features_for(clip_dir: Path, out_file: Path, total_length: int,
                 backgrounds: Sequence[str], impulses: Sequence[str],
                 rounds: int = 1, batch_size: int = 128,
                 ncpu: int = 1, overwrite: bool = False) -> int:
    """Augment a directory of clips and write openWakeWord features for them.

    Both halves are openWakeWord's: `augment_clips` adds room, noise, EQ,
    distortion, pitch and gain, and `compute_features_from_generator` runs the
    melspectrogram and embedding models the runtime uses. The result is a memory
    -mapped .npy, so a corpus bigger than RAM is not a problem.
    """
    from openwakeword.data import augment_clips
    from openwakeword.utils import compute_features_from_generator

    if out_file.is_file() and not overwrite:
        existing = np.load(out_file, mmap_mode="r")
        logger.info(f"  {out_file.name}: already there, {existing.shape[0]} examples")
        return int(existing.shape[0])

    clips = [str(p) for p in sorted(clip_dir.glob("*.wav"))] * rounds
    if not clips:
        raise FileNotFoundError(f"No clips in {clip_dir}.")
    logger.info(f"  {out_file.name}: {len(clips)} clips -> features")
    generator = augment_clips(clips, total_length=total_length,
                              batch_size=batch_size,
                              background_clip_paths=list(backgrounds),
                              RIR_paths=list(impulses))
    compute_features_from_generator(generator, n_total=len(clips),
                                    clip_duration=total_length,
                                    output_file=str(out_file),
                                    device="cpu", ncpu=ncpu)
    written = np.load(out_file, mmap_mode="r")
    return int(written.shape[0])


def continuous_features(clip_dir: Path, out_file: Path,
                        chunk_seconds: float = 300.0,
                        overwrite: bool = False) -> Dict[str, float]:
    """Features for one long unbroken stretch of negative audio.

    This is what the false-positive rate is measured on, and it has to be
    continuous rather than a pile of clips: a wake-word model fires on the
    boundary between two sounds as readily as on a word, and a corpus of clips
    padded to a fixed window hides exactly those boundaries. So the clips are
    concatenated into one stream and the embeddings are computed across it.

    Done in chunks only to bound memory. A chunk boundary loses the few frames of
    a 76-frame window that would have straddled it, which at five minutes a chunk
    is under a thousandth of the stream.
    """
    from openwakeword.utils import AudioFeatures

    clips = sorted(clip_dir.glob("*.wav"))
    if not clips:
        raise FileNotFoundError(f"No clips in {clip_dir}.")

    if out_file.is_file() and not overwrite:
        existing = np.load(out_file, mmap_mode="r")
        # 8 melspectrogram frames per embedding frame, 10 ms each.
        seconds = existing.shape[0] * 0.08
        logger.info(f"  {out_file.name}: already there, {seconds/3600:.2f} hours")
        return {"frames": int(existing.shape[0]), "hours": seconds / 3600.0}

    features = AudioFeatures(device="cpu")
    chunk_samples = int(chunk_seconds * SAMPLE_RATE)
    parts: List[np.ndarray] = []
    buffer: List[np.ndarray] = []
    held = 0
    total_samples = 0

    def flush() -> None:
        nonlocal buffer, held
        if not buffer:
            return
        stream = np.concatenate(buffer)
        parts.append(features._get_embeddings(stream))
        buffer, held = [], 0

    for path in clips:
        samples = _read_wav(path)
        if samples.size == 0:
            continue
        buffer.append(samples)
        held += samples.size
        total_samples += samples.size
        if held >= chunk_samples:
            flush()
            logger.info(f"  {out_file.name}: {total_samples/SAMPLE_RATE/3600:.2f} hours")
    flush()

    stacked = np.vstack(parts).astype(np.float32)
    np.save(out_file, stacked)
    hours = total_samples / SAMPLE_RATE / 3600.0
    logger.info(f"  {out_file.name}: {stacked.shape[0]} frames over {hours:.3f} hours")
    return {"frames": int(stacked.shape[0]), "hours": hours}


# How many sliding windows of the false-positive stream are scored at once.
# openWakeWord's own code uses one batch of the whole thing, which is right for
# its published 11.3-hour set of 500,000 windows only because it never has to hold
# a copy: two and a half hours of audio is 109,000 windows, and materialising them
# as one array is 670 MB plus another 670 MB while it is being built. The count it
# produces is a sum over batches either way - openWakeWord's train_model
# accumulates val_fp across the loader - so the batch size changes nothing except
# the peak.
FP_BATCH_WINDOWS = 16384


def _reshape_for_model(path: Path, window: int):
    """The sliding windows openWakeWord scores a continuous stream with.

    One window every frame - 80 ms - which is the same step the runtime takes, and
    the same windows openWakeWord's own train.py builds with

        [X[i:i+16] for i in range(0, X.shape[0]-16, 1)]

    A stride view instead, because that comprehension materialises every window:
    109,000 of them for two and a half hours of audio is 670 MB, plus another
    670 MB while numpy builds the array from the list. The view costs nothing, and
    only the rows in a batch are ever copied - by the collation, which was going to
    copy them anyway.

    torch prints one warning here - "the given NumPy array is not writable" -
    because sliding_window_view hands back a read-only view. That is correct and
    wanted: these windows are only ever read. Copying the array to silence it would
    cost the 670 MB this exists to avoid.
    """
    import torch

    frames = np.load(path)
    # sliding_window_view appends the window axis, giving (n, features, window);
    # the model wants (n, window, features). Both the view and the transpose are
    # views, so neither allocates.
    windows = np.lib.stride_tricks.sliding_window_view(
        frames, window_shape=window, axis=0).transpose(0, 2, 1)[:-1]
    labels = np.zeros(windows.shape[0], dtype=np.float32)
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(windows),
                                       torch.from_numpy(labels)),
        batch_size=FP_BATCH_WINDOWS)


def _validation_loader(positive_path: Path, negative_paths: Sequence[Path]):
    import torch

    positive = np.load(positive_path)
    negatives = [np.load(p) for p in negative_paths]
    data = np.vstack([positive] + negatives)
    labels = np.hstack([np.ones(positive.shape[0])]
                       + [np.zeros(n.shape[0]) for n in negatives]).astype(np.float32)
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(data),
                                       torch.from_numpy(labels)),
        batch_size=len(labels))


def train(config: Dict, work: Path, out_dir: Path,
          overwrite: bool = False, ncpu: int = 1) -> Dict:
    """Augment, featurise, train, merge, export. Returns what happened."""
    import torch
    from openwakeword.data import mmap_batch_generator
    from openwakeword.train import Model
    from openwakeword.utils import AudioFeatures

    torch.manual_seed(int(config["seed_torch"]))
    np.random.seed(int(config["seed_torch"]))

    clips = work / "clips"
    feats = work / "features"
    feats.mkdir(parents=True, exist_ok=True)
    backgrounds = sorted(str(p) for p in (work / "backgrounds").glob("*.wav"))
    impulses = sorted(str(p) for p in (work / "impulses").glob("*.wav"))
    if not backgrounds or not impulses:
        raise FileNotFoundError(
            "No backgrounds or impulse responses. Run prepare_negatives.py first.")
    logger.info(f"{len(backgrounds)} background clips, {len(impulses)} impulse responses")

    total_length = total_length_for(clips / "positive_val")
    logger.info(f"Training window: {total_length} samples "
                f"({total_length/SAMPLE_RATE:.2f} s)")

    corpora = {
        "positive_train": clips / "positive_train",
        "positive_val": clips / "positive_val",
        "adversarial_train": clips / "negative_train",
        "adversarial_val": clips / "negative_val",
        "negative_speech_train": clips / "negative_speech",
        "negative_speech_val": clips / "negative_speech_val",
    }
    counts = {}
    for name, folder in corpora.items():
        counts[name] = features_for(
            folder, feats / f"{name}.npy", total_length, backgrounds, impulses,
            rounds=int(config["augmentation_rounds"]),
            batch_size=int(config["augmentation_batch_size"]),
            ncpu=ncpu, overwrite=overwrite)

    fp = continuous_features(clips / "fp_stream", feats / "fp_stream.npy",
                             overwrite=overwrite)

    features = AudioFeatures(device="cpu")
    input_shape = features.get_embedding_shape(total_length / SAMPLE_RATE)
    logger.info(f"Model input shape: {input_shape}")

    model = Model(n_classes=1, input_shape=input_shape,
                  model_type=str(config["model_type"]),
                  layer_dim=int(config["layer_size"]),
                  seconds_per_example=1280 * input_shape[0] / SAMPLE_RATE)

    # openWakeWord's own reshaping function for negative corpora whose window
    # length does not match the model's, taken from train.py unchanged.
    def reshape(x, n=input_shape[0]):
        if n > x.shape[1] or n < x.shape[1]:
            x = np.vstack(x)
            return np.array([x[i:i + n, :] for i in range(0, x.shape[0] - n, n)])
        return x

    data_files = {
        "positive": str(feats / "positive_train.npy"),
        "adversarial_negative": str(feats / "adversarial_train.npy"),
        "negative_speech": str(feats / "negative_speech_train.npy"),
    }
    label_transforms = {key: (lambda x: [1 for _ in x]) if key == "positive"
                        else (lambda x: [0 for _ in x])
                        for key in data_files}
    batches = mmap_batch_generator(
        data_files,
        n_per_class={k: int(v) for k, v in config["batch_n_per_class"].items()},
        data_transform_funcs={k: reshape for k in data_files},
        label_transform_funcs=label_transforms)

    class _Iterable(torch.utils.data.IterableDataset):
        def __init__(self, generator):
            self.generator = generator

        def __iter__(self):
            return self.generator

    cores = max(1, (os.cpu_count() or 2) // 2)
    train_loader = torch.utils.data.DataLoader(
        _Iterable(batches), batch_size=None, num_workers=cores, prefetch_factor=16)
    val_loader = _validation_loader(
        feats / "positive_val.npy",
        [feats / "adversarial_val.npy", feats / "negative_speech_val.npy"])
    fp_loader = _reshape_for_model(feats / "fp_stream.npy", input_shape[0])

    best = _auto_train(model, train_loader, val_loader, fp_loader,
                       steps=int(config["steps"]),
                       max_negative_weight=int(config["max_negative_weight"]),
                       target_fp_per_hour=float(config["target_false_positives_per_hour"]),
                       val_set_hrs=fp["hours"])

    out_dir.mkdir(parents=True, exist_ok=True)
    model.export_model(model=best, model_name=str(config["model_name"]),
                       output_dir=str(out_dir))
    onnx_path = out_dir / f"{config['model_name']}.onnx"
    logger.info(f"Wrote {onnx_path} ({onnx_path.stat().st_size} bytes)")

    return {
        "model": str(onnx_path),
        "bytes": onnx_path.stat().st_size,
        "total_length_samples": total_length,
        "input_shape": list(input_shape),
        "example_counts": counts,
        "false_positive_set_hours": fp["hours"],
        "false_positive_set_frames": fp["frames"],
        "backgrounds": len(backgrounds),
        "impulse_responses": len(impulses),
        "history": {k: [float(x) for x in v] for k, v in model.history.items()},
    }


def _auto_train(model, train_loader, val_loader, fp_loader, steps: int,
                max_negative_weight: int, target_fp_per_hour: float,
                val_set_hrs: float):
    """openWakeWord's auto_train, with the real length of the validation set.

    See this module's docstring for why it is written out rather than called.
    """
    import torch

    weight = max_negative_weight
    learning_rate = 0.0001
    sequence_steps = steps

    for sequence in (1, 2, 3):
        if sequence == 2:
            learning_rate /= 10
            sequence_steps = steps / 10
        elif sequence == 3:
            learning_rate /= 10
        if sequence > 1 and model.best_val_fp > target_fp_per_hour:
            weight *= 2
            logger.info("Raising the weight on negative examples to cut false positives.")
        logger.info("#" * 50 + f"\nTraining sequence {sequence}\n" + "#" * 50)
        weights = np.linspace(1, weight, int(sequence_steps)).tolist()
        if sequence == 1:
            val_steps = np.linspace(sequence_steps - int(sequence_steps * 0.25),
                                    sequence_steps, 20).astype(np.int64)
        else:
            val_steps = np.linspace(1, sequence_steps, 20).astype(np.int64)
        model.train_model(
            X=train_loader, X_val=val_loader, false_positive_val_data=fp_loader,
            max_steps=sequence_steps, negative_weight_schedule=weights,
            val_steps=val_steps, warmup_steps=sequence_steps // 5,
            hold_steps=sequence_steps // 3, lr=learning_rate,
            val_set_hrs=val_set_hrs)

    logger.info("Merging the checkpoints above the 90th percentile.")
    accuracy_cut = np.percentile(model.history["val_accuracy"], 90)
    recall_cut = np.percentile(model.history["val_recall"], 90)
    fp_cut = np.percentile(model.history["val_fp_per_hr"], 10)
    chosen = [m for m, score in zip(model.best_models, model.best_model_scores)
              if score["val_accuracy"] >= accuracy_cut
              and score["val_recall"] >= recall_cut
              and score["val_fp_per_hr"] <= fp_cut]
    merged = model.average_models(models=chosen) if chosen else model.model
    logger.info(f"Merged {len(chosen)} checkpoints.")

    with torch.no_grad():
        for batch in val_loader:
            x, y = batch[0].to(model.device), batch[1].to(model.device)
            predictions = merged(x)
        recall = float(model.recall(predictions, y[..., None]).detach().cpu().numpy())
        accuracy = float(model.accuracy(predictions, y[..., None].to(torch.int64))
                         .detach().cpu().numpy())
        false_positives = 0
        for batch in fp_loader:
            x, y = batch[0].to(model.device), batch[1].to(model.device)
            false_positives += model.fp(merged(x), y[..., None])
        per_hour = float(false_positives) / val_set_hrs
    logger.info(f"\n################\nMerged model accuracy: {accuracy:.4f}"
                f"\nMerged model recall: {recall:.4f}"
                f"\nMerged model false positives per hour: {per_hour:.3f}"
                f"\n  (over {val_set_hrs:.3f} hours of negative audio)"
                "\n################")
    return merged


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--work", required=True,
                        help="where the clips are and the features go")
    parser.add_argument("--out", required=True, help="where to write the .onnx")
    parser.add_argument("--report", default=None, help="where to write a JSON record")
    parser.add_argument("--overwrite", action="store_true",
                        help="recompute features that already exist")
    parser.add_argument("--ncpu", type=int, default=1)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        stream=sys.stdout)
    logging.getLogger("speechbrain").setLevel(logging.WARNING)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    started = time.time()
    outcome = train(config, Path(args.work), Path(args.out),
                    overwrite=args.overwrite, ncpu=args.ncpu)
    outcome["seconds"] = round(time.time() - started, 1)
    outcome["config"] = config
    if args.report:
        Path(args.report).write_text(json.dumps(outcome, indent=2), encoding="utf-8")
        print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
