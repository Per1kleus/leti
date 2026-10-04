"""Generate every clip corpus the training needs, in bounded processes.

WHY THIS IS NOT ONE LONG RUN

A generation process grows. Measured over six rounds of forty clips: the live
Python object count stays at 137,423 and the live tensor count at 749, so nothing
is leaking - but resident memory climbs from 588 MB to 1,977 MB, because every
batch allocates tensors sized by the text it is speaking and the audio that comes
out, and glibc cannot hand a fragmented heap back. malloc_trim(0) returns some of
it and not enough: with a trim after every batch, 512 clips still took one process
to 2.8 GB.

Left alone, four of those at once is 11 GB and the kernel kills them. It did:
9,728 clips into a 24,000-clip corpus, three of four shards vanished with no error
in any log, which is exactly what an OOM kill looks like from inside the process
that did not get killed.

So each process generates a bounded number of clips and exits, and this script
runs the next one. Resident memory per process is then flat by construction, and
the cost is one model load - about six seconds - per chunk.

IT IS ALSO RESUMABLE

A chunk whose files are already on disk is skipped. Three hours of generation that
cannot survive one interruption is three hours spent twice.

REPRODUCIBILITY

Each chunk's seed is `seed_<corpus> + chunk index`, and each chunk's files are
prefixed `c<index>_`, so the corpus is the concatenation of a fixed list of
fixed-seed chunks. `clips_per_process` is in the config for that reason: change it
and you get a different corpus, which is why it is recorded rather than passed.
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml

logger = logging.getLogger("hey_leti.generate_all")

# name, text file key, output directory, how many, which half of the speakers
CORPORA = (
    ("positive_train", "positive_texts", "positive_train", "n_positive",
     "train", "seed_positive"),
    ("positive_val", "positive_texts", "positive_val", "n_positive_val",
     "validation", "seed_positive_val"),
    ("adversarial_train", "adversarial_texts", "negative_train", "n_adversarial",
     "train", "seed_adversarial"),
    ("adversarial_val", "adversarial_texts", "negative_val", "n_adversarial_val",
     "validation", "seed_adversarial_val"),
    ("negative_speech", "negative_texts", "negative_speech", "n_negative_speech",
     "train", "seed_negative_speech"),
    ("negative_speech_val", "negative_texts", "negative_speech_val",
     "n_negative_speech_val", "validation", "seed_negative_speech_val"),
    ("fp_stream", "fp_stream_texts", "fp_stream", "n_fp_stream_clips",
     "validation", "seed_fp_stream"),
)


def chunks_for(total: int, per_process: int) -> List[int]:
    """How many clips each process makes. The last one takes the remainder."""
    full, rest = divmod(total, per_process)
    sizes = [per_process] * full
    if rest:
        sizes.append(rest)
    return sizes


def _already_there(out_dir: Path, prefix: str) -> int:
    return len(list(out_dir.glob(f"{prefix}*.wav")))


def run_corpus(name: str, texts: Path, out_dir: Path, total: int, split: str,
               seed: int, config: Dict, work: Path, model: Path, psg: Path,
               python: str, concurrency: int, batch_size: int,
               dry_run: bool = False) -> Dict:
    """One corpus, as a sequence of bounded processes."""
    per_process = int(config["clips_per_process"])
    accents = ",".join(config["accents"])
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = chunks_for(total, per_process)
    logger.info(f"=== {name}: {total} clips in {len(sizes)} chunks "
                f"of up to {per_process} ({split} speakers) ===")

    pending = []
    for index, size in enumerate(sizes):
        prefix = f"c{index:03d}_"
        done = _already_there(out_dir, prefix)
        if done >= size:
            continue
        if done:
            # A chunk that was interrupted is redone rather than topped up: the
            # seeded draw is a sequence, and starting it again part way through
            # would not continue that sequence.
            for stale in out_dir.glob(f"{prefix}*.wav"):
                stale.unlink()
        pending.append((index, size, prefix))
    if not pending:
        logger.info(f"  all {len(sizes)} chunks already generated")
        return {"corpus": name, "clips": _already_there(out_dir, "c"),
                "chunks": len(sizes), "ran": 0}

    logger.info(f"  {len(pending)} chunk(s) to generate")
    if dry_run:
        return {"corpus": name, "chunks": len(sizes), "ran": 0, "dry_run": True}

    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = "1"       # one core each; the shards are the
    environment["MKL_NUM_THREADS"] = "1"       # parallelism
    environment["MALLOC_ARENA_MAX"] = "2"
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent.parent)

    started = time.time()
    running: List[tuple] = []
    failures: List[str] = []
    queue = list(pending)

    def launch(index: int, size: int, prefix: str):
        command = [
            python, "-m", "training.wake_word.generate_clips",
            "--model", str(model), "--psg", str(psg), "--texts", str(texts),
            "--out", str(out_dir), "--count", str(size),
            "--batch-size", str(batch_size), "--seed", str(seed + index),
            "--split", split, "--voices", accents, "--prefix", prefix,
            "--manifest", str(work / "manifests" / f"{name}.{prefix}jsonl"),
        ]
        (work / "manifests").mkdir(parents=True, exist_ok=True)
        log = (work / "logs" / f"{name}.{prefix}log")
        log.parent.mkdir(parents=True, exist_ok=True)
        handle = log.open("w", encoding="utf-8")
        return (subprocess.Popen(command, env=environment, stdout=handle,
                                 stderr=subprocess.STDOUT), handle, index, size,
                prefix)

    while queue or running:
        while queue and len(running) < concurrency:
            running.append(launch(*queue.pop(0)))
        time.sleep(2)
        for entry in list(running):
            process, handle, index, size, prefix = entry
            if process.poll() is None:
                continue
            running.remove(entry)
            handle.close()
            made = _already_there(out_dir, prefix)
            if process.returncode != 0 or made < size:
                # An OOM kill shows up here and nowhere else: no traceback, a
                # negative return code, and a short chunk.
                failures.append(f"{name} chunk {index}: exit {process.returncode}, "
                                f"{made}/{size} clips")
                logger.warning(f"  chunk {index} FAILED: exit {process.returncode}, "
                               f"{made}/{size} clips")
            else:
                total_now = _already_there(out_dir, "c")
                logger.info(f"  chunk {index} done ({made} clips); "
                            f"{total_now}/{total} total, "
                            f"{time.time()-started:.0f}s elapsed")

    outcome = {"corpus": name, "clips": _already_there(out_dir, "c"),
               "chunks": len(sizes), "ran": len(pending),
               "seconds": round(time.time() - started, 1),
               "failures": failures}
    if failures:
        logger.warning(f"  {len(failures)} chunk(s) failed; run this again to "
                       "pick up where it stopped")
    return outcome


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--model", required=True, help="the piper generator .pt")
    parser.add_argument("--psg", required=True, help="piper-sample-generator v2")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--concurrency", type=int, default=3,
                        help="how many generation processes at once. Three rather "
                             "than one per core: each one peaks around 2 GB")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--only", default=None,
                        help="a comma-separated list of corpus names")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    work = Path(args.work)
    wanted = {name.strip() for name in args.only.split(",")} if args.only else None

    results = []
    for name, texts_key, folder, count_key, split, seed_key in CORPORA:
        if wanted and name not in wanted:
            continue
        results.append(run_corpus(
            name, work / config[texts_key], work / "clips" / folder,
            int(config[count_key]), split, int(config[seed_key]), config, work,
            Path(args.model), Path(args.psg), args.python, args.concurrency,
            args.batch_size, dry_run=args.dry_run))

    print("\n=== generated ===")
    failed = 0
    for row in results:
        note = ""
        if row.get("failures"):
            failed += len(row["failures"])
            note = f"  ({len(row['failures'])} chunk(s) failed)"
        print(f"  {row['corpus']:22s} {row.get('clips', 0):6d} clips{note}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
