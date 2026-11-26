"""Score saved English-target holdouts with a fixed contextual encoder."""

import argparse
import csv
from hashlib import file_digest
from importlib.metadata import version
import json
from pathlib import Path
import tarfile

import numpy as np

from .prepare import write_json


ENCODER = "distilbert/distilbert-base-uncased"
REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
LAYER = 5
DIRECTIONS = {"kha-en": 1000, "brx-en": 637}
SYSTEMS = [(model, stage) for model in ("madlad", "gemma") for stage in ("base", "tuned")]


def load_holdouts(path):
    members = {f"{stage}/{model}/test.json": (model, stage) for model, stage in SYSTEMS}
    found = {}
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            name = member.name.removeprefix("./")
            if name not in members:
                continue
            key = members[name]
            if key in found or not member.isfile():
                raise ValueError(f"Invalid or duplicate archive member: {name}")
            with archive.extractfile(member) as stream:
                found[key] = json.load(stream)
    if set(found) != set(SYSTEMS):
        raise ValueError("Archive is missing complete baseline or fine-tuned holdouts")

    data = {}
    for direction, count in DIRECTIONS.items():
        groups = {}
        for system in SYSTEMS:
            rows = [row for row in found[system] if row["direction"] == direction]
            ids = [row["id"] for row in rows]
            if len(rows) != count or len(set(ids)) != count:
                raise ValueError(f"{system} {direction}: expected {count} unique holdout rows")
            groups[system] = {row["id"]: row for row in rows}
        first = groups[SYSTEMS[0]]
        keys = sorted(first)
        for system, rows in groups.items():
            if rows.keys() != first.keys() or any(
                (rows[key]["source"], rows[key]["reference"]) != (first[key]["source"], first[key]["reference"])
                for key in keys
            ):
                raise ValueError(f"Unaligned inputs or references in {system} {direction}")
        refs = [first[key]["reference"] for key in keys]
        if any(not ref.strip() for ref in refs):
            raise ValueError(f"{direction} includes an empty reference")
        predictions = [[groups[system][identifier]["translation"] for identifier in keys] for system in SYSTEMS]
        data[direction] = refs, predictions
    return data


def paired_interval(base, tuned, resamples, seed):
    base, tuned = [np.asarray(values, dtype=np.float64) for values in (base, tuned)]
    if base.shape != tuned.shape or base.ndim != 1 or not base.size:
        raise ValueError("Expected equally sized, nonempty sentence score vectors")
    differences = tuned - base
    if not np.all(np.isfinite(differences)):
        raise ValueError("Expected finite paired sentence scores")
    rng = np.random.default_rng(seed)
    bootstrap = []
    for start in range(0, resamples, 128):
        n = min(128, resamples - start)
        indices = rng.integers(differences.size, size=(n, differences.size))
        bootstrap.extend(differences[indices].mean(axis=1))
    return tuple(float(value) for value in np.quantile(bootstrap, [0.025, 0.975]))


def score_direction(scorer, refs, hypotheses, batch_size, chunk_size, resamples, seed):
    values = np.empty((len(SYSTEMS), len(refs)), dtype=np.float64)
    for start in range(0, len(refs), chunk_size):
        stop = min(len(refs), start + chunk_size)
        candidates = [text for group in hypotheses for text in group[start:stop]]
        _, _, f1 = scorer.score(candidates, refs[start:stop] * len(SYSTEMS), batch_size=batch_size)
        values[:, start:stop] = f1.numpy().reshape(len(SYSTEMS), stop - start)
    if not np.all(np.isfinite(values)):
        raise ValueError("Encoder returned non-finite F1 scores")
    rows = []
    for i in range(0, len(SYSTEMS), 2):
        model = SYSTEMS[i][0]
        base, tuned = values[i:i + 2]
        low, high = paired_interval(base, tuned, resamples, seed)
        rows.append({"model": model, "count": len(refs), "f1_base": float(base.mean()), "f1_tuned": float(tuned.mean()),
                     "f1_delta": float((tuned - base).mean()), "ci95_low": low, "ci95_high": high})
    return rows


def evaluate(archive, batch_size, chunk_size, threads, resamples, seed):
    import torch
    from bert_score import BERTScorer
    from huggingface_hub import snapshot_download

    data = load_holdouts(archive)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    snapshot = snapshot_download(ENCODER, revision=REVISION,
                                 allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                                                 "tokenizer_config.json", "vocab.txt"])
    scorer = BERTScorer(model_type=snapshot, num_layers=LAYER, lang="en", device="cpu",
                        idf=False, rescale_with_baseline=False, use_fast_tokenizer=True,
                        batch_size=batch_size, nthreads=threads)
    rows = []
    for direction, (refs, hypotheses) in data.items():
        print(f"Scoring {direction} ({len(refs)} pairs per system)", flush=True)
        for result in score_direction(scorer, refs, hypotheses, batch_size, chunk_size, resamples, seed):
            rows.append({"direction": direction, **result})
            print(rows[-1], flush=True)
    with archive.open("rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()
    return {"encoder": ENCODER, "encoder_revision": REVISION, "layers": LAYER, "target_language": "en",
            "tokenizer": "fast uncased", "max_input_tokens": 512, "idf": False, "rescale_with_baseline": False,
            "device": "cpu", "dtype": "float32", "batch_size": batch_size, "chunk_size": chunk_size,
            "aggregation": "mean sentence F1", "interval": "paired sentence bootstrap percentile",
            "resamples": resamples, "seed": seed, "archive_sha256": digest,
            "versions": {name: version(name) for name in ("bert-score", "torch", "transformers", "tokenizers", "numpy")},
            "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/metrics"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.batch_size, args.chunk_size, args.threads, args.resamples) < 1 or args.seed < 0:
        parser.error("Batch size, chunk size, threads and resamples must be positive, and the seed nonnegative")
    result = evaluate(args.archive, args.batch_size, args.chunk_size, args.threads, args.resamples, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "semantic-evaluation.json", result)
    with (args.output / "semantic-evaluation.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["rows"][0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(result["rows"])
