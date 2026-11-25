import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path

import numpy as np
from sacrebleu.metrics import CHRF, TER

from prepare import write_json


def analyze(job):
    model, direction, groups, resamples, seed = job
    rng = np.random.default_rng(seed)
    chrf = CHRF(word_order=2)
    ter = TER(case_sensitive=True)
    refs = [[row["reference"] for row in groups[0]]]
    predictions = [[row["translation"] for row in group] for group in groups]
    stats = [np.asarray(chrf._extract_corpus_statistics(text, refs), dtype=np.int64)
             for text in predictions]
    scores = [chrf._compute_score_from_stats(matrix.sum(axis=0).tolist()).score for matrix in stats]
    count = len(groups[0])
    deltas = []
    for indices in rng.integers(0, count, size=(resamples, count)):
        before, after = [chrf._compute_score_from_stats(matrix[indices].sum(axis=0).tolist()).score
                         for matrix in stats]
        deltas.append(after - before)
    low, high = np.quantile(deltas, [.025, .975])
    ter_base, ter_tuned = [ter.corpus_score(text, refs).score for text in predictions]
    return {"model": model, "direction": direction, "count": count,
            "chrf++_delta": scores[1] - scores[0], "ci95_low": float(low),
            "ci95_high": float(high), "ter_base": ter_base, "ter_tuned": ter_tuned,
            "ter_signature": str(ter.get_signature())}


def evaluate(root, resamples, seed):
    jobs = []
    for model in ("madlad", "gemma"):
        paths = [root / stage / model / "test.json" for stage in ("base", "tuned")]
        base, tuned = [json.loads(path.read_text(encoding="utf-8")) for path in paths]

        def identity(row):
            return row["id"], row["direction"], row["source"], row["reference"]

        if [identity(row) for row in base] != [identity(row) for row in tuned]:
            raise ValueError(f"Unaligned test rows for {model}")
        for direction in sorted({row["direction"] for row in base}):
            groups = [[row for row in stage if row["direction"] == direction] for stage in (base, tuned)]
            jobs.append((model, direction, groups, resamples, [seed, len(jobs)]))
    with ProcessPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(analyze, jobs))
    return {"seed": seed, "resamples": resamples, "rows": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("runs/results"))
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.resamples < 1:
        parser.error("--resamples must be positive")
    summary = evaluate(args.root, args.resamples, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "bonus-evaluation.json", summary)
    with (args.output / "bonus-evaluation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary["rows"][0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary["rows"])
