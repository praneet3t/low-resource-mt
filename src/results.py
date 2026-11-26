import argparse
from collections import Counter
import csv
from pathlib import Path
import re
import tarfile
import unicodedata as ud

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sacrebleu.metrics import CHRF

from .prepare import LANGUAGES, normalize, read_jsonl, scripts, write_json

from .report import DIRECTIONS, MODELS, REVIEW_CASES, build_report, read


COLORS = ["#32618e", "#ad532c"]


def finish(fig, output, name):
    for extension in ("pdf", "svg", "png"):
        fig.savefig(output / f"{name}.{extension}", bbox_inches="tight", dpi=180)
    plt.close(fig)


def eda(root, output):
    stats = read(root / "data/eda.json")
    audit = read(root / "data/inspection.json")
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.2), layout="constrained")
    for axis, (code, name) in zip(axes, LANGUAGES.items()):
        columns = list(zip(*audit[code]["lengths"]))
        for values, color, label in zip(columns, COLORS, ("English", name)):
            thresholds = range(1, 101)
            counts = Counter(values)
            cumulative, total = [], 0
            for x in thresholds:
                total += counts[x]
                cumulative.append(total / len(values))
            axis.step(thresholds, cumulative, color=color, label=label, linewidth=1.5)
        axis.set(title=f"English–{name}", xlabel="Whitespace tokens", xlim=(0, 100), ylim=(0, 1.02))
        axis.legend(frameon=False, loc="lower right")
    axes[0].set_ylabel("Fraction of sentences")
    finish(fig, output, "lengths")
    write_json(output / "eda.json", stats)
    return stats


def numeral_strings(text):
    return {"".join(str(ud.decimal(char)) for char in value)
            for value in re.findall(r"\d+", text)}


def diagnostic(row):
    text = row["translation"]
    normal = normalize(text)
    source = normalize(row["source"])
    common_numerals = numeral_strings(row["source"]) & numeral_strings(row["reference"])
    words = text.casefold().split()
    trigrams = list(zip(words, words[1:], words[2:]))
    expected = "DEVANAGARI" if row["direction"].endswith("-brx") else "LATIN"
    letters = scripts(text)
    return {
        "empty": not normal,
        "exact_normalized_source_copy": bool(normal) and normal == source,
        "embedded_source_copy": len(source) >= 20 and source in normal and source != normal,
        "missing_shared_numeral": bool(common_numerals - numeral_strings(text)),
        "short_output": len(words) < .5 * len(row["reference"].split()),
        "repeated_trigram": max(Counter(trigrams).values(), default=0) >= 3,
        "unexpected_majority_script": bool(letters) and letters[expected] * 2 < letters.total(),
        "replacement_character": "\ufffd" in text,
    }


def aggregate(root, output):
    scores, diagnostics, examples = {}, {}, {}
    chrf = CHRF(word_order=2)
    official = read_jsonl(root / "data/official.jsonl")
    test = read_jsonl(root / "data/test.jsonl")
    for model in MODELS:
        paths = {stage: root / stage / model for stage in ("base", "tuned")}
        scores[model] = {stage: read(path / "metrics.json") for stage, path in paths.items()}
        scores[model]["delta"] = {
            d: {m: scores[model]["tuned"][d][m] - scores[model]["base"][d][m]
                for m in ("bleu", "chrf++")}
            for d in DIRECTIONS
        }
        rows = {stage: read(path / "test.json") for stage, path in paths.items()}
        for stage, path in paths.items():
            for filename, expected, actual in (("test", test, rows[stage]),
                                               ("official", official, read(path / "official.json"))):
                observed = [{key: value for key, value in row.items() if key != "translation"} for row in actual]
                assert observed == expected, (model, stage, filename)
                assert all(isinstance(row["translation"], str) for row in actual)

        diagnostics[model], examples[model] = {}, {}
        for direction in DIRECTIONS:
            aligned = [(base, tuned) for base, tuned in zip(rows["base"], rows["tuned"])
                       if base["direction"] == direction]
            assert aligned
            assessed = [(base, tuned, diagnostic(base), diagnostic(tuned)) for base, tuned in aligned]
            diagnostics[model][direction] = {
                stage: {flag: sum(pair[index + 2][flag] for pair in assessed)
                        for flag in assessed[0][index + 2]}
                for index, stage in enumerate(("base", "tuned"))
            }
            ranked = []
            for base, tuned, _, _ in assessed:
                values = [chrf.sentence_score(row["translation"], [row["reference"]]).score
                          for row in (base, tuned)]
                ranked.append((values[1] - values[0], base, tuned, values))
            ranked.sort(key=lambda entry: entry[0])
            examples[model][direction] = {}
            for label, (change, base, tuned, values) in zip(
                    ("decrease", "median", "increase"), (ranked[0], ranked[len(ranked) // 2], ranked[-1])):
                examples[model][direction][label] = {
                    "id": base["id"], "source": base["source"], "reference": base["reference"],
                    "base": base["translation"], "tuned": tuned["translation"],
                    "sentence_chrf_base": values[0], "sentence_chrf_tuned": values[1], "delta": change,
                }
            reviewed_id = REVIEW_CASES.get((model, direction))
            if reviewed_id:
                case = next((entry for entry in ranked if entry[1]["id"] == reviewed_id), None)
                if case:
                    change, base, tuned, values = case
                    examples[model][direction]["reviewed"] = {
                        "id": base["id"], "source": base["source"], "reference": base["reference"],
                        "base": base["translation"], "tuned": tuned["translation"],
                        "sentence_chrf_base": values[0], "sentence_chrf_tuned": values[1], "delta": change,
                    }
            examples[model][direction]["flag_changes"] = {}
            for flag in ("exact_normalized_source_copy", "short_output", "repeated_trigram",
                         "unexpected_majority_script", "missing_shared_numeral"):
                for base, tuned, before, after in assessed:
                    if before[flag] and not after[flag]:
                        examples[model][direction]["flag_changes"][flag] = {
                            "id": base["id"], "source": base["source"], "reference": base["reference"],
                            "base": base["translation"], "tuned": tuned["translation"],
                        }
                        break
    improvement_figure(scores, output)
    write_json(output / "scores.json", scores)
    write_scores_csv(output / "scores.csv", scores)
    write_json(output / "diagnostics.json", diagnostics)
    write_json(output / "examples.json", examples)
    return scores, diagnostics, examples


def improvement_figure(scores, output):
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 1.82), layout="constrained", sharey=True)
    for axis, (model, title) in zip(axes, MODELS.items()):
        for index, name in enumerate(("bleu", "chrf++")):
            axis.bar([i + (.18 if index else -.18) for i in range(4)],
                     [scores[model]["delta"][d][name] for d in DIRECTIONS],
                     width=.32, color=COLORS[index], label=("BLEU", "chrF++")[index])
        axis.axhline(0, color=".3", linewidth=.7)
        axis.set(title=title, xticks=range(4), xticklabels=["en→kha", "kha→en", "en→brx", "brx→en"])
    axes[0].set_ylabel("Change in score (points)")
    axes[1].legend(frameon=False)
    finish(fig, output, "improvements")


def write_scores_csv(path, scores):
    metrics = ("bleu", "chrf++")
    stages = ("base", "tuned", "delta")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["model", "direction", "count"] +
                        [f"{metric}_{stage}" for metric in metrics for stage in stages])
        for model, score in scores.items():
            for direction, base in score["base"].items():
                values = [score[stage][direction][metric] for metric in metrics for stage in stages]
                writer.writerow([model, direction, base["count"], *values])


def curves(histories, output):
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 1.80), layout="constrained")
    for axis, (model, title) in zip(axes, MODELS.items()):
        history = histories[model]["history"]
        for metric, label, color in zip(("loss", "eval_loss"), ("Training", "Validation"), COLORS):
            xy = [(item["epoch"], item[metric]) for item in history if metric in item]
            axis.plot(*zip(*xy), label=label, color=color, linewidth=1.2,
                      marker="o" if metric == "eval_loss" else None, markersize=3)
        axis.set(title=title, xlabel="Epoch", ylabel="Cross-entropy")
        axis.legend(frameon=False)
    finish(fig, output, "training")


def training_architectures(output):
    fig, ax = plt.subplots(figsize=(6.6, 2.27))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0, 660), ylim=(0, 227))
    ax.axis("off")

    def label(x, y, words, size=9.25, **style):
        ax.text(x, y, words, ha="center", va="center", fontsize=size,
                fontfamily="serif", color="#181818", **style)

    def arrow(origin, target):
        ax.annotate("", target, origin, arrowprops={
            "arrowstyle": "->", "linewidth": .8, "color": "#454545",
        })

    label(157, 214, "MADLAD-400 3B  |  Encoder-decoder", 10, weight="bold")
    label(503, 214, "Gemma 3 1B  |  Decoder-only", 10, weight="bold")
    ax.plot([330, 330], [8, 218], color="#c5c5c5", linewidth=.6)

    label(87, 178, "Target tag + source")
    label(244, 178, "Shifted target tokens")
    label(87, 129, "Enc. self-attn. + LoRA", size=8.5)
    label(87, 106, "Encoder states", size=8.5)
    label(244, 129, "Dec. self-attn. + LoRA", size=8.5)
    label(244, 84, "Cross-attention + LoRA", size=8.5)
    label(244, 43, "Next-token logits")
    label(244, 10, "Target loss + EOS")
    arrow((87, 165), (87, 142))
    arrow((244, 165), (244, 142))
    arrow((244, 116), (244, 97))
    arrow((87, 97), (87, 84))
    arrow((87, 84), (164, 84))
    arrow((244, 71), (244, 56))
    arrow((244, 32), (244, 20))

    label(503, 178, "Chat prompt + source + target tokens")
    label(503, 129, "Causal decoder self-attention + LoRA")
    label(503, 84, "Next-token logits")
    label(503, 33, "Target loss + end of turn")
    label(503, 10, "Prompt and source labels masked", size=8.5)
    arrow((503, 165), (503, 142))
    arrow((503, 116), (503, 97))
    arrow((503, 71), (503, 47))
    finish(fig, output, "architectures")


def export_decoding_results(output):
    pilot = []
    for model in MODELS:
        for path in sorted((output.parent / "decoding-pilot" / model).glob("*.json")):
            if not path.name.endswith(".translations.json"):
                pilot.append(read(path))
    if not pilot:
        return
    write_json(output / "additional-decoding.json", pilot)
    with (output / "additional-decoding.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("model", "condition", "direction", "examples", "bleu", "chrf++",
                         "condition_decode_seconds"))
        for run in pilot:
            for direction, score in run["scores"].items():
                writer.writerow((run["model"], run["label"], direction, score["count"], score["bleu"],
                                 score["chrf++"], run["seconds"]))


def archive_results(root, archive):
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as bundle:
        def retain(info):
            if (".partial." in info.name or "/report-page-" in info.name
                    or info.name.startswith("train/") and info.name.endswith("/README.md")):
                return None
            return info

        bundle.add(root / "results", arcname="results", filter=retain)
        for directory in ("train", "base", "tuned", "benchmark"):
            for model in MODELS:
                name = f"{directory}/{model}"
                bundle.add(root / name, arcname=name, filter=retain)
        for model in MODELS:
            file = f"smoke/{model}/smoke.json"
            bundle.add(root / file, arcname=file)
        for name in ("eda.json", "inspection.json"):
            bundle.add(root / "data" / name, arcname=f"data-audit/{name}")
        source = Path(__file__).resolve().parent
        for path in sorted(source.iterdir()):
            if path.suffix in (".py", ".html"):
                bundle.add(path, arcname=f"reproduction/src/{path.name}")
        for name in ("README.md", "requirements.txt"):
            bundle.add(source.parent / name, arcname=f"reproduction/{name}")
        if (root / "decoding-pilot").is_dir():
            bundle.add(root / "decoding-pilot", arcname="decoding-pilot")
        for path in sorted(root.glob("*.log")):
            if "qwen" not in path.name:
                bundle.add(path, arcname=f"logs/{path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs"))
    parser.add_argument("--eda-only", action="store_true")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--report", action="store_true", help="Build the PDF after all evaluations")
    parser.add_argument("--browser", help="Path to Chrome, Chromium or Edge")
    args = parser.parse_args()
    output = args.root / "results"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "path", "pdf.fonttype": 42})
    eda(args.root, output)
    if not args.eda_only:
        curves({model: read(args.root / f"train/{model}/training.json") for model in MODELS}, output)
        _, _, examples = aggregate(args.root, output)
        export_decoding_results(output)
        training_architectures(output)
        if args.report:
            build_report(output, examples, output / "report.pdf", args.browser)
        if args.archive:
            archive_results(args.root, args.archive)
