import argparse
from collections import Counter
import csv
from html import escape
import json
from pathlib import Path
import re
import tarfile
import unicodedata as ud

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pymupdf
from sacrebleu.metrics import CHRF
from weasyprint import HTML

from prepare import LANGUAGES, normalize, read_jsonl, scripts, write_json


DIRECTIONS = {"en-kha": "English → Khasi", "kha-en": "Khasi → English",
              "en-brx": "English → Bodo", "brx-en": "Bodo → English"}
MODELS = {"madlad": "MADLAD-400 3B", "gemma": "Gemma 3 1B"}
COLORS = ["#32618e", "#ad532c"]
REVIEW_CASES = {("madlad", "en-brx"): "brx:13772", ("gemma", "kha-en"): "kha:19957"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


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
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.5), layout="constrained", sharey=True)
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
    write_json(output / "scores.json", scores)
    write_scores_csv(output / "scores.csv", scores)
    write_json(output / "diagnostics.json", diagnostics)
    write_json(output / "examples.json", examples)
    return scores, diagnostics, examples


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


def curves(root, output):
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.3), layout="constrained")
    for axis, (model, title) in zip(axes, MODELS.items()):
        history = read(root / f"train/{model}/training.json")["history"]
        for metric, label, color in zip(("loss", "eval_loss"), ("Training", "Validation"), COLORS):
            xy = [(item["epoch"], item[metric]) for item in history if metric in item]
            axis.plot(*zip(*xy), label=label, color=color, linewidth=1.2,
                      marker="o" if metric == "eval_loss" else None, markersize=3)
        axis.set(title=title, xlabel="Epoch", ylabel="Cross-entropy")
        axis.legend(frameon=False)
    finish(fig, output, "training")


def table(header, data):
    def cells(tag, values):
        return "".join(f"<{tag}>{escape(str(value))}</{tag}>" for value in values)
    return ("<table><thead><tr>" + cells("th", header) + "</tr></thead><tbody>" +
            "".join("<tr>" + cells("td", row) + "</tr>" for row in data) + "</tbody></table>")


def visible_text(text):
    excerpt = text[:360].replace("\ufffd", "[U+FFFD]")
    return excerpt + ("…" if len(text) > 360 else "")


def report(output, stats, scores, diagnostics, examples):
    def score_pair(model, direction, metric):
        base = scores[model]["base"][direction][metric]
        tuned = scores[model]["tuned"][direction][metric]
        return f"{base:.2f} → {tuned:.2f}"

    data_table = table(
        ("Corpus", "Raw pairs", "Train", "Validation", "Test"),
        [(f"{name} · Category {index}", f"{stats[code]['usable_rows']:,}",
          *(f"{stats[code][split + '_pairs']:,}" for split in ("train", "validation", "test")))
         for index, (code, name) in enumerate(LANGUAGES.items(), 1)],
    )
    score_table = table(
        ("Direction", "MADLAD BLEU", "MADLAD chrF++", "Gemma BLEU", "Gemma chrF++"),
        [(label, *(score_pair(model, direction, metric) for model in MODELS
                   for metric in ("bleu", "chrf++"))) for direction, label in DIRECTIONS.items()],
    )
    flags = ("exact_normalized_source_copy", "short_output",
             "repeated_trigram", "unexpected_majority_script")
    diagnostic_table = table(
        ("Model and direction", "Copies", "Short", "Repeated", "Script"),
        [(f"{name} · {direction}", *(f"{group['base'].get(flag, 0)} → {group['tuned'].get(flag, 0)}"
                                    for flag in flags))
         for model, name in (("madlad", "MADLAD"), ("gemma", "Gemma"))
         for direction, group in diagnostics[model].items()],
    )
    cases = []
    for model, direction in REVIEW_CASES:
        options = examples[model][direction]
        case = options.get("reviewed", options["median"])
        fields = "".join(f"<dt>{label}</dt><dd>{escape(visible_text(case[key]))}</dd>"
                         for label, key in (("Source", "source"), ("Reference", "reference"),
                                            ("Zero-shot", "base"), ("Fine-tuned", "tuned")))
        cases.append(f"<article><h3>{MODELS[model]} · {DIRECTIONS[direction]}</h3>"
                     f"<dl>{fields}</dl></article>")
    madlad_gain = scores["madlad"]["delta"]["en-brx"]["chrf++"]
    gemma_bodo = diagnostics["gemma"]["en-brx"]
    gemma_repetitions = gemma_bodo["tuned"].get("repeated_trigram", 0)
    script_base = gemma_bodo["base"]["unexpected_majority_script"]
    script_tuned = gemma_bodo["tuned"]["unexpected_majority_script"]
    html = f"""
    <!doctype html>
    <html lang="en"><head><meta charset="utf-8"><title>Low-resource machine translation</title>
    <style>
    @page {{
      size: A4; margin: 15mm 17mm 17mm;
      @bottom-left {{ content: "LOW-RESOURCE MACHINE TRANSLATION"; font: 8pt 'Noto Sans'; color: #52646d; }}
      @bottom-right {{ content: counter(page) " / " counter(pages); font: 8pt 'Noto Sans'; color: #52646d; }}
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font: 9.5pt/1.46 'Noto Sans', 'Noto Sans Devanagari', sans-serif; color: #172d36; }}
    section + section {{ break-before: page; }}
    header {{ border-top: 3pt solid #146b72; padding-top: 9pt; margin-bottom: 10pt; }}
    h1,h2,h3,p,figure,dl {{ margin: 0; }}
    h1 {{ font: bold 31pt/1.06 'Noto Serif', serif; letter-spacing: -.7pt; max-width: 160mm; }}
    h2 {{ font: bold 23pt/1.12 'Noto Serif', serif; letter-spacing: -.35pt; }}
    h3 {{ font-size: 10pt; line-height: 1.25; margin-bottom: 4pt; }}
    header p {{ margin-top: 6pt; }}
    .lead {{ color: #405760; font-size: 10pt; }}
    .grid {{ display: flex; gap: 12pt; }}
    .grid > * {{ flex: 1; min-width: 0; }}
    .method > div {{ padding-top: 7pt; border-top: .7pt solid #b9cbce; }}
    .method p + p {{ margin-top: 5pt; }}
    aside {{ background: #e9f3f2; border-left: 3pt solid #146b72; padding: 8pt 10pt; margin: 8pt 0 10pt; }}
    aside strong {{ color: #12585e; }}
    table {{ width: 100%; border-collapse: collapse; margin: 7pt 0 9pt; font-size: 8pt; }}
    th,td {{ text-align: left; padding: 4pt 5pt; border-bottom: .5pt solid #cfdbde; }}
    th {{ background: #143b46; color: white; font-weight: bold; }}
    tr:nth-child(even) td {{ background: #f1f6f6; }}
    figure {{ border: .6pt solid #ccd9dc; padding: 6pt 8pt; margin: 8pt 0 10pt; }}
    figure img {{ display: block; width: 100%; margin: 0 auto; }}
    .result-figure img {{ width: 84%; }}
    figcaption,.small {{ color: #405760; font-size: 8.2pt; line-height: 1.42; }}
    figcaption {{ margin-top: 4pt; }}
    article {{ background: #edf4f4; padding: 8pt; }}
    dt {{ font-weight: bold; color: #165d65; font-size: 7.5pt; }}
    dd {{ margin: 0 0 4pt; font-size: 8pt; overflow-wrap: anywhere; }}
    dd:last-child {{ margin-bottom: 0; }}
    .analysis {{ margin-top: 8pt; }}
    footer {{ border-top: .7pt solid #b9cbce; padding-top: 5pt; margin-top: 8pt; }}
    a {{ color: #145e66; }}
    </style></head><body>
    <section>
      <header><h1>Low-resource machine translation</h1>
      <p class="lead">English ↔ Khasi and English ↔ Bodo · WMT26 Indic MT</p></header>
      <p>I compared two pretrained models before and after one epoch of supervised fine-tuning.
      I used the same <strong>70,608 directed training examples</strong> for MADLAD-400 3B and
      Gemma 3 1B, then evaluated both versions on matching holdouts.</p>
      {data_table}
      <figure><img src="lengths.svg" alt="Two cumulative sentence-length charts for Khasi and Bodo">
      <figcaption><strong>Sentence lengths.</strong> Cumulative distributions of whitespace-token lengths.
      The horizontal axis shows up to 100 tokens.</figcaption></figure>
      <div class="grid method">
        <div><h3>Prepare the data</h3>
        <p>I inspected eight pairs per language. Khasi samples were mostly biblical. Bodo samples covered
        history, daily life, and news. The Bodo corpus contained {stats['brx']['normalized_pair_duplicates']:,}
        normalized duplicate pairs and {stats['brx']['excel_escape_rows']} rows with spreadsheet escape patterns.</p>
        <p>I screened for normalized duplicates and official-source overlap before creating fixed validation
        and test splits. The four official lists supplied 4,000 source sentences without references.</p></div>
        <div><h3>Check, train, and evaluate</h3>
        <p>I tested NF4 loading, optimizer updates, adapter reloads, and checkpoint recovery, then used
        A40 benchmarks to choose the batch sizes.</p>
        <p>I trained with BF16 NF4 QLoRA, rank 16, alpha 32, AdamW, learning rate 2 × 10⁻⁴, and seed 42.
        I used MADLAD target tags and Gemma’s chat template, masking Gemma’s prompt from the loss.
        I used an effective batch of 64 examples for each model, chose checkpoints by validation loss,
        and decoded greedily with up to 256 new tokens.</p></div>
      </div>
    </section>
    <section>
      <header><h2>What changed after fine-tuning</h2>
      <p class="lead">Corpus BLEU and chrF++ on 3,274 labeled test examples</p></header>
      {score_table}
      <p class="small">Each cell shows zero-shot → fine-tuned. I used SacreBLEU with 13a tokenization
      and chrF++ with word bigrams. I generated official predictions separately for all four systems.</p>
      <aside><strong>MADLAD improved on both metrics in all four directions.</strong>
      Its largest chrF++ gain was {madlad_gain:+.2f} for English → Bodo. Gemma’s results were mixed,
      including a drop in Khasi → English chrF++ and English → Bodo BLEU.</aside>
      <figure class="result-figure"><img src="improvements.svg" alt="Changes in BLEU and chrF++ by model and direction">
      <figcaption><strong>Score changes.</strong> Each bar shows the point change from the matching zero-shot baseline.</figcaption></figure>
      <figure class="result-figure"><img src="training.svg" alt="Training and validation loss for MADLAD and Gemma">
      <figcaption><strong>Training progress.</strong> Validation loss fell across the four evaluated checkpoints for each model.</figcaption></figure>
      <p class="small">Both tuned models averaged higher chrF++ into English. Both also averaged higher
      chrF++ on Khasi directions. Corpus size, domain, script, tokenization, model size, and pretraining
      differ across these comparisons. The study used one seed.</p>
    </section>
    <section>
      <header><h2>Where translations still failed</h2>
      <p class="lead">Automatic checks and examples from the labeled holdouts</p></header>
      {diagnostic_table}
      <p class="small">Counts show zero-shot → fine-tuned. A copy matches the normalized source.
      Short means under half the reference’s whitespace-token count. A repetition flag requires the
      same trigram at least three times. The script flag checks the majority letter script.</p>
      <aside>For Gemma in English → Bodo, script flags fell from <strong>{script_base} to {script_tuned}</strong>
      out of 637 outputs. After fine-tuning, <strong>{gemma_repetitions}</strong> outputs triggered the repetition check.</aside>
      <div class="grid">{''.join(cases)}</div>
      <p class="analysis">In the MADLAD example, fine-tuning removed the English copy and produced
      Devanagari text with recognizable forms of Michael Jordan and basketball. In the Gemma example,
      fine-tuning recovered Tom and 2:30 but still omitted the homecoming.</p>
      <p class="analysis">I found other meaning errors when comparing outputs with English references.
      MADLAD turned a song into a taxi company. Gemma changed a question about Bitcoin crossing a
      threshold into a question about whether it had already reached it. I did not conduct a bilingual adequacy review.</p>
      <footer class="small"><strong>Sources</strong> · <a href="https://aclanthology.org/2026.wmt-main.53/">WMT26 Indic MT findings</a>.
      Kudugunta et al., MADLAD-400 (2023). Google, Gemma 3 Technical Report (2025).
      Dettmers et al., QLoRA (2023). Post, BLEU reporting (2018). Popović, chrF++ (2017).</footer>
    </section>
    </body></html>
    """
    path = output / "report.pdf"
    HTML(string=html, base_url=str(output.resolve())).write_pdf(path)
    with pymupdf.open(path) as pdf:
        if len(pdf) != 3:
            raise ValueError(f"Report has {len(pdf)} pages")
        text = "".join(page.get_text() for page in pdf)
        assert any("DEVANAGARI" in ud.name(char, "") for char in text)
        assert "\ufffd" not in text
        for index, page in enumerate(pdf, 1):
            page.get_pixmap(dpi=110).save(output / f"report-page-{index}.png")


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
        for name in ("README.md", "requirements.txt", "prepare.py", "study.py", "results.py",
                     "model_loading.py", "fine_tuning.py", "evaluation.py", "checks.py"):
            bundle.add(Path(__file__).with_name(name), arcname=f"reproduction/{name}")
        for path in sorted(root.glob("*.log")):
            if "qwen" not in path.name:
                bundle.add(path, arcname=f"logs/{path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs"))
    parser.add_argument("--eda-only", action="store_true")
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    output = args.root / "results"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "path", "pdf.fonttype": 42})
    statistics = eda(args.root, output)
    if not args.eda_only:
        curves(args.root, output)
        report(output, statistics, *aggregate(args.root, output))
        if args.archive:
            archive_results(args.root, args.archive)
