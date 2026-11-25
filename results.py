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


def training_architectures(output):
    fig, ax = plt.subplots(figsize=(7, 2.75))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0, 700), ylim=(0, 275))
    ax.axis("off")

    def label(x, y, words, size=9, **style):
        ax.text(x, y, words, ha="center", va="center", fontsize=size,
                fontfamily="serif", **style)

    def arrow(x, start, end, dashed=False):
        ax.annotate("", (x, end), (x, start), arrowprops={
            "arrowstyle": "->", "linewidth": .75, "linestyle": "--" if dashed else "-",
        })

    madlad, gemma = 172, 528
    label(madlad, 261, "MADLAD-400 3B · Encoder–decoder", weight="bold")
    label(gemma, 261, "Gemma 3 1B · Decoder-only", weight="bold")

    label(madlad, 229, "Target tag + source tokens")
    arrow(madlad, 214, 196)
    label(madlad, 182, "Encoder self-attention + LoRA")
    arrow(madlad, 168, 149)
    label(madlad, 130, "Decoder self- and cross-attention + LoRA\nEncoder states and shifted targets")
    arrow(madlad, 105, 88)

    label(gemma, 229, "Chat instruction + source tokens")
    arrow(gemma, 214, 149)
    label(gemma, 130, "Causal decoder self-attention + LoRA\nTarget tokens appended during training")
    arrow(gemma, 105, 88)

    for xpos in (madlad, gemma):
        label(xpos, 77, "Next-token logits")
        ax.plot([xpos - 77, xpos + 77], [65, 65], color="#111", lw=.65)
        arrow(xpos - 77, 65, 42)
        arrow(xpos + 77, 65, 42, dashed=True)
        label(xpos + 77, 28, "Generated target tokens", size=8)
    label(madlad - 77, 28, "Target cross-entropy", size=8)
    label(gemma - 77, 24, "Target cross-entropy\nPrompt labels masked", size=8)
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


def report(output, stats, scores, diagnostics, examples):
    export_decoding_results(output)
    training_architectures(output)

    def val(model, direction, metric, stage):
        return scores[model][stage][direction][metric]

    data = table(("Language", "Category", "Raw pairs", "Train", "Validation", "Test", "Median EN / target"),
        [(name, index, *(f"{stats[code][key]:,}" for key in
         ("usable_rows", "train_pairs", "validation_pairs", "test_pairs")),
         f"{stats[code]['english_whitespace_tokens']['median']:.0f} / {stats[code]['target_whitespace_tokens']['median']:.0f}")
         for index, (code, name) in enumerate(LANGUAGES.items(), 1)])
    main = table(("Model", "Direction", "BLEU base", "Tuned", "Δ", "chrF++ base", "Tuned", "Δ"),
        [(label, name, *[f"{val(model, direction, metric, stage):+.2f}" if stage == "delta" else
          f"{val(model, direction, metric, stage):.2f}" for metric in ("bleu", "chrf++")
          for stage in ("base", "tuned", "delta")]) for model, label in MODELS.items()
         for direction, name in DIRECTIONS.items()])
    pilot_path = output / "additional-decoding.json"
    evaluation_path = output / "additional-evaluation.json"
    pilot = read(pilot_path) if pilot_path.exists() else []
    evaluation = read(evaluation_path)["rows"] if evaluation_path.exists() else []
    additional = ""
    if pilot:
        pilot_rows = []
        for row in sorted(pilot, key=lambda r: (r["model"], r["shots"], r["selection"], r["beams"])):
            label = (f"MADLAD, beam {row['beams']}" if row["model"] == "madlad" else
                     f"Gemma, {row['shots']}-shot {row['selection']}" if row["shots"] else "Gemma, zero-shot")
            means = [sum(x[metric] for x in row["scores"].values()) / len(row["scores"])
                     for metric in ("bleu", "chrf++")]
            pilot_rows.append((label, *(f"{value:.2f}" for value in means), f"{row['seconds']:.1f}"))
        pilot_table = table(("Condition", "BLEU", "chrF++", "Seconds"), pilot_rows)
        additional = f"""<h3>3.1 Additional evaluation</h3><div class="pilot">{pilot_table}</div>
        <p class="compact"><b>Table 4.</b> Decoding pilots. Scores average four directions with 12 examples each.
        Each pass used a 128-token cap. Times exclude model loading and demonstration retrieval.
        Gemma demonstrations came from same-direction training pairs, selected randomly or by weighted source-token overlap. Lexical selection led random at each shot count in chrF++. Beam 20 scored
        highest among the tested widths and took longer than greedy decoding. The sample is exploratory.</p>"""
    if evaluation:
        row = next(r for r in evaluation if r["model"] == "gemma" and r["direction"] == "en-brx")
        additional += (f'<p class="compact">For Gemma into Bodo, the full-holdout chrF++ gain was '
                       f'{row["chrf++_delta"]:+.2f} (95% paired interval '
                       f'{row["ci95_low"]:+.2f} to {row["ci95_high"]:+.2f}). '
                       'Intervals use 2,000 paired resamples with seed 42 and capture sentence sampling only. '
                       'See page 4 for translation edit rate (TER). Data augmentation was not tested.</p>')
    observations = diagnostics["gemma"]["en-brx"]
    cases = [examples[model][direction]["reviewed"] for model, direction in REVIEW_CASES]
    def case(case):
        return "".join(f"<p><b>{label}</b> {escape(case[key])}</p>" for label, key in
                       (("Source", "source"), ("Reference", "reference"), ("Base", "base"), ("Tuned", "tuned")))
    html = f"""
    <!doctype html><html lang="en"><head><meta charset="utf-8"><title>Machine Translation with Pretrained Models</title>
    <style>
    @page {{ size:A4; margin:16mm 19mm; @bottom-center {{ content:counter(page); font:9pt 'Noto Serif'; }} }}
    body {{ margin:0; font:9.5pt/1.38 'Noto Serif','Noto Sans Devanagari',serif; color:#111; }}
    section+section {{ break-before:page; }} section:not(.results) {{ font-size:10.3pt; line-height:1.43; }} h1 {{ text-align:center; font-size:19pt; margin:0 0 10pt; }}
    h2 {{ font-size:12.5pt; margin:10pt 0 4pt; }} h3 {{ font-size:10pt; margin:7pt 0 3pt; }}
    p {{ margin:0 0 7pt; }} figure {{ margin:7pt 0; break-inside:avoid; }} figure img {{ width:100%; display:block; margin:auto; }}
    .length img {{ width:92%; }} .training img {{ width:72%; }} figcaption {{ font-size:8.3pt; line-height:1.3; margin-top:3pt; }}
    table {{ width:100%; border-collapse:collapse; font-size:8pt; margin:5pt 0; border-top:.8pt solid; border-bottom:.8pt solid; }}
    th,td {{ padding:3pt 4pt; text-align:right; }} th {{ border-bottom:.6pt solid; }} td:first-child,th:first-child,td:nth-child(2),th:nth-child(2) {{ text-align:left; }}
    .pilot th,.pilot td {{ padding:1pt 4pt; }} .pilot td:nth-child(2),.pilot th:nth-child(2) {{ text-align:right; }}
    .compact {{ font-size:8.4pt; line-height:1.27; margin-bottom:4pt; }} .nowrap {{ white-space:nowrap; }}
    .cases {{ margin:7pt 0 10pt; }} .cases p {{ margin:3pt 0; font-size:9.3pt; }}
    .cases h3 {{ border-top:.5pt solid #444; padding-top:5pt; margin-top:9pt; }} .refs {{ font-size:8.5pt; line-height:1.35; }}
    </style></head><body>
    <section><h1>Machine Translation with Pretrained Models</h1>
    <p><b>Abstract.</b> This study compares zero-shot and one-epoch QLoRA translation with MADLAD-400 3B and Gemma 3 1B for English–Khasi and English–Bodo. On matched holdouts, MADLAD improved BLEU and chrF++ in all four directions. Gemma showed mixed results. Repetition increased in English→Bodo even as script use improved.</p>
    <h2>1 Introduction and dataset</h2>
    <p>The WMT26 Indic MT data includes Khasi in Category 1 and Bodo in Category 2. The study uses the same parallel examples in both directions for each model. Table 1 shows raw and retained pair counts. Khasi has approximately twice the training data, although language, script and domain also differ.</p>
    {data}<p><small><b>Table 1.</b> Pair counts after screening and splitting. Median lengths use raw whitespace-token counts.</small></p>
    <figure class="length"><img src="lengths.svg"><figcaption><b>Figure 1.</b> Sentence-length distributions in the original parallel corpora.</figcaption></figure>
    <p>Eight pairs per language were inspected. Khasi examples were predominantly biblical, with longer target sentences and occasional apparent verse-to-heading or verse-to-passage mismatches. Bodo samples covered history, daily life and news. The Bodo data contained 2,451 normalized duplicate pairs and 35 rows with spreadsheet escape patterns. Khasi targets use Latin script and Bodo targets predominantly use Devanagari.</p>
    <p>The length distributions also differ in their upper tails. The 95th-percentile English and target lengths are
    {stats["kha"]["english_whitespace_tokens"]["p95"]} and {stats["kha"]["target_whitespace_tokens"]["p95"]} tokens for Khasi pairs,
    compared with {stats["brx"]["english_whitespace_tokens"]["p95"]} and {stats["brx"]["target_whitespace_tokens"]["p95"]} for Bodo pairs.
    These differences matter for batching and highlight why pair counts alone do not characterize the training data.</p></section>
    <section><h2>2 Methodology</h2>
    <figure><img src="architectures.svg"><figcaption><b>Figure 2.</b> Training paths. Base weights are frozen. LoRA updates query and value projections in encoder and decoder attention, including MADLAD cross-attention. Dashed arrows show inference. Training labels include EOS for MADLAD and the end-of-turn token for Gemma.</figcaption></figure>
    <h3>2.1 Splitting and inputs</h3><p>Normalized repeated sides and overlaps with official sources were screened before deterministic validation and test holdouts were created. Both directions of each pair remain in the same split. Official workbooks provide 4,000 source sentences without references, so their translations are generated separately. MADLAD receives a target prefix (<code>&lt;2en&gt;</code>, <code>&lt;2kha&gt;</code>, or <code>&lt;2brx&gt;</code>). Gemma receives a translation instruction through its chat template, with instruction and source labels masked.</p>
    <h3>2.2 Fine-tuning and evaluation</h3>
    {table(("Setting", "MADLAD-400 3B", "Gemma 3 1B"), (("Epochs / seed","1 / 42","1 / 42"),("Learning rate / optimizer","2 × 10⁻⁴ / AdamW","2 × 10⁻⁴ / AdamW"),("LoRA rank / alpha","16 / 32","16 / 32"),("Microbatch / accumulation","32 / 2","16 / 4"),("Precision","NF4 + BF16","NF4 + BF16")))}
    <p><small><b>Table 2.</b> Shared effective batch size is 64. Frozen linear weights use double-quantized NF4.</small></p>
    <p>Adapters were selected by validation loss at quarter-epoch checkpoints. Both models selected the final checkpoint. Source and target segments were capped at 256 tokens in training. Both baseline and adapted models were decoded in BF16 with greedy search and at most 256 new tokens. Evaluation uses case-sensitive corpus BLEU with 13a tokenization and chrF++ with word bigrams. Loading, optimizer updates, adapter reloads and checkpoint recovery were tested before the full runs.</p></section>
    <section class="results"><h2>3 Results and discussion</h2>
    {main}<p><small><b>Table 3.</b> Full labeled holdout. Δ is tuned minus baseline, calculated before rounding.</small></p>
    <p>MADLAD improved both metrics in all directions and led Gemma on the same test sets. Supervision may have adapted its existing translation objective and language-tag interface. Gemma's gains were smaller and mixed across metrics. Model size, pretraining, tokenizer and input format also differ, so architecture alone cannot explain the gap. Both tuned models scored higher on Khasi than Bodo on average, but the two categories also differ in language and domain.</p>
    <figure class="training"><img src="training.svg"><figcaption><b>Figure 3.</b> Training and validation loss. Trends should be compared within models because their objectives and tokenization differ.</figcaption></figure>
    {additional}</section>
    <section><h2>4 Error analysis and conclusion</h2>
    <p>For English→Bodo, Gemma's majority-script flags fell from {observations['base']['unexpected_majority_script']} to {observations['tuned']['unexpected_majority_script']} of 637 outputs, while repeated-trigram flags rose from {observations['base']['repeated_trigram']} to {observations['tuned']['repeated_trigram']}. The adapted output can use the expected script without preserving meaning. <span class="nowrap">chrF++</span> rose from 1.57 to 3.83, while TER increased from 109.77 to 496.57. Repetition can increase character overlap and also require many edits.</p>
    <h3>4.1 Translation examples</h3><div class="cases"><h3>MADLAD · English → Bodo</h3>{case(cases[0])}<h3>Gemma · Khasi → English</h3>{case(cases[1])}</div>
    <p>MADLAD initially copied the English source. Its adapted output uses Devanagari and retains recognizable names. Gemma's adapted output recovers Tom and the time but omits coming home. These observations compare visible outputs with the supplied references. The study used automatic metrics and qualitative inspection without a bilingual adequacy assessment.</p>
    <h3>4.2 Limitations and conclusion</h3><p>The labeled holdout comes from the training-corpus distribution and has one reference per source. Near-duplicate and pretraining overlap were not assessed. Results use one training seed and one language per resource category. The source-only official sets cannot support reference-based scores. MADLAD benefits consistently from adaptation on these holdouts. Gemma's script improvement coexists with repetition and omissions, making multiple metrics and example inspection important.</p>
    <h3>References</h3><p class="refs">Pakray et al. (2026), WMT26 Indic MT findings. Kudugunta et al. (2023), MADLAD-400. Gemma Team (2025), Gemma 3 technical report. Dettmers et al. (2023), QLoRA. Post (2018), BLEU reporting. Popović (2017), chrF++. Koehn (2004), paired bootstrap resampling. Snover et al. (2006), translation edit rate.</p></section>
    </body></html>
    """
    HTML(string=html, base_url=str(output.resolve())).write_pdf(output / "report.pdf")
    with pymupdf.open(output / "report.pdf") as pdf:
        if len(pdf) != 4:
            raise ValueError(f"Report has {len(pdf)} pages")
        for i, page in enumerate(pdf, 1):
            page.get_pixmap(dpi=150).save(output / f"report-page-{i}.png")



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
                     "model_loading.py", "fine_tuning.py", "evaluation.py", "checks.py",
                     "additional_evaluation.py", "decoding_pilot.py"):
            bundle.add(Path(__file__).with_name(name), arcname=f"reproduction/{name}")
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
