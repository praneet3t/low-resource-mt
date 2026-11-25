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
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(7.4, 2.2))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0, 740), ylim=(0, 220))
    ax.axis("off")

    def node(x, y, width, text):
        ax.add_patch(Rectangle((x, y - 15), width, 30, fill=False, linewidth=.65))
        ax.text(x + width / 2, y, text, va="center", ha="center", fontsize=8)

    def arrow(x, y, end_x, end_y, style="-"):
        ax.annotate("", (end_x, end_y), (x, y),
                    arrowprops={"arrowstyle": "->", "linewidth": .7, "shrinkA": 0, "shrinkB": 0, "linestyle": style})

    ax.text(4, 204, "(a) MADLAD-400 3B", weight="bold", fontsize=9)
    ax.text(4, 166, "Target tag +\nsource tokens", va="center", fontsize=8)
    arrow(86, 166, 108, 166)
    node(108, 166, 120, "Encoder\nSelf-attn. + LoRA")
    arrow(228, 166, 294, 166)
    ax.text(260, 173, "states", ha="center", fontsize=7)
    node(294, 166, 183, "Decoder + LoRA\nSelf- and cross-attn.")
    ax.text(385, 119, "Shifted target tokens", ha="center", fontsize=8)
    arrow(385, 129, 385, 151)
    arrow(477, 166, 503, 166)
    ax.text(508, 166, "Target logits", va="center", fontsize=8)
    arrow(583, 166, 602, 166)
    ax.text(608, 166, "CE against\ntarget tokens", va="center", fontsize=8)
    arrow(540, 156, 540, 128, "--")
    ax.text(551, 125, "Generated target tokens", va="center", fontsize=8)

    ax.text(4, 93, "(b) Gemma 3 1B", weight="bold", fontsize=9)
    ax.text(4, 55, "Chat instruction + source\n+ targets (training)", va="center", fontsize=8)
    arrow(176, 55, 208, 55)
    node(208, 55, 183, "Causal decoder\nself-attention + LoRA")
    arrow(391, 55, 424, 55)
    ax.text(429, 55, "Next-token logits", va="center", fontsize=8)
    arrow(535, 55, 555, 55)
    ax.text(561, 55, "CE on target positions\nPrompt labels ignored", va="center", fontsize=8)
    arrow(474, 46, 474, 22, "--")
    ax.text(486, 18, "Generated target tokens", va="center", fontsize=8)
    ax.text(4, 12, "Frozen base weights. Dashed arrows denote inference.", fontsize=8)
    finish(fig, output, "architectures")


def export_decoding_results(output):
    pilot = []
    for model in MODELS:
        for path in sorted((output.parent / "bonus" / model).glob("*.json")):
            if not path.name.endswith(".translations.json"):
                pilot.append(read(path))
    if not pilot:
        return
    write_json(output / "bonus-decoding.json", pilot)
    with (output / "bonus-decoding.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("model", "condition", "direction", "examples", "bleu", "chrf++",
                         "condition_decode_seconds"))
        for run in pilot:
            for direction, score in run["scores"].items():
                writer.writerow((run["model"], run["label"], direction, score["count"], score["bleu"],
                                 score["chrf++"], run["seconds"]))


def report(output, stats, scores, diagnostics, examples):
    export_decoding_results(output)
    with plt.rc_context({"font.family": "serif", "svg.fonttype": "path", "pdf.fonttype": 42}):
        training_architectures(output)

    def render_table(headers, rows, caption):
        return f'<figure class="tabular"><figcaption>{caption}</figcaption>{table(headers, rows)}</figure>'

    corpus = render_table(
        ("Language", "Category", "Raw", "Train", "Dev", "Test", "EN / target median"),
        [(name, i, *(f"{stats[code][key]:,}" for key in ("usable_rows", "train_pairs",
                                                       "validation_pairs", "test_pairs")),
          f"{stats[code]['english_whitespace_tokens']['median']:.0f} / "
          f"{stats[code]['target_whitespace_tokens']['median']:.0f}")
         for i, (code, name) in enumerate(LANGUAGES.items(), 1)],
        "<b>Table 1:</b> Parallel pair counts and median whitespace-token lengths in the raw corpora.",
    )
    main_rows = []
    for model, label in MODELS.items():
        for direction, name in DIRECTIONS.items():
            values = scores[model]
            main_rows.append((label, name, *
                              [f"{values[stage][direction][metric]:{'+.2f' if stage == 'delta' else '.2f'}}"
                               for metric in ("bleu", "chrf++") for stage in ("base", "tuned", "delta")]))
    headings = ('<thead><tr><th rowspan="2">Model</th><th rowspan="2">Direction</th>'
                '<th colspan="3">BLEU</th><th colspan="3">chrF++</th></tr>'
                '<tr><th>Base</th><th>Tuned</th><th>Δ</th><th>Base</th><th>Tuned</th><th>Δ</th></tr></thead>')
    comparison = table((), main_rows).replace("<thead><tr></tr></thead>", headings)
    cpu_file = output / "bonus-evaluation.json"
    pilot_file = output / "bonus-decoding.json"
    cpu = read(cpu_file) if cpu_file.exists() else {"rows": []}
    pilot = read(pilot_file) if pilot_file.exists() else []
    intervals = render_table(
        ("Model", "Direction", "Δ chrF++ [95% CI]", "TER base", "TER tuned"),
        [(MODELS[row["model"]], row["direction"],
          f"{row['chrf++_delta']:+.2f} [{row['ci95_low']:+.2f}, {row['ci95_high']:+.2f}]",
          f"{row['ter_base']:.2f}", f"{row['ter_tuned']:.2f}") for row in cpu["rows"]],
        "<b>Table 3:</b> Paired percentile intervals and case-sensitive TER on the full holdout.",
    )
    pilot_table = render_table(
        ("Model / condition", "BLEU", "chrF++", "Time (s)"),
        [((f"MADLAD, beam {row['beams']}" if row["model"] == "madlad" else
           f"Gemma, {row['shots']}-shot {row['selection']}" if row["shots"] else "Gemma, zero-shot"),
          *(f"{sum(x[metric] for x in row['scores'].values()) / len(row['scores']):.2f}"
            for metric in ("bleu", "chrf++")), f"{row['seconds']:.1f}")
         for row in sorted(pilot, key=lambda x: (x["model"], x["beams"], x["shots"], x["selection"]))],
        "<b>Table 4:</b> Decoding pilots on 48 matched examples. Scores are means of the four directional "
        "corpus scores. Time is for one pass and excludes loading and retrieval.",
    )
    case_blocks = []
    for model, direction in REVIEW_CASES:
        group = examples[model][direction]
        case = group.get("reviewed", group["median"])
        entries = "".join(f'<dt>{label}</dt><dd>{escape(visible_text(case[key]))}</dd>'
                          for label, key in (("Source", "source"), ("Reference", "reference"),
                                             ("Base", "base"), ("Tuned", "tuned")))
        case_blocks.append(f'<div><h3>{MODELS[model]} · {DIRECTIONS[direction]}</h3>'
                           f'<dl>{entries}</dl></div>')
    gemma = diagnostics["gemma"]["en-brx"]
    madlad = diagnostics["madlad"]["en-brx"]
    gain = scores["madlad"]["delta"]["en-brx"]["chrf++"]
    has_supplement = bool(cpu["rows"] and pilot)
    additional = f"""
      <h2>5 Additional experiments</h2>
      <p>On the full holdout, I resampled matched sentences within each direction 2,000 times
      with seed 42 and recomputed corpus chrF++ (Koehn, 2004). These intervals reflect sentence
      sampling for the fitted systems. They omit variation across training seeds and are
      unadjusted for multiple comparisons. TER measures edits per reference word and can
      exceed 100 (Snover et al., 2006).</p>
      {intervals}
      <p>In separate pilots with 12 test sentences per direction, I used pretrained Gemma
      with zero, one, three, or five demonstrations drawn from same-direction training pairs.
      Selection used seeded random sampling or inverse-frequency-weighted source-token overlap.
      With pretrained MADLAD, I varied beam width at a fixed batch size. Both pilots capped
      generation at 128 new tokens.</p>
      {pilot_table}
    """ if has_supplement else ""
    pilot_discussion = f"""
      <p>Lexical selection outscored random selection at every tested shot count on chrF++.
      The best BLEU and chrF++ occurred at different counts. Retrieved near-matches may
      contribute to these gains. Beam width 20 gave the highest scores among the tested
      widths but increased decoding time. Widths 5 and 10 reduced chrF++ relative to greedy
      decoding. These rankings are specific to the small sample. I did not test data augmentation.</p>
    """ if has_supplement else ""
    html = f"""
    <!doctype html><html lang="en"><head><meta charset="utf-8">
    <title>Machine Translation with Pretrained Models</title>
    <style>
    @page {{ size: A4; margin: 17mm 18mm 17mm; @bottom-center {{
      content: counter(page); font: 8.5pt 'Noto Serif'; }} }}
    body {{ margin:0; color:#111; font:9.8pt/1.31 'Noto Serif','Noto Sans Devanagari',serif; }}
    section + section {{ break-before:page; }}
    h1 {{ margin:3mm 0 8mm; text-align:center; font-size:19pt; line-height:1.18; }}
    h2 {{ margin:0 0 5pt; font-size:12pt; line-height:1.25; }}
    h3 {{ margin:0 0 4pt; font-size:9.4pt; }}
    p {{ margin:0 0 6pt; widows:2; orphans:2; }}
    .columns {{ display:flex; gap:6mm; align-items:flex-start; }}
    .columns > div {{ flex:1; min-width:0; }}
    figure {{ margin:8pt 0 10pt; break-inside:avoid; }}
    img {{ display:block; margin:auto; width:94%; }}
    .architecture img {{ width:100%; }}
    .length img {{ width:76%; }}
    .supplement td,.supplement th {{ padding-top:2.2pt; padding-bottom:2.2pt; }}
    figcaption {{ margin-top:4pt; font-size:8.8pt; line-height:1.34; }}
    .tabular figcaption {{ margin:0 0 5pt; }}
    table {{ border-collapse:collapse; width:100%; font-size:8.4pt; line-height:1.3;
             border-top:.8pt solid #171717; border-bottom:.8pt solid #171717; }}
    thead {{ border-bottom:.55pt solid #171717; }}
    th,td {{ text-align:right; padding:2.8pt 4pt; white-space:nowrap; }}
    th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) {{ text-align:left; }}
    .primary tbody tr:nth-child(5) {{ border-top:.5pt solid #333; }}
    .primary th[colspan] {{ text-align:center; }}
    .example {{ margin:10pt 0; }}
    dl {{ margin:0; font-size:9pt; line-height:1.35; }}
    dt {{ font-weight:bold; }}
    dd {{ margin:0 0 4pt; overflow-wrap:anywhere; }}
    .refs {{ font-size:8.6pt; line-height:1.4; }}
    a {{ color:#111; }}
    </style></head><body>
    <section>
      <h1>Machine Translation with Pretrained Models</h1>
      <div class="columns"><div>
      <h2>1 Introduction</h2>
      <p>I study how one epoch of supervised adaptation changes translation between English
      and two low-resource languages, Khasi and Bodo. I compare a pretrained translation model
      with an instruction-tuned causal model, using the same directed examples and evaluating
      each checkpoint before and after adaptation. The comparison asks which directions
      improve and whether gains in reference overlap coincide with fewer observable errors.</p>
      <h2>2 Dataset and exploratory analysis</h2>
      <p>I used the WMT26 Indic MT workbooks (Pakray et al., 2026), with Khasi in Category 1
      and Bodo in Category 2. I inspected eight evenly spaced parallel pairs per language.
      The Khasi examples include biblical passages and archaic English. The Bodo examples span
      history, everyday descriptions, and news. Khasi uses Latin script, whereas the Bodo
      targets are predominantly Devanagari.</p>
      </div><div>
      <p>Khasi reference sentences are generally longer than their English counterparts
      (Table 1, Figure 1). I found extreme Khasi length ratios, including examples where a
      verse appeared paired with a heading or a longer passage. In Bodo, 2,451 rows repeated
      an earlier normalized pair, and 35 rows contained spreadsheet escape patterns.
      The larger Khasi corpus may help adaptation, although its domain concentration can
      limit transfer to other material.</p>
      <p>I screened for repeated normalized sides and overlap with official sources before
      forming fixed development and test splits. I held out both directions of each selected
      pair together. The four official test lists supplied 4,000 sources without references.
      I generated official predictions separately for each system. All reported scores use
      the labeled holdouts.</p>
      </div></div>
      {corpus}
      <figure class="length"><img src="lengths.svg"><figcaption><b>Figure 1:</b> Empirical cumulative distributions
      of English and target lengths in whitespace tokens. The horizontal axis ends at 100.</figcaption></figure>
      <h2>3 Models and methodology</h2>
      <div class="columns"><div>
      <p>MADLAD-400 3B is a T5 encoder–decoder trained for multilingual translation
      (Kudugunta et al., 2023). I prepended <code>&lt;2en&gt;</code>, <code>&lt;2kha&gt;</code>,
      or <code>&lt;2brx&gt;</code>. Gemma 3 1B is a causal decoder. I used its chat template with
      a translation-only instruction and masked prompt positions in training.</p>
      </div><div>
      <p>I first tested loading, optimization, adapter reloads and checkpoint recovery, then
      selected microbatches of 32 for MADLAD and 16 for Gemma using A40 measurements.
      Both used an effective batch of 64, AdamW at an initial learning rate of 2 × 10⁻⁴,
      and seed 42. I used BF16 with frozen base weights, double-quantized NF4 linear layers, and rank-16, α=32
      LoRA adapters on attention query and value projections (Dettmers et al., 2023).</p>
      </div></div>
    </section>
    <section>
      <figure class="architecture"><img src="architectures.svg"><figcaption><b>Figure 2:</b> Token flow
      during adaptation. CE denotes cross-entropy. MADLAD applies LoRA in encoder self-attention,
      decoder self-attention and decoder cross-attention. Gemma applies LoRA in causal self-attention.
      The target labels include EOS for MADLAD and the end-of-turn token for Gemma.</figcaption></figure>
      <div class="columns"><div><p>I trained each model for one epoch with a 256-token cap on each
      source and target segment. I selected checkpoints by development loss measured every quarter
      epoch. The final checkpoint had the lowest loss for both models (Figure 3).</p></div>
      <div><p>I decoded both pretrained and adapted checkpoints in BF16 with greedy decoding and a
      cap of 256 new tokens. I scored the same 3,274 examples with SacreBLEU (13a, case-sensitive)
      and chrF++ with word bigrams.</p></div></div>
      <figure><img src="training.svg"><figcaption><b>Figure 3:</b> Logged training loss and quarterly
      development loss. Compare trends within each model because tokenization and objectives differ.</figcaption></figure>
      <h2>4 Results and discussion</h2>
      <figure class="tabular primary"><figcaption><b>Table 2:</b> Full-holdout scores. Δ is the fine-tuned
      score minus the zero-shot score, computed before rounding.</figcaption>{comparison}</figure>
    </section>
    <section class="supplement">
      <div class="columns"><div>
      <p>MADLAD improved on both metrics in all directions and led tuned Gemma in each
      comparison (Table 2). Its largest chrF++ gain was {gain:.2f} into Bodo, where exact
      English copies fell from {madlad['base']['exact_normalized_source_copy']} to
      {madlad['tuned']['exact_normalized_source_copy']}. Parallel supervision may have helped
      adapt an existing translation objective and target-tag interface to these corpora.
      Model size, pretraining, tokenization, and input format vary alongside architecture.</p>
      <p>Both tuned models averaged higher chrF++ on Category 1 Khasi than on Category 2
      Bodo. Khasi had roughly twice as many training pairs, but one language represents each
      category. Differences in script, sentence length, and domain prevent isolation of a
      resource-level effect.</p>
      </div><div>
      <p>Gemma’s gains were smaller and depended on the metric. English→Bodo BLEU and
      Khasi→English chrF++ fell. The autoregressive objective can reward locally plausible
      continuations without ensuring that greedy generation maintains source coverage.
      The observed repetition and omissions are consistent with this account. The
      experiment cannot establish their cause.</p>
      </div></div>
      {additional}
    </section>
    <section>
      {pilot_discussion}
      <h2>6 Error analysis and limitations</h2>
      <div class="columns"><div>
      <p>In English→Bodo, Gemma’s script flags fell from {gemma['base']['unexpected_majority_script']}
      to {gemma['tuned']['unexpected_majority_script']} of 637 outputs. The flag identifies a mismatch
      in majority letter script and does not verify linguistic adequacy. In the same direction,
      repeated-trigram flags rose from {gemma['base']['repeated_trigram']} to
      {gemma['tuned']['repeated_trigram']}. A trigram triggers the flag when it appears at least
      three times.</p>
      </div><div>
      <p>The chrF++ gain into Bodo is compatible with increased character overlap once outputs
      use Devanagari. Repeated material can still require many edits, which helps explain why
      TER worsened sharply{" (Table 3)" if has_supplement else ""}. Neither measure judges whether the generated proposition
      preserves the meaning of the source.</p>
      </div></div>
      <figure class="example"><div class="columns">{''.join(case_blocks)}</div>
      <figcaption><b>Examples 1–2:</b> Labeled holdout sources, references, and system outputs.</figcaption></figure>
      <div class="columns"><div>
      <p>In the first example, MADLAD reproduces the entire English source before tuning.
      The adapted text uses Devanagari and contains recognizable forms of Michael Jordan
      and basketball. The Gemma example shows a different failure: the tuned model recovers
      Tom and 2:30 but omits the action of coming home. In other inspected cases, MADLAD
      rendered a song as a taxi company. Gemma changed whether a Bitcoin threshold would
      be crossed into whether it had already been reached.</p>
      </div><div>
      <p>I inspected examples and used automatic diagnostics. I did not conduct a bilingual
      adequacy study. The holdout comes from the available parallel workbooks and has one
      reference per source. Possible near-duplicates and pretraining overlap were not
      assessed. Results come from a single training seed, and the source-only official
      lists cannot support reference-based claims about out-of-domain translation.</p>
      </div></div>
      <h2>7 Conclusion</h2>
      <p>On these holdouts, one epoch of QLoRA consistently improved MADLAD’s reference-based
      scores. Gemma’s improved script use coexisted with repetition and missing source content.
      Comparing chrF++ with TER and inspecting outputs was necessary to characterize that result.
      The decoding pilots show what was possible on a small sample and leave full-test comparisons open.</p>
      <h2>References</h2>
      <p class="refs">Pakray et al. (2026). <a href="https://aclanthology.org/2026.wmt-main.53/">WMT26 Indic MT findings</a>.
      Kudugunta et al. (2023). <a href="https://arxiv.org/abs/2309.04662">MADLAD-400</a>.
      Gemma Team (2025). Gemma 3 technical report.
      Dettmers et al. (2023). <a href="https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract.html">QLoRA</a>.
      Post (2018). <a href="https://aclanthology.org/W18-6319/">A call for clarity in reporting BLEU scores</a>.
      Popović (2017). chrF++.<br>
      Koehn (2004). <a href="https://aclanthology.org/W04-3250/">Statistical significance tests for machine translation evaluation</a>.
      Snover et al. (2006). <a href="https://aclanthology.org/2006.amta-papers.25/">A study of translation edit rate with targeted human annotation</a>.</p>
    </section>
    </body></html>
    """
    path = output / "report.pdf"
    HTML(string=html, base_url=str(output.resolve())).write_pdf(path)
    with pymupdf.open(path) as pdf:
        if len(pdf) != 4:
            raise ValueError(f"Report has {len(pdf)} pages")
        text = "".join(page.get_text() for page in pdf)
        assert any("DEVANAGARI" in ud.name(char, "") for char in text)
        assert "\ufffd" not in text
        for index, page in enumerate(pdf, 1):
            page.get_pixmap(dpi=180).save(output / f"report-page-{index}.png")


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
                     "bonus_evaluation.py", "bonus_decoding.py"):
            bundle.add(Path(__file__).with_name(name), arcname=f"reproduction/{name}")
        if (root / "bonus").is_dir():
            bundle.add(root / "bonus", arcname="bonus")
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
