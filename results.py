import argparse
from collections import Counter
from html import escape
import json
from pathlib import Path
import unicodedata as ud

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pymupdf
from sacrebleu.metrics import CHRF
from weasyprint import HTML

from prepare import LANGUAGES, normalize, scripts, write_json


DIRECTIONS = {"en-kha": "English → Khasi", "kha-en": "Khasi → English",
              "en-brx": "English → Bodo", "brx-en": "Bodo → English"}
MODELS = {"madlad": "MADLAD-400 3B", "qwen": "Qwen3-1.7B"}
COLORS = ["#32618e", "#ad532c"]


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


def diagnostic(row):
    text = row["translation"]
    normal = normalize(text)
    words = text.casefold().split()
    trigrams = list(zip(words, words[1:], words[2:]))
    expected = "DEVANAGARI" if row["direction"].endswith("-brx") else "LATIN"
    letters = scripts(text)
    return {
        "empty": not normal,
        "exact_normalized_source_copy": bool(normal) and normal == normalize(row["source"]),
        "short_output": len(words) < .5 * len(row["reference"].split()),
        "repeated_trigram": max(Counter(trigrams).values(), default=0) >= 3,
        "unexpected_majority_script": bool(letters) and letters[expected] * 2 < letters.total(),
        "replacement_character": "\ufffd" in text,
    }


def aggregate(root, output):
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.5), layout="constrained", sharey=True)
    scores, diagnostics, examples = {}, {}, {}
    chrf = CHRF(word_order=2)
    for model in MODELS:
        paths = {stage: root / stage / model for stage in ("base", "tuned")}
        scores[model] = {stage: read(path / "metrics.json") for stage, path in paths.items()}
        scores[model]["delta"] = {
            d: {m: scores[model]["tuned"][d][m] - scores[model]["base"][d][m]
                for m in ("bleu", "chrf++")}
            for d in DIRECTIONS
        }
        rows = {stage: read(path / "test.json") for stage, path in paths.items()}
        for key in ("id", "direction", "source", "reference"):
            assert [row[key] for row in rows["base"]] == [row[key] for row in rows["tuned"]]
        diagnostics[model], examples[model] = {}, {}
        for direction in DIRECTIONS:
            aligned = [(base, tuned) for base, tuned in zip(rows["base"], rows["tuned"])
                       if base["direction"] == direction]
            assert aligned
            diagnostics[model][direction] = {
                stage: dict(sum((Counter({k: int(v) for k, v in diagnostic(pair[index]).items()})
                                 for pair in aligned), Counter()))
                for index, stage in enumerate(("base", "tuned"))
            }
            ranked = []
            for base, tuned in aligned:
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
    for axis, (model, title) in zip(axes, MODELS.items()):
        for index, name in enumerate(("bleu", "chrf++")):
            axis.bar([i + (.18 if index else -.18) for i in range(4)],
                     [scores[model]["delta"][d][name] for d in DIRECTIONS],
                     width=.32, color=COLORS[index], label=("BLEU", "chrF++")[index])
        axis.axhline(0, color=".3", linewidth=.7)
        axis.set(title=title, xticks=range(4), xticklabels=["EN–KH", "KH–EN", "EN–BO", "BO–EN"])
    axes[0].set_ylabel("Change in score (points)")
    axes[1].legend(frameon=False)
    finish(fig, output, "improvements")
    write_json(output / "scores.json", scores)
    write_json(output / "diagnostics.json", diagnostics)
    write_json(output / "examples.json", examples)
    return scores, diagnostics, examples


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
    rows = []
    for model, title in MODELS.items():
        for direction, label in DIRECTIONS.items():
            entry = []
            for metric in ("bleu", "chrf++"):
                entry.extend(f"{scores[model][stage][direction][metric]:.2f}"
                             for stage in ("base", "tuned", "delta"))
            rows.append([title, label] + entry)
    data_table = table(("Language", "Category", "Raw pairs", "Train", "Validation", "Test"),
                       [(name, index, stats[code]["usable_rows"], stats[code]["train_pairs"],
                         stats[code]["validation_pairs"], stats[code]["test_pairs"])
                        for index, (code, name) in enumerate(LANGUAGES.items(), 1)])
    score_table = table(("Model", "Direction", "BLEU base", "Tuned", "Δ", "chrF++ base", "Tuned", "Δ"), rows)
    notes = []
    for model, title in MODELS.items():
        ranked = sorted(DIRECTIONS, key=lambda d: scores[model]["delta"][d]["chrf++"])
        largest, smallest = ranked[-1], ranked[0]
        notes.append(f"{title}: the largest chrF++ change was {scores[model]['delta'][largest]['chrf++']:+.2f} "
                     f"for {DIRECTIONS[largest]}. The smallest was "
                     f"{scores[model]['delta'][smallest]['chrf++']:+.2f} for {DIRECTIONS[smallest]}.")
    diagnostic_rows = []
    for model, title in MODELS.items():
        for direction, label in DIRECTIONS.items():
            vals = diagnostics[model][direction]
            diagnostic_rows.append([title, label] + [
                f"{vals['base'].get(key, 0)} → {vals['tuned'].get(key, 0)}"
                for key in ("exact_normalized_source_copy", "short_output",
                            "repeated_trigram", "unexpected_majority_script")
            ])
    diagnostic_table = table(("Model", "Direction", "Source copy", "Short", "Repetition", "Script"), diagnostic_rows)
    case_html = []
    for model, direction in (("madlad", "en-kha"), ("qwen", "en-brx")):
        case = examples[model][direction]["median"]
        fields = "".join(f"<p><b>{label}</b> {escape(visible_text(case[key]))}</p>"
                         for label, key in (("Source", "source"), ("Reference", "reference"),
                                            ("Base", "base"), ("Tuned", "tuned")))
        case_html.append(f"<div class='case'><h3>{MODELS[model]}, {DIRECTIONS[direction]}, "
                         f"{case['id']}</h3>{fields}</div>")
    manual = output / "linguistic-observations.txt"
    observations = escape(manual.read_text().strip()) if manual.exists() else (
        "Examples are ranked using sentence chrF++. Their comparison can expose script and copying "
        "failures, but validating omissions or semantic equivalence requires bilingual review.")
    html = f"""
    <style>
    @page {{ size: A4; margin: 17mm 19mm; @bottom-center {{ content: counter(page); }} }}
    body {{ font: 10.6pt/1.43 'Noto Serif', 'Noto Sans Devanagari', serif; color: #171b20; }}
    h1 {{ font-size: 24pt; margin: 0 0 10pt; line-height: 1.15; }}
    h2 {{ font-size: 14pt; margin: 13pt 0 4pt; }}
    h3 {{ font-size: 10.4pt; margin: 8pt 0 3pt; }}
    p {{ margin: 4pt 0 7pt; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 8.5pt; margin: 8pt 0; }}
    th,td {{ text-align: left; padding: 3pt; }}
    th {{ border-top: 1pt solid; border-bottom: .5pt solid; }}
    tr:last-child td {{ border-bottom: .5pt solid; }}
    figure {{ margin: 10pt 0; }}
    figcaption {{ font-size: 9pt; }}
    img {{ width: 100%; }}
    section {{ break-before: page; }}
    .case {{ break-inside: avoid; }} .case p {{ margin: 2pt 0; font-size: 9pt; overflow-wrap: anywhere; }}
    </style>
    <h1>One-epoch adaptation for English–Khasi and English–Bodo translation</h1>
    <p><b>Abstract.</b> We compare zero-shot and supervised translation with a pretrained
    encoder–decoder and a causal language model. Both receive the same English-centric pairs
    in both directions. {' '.join(notes)}</p>
    <h2>1. Data</h2>
    <p>The WMT26 archive supplies Khasi in Category 1 and Bodo in Category 2.
    We excluded repeated normalized sides and overlap with the official sources before creating
    deterministic validation and test holdouts. The four official lists contain 1,000 IDs each
    and have no supplied references. The internal scores estimate performance on the training
    corpus distribution. Official translations are released separately from this evaluation.</p>
    {data_table}
    <figure><img src="lengths.svg"><figcaption>Figure 1. Empirical whitespace-token length
    distributions. The horizontal axis displays lengths up to 100 tokens.</figcaption></figure>
    <p>Eight evenly spaced pairs per language were inspected. Khasi samples are mostly biblical,
    with archaic English and occasional spacing around punctuation. The sampled Bodo rows include
    history, everyday descriptions, regional culture, headlines and economic text. Khasi had 91
    English-to-Khasi length ratios outside [0.33, 3] and two extreme inspected rows appeared to pair
    verse text with a heading or larger passage. Bodo had 2,451 normalized duplicate pairs and
    at least one sampled English row contained an escaped spreadsheet character.
    Khasi primarily uses Latin letters and Bodo primarily uses Devanagari. The official English
    sources include contemporary news, suggesting domain shift for Khasi.</p>
    <section><h2>2. Method and results</h2>
    <p>MADLAD uses target prefixes &lt;2en&gt;, &lt;2kha&gt; and &lt;2brx&gt;. Qwen uses
    its official chat template with thinking disabled and a translation-only instruction.
    Each model trains one epoch using BF16 and NF4 QLoRA (rank 16, alpha 32, AdamW, initial
    learning rate 2 × 10⁻⁴, seed 42). Causal loss covers response tokens and EOS. We select
    adapters by validation loss and decode both versions greedily in BF16 with a 256-token limit.
    BLEU uses 13a tokenization, and chrF++ includes word bigrams.</p>
    {score_table}
    <figure><img src="improvements.svg"><figcaption>Figure 2. Changes relative to zero-shot
    performance on identical internal test examples.</figcaption></figure>
    <figure><img src="training.svg"><figcaption>Figure 3. Training logs and held-out
    validation loss. Training and validation series are sampled at different intervals.</figcaption></figure>
    <p>Khasi provides twice as many training pairs, but distinct domains, scripts, pretraining
    coverage, model sizes and tokenizers also affect results. This two-checkpoint comparison
    cannot isolate the effects of architecture or resource level. Results use one random seed.</p></section>
    <section><h2>3. Translation errors</h2>
    <p>The table counts automatic flags. A short output has fewer than half the reference's
    whitespace tokens. Repetition means the same trigram occurs at least three times. The script
    flag detects when over half the letters fall outside the expected script. Source copying
    requires full normalized equality. These checks can miss partial copying and are sensitive
    to differences in word segmentation.</p>
    {diagnostic_table}
    {''.join(case_html)}
    <p>{observations}</p>
    <h2>References</h2>
    <p>WMT 2026, Low-Resource Indic Language Translation shared task. Kudugunta et al.,
    MADLAD-400 (2023). Qwen Team, Qwen3 Technical Report (2025). Dettmers et al., QLoRA (2023).
    Post, A Call for Clarity in Reporting BLEU Scores (2018). Popović, chrF++ (2017).</p></section>
    """
    path = output / "report.pdf"
    HTML(string=html, base_url=str(output.resolve())).write_pdf(path)
    with pymupdf.open(path) as pdf:
        if not 2 <= len(pdf) <= 4:
            raise ValueError(f"Report has {len(pdf)} pages")
        text = "".join(page.get_text() for page in pdf)
        assert any("DEVANAGARI" in ud.name(char, "") for char in text)
        assert "\ufffd" not in text
        for index, page in enumerate(pdf, 1):
            page.get_pixmap(dpi=120).save(output / f"report-page-{index}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs"))
    parser.add_argument("--eda-only", action="store_true")
    args = parser.parse_args()
    output = args.root / "results"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "path", "pdf.fonttype": 42})
    statistics = eda(args.root, output)
    if not args.eda_only:
        curves(args.root, output)
        report(output, statistics, *aggregate(args.root, output))
