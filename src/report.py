"""Build the four-page report from saved metrics, figures and translations."""

import argparse
from html import escape
import json
import os
from pathlib import Path
import shutil
from string import Template
import subprocess
import tarfile
from tempfile import TemporaryDirectory

import pymupdf


DIRECTIONS = {
    "en-kha": "English → Khasi", "kha-en": "Khasi → English",
    "en-brx": "English → Bodo", "brx-en": "Bodo → English",
}
MODELS = {"madlad": "MADLAD-400 3B", "gemma": "Gemma 3 1B"}
REVIEW_CASES = {("madlad", "en-brx"): "brx:13772", ("gemma", "kha-en"): "kha:19957"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def archived_cases(path):
    members = {f"{stage}/{model}/test.json": (model, stage)
               for model, _ in REVIEW_CASES for stage in ("base", "tuned")}
    rows = {}
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            if member.name in members:
                with archive.extractfile(member) as handle:
                    rows[members[member.name]] = json.load(handle)
                if len(rows) == len(members):
                    break
    if len(rows) != len(members):
        raise ValueError("The archive must contain baseline and tuned holdout translations")

    cases = {model: {} for model in MODELS}
    for (model, direction), identifier in REVIEW_CASES.items():
        base, tuned = [next(row for row in rows[model, stage]
                            if row["id"] == identifier and row["direction"] == direction)
                       for stage in ("base", "tuned")]
        fields = ("id", "source", "reference", "direction")
        if any(base[field] != tuned[field] for field in fields):
            raise ValueError(f"Unaligned example {identifier} ({direction})")
        cases[model][direction] = {"reviewed": {
            "source": base["source"], "reference": base["reference"],
            "base": base["translation"], "tuned": tuned["translation"],
        }}
    return cases


def cells(values, first="td"):
    head, *tail = map(lambda value: escape(str(value)), values)
    return f"<tr><{first}>{head}</{first}>" + "".join(f"<td>{v}</td>" for v in tail) + "</tr>"


def case_html(case, source_lang, target_lang):
    return "".join(
        f'<dt>{label}</dt><dd lang="{lang}">{escape(case[key])}</dd>'
        for label, key, lang in (("Source", "source", source_lang),
                                 ("Reference", "reference", target_lang),
                                 ("Pretrained", "base", target_lang),
                                 ("Fine-tuned", "tuned", target_lang))
    )


def browser_path(browser):
    candidates = [browser] if browser else [
        shutil.which("chromium"), shutil.which("google-chrome"), shutil.which("chrome"),
        shutil.which("msedge"), "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    for directory in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
        if not browser and (root := os.environ.get(directory)):
            candidates += [Path(root) / relative for relative in (
                "Google/Chrome/Application/chrome.exe", "Microsoft/Edge/Application/msedge.exe",
            )]
    for candidate in candidates:
        if candidate and (Path(candidate).is_file() or shutil.which(candidate)):
            return str(candidate)
    raise FileNotFoundError("Install Chrome or Chromium, or pass the browser executable with --browser")


def semantic_sections(path, khasi):
    values = dict.fromkeys(("semantic_method", "semantic_table", "semantic_discussion", "semantic_reference",
                            "semantic_class"), "")
    if not path.is_file():
        return values
    report = read(path)
    rows = {(row["direction"], row["model"]): row for row in report["rows"]}
    body = []
    for direction in ("kha-en", "brx-en"):
        line = f'<tr><th scope="row">{DIRECTIONS[direction]}</th>'
        for model in MODELS:
            row = rows[direction, model]
            line += (f"<td>{row['f1_base']:.3f} → {row['f1_tuned']:.3f}<br>"
                     f"<small>{row['f1_delta']:+.3f} [{row['ci95_low']:+.3f}, {row['ci95_high']:+.3f}]</small></td>")
        body.append(line + "</tr>")
    gemma = rows["kha-en", "gemma"]
    if khasi["chrf++_delta"] < 0 and khasi["ter_tuned"] > khasi["ter_base"] and gemma["ci95_low"] > 0:
        interpretation = ("For Gemma on Khasi→English, BERTScore rose despite lower chrF++ and higher TER. "
                          "As a source-blind measure, it cannot establish improved adequacy.")
    else:
        interpretation = ("The English-target BERTScore intervals quantify holdout sampling uncertainty. "
                          "Contextual similarity alone cannot establish translation adequacy.")
    values.update(
        semantic_method=("<p>English-target BERTScore [9] averages sentence F1 using uncased DistilBERT "
                         f"(layer {report['layers']}, raw F1 without IDF or baseline rescaling). "
                         "Khasi and Bodo targets were outside this encoder’s documented coverage.</p>"),
        semantic_table=('<table class="semantic-scores"><caption><strong>Table 4.</strong> English-target mean '
                        'BERTScore F1, pretrained → fine-tuned. Changes show paired 95% intervals '
                        f'({report["resamples"]:,} resamples).</caption><thead><tr><th scope="col">Direction</th>'
                        '<th scope="col">MADLAD-400 3B</th><th scope="col">Gemma 3 1B</th></tr></thead><tbody>'
                        + "".join(body) + "</tbody></table>"),
        semantic_discussion=f"<p>{interpretation}</p>",
        semantic_reference=('<li>[9] Zhang et al. (2020). <a href="https://arxiv.org/abs/1904.09675">'
                            'BERTScore: Evaluating Text Generation with BERT</a>.</li>'),
        semantic_class=" semantic",
    )
    return values


def build_report(results, examples, output, browser=None):
    def artifact(group, filename):
        grouped = results / group / filename
        return grouped if grouped.is_file() else results / filename

    stats = read(artifact("diagnostics", "eda.json"))
    diagnostics = read(artifact("diagnostics", "diagnostics.json"))
    scores = read(artifact("metrics", "scores.json"))
    extra = read(artifact("metrics", "additional-evaluation.json"))
    pilots = {row["label"]: row for row in read(artifact("metrics", "additional-decoding.json"))}
    extra_rows = {(row["model"], row["direction"]): row for row in extra["rows"]}
    bodo = extra_rows["gemma", "en-brx"]
    khasi = extra_rows["gemma", "kha-en"]
    bodo_flags = diagnostics["gemma"]["en-brx"]

    def pair(model, direction, metric):
        score = scores[model]
        return " → ".join(f"{score[stage][direction][metric]:.2f}" for stage in ("base", "tuned"))

    def mean_pilot(run):
        by_direction = pilots[run]["scores"]
        return sum(value["chrf++"] for value in by_direction.values()) / len(by_direction)

    def pilot_range(method):
        means = [mean_pilot(f"{method}-{shots}") for shots in (1, 3, 5)]
        return f"{min(means):.2f}–{max(means):.2f}"

    def median_p95(code, side):
        lengths = stats[code][f"{side}_whitespace_tokens"]
        return f"{lengths['median']:.0f} / {lengths['p95']}"

    values = {
        "corpus_rows": "".join(cells((
            label, f"{stats[code]['usable_rows']:,}", f"{stats[code]['train_pairs']:,}",
            f"{stats[code]['validation_pairs']:,}", f"{stats[code]['test_pairs']:,}",
            median_p95(code, "english"), median_p95(code, "target"),
        ), "th") for code, label in (("kha", "Khasi (1)"), ("brx", "Bodo (2)"))),
        "score_rows": "".join(cells((
            label, f"{scores['madlad']['base'][direction]['count']:,}",
            *(pair(model, direction, metric) for model in MODELS for metric in ("bleu", "chrf++")),
        ), "th") for direction, label in DIRECTIONS.items()),
        "train_count": f"{2 * sum(stats[code]['train_pairs'] for code in ('kha', 'brx')):,}",
        "official_count": f"{sum(stats['official'].values()):,}",
        "duplicates": f"{stats['brx']['normalized_pair_duplicates']:,}",
        "escapes": stats["brx"]["excel_escape_rows"],
        "archaic": f"{stats['kha']['archaic_english_marker_rows']:,}",
        "shift": f"{scores['madlad']['delta']['en-brx']['chrf++']:+.2f}",
        "resamples": f"{extra['resamples']:,}",
        "bodo_ci": f"{bodo['ci95_low']:+.2f} to {bodo['ci95_high']:+.2f}",
        "khasi_ci": f"{khasi['ci95_low']:+.2f} to {khasi['ci95_high']:+.2f}",
        "bodo_chrf": pair("gemma", "en-brx", "chrf++"),
        "bodo_ter": f"{bodo['ter_base']:.2f} → {bodo['ter_tuned']:.2f}",
        "bodo_n": bodo["count"],
        "wrong_script": " → ".join(str(bodo_flags[stage]["unexpected_majority_script"])
                                   for stage in ("base", "tuned")),
        "repetition": " → ".join(str(bodo_flags[stage]["repeated_trigram"])
                                 for stage in ("base", "tuned")),
        "zero_shot": f"{mean_pilot('none-0'):.2f}",
        "lexical": pilot_range("lexical"),
        "random": pilot_range("random"),
        "beam_1": f"{mean_pilot('beam-1'):.2f}",
        "beam_20": f"{mean_pilot('beam-20'):.2f}",
        "beam_seconds_1": f"{pilots['beam-1']['seconds']:.1f}",
        "beam_seconds_20": f"{pilots['beam-20']['seconds']:.1f}",
        "pilot_n": pilots["beam-1"]["scores"]["en-brx"]["count"],
        "pilot_cap": pilots["beam-1"]["max_new_tokens"],
        "madlad_example": case_html(examples["madlad"]["en-brx"]["reviewed"], "en", "brx"),
        "gemma_example": case_html(examples["gemma"]["kha-en"]["reviewed"], "kha", "en"),
    }
    values.update(semantic_sections(artifact("metrics", "semantic-evaluation.json"), khasi))
    template = Template(Path(__file__).with_name("report.html").read_text(encoding="utf-8"))
    output = output.resolve()
    with TemporaryDirectory(prefix="mt-report-") as tmp:
        work = Path(tmp)
        for name in ("lengths", "architectures", "training", "improvements"):
            shutil.copyfile(artifact("figures", f"{name}.svg"), work / f"{name}.svg")
        html = work / "report.html"
        html.write_text(template.substitute(values), encoding="utf-8")
        rendered = work / "report.pdf"
        subprocess.run([
            browser_path(browser), "--headless", "--disable-gpu",
            f"--user-data-dir={work / 'browser'}", "--no-pdf-header-footer",
            f"--print-to-pdf={rendered}", html.as_uri(),
        ], check=True, capture_output=True, timeout=90)
        with pymupdf.open(rendered) as pdf:
            if len(pdf) != 4 or any(
                abs(page.rect.width - 595.28) > 1 or abs(page.rect.height - 841.89) > 1
                for page in pdf
            ):
                raise ValueError(f"Expected four A4 pages, received {len(pdf)} pages")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(rendered, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--archive", type=Path, help="Published results archive")
    inputs.add_argument("--examples", type=Path, help="Saved examples.json from src.results")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("report.pdf"))
    parser.add_argument("--browser", help="Path to Chrome, Chromium or Edge")
    args = parser.parse_args()
    build_report(args.results, archived_cases(args.archive) if args.archive else read(args.examples),
                 args.output, args.browser)
