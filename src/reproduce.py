"""Run one reproduction workflow without changing the saved study results."""

import argparse
from hashlib import file_digest
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


ROOT = Path(__file__).resolve().parents[1]
RELEASE = "study-v1"
REPOSITORY = "praneet3t/low-resource-mt"
MODELS = ("madlad", "gemma")
CPU = ("numpy==2.2.6", "sacrebleu==2.6.0", "openpyxl==3.1.5", "pymupdf==1.27.1")
ARCHIVE = "low-resource-mt-results.tar.gz"
PILOT_ARCHIVE = "decoding-pilot-translations.tar.gz"
TRANSLATIONS = ROOT / "results/translations"


def run(module, *args, env=None):
    subprocess.run([sys.executable, "-m", module, *map(str, args)], cwd=ROOT, env=env, check=True)


def require(packages):
    missing = []
    for package in packages:
        name, _, expected = package.partition("==")
        try:
            installed = version(name)
            if expected and installed != expected:
                missing.append(package)
        except PackageNotFoundError:
            missing.append(package)
    if missing:
        raise ValueError("Install compatible dependencies with python -m src.reproduce setup "
                         f"(missing or different: {', '.join(missing)}).")


def setup(args):
    if args.with_gpu:
        run("pip", "install", "torch==2.8.0", "--index-url", args.torch_index)
        run("pip", "install", "-r", ROOT / "requirements.txt")
    else:
        run("pip", "install", *CPU)
        if args.with_semantic:
            run("pip", "install", "torch==2.8.0", "--index-url", "https://download.pytorch.org/whl/cpu")
            run("pip", "install", "bert-score==0.3.13", "transformers==4.51.3",
                "tokenizers==0.21.4", "huggingface-hub==0.36.2")


def release_archive(path, directory, name):
    if path is not None:
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    path = directory / name
    if not path.is_file():
        if not shutil.which("gh"):
            raise ValueError("Install GitHub CLI or supply --archive with a downloaded archive.")
        directory.mkdir(parents=True, exist_ok=True)
        subprocess.run(["gh", "release", "download", RELEASE, "--repo", REPOSITORY,
                        "--pattern", name, "--dir", str(directory)], check=True)
    return path


def json_members(path, names):
    found = {}
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            name = member.name.removeprefix("./")
            if name in names:
                if name in found or not member.isfile():
                    raise ValueError(f"Invalid or duplicate member {name}")
                with archive.extractfile(member) as stream:
                    found[name] = json.load(stream)
    missing = set(names) - found.keys()
    if missing:
        raise ValueError(f"Missing archive members: {', '.join(sorted(missing))}")
    return found


def pilot_metrics(path):
    from .metrics import metrics

    names = [f"bonus/madlad/beam-{beam}" for beam in (1, 5, 10, 20)]
    names += ["bonus/gemma/none-0", *(f"bonus/gemma/{method}-{shots}" for shots in (1, 3, 5)
                                    for method in ("random", "lexical"))]
    records = json_members(path, {f"{name}{suffix}.json" for name in names for suffix in ("", ".translations")})
    return [records[f"{name}.json"] | {"scores": metrics(records[f"{name}.translations.json"])} for name in names]


def with_references(path, references, *, subset=False):
    rows = json.loads(path.read_text(encoding="utf-8"))
    keys = [(row["direction"], row["id"]) for row in rows]
    if not keys or len(set(keys)) != len(keys) or not set(keys) <= references.keys():
        raise ValueError(f"Missing, duplicate or unknown prediction identifiers in {path}")
    if not subset and len(keys) != len(references):
        raise ValueError(f"Incomplete holdout predictions in {path}")
    return [row | references[key] for key, row in zip(keys, rows, strict=True)]


def cpu_evaluate(args):
    require(("sacrebleu==2.6.0", "numpy", "openpyxl"))
    from .holdout_analysis import evaluate_predictions
    from .metrics import metrics
    from .prepare import read_jsonl, write_json

    archive, archived = None, None
    prepared = (args.data or args.runs / "data") / "test.jsonl"
    if not args.archive and (args.data or prepared.is_file()):
        reference_rows = read_jsonl(prepared)
    else:
        archive = release_archive(args.archive, args.runs, ARCHIVE)
        members = {f"{stage}/{model}/test.json": (stage, model)
                   for stage in ("base", "tuned") for model in MODELS}
        archived = {members[name]: data for name, data in json_members(archive, members).items()}
        reference_rows = archived["base", "madlad"]
    references = {(r["direction"], r["id"]): {field: r[field] for field in ("source", "reference")}
                  for r in reference_rows}
    if len(references) != len(reference_rows):
        raise ValueError("Duplicate identifiers in the labeled holdout")
    if args.archive:
        predictions = archived
    else:
        predictions = {(stage, model): with_references(TRANSLATIONS / model / condition / "holdout.json", references)
                       for stage, condition in (("base", "pretrained"), ("tuned", "fine-tuned")) for model in MODELS}
    scores = {model: {stage: metrics(predictions[stage, model]) for stage in ("base", "tuned")}
              for model in MODELS}
    for score in scores.values():
        score["delta"] = {d: {name: score["tuned"][d][name] - score["base"][d][name]
                              for name in ("bleu", "chrf++")} for d in score["base"]}
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "scores.json", scores)
    write_json(args.output / "additional-evaluation.json",
               evaluate_predictions(predictions, 2000, 42, args.workers))
    if args.pilot_archive:
        pilots = pilot_metrics(args.pilot_archive)
    elif args.archive:
        pilots = pilot_metrics(release_archive(None, args.runs, PILOT_ARCHIVE))
    else:
        recorded = json.loads((ROOT / "results/metrics/additional-decoding.json").read_text(encoding="utf-8"))
        pilots = []
        for row in recorded:
            folder = "beam-search" if row["model"] == "madlad" else "few-shot"
            path = TRANSLATIONS / row["model"] / folder / f"{row['label']}.json"
            pilots.append(row | {"scores": metrics(with_references(path, references, subset=True))})
    write_json(args.output / "additional-decoding.json", pilots)
    semantic = ROOT / "results/metrics/semantic-evaluation.json"
    if archive and predictions == archived and semantic.is_file():
        with archive.open("rb") as stream:
            digest = file_digest(stream, "sha256").hexdigest()
        if json.loads(semantic.read_text(encoding="utf-8"))["archive_sha256"] == digest:
            shutil.copyfile(semantic, args.output / semantic.name)
            print("Reused BERTScore for the matching archive.")
    print(f"Metrics: {args.output}")


def report(args):
    require(("pymupdf",))
    archive = release_archive(args.archive, args.runs, ARCHIVE)
    options = ["--browser", args.browser] if args.browser else []
    run("src.report", "--archive", archive, "--results", args.results, "--output", args.output, *options)


def gpu(args):
    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)
    require(("torch", "transformers==4.57.6", "bitsandbytes==0.49.2", "peft==0.18.1",
             "accelerate==1.12.0", "openpyxl", "sacrebleu==2.6.0", "matplotlib", "pymupdf"))
    import torch
    from huggingface_hub import get_token

    if not torch.cuda.is_available():
        raise ValueError("The full experiment requires CUDA.")
    if not args.models and not get_token():
        raise ValueError("Authenticate to Hugging Face and accept the Gemma model terms, or supply --models.")
    from .report import browser_path
    browser = browser_path(args.browser)
    dataset = args.runs / "data"
    with args.dataset.open("rb") as stream:
        dataset_hash = file_digest(stream, "sha256").hexdigest()
    settings = {"dataset_sha256": dataset_hash, "models": str(args.models) if args.models else None,
                **{name: getattr(args, name) for name in ("eval_batch", "madlad_batch", "gemma_batch",
                   "madlad_accumulation", "gemma_accumulation", "madlad_pilot_batch", "gemma_pilot_batch",
                   "max_tokens", "gradient_checkpointing")}}
    signature = args.runs / "experiment.json"
    if signature.is_file() and json.loads(signature.read_text(encoding="utf-8")) != settings:
        raise ValueError("Run settings changed. Use a separate --runs directory.")
    args.runs.mkdir(parents=True, exist_ok=True)
    signature.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    if not all((dataset / name).is_file() for name in
               ("train.jsonl", "validation.jsonl", "test.jsonl", "official.jsonl", "eda.json", "inspection.json")):
        run("src.prepare", args.dataset, "--output", dataset)
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    shared = ["--data", dataset] + (["--models", args.models] if args.models else [])
    for stage in ("base", "train", "tuned", "decoding-pilot"):
        for model in MODELS:
            dest = args.runs / stage / model
            trained = args.runs / "train" / model
            if stage == "train":
                if (trained / "training.json").is_file() and (trained / "adapter/adapter_model.safetensors").is_file():
                    continue
                resume = ["--resume", args.resume] if args.resume and args.resume_model == model else []
                if not resume and any(trained.glob("checkpoint-*")):
                    resume = ["--resume"]
                checkpoint = ["--gradient-checkpointing"] if args.gradient_checkpointing else []
                run("src.study", "train", model, *shared, "--output", dest,
                    "--batch", getattr(args, f"{model}_batch"),
                    "--accumulation", getattr(args, f"{model}_accumulation"), *checkpoint, *resume, env=env)
            elif stage == "decoding-pilot":
                labels = ("beam-1", "beam-5", "beam-10", "beam-20") if model == "madlad" else (
                    "none-0", *(f"{s}-{n}" for s in ("random", "lexical") for n in (1, 3, 5)))
                if all((dest / f"{label}{suffix}.json").is_file() for label in labels for suffix in ("", ".translations")):
                    continue
                run("src.decoding_pilot", model, *shared, "--output", dest,
                    "--batch", getattr(args, f"{model}_pilot_batch"), env=env)
            else:
                if all((dest / name).is_file() for name in ("test.json", "official.json", "metrics.json")):
                    continue
                adapter = ["--adapter", trained / "adapter"] if stage == "tuned" else []
                run("src.study", "evaluate", model, *shared, "--output", dest, "--batch", args.eval_batch,
                    "--max-tokens", args.max_tokens, *adapter, env=env)
    run("src.holdout_analysis", "--root", args.runs, "--output", args.runs / "results")
    run("src.results", "--root", args.runs, "--report", "--browser", browser)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup_args = commands.add_parser("setup", help="Install CPU dependencies unless --with-gpu is set.")
    profiles = setup_args.add_mutually_exclusive_group()
    profiles.add_argument("--with-gpu", action="store_true")
    profiles.add_argument("--with-semantic", action="store_true")
    setup_args.add_argument("--torch-index", default="https://download.pytorch.org/whl/cu129")
    for name in ("cpu-eval", "report", "gpu"):
        p = commands.add_parser(name)
        p.add_argument("--runs", type=lambda value: Path(value).resolve(), default=ROOT / "runs")
        p.add_argument("--dry-run", action="store_true", help="Show the workflow without running it.")
        if name in ("cpu-eval", "report"):
            p.add_argument("--archive", type=lambda value: Path(value).resolve())
            p.add_argument("--output", type=lambda value: Path(value).resolve())
        if name in ("report", "gpu"):
            p.add_argument("--browser")
        if name == "report":
            p.add_argument("--results", type=lambda value: Path(value).resolve(), default=ROOT / "results")
        if name == "cpu-eval":
            p.add_argument("--workers", type=int, default=2)
            p.add_argument("--pilot-archive", type=lambda value: Path(value).resolve())
            p.add_argument("--data", type=lambda value: Path(value).resolve(),
                           help="Use the prepared test.jsonl for references without downloading the archive.")
        if name == "gpu":
            p.add_argument("--dataset", type=lambda value: Path(value).resolve(), required=True)
            p.add_argument("--models", type=lambda value: Path(value).resolve())
            p.add_argument("--resume", type=lambda value: Path(value).resolve())
            p.add_argument("--resume-model", choices=MODELS)
            for option, default in (("eval-batch", 64), ("madlad-batch", 32), ("gemma-batch", 16),
                                    ("madlad-accumulation", 2), ("gemma-accumulation", 4),
                                    ("madlad-pilot-batch", 4), ("gemma-pilot-batch", 16), ("max-tokens", 256)):
                p.add_argument(f"--{option}", type=int, default=default)
            p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.command == "setup":
        setup(args)
        return
    stages = {"cpu-eval": "saved holdout and pilot predictions → BLEU, chrF++, TER, paired intervals and matching BERTScore",
              "report": "saved metrics, figures and examples → PDF",
              "gpu": "prepare → pretrained translation → fine-tune → adapted translation → decoding pilots → analysis → PDF"}
    if args.dry_run:
        print(stages[args.command])
        return
    if args.command == "gpu":
        if bool(args.resume) != bool(args.resume_model):
            parser.error("--resume and --resume-model must be set together")
        if any(val < 1 for name, val in vars(args).items()
               if name.endswith(("_batch", "_accumulation")) or name == "max_tokens"):
            parser.error("Batch, accumulation and token limits must be positive")
        gpu(args)
    elif args.command == "cpu-eval":
        if args.workers < 1:
            parser.error("--workers must be positive")
        if args.data and args.archive:
            parser.error("Choose --data or --archive.")
        args.output = args.output or args.runs / "cpu-metrics"
        cpu_evaluate(args)
    else:
        args.output = args.output or args.runs / "report.pdf"
        report(args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
