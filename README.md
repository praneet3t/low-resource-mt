# Machine Translation with Pretrained Models

A comparison of pretrained and one-epoch QLoRA-adapted MADLAD-400 3B and Gemma 3 1B on English–Khasi and English–Bodo. [Four-page report](report.pdf)

| Direction | Pairs | MADLAD BLEU | MADLAD chrF++ | MADLAD BERTScore F1 | Gemma BLEU | Gemma chrF++ | Gemma BERTScore F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| English → Khasi | 1,000 | 3.47 → 10.88 | 22.43 → 32.15 | — | 0.05 → 0.36 | 5.09 → 9.17 | — |
| Khasi → English | 1,000 | 7.65 → 18.77 | 28.02 → 38.80 | 0.799 → 0.858 | 0.70 → 0.83 | 17.17 → 14.71 | 0.707 → 0.734 |
| English → Bodo | 637 | 0.09 → 2.01 | 1.60 → 18.21 | — | 0.18 → 0.04 | 1.57 → 3.83 | — |
| Bodo → English | 637 | 3.72 → 8.72 | 23.75 → 29.78 | 0.735 → 0.797 | 1.16 → 2.84 | 15.15 → 18.71 | 0.698 → 0.734 |

Scores are pretrained → fine-tuned on the fixed labeled holdouts. Official inputs have no references and are translated separately. BERTScore uses English DistilBERT layer 5, raw sentence F1, and 2,000 paired bootstrap resamples. Khasi and Bodo targets are outside the encoder's documented coverage. [Full scores](results/metrics/scores.csv) · [TER and intervals](results/metrics/additional-evaluation.csv) · [BERTScore configuration and revision](results/metrics/semantic-evaluation.json) · [Decoding pilots](results/metrics/additional-decoding.csv)

## Artifacts

- [Source](src/) · [Metrics](results/metrics/) · [Figures](results/figures/) · [Training histories](results/training/) · [Diagnostics](results/diagnostics/)
- [Translations, adapters and training checkpoints](https://github.com/praneet3t/low-resource-mt/releases/download/study-v1/low-resource-mt-results.tar.gz) · [Exploratory pilot translations](https://github.com/praneet3t/low-resource-mt/releases/download/study-v1/decoding-pilot-translations.tar.gz)
- Authorized Hugging Face mirrors for [MADLAD](https://huggingface.co/praneet3/low-resource-mt-madlad/tree/main) and [Gemma](https://huggingface.co/praneet3/low-resource-mt-gemma/tree/main).

## Reproduce

Use Python 3.12 from the repository root, preferably in a virtual environment. The CPU commands work on Linux and Windows with GitHub CLI authenticated, or accept `--archive` for a downloaded archive. PDF rendering requires Chrome, Chromium or Edge and Noto Serif plus a Devanagari font. The report command writes `runs/report.pdf`.

| Workflow | Command |
|---|---|
| CPU environment | `python -m src.reproduce setup` |
| Evaluate saved predictions | `python -m src.reproduce cpu-eval` |
| Regenerate the PDF | `python -m src.reproduce report` |
| CUDA environment | `python -m src.reproduce setup --with-gpu` |
| Full study, including training | `python -m src.reproduce gpu --dataset /path/to/data.zip` |

Saved-prediction evaluation recomputes BLEU, chrF++, TER and paired intervals. It reuses BERTScore only when the archive hash matches. For a separate semantic environment, use `python -m src.reproduce setup --with-semantic`. Use `python -m src.semantic_evaluation --archive runs/low-resource-mt-results.tar.gz` for an explicit CPU BERTScore rerun. Its dependency versions are recorded in the metrics JSON.

The GPU workflow requires the original task ZIP and authenticated Gemma access, or local snapshots supplied with `--models`. It reuses completed stages and resumes from a complete checkpoint. Use `gpu --help` for dataset, checkpoint, batch, accumulation and output controls. Use `--dry-run` to inspect a workflow without starting it. [Task data](https://www2.statmt.org/wmt26/indic-mt-task.html) · [Gemma terms](https://huggingface.co/google/gemma-3-1b-it)
