# Machine Translation with Pretrained Models

This study compares pretrained and one-epoch QLoRA-adapted MADLAD-400 3B and Gemma 3 1B for bidirectional English–Khasi and English–Bodo translation. Both models used the same supervised training split and labeled holdouts. Predictions for the official source-only sets are included in the release.

[Read the four-page report](report.pdf).

| Direction | Holdout pairs | MADLAD BLEU | MADLAD chrF++ | Gemma BLEU | Gemma chrF++ |
|---|---:|---:|---:|---:|---:|
| English → Khasi | 1,000 | 3.47 → 10.88 | 22.43 → 32.15 | 0.05 → 0.36 | 5.09 → 9.17 |
| Khasi → English | 1,000 | 7.65 → 18.77 | 28.02 → 38.80 | 0.70 → 0.83 | 17.17 → 14.71 |
| English → Bodo | 637 | 0.09 → 2.01 | 1.60 → 18.21 | 0.18 → 0.04 | 1.57 → 3.83 |
| Bodo → English | 637 | 3.72 → 8.72 | 23.75 → 29.78 | 1.16 → 2.84 | 15.15 → 18.71 |

Each cell shows pretrained → fine-tuned scores. See [full scores](results/metrics/scores.csv), [TER and confidence intervals](results/metrics/additional-evaluation.csv), [decoding pilot scores](results/metrics/additional-decoding.csv), and [English-target BERTScore](results/metrics/semantic-evaluation.csv).

## English-target semantic evaluation

BERTScore is mean sentence F1 on the full Khasi→English and Bodo→English labeled holdouts. It uses uncased English DistilBERT at revision `12040accade4e8a0f71eabdb258fecc2e7e948be`, layer 5, a fast tokenizer (512 tokens), and raw cosine matching without IDF weighting or baseline rescaling. The 95% intervals use 2,000 paired sentence resamples with seed 42. The English encoder has no validated coverage for Khasi or Bodo targets. Neither language appears in the checked WMT22 COMET model's coverage list. [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased) · [COMET coverage](https://huggingface.co/Unbabel/wmt22-comet-da) · [Scoring configuration](results/metrics/semantic-evaluation.json)

## Repository and artifacts

- [src/](src) contains data preparation, model loading, fine-tuning, evaluation, decoding and report generation.
- [results/](results) contains [metrics](results/metrics), [figures](results/figures), [training histories](results/training) and [diagnostics](results/diagnostics).
- The [complete results archive](https://github.com/praneet3t/low-resource-mt/releases/download/study-v1/low-resource-mt-results.tar.gz) contains labeled and official translation JSON under `base/` and `tuned/`, plus adapters and resumable checkpoints under `train/`. The [pilot translation archive](https://github.com/praneet3t/low-resource-mt/releases/download/study-v1/decoding-pilot-translations.tar.gz) contains the few-shot and beam-search outputs.
- Hugging Face adapter and checkpoint mirrors are available to authorized users for [MADLAD](https://huggingface.co/praneet3/low-resource-mt-madlad/tree/main) and [Gemma](https://huggingface.co/praneet3/low-resource-mt-gemma/tree/main). Repository and release downloads require access to this GitHub repository.

## Reproduce

Run all commands from the repository root with Python 3.12. To rebuild the report on CPU, install Chrome, Chromium or Edge with Noto Serif and Noto Sans Devanagari (Nirmala UI also works for Devanagari on Windows). Authenticate GitHub CLI with an account that has repository access.

```sh
python -m pip install pymupdf==1.27.1
gh release download study-v1 --pattern low-resource-mt-results.tar.gz --dir runs
python -m src.report --archive runs/low-resource-mt-results.tar.gz
```

To regenerate the semantic scores from the same archive, use a separate CPU environment:

```sh
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install bert-score==0.3.13 transformers==4.51.3 tokenizers==0.21.4 numpy==2.2.6 huggingface-hub==0.36.2 openpyxl==3.1.5
python -m src.semantic_evaluation --archive runs/low-resource-mt-results.tar.gz
```

To reproduce the model runs, use a CUDA GPU, obtain the original [WMT26 task data](https://www2.statmt.org/wmt26/indic-mt-task.html), and accept the [Gemma model terms](https://huggingface.co/google/gemma-3-1b-it). The commands below use a POSIX shell.

```sh
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
python -m pip install -r requirements.txt
hf auth login
python -m src.prepare /path/to/Low_Resource_MT_Data-20261123T221212Z-1-001.zip
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for model in madlad gemma; do
  python -m src.study evaluate "$model" --batch 64 --output "runs/base/$model"
done
python -m src.study train madlad --batch 32 --accumulation 2 --gradient-checkpointing --output runs/train/madlad
python -m src.study train gemma --batch 16 --accumulation 4 --gradient-checkpointing --output runs/train/gemma
for model in madlad gemma; do
  python -m src.study evaluate "$model" --batch 64 --adapter "runs/train/$model/adapter" --output "runs/tuned/$model"
done
python -m src.holdout_analysis --seed 42 --resamples 2000
python -m src.decoding_pilot gemma --batch 16 --output runs/decoding-pilot/gemma
python -m src.decoding_pilot madlad --batch 4 --output runs/decoding-pilot/madlad
python -m src.results --report
```

The last command writes figures, metrics, diagnostics and the rebuilt PDF to `runs/results/`. The holdout analysis also accepts `--root /path/to/extracted-results` to score the archived translations on CPU.
