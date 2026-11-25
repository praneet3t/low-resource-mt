# Machine Translation with Pretrained Models

One epoch of NF4 QLoRA on the same 70,608 directed WMT26 training examples per model, with seed 42. Evaluation uses fixed labeled holdouts. The official sets contain sources only.

| Direction | MADLAD BLEU | MADLAD chrF++ | Gemma BLEU | Gemma chrF++ |
|---|---:|---:|---:|---:|
| English → Khasi | 3.47 → 10.88 | 22.43 → 32.15 | 0.05 → 0.36 | 5.09 → 9.17 |
| Khasi → English | 7.65 → 18.77 | 28.02 → 38.80 | 0.70 → 0.83 | 17.17 → 14.71 |
| English → Bodo | 0.09 → 2.01 | 1.60 → 18.21 | 0.18 → 0.04 | 1.57 → 3.83 |
| Bodo → English | 3.72 → 8.72 | 23.75 → 29.78 | 1.16 → 2.84 | 15.15 → 18.71 |

Scores are zero-shot → fine-tuned. See [report](report.pdf), [CSV](results/scores.csv), [JSON](results/scores.json), [figures](results/), and [training histories](results/history/). The [original archive](https://github.com/praneet3t/low-resource-mt/releases/download/study-v1/low-resource-mt-results.tar.gz) contains the labeled and official translation JSON files for all four systems under `base/` and `tuned/`. Adapters and resumable checkpoints are also on Hugging Face for [MADLAD](https://huggingface.co/praneet3/low-resource-mt-madlad/tree/main) and [Gemma](https://huggingface.co/praneet3/low-resource-mt-gemma/tree/main).

## Reproduce

Use Python 3.12, a CUDA GPU, and Pango with Noto Serif and Noto Sans Devanagari. Accept the Gemma terms and sign in with `hf auth login`.
The scripts cover [preparation](prepare.py), [EDA and reporting](results.py), [loading and prompts](model_loading.py), [fine-tuning](fine_tuning.py), and [baseline and tuned evaluation](evaluation.py). Run `python results.py --eda-only` after preparation to regenerate corpus figures.

```sh
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
python -m pip install -r requirements.txt
python prepare.py /path/to/Low_Resource_MT_Data-20261123T221212Z-1-001.zip
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for model in madlad gemma; do
  python study.py evaluate "$model" --batch 64 --output "runs/base/$model"
done
python study.py train madlad --batch 32 --accumulation 2 --gradient-checkpointing --output runs/train/madlad
python study.py train gemma --batch 16 --accumulation 4 --gradient-checkpointing --output runs/train/gemma
for model in madlad gemma; do
  python study.py evaluate "$model" --batch 64 --adapter "runs/train/$model/adapter" --output "runs/tuned/$model"
done
python results.py
```

## Bonus evaluation

The pilot uses 12 fixed test examples per direction, seed 42, and a 128-token decoding cap. See [pilot scores](results/bonus-decoding.csv), [TER and bootstrap intervals](results/bonus-evaluation.csv), and [pilot translations](https://github.com/praneet3t/low-resource-mt/releases/download/study-v1/bonus-translations.tar.gz). The full-holdout TER and bootstrap analysis only reads saved translations. To run it without retraining, unpack the [original archive](https://github.com/praneet3t/low-resource-mt/releases/download/study-v1/low-resource-mt-results.tar.gz) and pass its extracted directory as `--root`.

```sh
python bonus_evaluation.py --seed 42 --resamples 2000
python bonus_decoding.py gemma --batch 16 --output runs/bonus/gemma
python bonus_decoding.py madlad --batch 4 --output runs/bonus/madlad
python results.py
```
