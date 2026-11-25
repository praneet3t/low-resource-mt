# Low-resource machine translation

English↔Khasi and English↔Bodo supervised adaptation on the supplied WMT26 Indic MT archive. Compare pinned MADLAD-400 3B and Gemma 3 1B instruction-tuned checkpoints. Accept the Gemma terms on Hugging Face and authenticate using `hf auth login`. MADLAD uses the Bodo tag `<2brx>`, while the source workbook uses `bodo`.

## Reproduce

Use Python 3.12, CUDA-compatible NVIDIA hardware, PyTorch 2.8.0 with CUDA 12.9, and Noto Latin and Devanagari fonts with Pango for the report.

```sh
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
python -m pip install -r requirements.txt
python prepare.py /path/to/Low_Resource_MT_Data-20261123T221212Z-1-001.zip
python results.py --eda-only

for model in madlad gemma; do
  python study.py smoke "$model" --output "runs/smoke/$model"
  python study.py benchmark "$model" --output "runs/benchmark/$model"
done
```

The A40 run uses 64-example effective training batches. Use `--resume` to continue training from the most recent checkpoint, or pass a checkpoint path. Evaluation resumes from its partial output automatically.

```sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for model in madlad gemma; do
  python study.py evaluate "$model" --batch 64 --output "runs/base/$model"
done
python study.py train madlad --batch 32 --accumulation 2 --gradient-checkpointing --output runs/train/madlad
python study.py train gemma --batch 16 --accumulation 4 --gradient-checkpointing --output runs/train/gemma
for model in madlad gemma; do
  python study.py evaluate "$model" --batch 64 --adapter "runs/train/$model/adapter" --output "runs/tuned/$model"
done
python results.py --archive runs/low-resource-mt-results.tar.gz
```

## Protocol

The archive has parallel training workbooks and source-only official test workbooks. After excluding official-source overlap and duplicate sentences, reserve 5% of surviving pairs per language for validation and another 5% for an internal test, each capped at 1,000 pairs. Matching uses NFKC, case folding, and removal of characters outside the Unicode letter, mark, and number classes. The four official source lists retain every ID in workbook order. Scores on the internal test are separate from official predictions.

Train one epoch on both directions of every retained pair, with seed 42, BF16 on the A40, double-quantized NF4, AdamW, a linear learning-rate schedule starting at 2e-4, and rank-16/alpha-32 adapters on the attention query and value projections. The training and validation input and target token budgets are 256 each. Gemma uses a target-only causal loss, including its end-of-turn token. Select by lowest validation loss among quarter-epoch checkpoints including the final step.

MADLAD input is `<2{target}> {source}`. Gemma uses its official chat template and a single user message. The message is `Translate the following text from {source language} to {target language}. Return only the translation.` followed by two newlines and the source. Both baseline and adapter evaluations use BF16 greedy decoding with at most 256 new tokens. Report case-sensitive corpus BLEU with 13a tokenization and chrF++ with word order 2.

## Study results

- [Aggregate scores](results/scores.json), [corpus statistics](results/eda.json), and [error diagnostics](results/diagnostics.json)
- [Sentence lengths](results/lengths.pdf), [score changes](results/improvements.pdf), and [training curves](results/training.pdf)
- [Report and complete results archive](https://github.com/praneet3t/low-resource-mt/releases/tag/study-v1)
