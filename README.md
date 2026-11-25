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

Use the benchmark to set the microbatch and gradient accumulation for each model. Enable `--gradient-checkpointing` if required by memory. Run with `--resume` to load the latest Trainer checkpoint or pass an explicit checkpoint path.

```sh
for model in madlad gemma; do
  python study.py evaluate "$model" --batch 32 --output "runs/base/$model"
  python study.py train "$model" --batch 16 --accumulation 2 --output "runs/train/$model"
  python study.py evaluate "$model" --batch 32 --adapter "runs/train/$model/adapter" --output "runs/tuned/$model"
done
python results.py
```

## Protocol

The archive has parallel training workbooks and source-only official test workbooks. After excluding official-source overlap and duplicate sentences, reserve 5% of surviving pairs per language for validation and another 5% for an internal test, each capped at 1,000 pairs. Matching uses NFKC, case folding, and removal of characters outside the Unicode letter, mark, and number classes. The four official source lists retain every ID in workbook order. Scores on the internal test are separate from official predictions.

Train one epoch on both directions of every retained pair, with seed 42, BF16 on the A40, double-quantized NF4, AdamW, a linear learning-rate schedule starting at 2e-4, and rank-16/alpha-32 adapters on the attention query and value projections. The training and validation input and target token budgets are 256 each. Gemma uses a target-only causal loss, including its end-of-turn token. Select by lowest validation loss among quarter-epoch checkpoints including the final step.

MADLAD input is `<2{target}> {source}`. Gemma uses its official chat template and a single user message. The message is `Translate the following text from {source language} to {target language}. Return only the translation.` followed by two newlines and the source. Both baseline and adapter evaluations use BF16 greedy decoding with at most 256 new tokens. Report case-sensitive corpus BLEU with 13a tokenization and chrF++ with word order 2.
