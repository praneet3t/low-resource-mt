from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import time

from peft import PeftModel
from sacrebleu.metrics import BLEU, CHRF
import torch

from model_loading import MODELS, encode_source, load
from prepare import read_jsonl, write_json


def metrics(rows):
    scores = {}
    for direction in sorted({row["direction"] for row in rows}):
        group = [row for row in rows if row["direction"] == direction]
        predictions = [row["translation"] for row in group]
        references = [[row["reference"] for row in group]]
        bleu, chrf = BLEU(tokenize="13a"), CHRF(word_order=2)
        scores[direction] = {
            "count": len(group),
            "bleu": bleu.corpus_score(predictions, references).score,
            "chrf++": chrf.corpus_score(predictions, references).score,
            "bleu_signature": str(bleu.get_signature()),
            "chrf++_signature": str(chrf.get_signature()),
        }
    return scores


def decode_batch(model, tokenizer, sequences, seq2seq, max_tokens):
    inputs = tokenizer.pad([{"input_ids": ids} for ids in sequences],
                           pad_to_multiple_of=8, return_tensors="pt").to("cuda")
    result = model.generate(**inputs, do_sample=False, num_beams=1, use_cache=True,
                            max_new_tokens=max_tokens, pad_token_id=tokenizer.pad_token_id)
    if not seq2seq:
        result = result[:, inputs.input_ids.shape[1]:]
    return tokenizer.batch_decode(result, skip_special_tokens=True)


def decode_with_backoff(model, tokenizer, sequences, seq2seq, max_tokens):
    try:
        return decode_batch(model, tokenizer, sequences, seq2seq, max_tokens)
    except torch.OutOfMemoryError:
        if len(sequences) == 1:
            raise
    torch.cuda.empty_cache()
    middle = len(sequences) // 2
    return (decode_with_backoff(model, tokenizer, sequences[:middle], seq2seq, max_tokens)
            + decode_with_backoff(model, tokenizer, sequences[middle:], seq2seq, max_tokens))


@torch.inference_mode()
def generate(model, tokenizer, rows, seq2seq, batch_size, max_tokens):
    with ThreadPoolExecutor(max_workers=8) as workers:
        encoded = list(workers.map(lambda row: encode_source(tokenizer, row, seq2seq), rows))
    order = sorted(range(len(rows)), key=lambda i: len(encoded[i]))
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        decoded = decode_with_backoff(model, tokenizer, [encoded[i] for i in indices], seq2seq, max_tokens)
        for i, prediction in zip(indices, decoded):
            yield i, prediction.strip()


def output_rows(model, tokenizer, rows, args, file):
    signature = hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode()).hexdigest()
    adapter_hash = None
    if args.adapter:
        with (args.adapter / "adapter_model.safetensors").open("rb") as handle:
            adapter_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    settings = {"model": MODELS[args.model], "adapter_sha256": adapter_hash, "batch": args.batch,
                "max_new_tokens": args.max_tokens, "rows_sha256": signature}
    metadata = file.with_suffix(".settings.json")
    settings = json.loads(json.dumps(settings))
    if metadata.exists() and json.loads(metadata.read_text()) != settings:
        raise ValueError(f"Settings differ from {metadata}")
    write_json(metadata, settings)
    cache = file.with_suffix(".partial.jsonl")
    completed = {}
    if cache.exists():
        with cache.open("rb+") as handle:
            while line := handle.readline():
                if not line.endswith(b"\n"):
                    handle.seek(-len(line), os.SEEK_CUR)
                    handle.truncate()
                    break
                index, translation = json.loads(line)
                completed[index] = translation
    pending = [index for index in range(len(rows)) if index not in completed]
    started = time.monotonic()
    with cache.open("a", encoding="utf-8", buffering=1) as handle:
        for inner, translation in generate(model, tokenizer, [rows[i] for i in pending],
                                            args.model == "madlad", args.batch, args.max_tokens):
            index = pending[inner]
            completed[index] = translation
            handle.write(json.dumps([index, translation], ensure_ascii=False) + "\n")
    outputs = [row | {"translation": completed[index]} for index, row in enumerate(rows)]
    write_json(file, outputs)
    return outputs, (time.monotonic() - started) / max(1, len(pending))


def evaluate(args):
    model, tokenizer = load(args)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    for split in ("test", "official"):
        rows = read_jsonl(args.data / f"{split}.jsonl")
        predictions, seconds = output_rows(model, tokenizer, rows, args, args.output / f"{split}.json")
        if split == "test":
            write_json(args.output / "metrics.json", metrics(predictions))
        print(f"{split}: {len(predictions)} translations, {seconds:.3f} seconds per new row", flush=True)
