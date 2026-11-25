import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import bitsandbytes as bnb
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from sacrebleu.metrics import BLEU, CHRF
from transformers import (
    AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig,
    Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback, set_seed,
)

from prepare import LANGUAGES, SEED, read_jsonl, write_json


MODELS = {
    "madlad": ("google/madlad400-3b-mt", "fa184c675da0b5c9e1c8694fccd4e12e2d422094"),
    "gemma": ("google/gemma-3-1b-it", "dcc83ea841ab6100d6b47a070329e1ba4cf78752"),
}
NAMES = {"en": "English", **LANGUAGES}


def load(args, quantized=False):
    repo, revision = MODELS[args.model]
    path = args.models / repo.split("/")[-1] if args.models else repo
    revision_args = {} if args.models else {"revision": revision}
    seq2seq = args.model == "madlad"
    tokenizer = AutoTokenizer.from_pretrained(path, **revision_args)
    tokenizer.padding_side = "right" if seq2seq else "left"
    if seq2seq:
        for code in NAMES:
            tag = f"<2{code}>"
            assert tokenizer.convert_tokens_to_ids(tag) in tokenizer.encode(tag)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    cls = AutoModelForSeq2SeqLM if seq2seq else AutoModelForCausalLM
    model = cls.from_pretrained(path, device_map={"": 0}, dtype=dtype,
                                quantization_config=quantization if quantized else None,
                                **revision_args)
    if quantized:
        layers = [m for m in model.modules() if isinstance(m, bnb.nn.Linear4bit)]
        assert layers and all(m.weight.quant_state.quant_type == "nf4" for m in layers)
    return model, tokenizer


def encode_source(tokenizer, row, seq2seq):
    source, target = row["direction"].split("-")
    if seq2seq:
        return tokenizer.encode(f"<2{target}> {row['source']}")
    prompt = (f"Translate the following text from {NAMES[source]} to {NAMES[target]}. "
              f"Return only the translation.\n\n{row['source']}")
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                         tokenize=True, add_generation_prompt=True)


def encode_target(tokenizer, row, seq2seq):
    source_row = row
    source = encode_source(tokenizer, source_row, seq2seq)
    while len(source) > 256:
        text = source_row["source"]
        shortened = text[:max(1, int(len(text) * 224 / len(source)))]
        source_row = row | {"source": shortened}
        source = encode_source(tokenizer, source_row, seq2seq)
    reference = tokenizer.encode(row["reference"], add_special_tokens=False)
    end = tokenizer.eos_token_id if seq2seq else tokenizer.convert_tokens_to_ids("<end_of_turn>")
    assert end is not None and end != tokenizer.unk_token_id
    labels = reference[:255] + [end]
    if seq2seq:
        return {"input_ids": source, "labels": labels}
    return {"input_ids": source + labels, "labels": [-100] * len(source) + labels}


def encode_all(tokenizer, rows, seq2seq):
    with ThreadPoolExecutor(max_workers=8) as workers:
        return list(workers.map(lambda row: encode_target(tokenizer, row, seq2seq), rows,
                                chunksize=128))


def collate(tokenizer, batch, seq2seq):
    padded = tokenizer.pad([{"input_ids": item["input_ids"]} for item in batch],
                           pad_to_multiple_of=8, return_tensors="pt")
    size = math.ceil(max(len(item["labels"]) for item in batch) / 8) * 8
    labels = []
    for item in batch:
        padding = [-100] * (size - len(item["labels"]))
        labels.append(item["labels"] + padding if seq2seq else padding + item["labels"])
    padded["labels"] = torch.tensor(labels)
    return padded


def attach(model, name):
    seq2seq = name == "madlad"
    adapter = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=["q", "v"] if seq2seq else ["q_proj", "v_proj"],
        task_type="SEQ_2_SEQ_LM" if seq2seq else "CAUSAL_LM",
    ))
    config = adapter.peft_config["default"]
    config.base_model_name_or_path, config.revision = MODELS[name]
    return adapter


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


class FinalValidation(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == state.max_steps:
            control.should_evaluate = True


def trainer_for(model, tokenizer, train, validation, args, output, smoke=False):
    training = Seq2SeqTrainingArguments(
        output_dir=str(output), num_train_epochs=1, max_steps=3 if smoke else -1,
        learning_rate=2e-4, seed=SEED, data_seed=SEED,
        bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
        per_device_train_batch_size=2 if smoke else args.batch,
        per_device_eval_batch_size=2 if smoke else args.batch,
        gradient_accumulation_steps=1 if smoke else args.accumulation,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch", group_by_length=not smoke,
        eval_strategy="steps", eval_steps=2 if smoke else .25,
        save_strategy="steps", save_steps=2 if smoke else .25, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        eval_accumulation_steps=1, prediction_loss_only=True,
        logging_steps=1 if smoke else 20, logging_nan_inf_filter=False, report_to="none",
    )
    model.config.use_cache = False
    return Seq2SeqTrainer(
        model=model, args=training, train_dataset=train, eval_dataset=validation,
        processing_class=tokenizer, callbacks=[] if smoke else [FinalValidation()],
        data_collator=lambda features: collate(tokenizer, features, args.model == "madlad"),
    )


def sample_directions(rows, count):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["direction"]].append(row)
    return [row for group in grouped.values() for row in group[:count]]


def smoke(args):
    seq2seq = args.model == "madlad"
    rows = read_jsonl(args.data / "train.jsonl")
    subset = sample_directions(rows, 2)
    model, tokenizer = load(args, quantized=True)
    examples = list(generate(model.eval(), tokenizer, subset, seq2seq, 8, 96))
    outputs = [subset[i] | {"translation": prediction} for i, prediction in examples]
    model = attach(model, args.model)
    encoded = encode_all(tokenizer, subset, seq2seq)
    trainer = trainer_for(model, tokenizer, encoded, encoded, args, args.output, smoke=True)
    adapter = next(param for name, param in model.named_parameters() if "lora_B" in name)
    before = adapter.detach().clone()
    trainer.train()
    assert not torch.equal(before, adapter)
    checkpoint = args.output / "checkpoint-2"
    restored_state = trainer.train(resume_from_checkpoint=str(checkpoint))
    assert restored_state.global_step == 3
    assert all(math.isfinite(item[key]) for item in trainer.state.log_history
               for key in ("loss", "eval_loss", "train_loss", "grad_norm") if key in item)
    trainer.save_model(str(args.output / "adapter"))
    expected = adapter.detach().cpu().clone()
    del trainer, model, adapter, before
    gc.collect()
    torch.cuda.empty_cache()
    model, _ = load(args, quantized=True)
    model = PeftModel.from_pretrained(model, args.output / "adapter")
    restored = next(p for n, p in model.named_parameters() if "lora_B" in n)
    assert torch.equal(restored.detach().cpu(), expected)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    write_json(args.output / "smoke.json", {"metrics": metrics(outputs), "examples": outputs,
                                          "optimizer_step": restored_state.global_step,
                                          "adapter_changed": True, "checkpoint_recovered": True})


def benchmark(args):
    data = read_jsonl(args.data / "train.jsonl")
    rng = random.Random(SEED)
    rows = rng.sample(data, min(768, len(data)))
    model, tokenizer = load(args, quantized=True)
    model = attach(model, args.model)
    model.config.use_cache = False
    model.train()
    seq2seq = args.model == "madlad"
    encoded = sorted(encode_all(tokenizer, rows, seq2seq), key=lambda x: len(x["input_ids"]))
    timings = []
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4)
    dtype = model.get_input_embeddings().weight.dtype
    for checkpointing in (False, True):
        if checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        for size in (8, 16, 32, 48):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.monotonic()
            try:
                for fraction in (.15, .45, .75, .9, 1):
                    end = max(size, int(len(encoded) * fraction))
                    batch = collate(tokenizer, encoded[end-size:end], seq2seq).to("cuda")
                    with torch.autocast("cuda", dtype=dtype):
                        loss = model(**batch).loss
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                duration = time.monotonic() - started
                timings.append({"batch": size, "checkpointing": checkpointing,
                                "examples_per_second": 5 * size / duration,
                                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30})
            except torch.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                timings.append({"batch": size, "checkpointing": checkpointing, "oom": True})
                break
    write_json(args.output / "training-benchmark.json", timings)
    del model, optimizer, batch, loss
    gc.collect()
    torch.cuda.empty_cache()
    model, tokenizer = load(args)
    sample = sample_directions(read_jsonl(args.data / "test.jsonl"), 16) + sample_directions(
        read_jsonl(args.data / "official.jsonl"), 16)
    timing = []
    for size in (16, 32, 64):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        list(generate(model.eval(), tokenizer, sample, seq2seq, size, args.max_tokens))
        timing.append({"batch": size, "examples_per_second": len(sample) / (time.monotonic() - started),
                       "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30})
    write_json(args.output / "inference-benchmark.json", timing)


def train(args):
    seq2seq = args.model == "madlad"
    model, tokenizer = load(args, quantized=True)
    data = read_jsonl(args.data / "train.jsonl")
    validation = read_jsonl(args.data / "validation.jsonl")
    trainer = trainer_for(attach(model, args.model), tokenizer,
                          encode_all(tokenizer, data, seq2seq),
                          encode_all(tokenizer, validation, seq2seq), args, args.output)
    result = trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(args.output / "adapter"))
    write_json(args.output / "training.json", {"metrics": result.metrics,
                "history": trainer.state.log_history, "checkpoint": trainer.state.best_model_checkpoint,
                "batch": args.batch, "gradient_accumulation": args.accumulation,
                "gradient_checkpointing": args.gradient_checkpointing})


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["smoke", "benchmark", "train", "evaluate"])
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--data", type=Path, default=Path("runs/data"))
    parser.add_argument("--models", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--resume", nargs="?", const=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    set_seed(SEED)
    globals()[args.action](args)
