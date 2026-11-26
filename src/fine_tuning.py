from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path

import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback

from .model_loading import attach, encode_source, load
from .prepare import SEED, read_jsonl, write_json


def encode_target(tokenizer, row, seq2seq):
    source_row = row
    source = encode_source(tokenizer, source_row, seq2seq)
    source_clipped = len(source) > 256
    while len(source) > 256:
        text = source_row["source"]
        shortened = text[:max(1, int(len(text) * 224 / len(source)))]
        source_row = row | {"source": shortened}
        source = encode_source(tokenizer, source_row, seq2seq)
    reference = tokenizer.encode(row["reference"], add_special_tokens=False)
    end = tokenizer.eos_token_id if seq2seq else tokenizer.convert_tokens_to_ids("<end_of_turn>")
    assert end is not None and end != tokenizer.unk_token_id
    labels = reference[:255] + [end]
    clipped = {"source_clipped": source_clipped, "target_clipped": len(reference) > 255}
    if seq2seq:
        return {"input_ids": source, "labels": labels} | clipped
    return {"input_ids": source + labels, "labels": [-100] * len(source) + labels} | clipped


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


def resume_checkpoint(output, request):
    if not request:
        return None
    required = ("adapter_config.json", "adapter_model.safetensors", "trainer_state.json",
                "optimizer.pt", "scheduler.pt", "rng_state.pth")
    if request is True:
        paths = [path for path in output.glob("checkpoint-*") if path.name[11:].isdigit()]
        paths.sort(key=lambda path: int(path.name[11:]), reverse=True)
    else:
        paths = [Path(request)]
    for path in paths:
        if all((path / name).is_file() and (path / name).stat().st_size for name in required):
            try:
                json.loads((path / "trainer_state.json").read_text())
            except json.JSONDecodeError:
                continue
            return str(path)
    raise FileNotFoundError(f"No complete checkpoint for {request!r} in {output}")


def train(args):
    seq2seq = args.model == "madlad"
    model, tokenizer = load(args, quantized=True)
    data = read_jsonl(args.data / "train.jsonl")
    validation = read_jsonl(args.data / "validation.jsonl")
    encoded_train = encode_all(tokenizer, data, seq2seq)
    encoded_validation = encode_all(tokenizer, validation, seq2seq)
    token_limits = {
        name: {key: sum(item[key] for item in encoded) for key in ("source_clipped", "target_clipped")}
        for name, encoded in (("train", encoded_train), ("validation", encoded_validation))
    }
    trainer = trainer_for(attach(model, args.model), tokenizer,
                          encoded_train, encoded_validation, args, args.output)
    result = trainer.train(resume_from_checkpoint=resume_checkpoint(args.output, args.resume))
    expected_steps = math.ceil(len(data) / (args.batch * args.accumulation))
    assert trainer.state.global_step == expected_steps
    assert math.isclose(trainer.state.epoch, 1.0)
    assert trainer.state.best_model_checkpoint is not None
    assert all(math.isfinite(item[key]) for item in trainer.state.log_history
               for key in ("loss", "eval_loss", "train_loss", "grad_norm") if key in item)
    trainer.save_model(str(args.output / "adapter"))
    write_json(args.output / "training.json", {"metrics": result.metrics,
                "history": trainer.state.log_history, "checkpoint": trainer.state.best_model_checkpoint,
                "batch": args.batch, "gradient_accumulation": args.accumulation,
                "token_limits": token_limits, "training_examples": len(data),
                "validation_examples": len(validation), "optimizer_steps": expected_steps,
                "gradient_checkpointing": args.gradient_checkpointing})
