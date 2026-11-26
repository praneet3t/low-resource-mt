from collections import defaultdict
import gc
import math
import random
import time

from peft import PeftModel
import torch

from .evaluation import generate, metrics
from .fine_tuning import collate, encode_all, trainer_for
from .model_loading import attach, load
from .prepare import SEED, read_jsonl, write_json


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
