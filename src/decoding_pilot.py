import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path
import random
import re
import time

import torch

from .evaluation import metrics
from .model_loading import NAMES, encode_source, load
from .prepare import normalize, read_jsonl, write_json


def terms(text):
    return set(re.findall(r"[^\W_]+", text.casefold()))


def examples_for(training, test, rng):
    by_direction = defaultdict(list)
    for row in training:
        by_direction[row["direction"]].append(row)
    indexes = {}
    for direction, pool in by_direction.items():
        index = defaultdict(list)
        for i, row in enumerate(pool):
            for word in terms(row["source"]):
                index[word].append(i)
        indexes[direction] = index
    selections = []
    for query in test:
        direction, source = query["direction"], query["source"]
        pool, index = by_direction[direction], indexes[direction]
        weights = Counter()
        for word in sorted(terms(source)):
            matches = index.get(word, ())
            weight = math.log((len(pool) + 1) / (len(matches) + 1))
            for i in matches:
                weights[i] += weight
        random_examples = rng.sample(pool, 5)
        ranked = [pool[i] for i in sorted(weights, key=lambda i: (-weights[i], i))[:5]]
        ranked.extend(row for row in random_examples if row not in ranked)
        assert all(normalize(row["source"]) != normalize(source) for row in ranked[:5] + random_examples)
        selections.append({"random": random_examples, "lexical": ranked[:5]})
    return selections


def prompt_tokens(tokenizer, row, demonstrations):
    if not demonstrations:
        return encode_source(tokenizer, row, False)
    source, target = row["direction"].split("-")
    examples = "\n\n".join(f"{NAMES[source]}: {demo['source']}\n{NAMES[target]}: {demo['reference']}"
                            for demo in demonstrations)
    prompt = (f"Translate the following text from {NAMES[source]} to {NAMES[target]}. "
              f"Return only the translation.\n\nExamples:\n{examples}\n\nText:\n{row['source']}")
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                         tokenize=True, add_generation_prompt=True)


@torch.inference_mode()
def decode(model, tokenizer, encoded, seq2seq, beams, batch, max_tokens):
    predictions = [""] * len(encoded)
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]))
    torch.cuda.synchronize()
    start = time.monotonic()
    for offset in range(0, len(order), batch):
        indices = order[offset:offset + batch]
        inputs = tokenizer.pad([{"input_ids": encoded[i]} for i in indices],
                               pad_to_multiple_of=8, return_tensors="pt").to("cuda")
        output = model.generate(**inputs, do_sample=False, num_beams=beams, use_cache=True,
                                max_new_tokens=max_tokens, pad_token_id=tokenizer.pad_token_id)
        if not seq2seq:
            output = output[:, inputs.input_ids.shape[1]:]
        for i, prediction in zip(indices, tokenizer.batch_decode(output, skip_special_tokens=True)):
            predictions[i] = prediction.strip()
    torch.cuda.synchronize()
    return predictions, time.monotonic() - start


def pilot(args):
    rng = random.Random(args.seed)
    data = read_jsonl(args.data / "test.jsonl")
    test = [row for direction in sorted({row["direction"] for row in data})
            for row in rng.sample([x for x in data if x["direction"] == direction], args.per_direction)]
    seq2seq = args.model == "madlad"
    if seq2seq:
        settings = [(beam, "none", 0) for beam in (1, 5, 10, 20)]
        selected = [{} for _ in test]
    else:
        train = read_jsonl(args.data / "train.jsonl")
        selected = examples_for(train, test, rng)
        settings = [(1, "none", 0)] + [(1, method, shots) for shots in (1, 3, 5)
                                      for method in ("random", "lexical")]
    model, tokenizer = load(args)
    model.eval()
    args.output.mkdir(parents=True, exist_ok=True)
    for beams, method, shots in settings:
        label = f"beam-{beams}" if seq2seq else f"{method}-{shots}"
        destination = args.output / f"{label}.json"
        if destination.exists():
            continue
        demonstrations = [choices.get(method, [])[:shots] for choices in selected]
        encoded = [encode_source(tokenizer, row, True) if seq2seq else prompt_tokens(tokenizer, row, demos)
                   for row, demos in zip(test, demonstrations)]
        predictions, seconds = decode(model, tokenizer, encoded, seq2seq, beams, args.batch, args.max_tokens)
        rows = [row | {"translation": prediction, "demonstration_ids": [demo["id"] for demo in demos]}
                for row, prediction, demos in zip(test, predictions, demonstrations)]
        write_json(args.output / f"{label}.translations.json", rows)
        write_json(destination, {"model": args.model, "label": label, "seed": args.seed, "beams": beams,
                                  "shots": shots, "selection": method, "batch": args.batch,
                                  "seconds": seconds, "max_new_tokens": args.max_tokens,
                                  "scores": metrics(rows)})
        print(f"{args.model} {label}: {seconds:.2f} s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["madlad", "gemma"])
    parser.add_argument("--data", type=Path, default=Path("runs/data"))
    parser.add_argument("--models", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-direction", type=int, default=12)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    pilot(parser.parse_args())
