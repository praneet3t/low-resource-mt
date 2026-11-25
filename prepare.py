import argparse
from collections import Counter
from io import BytesIO
import json
from pathlib import Path
import random
import statistics
import unicodedata as ud
from zipfile import ZipFile

from openpyxl import load_workbook


LANGUAGES = {"kha": "Khasi", "brx": "Bodo"}
CODES = {"kha": "kha", "brx": "bodo"}
SEED = 42


def normalize(text):
    return "".join(c for c in ud.normalize("NFKC", text).casefold()
                   if ud.category(c)[0] in "LMN")


def rows(archive, name, sheet):
    book = load_workbook(BytesIO(archive.read(name)), read_only=True, data_only=True)
    try:
        yield from book[sheet].values
    finally:
        book.close()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, values):
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def scripts(text):
    counts = Counter()
    for char in text:
        if not char.isalpha():
            continue
        name = ud.name(char, "")
        counts[next((script for script in ("LATIN", "DEVANAGARI", "BENGALI", "CYRILLIC")
                     if name.startswith(script)), "OTHER")] += 1
    return counts


def distribution(values):
    ordered = sorted(values)
    return {"min": ordered[0], "median": statistics.median(ordered),
            "mean": statistics.mean(ordered), "p95": ordered[int(.95 * (len(ordered) - 1))],
            "max": ordered[-1]}


def summarize(pairs, language):
    ratios = [len(right.split()) / len(left.split()) for _, left, right in pairs]
    expected = "LATIN" if language == "kha" else "DEVANAGARI"
    return {
        "usable_rows": len(pairs),
        "exact_pair_duplicates": len(pairs) - len({(a, b) for _, a, b in pairs}),
        "normalized_pair_duplicates": len(pairs) - len({(normalize(a), normalize(b)) for _, a, b in pairs}),
        "normalized_identical_sides": sum(normalize(a) == normalize(b) for _, a, b in pairs),
        "word_ratio_outliers": sum(r < .33 or r > 3 for r in ratios),
        "english_whitespace_tokens": distribution([len(a.split()) for _, a, _ in pairs]),
        "target_whitespace_tokens": distribution([len(b.split()) for _, _, b in pairs]),
        "english_characters": distribution([len(a) for _, a, _ in pairs]),
        "target_characters": distribution([len(b) for _, _, b in pairs]),
        "english_scripts": dict(scripts(" ".join(a for _, a, _ in pairs))),
        "target_scripts": dict(scripts(" ".join(b for _, _, b in pairs))),
        "target_without_expected_letters": sum(not scripts(b)[expected] for _, _, b in pairs),
    }


def official_rows(archive, code):
    for source, target in (("en", code), (code, "en")):
        suffix = f"/{CODES.get(source, source)}-{CODES.get(target, target)} Test.xlsx"
        name = next(n for n in archive.namelist() if "/Test_Data/" in n and n.endswith(suffix))
        result = []
        for row in list(rows(archive, name, "Final Data"))[1:]:
            if not any(value is not None for value in row):
                continue
            identifier, sentence = row[:2]
            if not isinstance(sentence, str) or not sentence.strip() or identifier is None:
                raise ValueError(f"Invalid official entry in {name}: {identifier!r}")
            result.append({"id": identifier, "source": sentence, "direction": f"{source}-{target}"})
        if len({row["id"] for row in result}) != len(result):
            raise ValueError(f"Duplicate official IDs in {name}")
        yield from result


def prepare(archive_path, output):
    output.mkdir(parents=True, exist_ok=True)
    stats, official, candidates, analysis = {}, [], [], {}
    with ZipFile(archive_path) as archive:
        for code, name in LANGUAGES.items():
            official.extend(official_rows(archive, code))
            member = next(n for n in archive.namelist() if "/Train_Data/" in n and f"-{name} " in n)
            pairs, rejected = [], Counter()
            for index, row in enumerate(list(rows(archive, member, "Sheet1"))[1:], 2):
                english, target = row[:2]
                if english is None and target is None:
                    rejected["blank"] += 1
                elif not all(isinstance(s, str) and normalize(s) for s in (english, target)):
                    rejected["empty_or_invalid_side"] += 1
                else:
                    pairs.append((index, english.strip(), target.strip()))
            stats[code] = summarize(pairs, code) | {"rejected": dict(rejected)}
            samples = [pairs[round(i * (len(pairs) - 1) / 7)] for i in range(8)]
            analysis[code] = {"samples": samples,
                              "lengths": [[len(a.split()), len(b.split())] for _, a, b in pairs]}
            for index, english, target in pairs:
                candidates.append((code, f"{code}:{index}", english, target))
    rng = random.Random(SEED)
    rng.shuffle(candidates)
    seen = {normalize(row["source"]) for row in official}
    clean = {code: [] for code in LANGUAGES}
    for code, identifier, english, target in candidates:
        keys = {normalize(english), normalize(target)}
        if keys & seen:
            continue
        seen.update(keys)
        clean[code].append((identifier, english, target))
    splits = {name: [] for name in ("train", "validation", "test")}
    for code, pairs in clean.items():
        holdout = min(1000, len(pairs) // 20)
        stats[code]["excluded_for_overlap"] = stats[code]["usable_rows"] - len(pairs)
        for name, group in zip(splits, (pairs[2 * holdout:], pairs[:holdout], pairs[holdout:2 * holdout])):
            stats[code][name + "_pairs"] = len(group)
            for identifier, english, target in group:
                for direction, source, reference in ((f"en-{code}", english, target),
                                                      (f"{code}-en", target, english)):
                    splits[name].append({"id": identifier, "direction": direction,
                                         "source": source, "reference": reference})
    for name, data in splits.items():
        rng.shuffle(data)
        write_jsonl(output / f"{name}.jsonl", data)
    write_jsonl(output / "official.jsonl", official)
    write_json(output / "eda.json", stats | {"official": dict(Counter(row["direction"] for row in official))})
    write_json(output / "inspection.json", analysis)
    keys = {name: {normalize(row["source"]) for row in data} for name, data in splits.items()}
    keys["official"] = {normalize(row["source"]) for row in official}
    for name, values in keys.items():
        for other in keys.keys() - {name}:
            assert values.isdisjoint(keys[other]), (name, other)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/data"))
    args = parser.parse_args()
    prepare(args.archive, args.output)
