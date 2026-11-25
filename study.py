import argparse
from pathlib import Path

import torch
from transformers import set_seed

from checks import benchmark, smoke
from evaluation import evaluate
from fine_tuning import train
from model_loading import MODELS
from prepare import SEED


if __name__ == "__main__":
    actions = {"smoke": smoke, "benchmark": benchmark, "train": train, "evaluate": evaluate}
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=actions)
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
    actions[args.action](args)
