"""Dataset and evaluation settings shared by the NNWHR experiment scripts."""

import os
from pathlib import Path

from torch.utils.data import Subset

DATA_ROOT = Path(os.environ.get("NNWHR_DATA_ROOT", Path(__file__).resolve().parents[1] / "data"))
FASHION_MNIST = DATA_ROOT / "fashion-mnist"
CIFAR10 = DATA_ROOT / "cifar10"
TINY_IMAGENET = DATA_ROOT / "tiny-imagenet-200"


def positive_env(name, default=None):
    value = os.environ.get(name)
    if value is None:
        return default
    number = int(value)
    if number < 1:
        raise ValueError(f"{name} must be a positive integer")
    return number


def limit_dataset(dataset, setting):
    limit = positive_env(setting)
    return dataset if limit is None else Subset(dataset, range(min(limit, len(dataset))))


def evaluation_dataset(dataset):
    return limit_dataset(dataset, "NNWHR_EVAL_SAMPLES")
