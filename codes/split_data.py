import os
import random
import shutil
from pathlib import Path


data_root = Path("../datas/train")
train_split_root = Path("../datas/train_split")
val_split_root = Path("../datas/val_split")
train_split_root.mkdir(parents=True, exist_ok=True)
val_split_root.mkdir(parents=True, exist_ok=True)

split_ratio = 0.2
random.seed(42)


all_samples = sorted([f for f in data_root.iterdir() if f.is_dir()])
random.shuffle(all_samples)

val_count = int(len(all_samples) * split_ratio)
val_samples = all_samples[:val_count]
train_samples = all_samples[val_count:]


def copy_samples(samples, target_root):
    for sample in samples:
        target = target_root / sample.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(sample, target)

copy_samples(train_samples, train_split_root)
copy_samples(val_samples, val_split_root)
