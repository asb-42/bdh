"""Byte-level datasets for the pipeline (no tokenizer; vocab is the raw 256 bytes)."""

import os

import numpy as np
import requests
import torch

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
# HuggingFace datasets-server parquet mirrors of Salesforce/wikitext (the original
# research.metamind.io S3 bucket is not reachable from all environments).
WIKITEXT_PARQUET_BASE = (
    "https://huggingface.co/datasets/Salesforce/wikitext/resolve/"
    "refs%2Fconvert%2Fparquet/wikitext-2-raw-v1"
)


def _download(url: str, dest: str) -> None:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        f.write(resp.content)


def _prepare_shakespeare(data_dir: str):
    path = os.path.join(data_dir, "tinyshakespeare.txt")
    if not os.path.exists(path):
        _download(SHAKESPEARE_URL, path)
    raw = open(path, "rb").read()
    split = int(0.9 * len(raw))
    return raw[:split], raw[split:], None


def _prepare_wikitext2(data_dir: str):
    import pyarrow.parquet as pq

    def load_split(split):
        path = os.path.join(data_dir, f"wikitext2-{split}.parquet")
        if not os.path.exists(path):
            _download(f"{WIKITEXT_PARQUET_BASE}/{split}/0000.parquet", path)
        table = pq.read_table(path)
        lines = [row.rstrip("\n") for row in table.column("text").to_pylist()]
        return "\n".join(lines).encode("utf-8")

    return load_split("train"), load_split("validation"), load_split("test")


class ByteDataset:
    def __init__(self, train: bytes, val: bytes, test: bytes | None = None):
        self.train = np.frombuffer(train, dtype=np.uint8)
        self.val = np.frombuffer(val, dtype=np.uint8)
        self.test = np.frombuffer(test, dtype=np.uint8) if test is not None else None

    def splits(self):
        return ["train", "val"] + (["test"] if self.test is not None else [])

    def get_batch(self, split, block_size, batch_size, device):
        data = getattr(self, split)
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack(
            [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix]
        )
        y = torch.stack(
            [torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix]
        )
        return x.to(device), y.to(device)


def load_dataset(cfg) -> ByteDataset:
    os.makedirs(cfg.data_dir, exist_ok=True)
    if cfg.dataset == "shakespeare":
        train, val, test = _prepare_shakespeare(cfg.data_dir)
    elif cfg.dataset == "wikitext2":
        train, val, test = _prepare_wikitext2(cfg.data_dir)
    else:
        raise ValueError(f"unknown dataset: {cfg.dataset}")
    return ByteDataset(train, val, test)
