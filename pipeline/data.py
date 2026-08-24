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
EUROPARL_BASE = "https://www.statmt.org/europarl/v7"


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
        # Intentional normalization: strip each row's trailing newlines and rejoin
        # with single newlines so blank-line runs collapse to one.
        lines = [row.rstrip("\n") for row in table.column("text").to_pylist()]
        return "\n".join(lines).encode("utf-8")

    return load_split("train"), load_split("validation"), load_split("test")


def _prepare_europarl(data_dir: str, lang_bytes: int):
    """Multilingual Europarl v7 (EN/DE/ES) as contiguous per-language blocks.

    Downloads the de-en and es-en parallel tarballs once, extracts the three
    monolingual sides, and returns (train, val, test) where each split is the
    concatenation of equal-sized EN, DE, ES blocks in that order. The last
    2 MB of each language are held out (1 MB val + 1 MB test) before the
    train cap, so evaluation text is never seen in training.
    """
    import tarfile

    edir = os.path.join(data_dir, "europarl")
    os.makedirs(edir, exist_ok=True)
    sources = {"en": "de-en", "de": "de-en", "es": "es-en"}
    sides = {}
    for lang, pair in sources.items():
        member = f"europarl-v7.{pair}.{lang}"
        txt = os.path.join(edir, f"{member}.txt")
        if not os.path.exists(txt):
            tgz = os.path.join(edir, f"{pair}.tgz")
            if not os.path.exists(tgz):
                _download(f"{EUROPARL_BASE}/{pair}.tgz", tgz)
            with tarfile.open(tgz, "r:gz") as tf:
                src = tf.extractfile(member)
                if src is None:
                    raise FileNotFoundError(f"{member} not in {pair}.tgz")
                with open(txt, "wb") as f:
                    while chunk := src.read(1 << 24):
                        f.write(chunk)
        sides[lang] = open(txt, "rb").read()

    train_parts, val_parts, test_parts = [], [], []
    for lang in ("en", "de", "es"):
        raw = sides[lang]
        need = lang_bytes + 2_000_000
        if len(raw) < need:
            raise ValueError(f"europarl {lang}: {len(raw)} bytes < needed {need}")
        block = raw[-need:]
        train_parts.append(block[:-2_000_000])
        val_parts.append(block[-2_000_000:-1_000_000])
        test_parts.append(block[-1_000_000:])
        print(f"europarl {lang}: train {len(block[:-2_000_000]):,} B | "
              f"val 1,000,000 B | test 1,000,000 B")
    sep = b"\n\n"
    return (sep.join(train_parts), sep.join(val_parts), sep.join(test_parts))


class ByteStream:
    """Sequential (temporally coherent) batch stream over one split.

    Serves batch b as the continuation of batch b-1 along `batch_size`
    independent streams, mirroring the paper's training setup (Appendix B):
    subsequent minibatches are related, so a carried attention state can
    meaningfully persist across them.
    """

    def __init__(self, data: np.ndarray, block_size: int, batch_size: int):
        self.data = data
        self.block_size = block_size
        self.batch_size = batch_size
        span = max(1, (len(data) - 2 * block_size - 1) // batch_size)
        self.cursors = np.arange(batch_size) * span

    def next_batch(self, device):
        xs, ys = [], []
        for i in range(self.batch_size):
            c = int(self.cursors[i])
            xs.append(torch.from_numpy(self.data[c : c + self.block_size].astype(np.int64)))
            ys.append(
                torch.from_numpy(self.data[c + 1 : c + 1 + self.block_size].astype(np.int64))
            )
            nxt = c + self.block_size
            self.cursors[i] = nxt if nxt + self.block_size + 1 < len(self.data) else 0
        return torch.stack(xs).to(device), torch.stack(ys).to(device)


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

    def make_stream(self, split, block_size, batch_size) -> ByteStream:
        return ByteStream(getattr(self, split), block_size, batch_size)


def load_dataset(cfg) -> ByteDataset:
    os.makedirs(cfg.data_dir, exist_ok=True)
    if cfg.dataset == "shakespeare":
        train, val, test = _prepare_shakespeare(cfg.data_dir)
    elif cfg.dataset == "wikitext2":
        train, val, test = _prepare_wikitext2(cfg.data_dir)
    elif cfg.dataset == "europarl":
        train, val, test = _prepare_europarl(cfg.data_dir, cfg.europarl_lang_mb * 1_000_000)
    else:
        raise ValueError(f"unknown dataset: {cfg.dataset}")
    if getattr(cfg, "train_slice", ""):
        a, b = (float(p) for p in cfg.train_slice.split(":"))
        if not (0 <= a < b <= 1):
            raise ValueError(f"invalid --train-slice {cfg.train_slice!r} (need 0 <= a < b <= 1)")
        train = train[int(a * len(train)):int(b * len(train))]
    return ByteDataset(train, val, test)
