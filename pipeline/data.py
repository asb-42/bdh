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


def _europarl_blocks(data_dir: str, lang_bytes, langs=("en", "de", "es")):
    """Per-language Europarl v7 blocks: {lang: {train, val, test} bytes}.

    lang_bytes: cap in BYTES — int (uniform) or per-language sequence.
    The last 2 MB of each language are held out (1 MB val + 1 MB test) before
    the train cap, so evaluation text is never seen in training.
    """
    import tarfile

    edir = os.path.join(data_dir, "europarl")
    os.makedirs(edir, exist_ok=True)
    sources = {
        "en": "de-en", "de": "de-en", "es": "es-en", "fr": "fr-en", "pt": "pt-en",
        "it": "it-en", "da": "da-en", "cs": "cs-en", "nl": "nl-en", "pl": "pl-en",
        "ro": "ro-en", "sv": "sv-en", "el": "el-en", "hu": "hu-en", "bg": "bg-en",
        "fi": "fi-en", "sk": "sk-en", "sl": "sl-en", "et": "et-en", "lt": "lt-en",
    }
    caps = list(lang_bytes) if isinstance(lang_bytes, (list, tuple)) else [lang_bytes] * len(langs)
    if len(caps) != len(langs):
        raise ValueError(f"{len(caps)} byte caps for {len(langs)} languages")
    blocks = {}
    for li, lang in enumerate(langs):
        need = caps[li] + 2_000_000
        pair = sources[lang]
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
        raw = open(txt, "rb").read()
        if len(raw) < need:
            raise ValueError(f"europarl {lang}: {len(raw)} bytes < needed {need}")
        blk = raw[-need:]
        blocks[lang] = {
            "train": blk[:-2_000_000],
            "val": blk[-2_000_000:-1_000_000],
            "test": blk[-1_000_000:],
        }
        print(f"europarl {lang}: train {len(blocks[lang]['train']):,} B | "
              f"val 1,000,000 B | test 1,000,000 B")
    return blocks


def _prepare_europarl(data_dir: str, lang_bytes, langs=("en", "de", "es")):
    """Multilingual Europarl v7 as contiguous per-language blocks."""
    blocks = _europarl_blocks(data_dir, lang_bytes, langs)
    sep = b"\n\n"
    return (sep.join(blocks[l]["train"] for l in langs),
            sep.join(blocks[l]["val"] for l in langs),
            sep.join(blocks[l]["test"] for l in langs))


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


def _prepare_textmix(spec: str, mb: int):
    """Multi-file byte corpus: spec = comma-separated "name:path" pairs.

    Per domain: train = first `mb` MB, val = following 1 MB, test = next 1 MB.
    Returns per-domain blocks plus a combined ByteDataset triple.
    """
    blocks, trains, vals, tests = {}, [], [], []
    for item in filter(None, map(str.strip, spec.split(","))):
        name, path = item.split(":", 1)
        with open(path, "rb") as f:
            raw = f.read(mb * 1_000_000)
        # carve splits inside the per-domain allocation: train | val 1MB | test 1MB
        t_end = max(0, len(raw) - 2_000_000)
        v_end = min(len(raw), t_end + 1_000_000)
        train, val, test = raw[:t_end], raw[t_end:v_end], raw[v_end:]
        blocks[name] = {"train": train, "val": val, "test": test}
        trains.append(train); vals.append(val); tests.append(test)
        print(f"textmix {name}: train {len(train):,} B | val {len(val):,} B | test {len(test):,} B")
    return blocks, b"".join(trains), b"".join(vals), b"".join(tests)


def load_dataset(cfg) -> ByteDataset:
    os.makedirs(cfg.data_dir, exist_ok=True)
    if cfg.dataset == "shakespeare":
        train, val, test = _prepare_shakespeare(cfg.data_dir)
    elif cfg.dataset == "wikitext2":
        train, val, test = _prepare_wikitext2(cfg.data_dir)
    elif cfg.dataset == "europarl":
        langs = tuple(s.strip() for s in cfg.europarl_langs.split(",") if s.strip())
        caps = [int(m) for m in str(cfg.europarl_lang_mb).split(",")]
        if len(caps) == 1:
            caps = caps * len(langs)
        train, val, test = _prepare_europarl(
            cfg.data_dir, [m * 1_000_000 for m in caps], langs)
    elif cfg.dataset == "textmix":
        _, train, val, test = _prepare_textmix(cfg.text_mix, cfg.text_mix_mb)
    else:
        raise ValueError(f"unknown dataset: {cfg.dataset}")
    if getattr(cfg, "train_slice", ""):
        a, b = (float(p) for p in cfg.train_slice.split(":"))
        if not (0 <= a < b <= 1):
            raise ValueError(f"invalid --train-slice {cfg.train_slice!r} (need 0 <= a < b <= 1)")
        train = train[int(a * len(train)):int(b * len(train))]
    return ByteDataset(train, val, test)
