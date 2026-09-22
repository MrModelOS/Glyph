#!/usr/bin/env python3
"""Generate the NeuMoE study corpus (port of neu_moe/data.py to a plain
binary format consumable by the C++ host driver).

Output files (little-endian):
  header  : u32 magic=0x4E4E4D31 ("NNM1"), u32 n_seqs, u32 seq_len
  payload : n_seqs * seq_len u16 tokens, row-major

Files:
  train.bin         – mixed prose/code/qa (weights 0.40/0.40/0.20)
  val_prose.bin     – 512 prose sequences
  val_code.bin      – 512 code sequences
  val_qa.bin        – 512 qa sequences
"""
import random
import struct
import sys

PAD, BOS, EOS, UNK = 0, 1, 2, 3
QMARK, AMARK = 4, 5
PROSE_LO, PROSE_HI = 6, 3685
CODE_LO, CODE_HI = 3686, 7365
QA_LO, QA_HI = 7366, 10239
VOCAB = 10240

DOMAIN_MIX = ["prose", "code", "qa"]
DOMAIN_W = [0.40, 0.40, 0.20]


def gen_prose(rng):
    toks = [BOS]
    n = rng.randint(10, 24)
    for _ in range(n):
        r = rng.random()
        if r < 0.15:
            toks.append(rng.randint(PROSE_LO, PROSE_LO + 60))
        elif r < 0.30:
            toks.append(rng.randint(PROSE_LO + 60, PROSE_LO + 300))
        else:
            toks.append(rng.randint(PROSE_LO + 300, PROSE_HI - 1))
    toks.append(EOS)
    return toks


def gen_code(rng):
    toks = [BOS]
    n = rng.randint(8, 18)
    indent = 0
    for _ in range(n):
        r = rng.random()
        if r < 0.1:
            toks.append(rng.randint(CODE_LO, CODE_LO + 20))
        elif r < 0.5:
            toks.append(rng.randint(CODE_LO + 20, CODE_LO + 500))
        else:
            toks.append(rng.randint(CODE_LO + 500, CODE_HI - 1))
        if rng.random() < 0.2:
            indent = min(indent + 1, 4)
        elif rng.random() < 0.3:
            indent = max(indent - 1, 0)
    toks.append(EOS)
    return toks


def gen_qa(rng):
    toks = [BOS, QMARK]
    n = rng.randint(4, 9)
    toks += [rng.randint(QA_LO, QA_HI) for _ in range(n)]
    toks.append(AMARK)
    m = rng.randint(4, 9)
    toks += [rng.randint(QA_LO, QA_HI) for _ in range(m)]
    toks.append(EOS)
    return toks


GENERATORS = {"prose": gen_prose, "code": gen_code, "qa": gen_qa}


def gen_one(rng):
    d = rng.choices(DOMAIN_MIX, weights=DOMAIN_W)[0]
    return GENERATORS[d](rng)


def _seqs(seqs, seq_len):
    out = []
    for s in seqs:
        s = list(s)
        if len(s) < seq_len:
            s = s + [PAD] * (seq_len - len(s))
        out.append(s[:seq_len])
    return out


def write_bin(path, rows, seq_len):
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x4E4E4D31, len(rows), seq_len))
        buf = bytearray()
        for r in rows:
            for t in r:
                buf += struct.pack("<H", t)
        f.write(bytes(buf))
    print(f"wrote {path}: {len(rows)} seqs x {seq_len}")


def main():
    seq_len = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    n_val = 512
    seed = 42

    rng = random.Random(seed)
    train = _seqs([gen_one(rng) for _ in range(n_train)], seq_len)
    write_bin("train.bin", train, seq_len)

    for dom, fn in GENERATORS.items():
        r = random.Random(f"{seed}-{dom}")
        rows = _seqs([fn(r) for _ in range(n_val)], seq_len)
        write_bin(f"val_{dom}.bin", rows, seq_len)


if __name__ == "__main__":
    main()