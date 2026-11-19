"""
prepare_data.py — turn text into the train.bin / val.bin files train_gpt.py
memmaps.

DOCUMENT BOUNDARIES (the part that matters for real corpora):
Training samples random windows from one long token stream. If that stream
is many concatenated documents (TinyStories: millions of separate stories),
windows will straddle documents — the tail of one story followed by the
head of an unrelated one. Left unmarked, the model wastes capacity trying
to predict across a seam that is genuinely unpredictable, and it never
learns that stories END.

The fix is one token: append an end-of-text (EOT) id after every document
(encode_documents below). Windows still cross boundaries, but now the
model SEES the boundary, learns "attend less to what precedes EOT", and
learns to EMIT it when a story finishes — which generate(stop_token=...)
turns into clean termination.

Conventions: GPT-2 uses <|endoftext|> as the LAST vocab id. For a 32768
tokenizer, train BPE with special_tokens=["<|endoftext|>"] (HF tokenizers
puts it at id 0) or reserve the top id yourself — either works, just be
consistent between prep and sampling.

SMOKE TEST (runs now): Shakespeare via our own BPE ids. It's one continuous
document, so the default path writes it EOT-free, same as before.

REAL RUN (vocab 32768):
    pip install tokenizers datasets
    from datasets import load_dataset
    from tokenizers import ByteLevelBPETokenizer
    ds = load_dataset("roneneldan/TinyStories")
    tok = ByteLevelBPETokenizer()
    tok.train(files=["tinystories.txt"], vocab_size=32768,
              special_tokens=["<|endoftext|>"])       # -> id 0
    eot = tok.token_to_id("<|endoftext|>")
    ids = encode_documents((s["text"] for s in ds["train"]),
                           lambda t: tok.encode(t).ids, eot)
    write_bins(ids, "data")
"""

import os

import numpy as np


def encode_documents(docs, encode_fn, eot_id) -> np.ndarray:
    """Encode each document and append eot_id after it, producing the
    single token stream train_gpt.py samples windows from."""
    out = []
    for d in docs:
        out.extend(encode_fn(d))
        out.append(eot_id)
    return np.array(out)


def write_bins(ids: np.ndarray, out_dir: str, val_frac: float = 0.05):
    os.makedirs(out_dir, exist_ok=True)
    assert ids.max() < 65536, "uint16 storage requires vocab < 65536"
    split = int(len(ids) * (1 - val_frac))
    ids[:split].astype(np.uint16).tofile(os.path.join(out_dir, "train.bin"))
    ids[split:].astype(np.uint16).tofile(os.path.join(out_dir, "val.bin"))
    print(f"{out_dir}/train.bin: {split:,} tokens")
    print(f"{out_dir}/val.bin:   {len(ids) - split:,} tokens")


def main():
    ids = np.load("shake_ids.npy")     # one continuous document: no EOT
    write_bins(ids, "data")
    print("\nready: uv run train-gpt --preset tiny")


if __name__ == "__main__":
    main()
