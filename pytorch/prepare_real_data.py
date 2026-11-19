"""
prepare_real_data.py — TinyStories -> data/train.bin + data/val.bin, using a
fast Rust-backed BPE tokenizer instead of the hand-built one in
numpy/bpe_tokenizer_v2.py.

Why not the hand-built tokenizer: its train() recomputes pair counts over
every unique chunk on every merge (no incremental updates). That's fine for
Shakespeare's 256-merge toy tokenizer, but a 32,768-vocab tokenizer (32.5k
merges) over a real corpus is the wrong algorithmic shape for it — likely
hours-to-days rather than minutes. tokenizers.ByteLevelBPETokenizer trains
and encodes the same corpus in seconds to minutes.

Consequence for sampling: train_gpt.py's --merges/load_bpe only understands
the hand-built tokenizer's format (raw bytes 0-255 + an ordered merge list,
same as shake_merges.json). A checkpoint trained on data from this script
must be sampled with --tokenizer_json pointing at the tokenizer.json this
script saves, not --merges.

Uses HF `datasets` in streaming mode so only the requested story count is
pulled over the network, not the full ~2M-story corpus — useful for a
smaller first real run before committing to the whole thing.
"""

import argparse
import itertools

import numpy as np
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

from prepare_data import write_bins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_stories", type=int, default=300_000,
                     help="how many TinyStories train examples to use")
    ap.add_argument("--vocab_size", type=int, default=32768)
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--tokenizer_out", default="tinystories_tokenizer.json")
    args = ap.parse_args()

    print(f"streaming {args.n_stories:,} TinyStories examples...", flush=True)
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    stories = [ex["text"] for ex in itertools.islice(ds, args.n_stories)]
    print(f"loaded {len(stories):,} stories, "
          f"{sum(len(s) for s in stories):,} chars", flush=True)

    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(stories, vocab_size=args.vocab_size,
                             special_tokens=["<|endoftext|>"])
    eot = tok.token_to_id("<|endoftext|>")
    print(f"trained tokenizer: vocab {tok.get_vocab_size():,}, eot id {eot}",
          flush=True)

    ids = []
    for enc in tok.encode_batch(stories):
        ids.extend(enc.ids)
        ids.append(eot)
    ids = np.array(ids)

    write_bins(ids, args.out_dir)
    tok.save(args.tokenizer_out)
    print(f"tokenizer -> {args.tokenizer_out}", flush=True)
    print(f"eot_id = {eot}  (pass as --stop_token {eot} when sampling, and "
          f"--tokenizer_json {args.tokenizer_out} instead of --merges)",
          flush=True)


if __name__ == "__main__":
    main()
