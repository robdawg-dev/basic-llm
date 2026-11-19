"""
prepare_fineweb.py — FineWeb-Edu -> data/train.bin + data/val.bin.

A sibling of prepare_real_data.py, not a replacement: that script's approach
(collect every story into a Python list, then encode_batch the whole thing,
then np.array() the whole token stream) is fine at TinyStories' scale
(hundreds of millions of tokens) but doesn't scale to FineWeb-Edu, where even
the smallest pre-shuffled sample config is billions of tokens. Holding that
much text and that many Python ints in memory at once is the wrong shape
regardless of how much RAM you have.

What's different here:

1. STREAMING WRITES. Documents are pulled and encoded in small batches; each
   batch is converted to a uint16 array and appended straight to train.bin /
   val.bin, then discarded. Peak memory is one batch, not the whole corpus.

2. A FIXED TOKEN BUDGET, decided up front (--n_tokens), not "however much the
   sample config happens to contain". FineWeb-Edu's sample-10BT config alone
   is ~10B tokens (counted with GPT-2's tokenizer) — training even one epoch
   over it is many GPU-days on a single desktop card. Pick a budget that
   matches your actual time budget; unlike TinyStories, less than one pass
   over the corpus is normal here (see RUNBOOK.md).

3. TOKENIZER IS A CHOICE, not always "train one". --tokenizer smollm2
   (default) reuses SmolLM2's 49,152-vocab tokenizer — SmolLM2 was itself
   trained substantially on FineWeb-Edu, so its vocabulary is shaped by
   this exact corpus's distribution, unlike GPT-2's 2019 WebText-derived
   one (still available via --tokenizer gpt2). Bonus: 49,152 = 384x128 is
   already a GPU-friendly size, unlike GPT-2's 50,257 (see gpt2ish_fw's
   history in RUNBOOK.md for why that mattered). --tokenizer train fits a
   fresh ByteLevelBPETokenizer instead, but only on a bounded sample of
   documents (--tokenizer_sample_docs), not the whole target corpus — a
   BPE vocab this size doesn't need billions of tokens to stabilize.

Sanity checks before burning GPU-hours (same spirit as prepare_real_data.py):
    uv run python -c "
    import numpy as np
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file('smollm2_tokenizer.json')
    train = np.memmap('data/train.bin', dtype=np.uint16, mode='r')
    print(tok.decode(train[:80].tolist()))"
"""

import argparse
import itertools
import time

import numpy as np
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer, Tokenizer

DATASET = "HuggingFaceFW/fineweb-edu"


TOKENIZER_REPOS = {
    "smollm2": "HuggingFaceTB/SmolLM2-1.7B",
    "gpt2": "gpt2",
}


def build_tokenizer(args, doc_iter):
    if args.tokenizer in TOKENIZER_REPOS:
        repo = TOKENIZER_REPOS[args.tokenizer]
        tok = Tokenizer.from_pretrained(repo)
        eot = tok.token_to_id("<|endoftext|>")
        print(f"using {repo}'s tokenizer: vocab {tok.get_vocab_size():,}, "
              f"eot id {eot}", flush=True)
        return tok, eot

    print(f"training a fresh tokenizer on {args.tokenizer_sample_docs:,} "
          f"sample documents...", flush=True)
    sample = list(itertools.islice(doc_iter, args.tokenizer_sample_docs))
    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(sample, vocab_size=args.vocab_size,
                             special_tokens=["<|endoftext|>"])
    eot = tok.token_to_id("<|endoftext|>")
    print(f"trained tokenizer: vocab {tok.get_vocab_size():,}, eot id {eot}",
          flush=True)
    return tok, eot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="sample-10BT",
                     help="FineWeb-Edu HF config: sample-10BT / "
                          "sample-100BT / sample-350BT / default (full "
                          "~1.3T-token dump) / a specific CC-MAIN-* dump")
    ap.add_argument("--n_tokens", type=int, default=2_000_000_000,
                     help="total token budget, train+val combined")
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--tokenizer", choices=["smollm2", "gpt2", "train"],
                     default="smollm2")
    ap.add_argument("--tokenizer_sample_docs", type=int, default=200_000,
                     help="--tokenizer train only: how many documents to "
                          "fit the fresh BPE on")
    ap.add_argument("--vocab_size", type=int, default=49152,
                     help="--tokenizer train only; kept a multiple of 128 "
                          "(GPU-friendly, see RUNBOOK.md) unless you have "
                          "a reason not to")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--tokenizer_out", default=None,
                     help="default: <tokenizer>_tokenizer.json")
    ap.add_argument("--batch_docs", type=int, default=2000,
                     help="documents encoded per streaming batch")
    args = ap.parse_args()
    if args.tokenizer_out is None:
        args.tokenizer_out = f"{args.tokenizer}_tokenizer.json"

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"streaming {DATASET} [{args.config}]...", flush=True)
    ds = load_dataset(DATASET, name=args.config, split="train",
                       streaming=True)
    doc_iter = (ex["text"] for ex in ds)

    tok, eot = build_tokenizer(args, doc_iter)
    tok.save(args.tokenizer_out)
    print(f"tokenizer -> {args.tokenizer_out}", flush=True)

    # --tokenizer train already consumed tokenizer_sample_docs documents
    # from doc_iter; the main pass continues from wherever that left off
    # (those docs are simply not re-used for training data, which is fine
    # at this scale).

    train_target = round(args.n_tokens * (1 - args.val_frac))
    val_target = args.n_tokens - train_target

    train_path = os.path.join(args.out_dir, "train.bin")
    val_path = os.path.join(args.out_dir, "val.bin")

    total_written = 0
    docs_seen = 0
    t0 = time.time()
    last_print = t0

    with open(train_path, "wb") as train_f, open(val_path, "wb") as val_f:
        while total_written < args.n_tokens:
            batch = list(itertools.islice(doc_iter, args.batch_docs))
            if not batch:
                print("stream exhausted before reaching --n_tokens", flush=True)
                break
            docs_seen += len(batch)

            ids = []
            for enc in tok.encode_batch(batch):
                ids.extend(enc.ids)
                ids.append(eot)
            ids = np.array(ids, dtype=np.uint16)

            remaining = args.n_tokens - total_written
            if len(ids) > remaining:
                ids = ids[:remaining]

            if total_written < train_target:
                train_room = train_target - total_written
                train_f.write(ids[:train_room].tobytes())
                if len(ids) > train_room:
                    val_f.write(ids[train_room:].tobytes())
            else:
                val_f.write(ids.tobytes())

            total_written += len(ids)

            now = time.time()
            if now - last_print > 15:
                pct = 100 * total_written / args.n_tokens
                rate = total_written / (now - t0)
                print(f"  {total_written:,} / {args.n_tokens:,} tokens "
                      f"({pct:.1f}%) | {docs_seen:,} docs | "
                      f"{rate / 1e3:.0f}k tok/s | "
                      f"{(now - t0) / 60:.1f} min elapsed", flush=True)
                last_print = now

    train_tokens = min(total_written, train_target)
    val_tokens = max(0, total_written - train_target)
    print(f"{train_path}: {train_tokens:,} tokens", flush=True)
    print(f"{val_path}:   {val_tokens:,} tokens", flush=True)
    print(f"docs processed: {docs_seen:,}", flush=True)
    print(f"eot_id = {eot}  (pass as --stop_token {eot} when sampling, and "
          f"--tokenizer_json {args.tokenizer_out} instead of --merges)",
          flush=True)


if __name__ == "__main__":
    main()
