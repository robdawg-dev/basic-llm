"""
Graduation: the full stack on real text (tiny Shakespeare, 1.1 MB).

Everything we built, working together: BPE tokenizer (regex + heap),
embeddings, RoPE, GQA attention, SwiGLU, RMSNorm, pre-norm blocks, tied
head, AdamW, warmup+cosine schedule, checkpointing. Plus one new
discipline that only matters on real data: a held-out VALIDATION split.
On the toy corpus, train loss was the whole story — the data was the
pattern. Real text can be memorized, so we watch loss on 5% of the corpus
the model never trains on.
"""

import json
import sys
import time

import numpy as np

from bpe_tokenizer_v2 import BPETokenizer
from embedding_layer import softmax
from gqa import GQALLM
from adam import AdamW, collect_params
from lr_schedule import lr_schedule
from checkpoint import save_checkpoint
from paths import data_path, out_path

rng = np.random.default_rng(0)

# ---- data ------------------------------------------------------------ #
ids = np.load(data_path("shake_ids.npy"))
tok = BPETokenizer()
for i, (a, b) in enumerate(json.load(open(data_path("shake_merges.json")))):
    tok.merges[(a, b)] = 256 + i
    tok.vocab[256 + i] = tok.vocab[a] + tok.vocab[b]

split = int(len(ids) * 0.95)
train_ids, val_ids = ids[:split], ids[split:]
print(f"tokens: {len(train_ids):,} train / {len(val_ids):,} val", flush=True)

# ---- model ----------------------------------------------------------- #
VOCAB, D, H, KV, FF, LAYERS, SEQ = 512, 64, 4, 2, 128, 3, 64
BATCH, STEPS, WARMUP, PEAK_LR = 24, 2500, 150, 0.02
model = GQALLM(VOCAB, n_kv_heads=KV, d_model=D, n_heads=H, d_ff=FF,
               n_layers=LAYERS, max_seq=256)
opt = AdamW(collect_params(model), lr=0.0)
n_params = sum(p.size for _, p, _ in collect_params(model))
print(f"model: {n_params:,} params | vocab {VOCAB} | seq {SEQ} | "
      f"{LAYERS} layers | GQA {H}q/{KV}kv", flush=True)
print(f"uniform baseline loss = ln({VOCAB}) = {np.log(VOCAB):.3f}", flush=True)

bi, si = np.arange(BATCH)[:, None], np.arange(SEQ)[None, :]

def batch_loss(data, starts, backward=False):
    xb = np.stack([data[s: s + SEQ] for s in starts])
    yb = np.stack([data[s + 1: s + SEQ + 1] for s in starts])
    probs = softmax(model.forward(xb))
    loss = -np.log(probs[bi, si, yb] + 1e-12).mean()
    if backward:
        d = probs.copy(); d[bi, si, yb] -= 1.0; d /= BATCH * SEQ
        model.zero_grad(); model.backward(d)
    return float(loss)

val_rng = np.random.default_rng(99)
VAL_STARTS = [val_rng.integers(0, len(val_ids) - SEQ - 1, size=BATCH)
              for _ in range(4)]                    # fixed for comparability

def val_loss():
    return float(np.mean([batch_loss(val_ids, s) for s in VAL_STARTS]))

def sample(prompt, n=120, temperature=0.8, seed=1):
    g = np.random.default_rng(seed)
    out = tok.encode(prompt)
    for _ in range(n):
        logits = model.forward(np.array([out[-SEQ:]]))[0, -1]
        p = softmax(logits / temperature)
        out.append(int(g.choice(len(p), p=p)))
    return tok.decode(out)

# ---- train ----------------------------------------------------------- #
t0 = time.perf_counter()
for step in range(STEPS):
    opt.lr = lr_schedule(step, STEPS, PEAK_LR, WARMUP)
    starts = rng.integers(0, len(train_ids) - SEQ - 1, size=BATCH)
    tl = batch_loss(train_ids, starts, backward=True)
    opt.step()

    if step % 250 == 0 or step == STEPS - 1:
        vl = val_loss()
        mins = (time.perf_counter() - t0) / 60
        print(f"step {step:>5}  train {tl:.3f}  val {vl:.3f}  "
              f"lr {opt.lr:.4f}  [{mins:.1f} min]", flush=True)
    if step > 0 and (step % 1000 == 0 or step == STEPS - 1):
        save_checkpoint(model, tok, out_path("shake_ckpt"),
                        extra={"step": step, "val_loss": vl})
        print(f"  checkpoint saved at step {step}", flush=True)
        print(f"  sample: {sample('ROMEO:', n=60)!r}", flush=True)

print("\n=== FINAL SAMPLES ===", flush=True)
for prompt in ["ROMEO:", "First Citizen:\n", "QUEEN "]:
    print(f"--- prompt {prompt!r} ---", flush=True)
    print(sample(prompt, n=150, temperature=0.8), flush=True)
    print(flush=True)
print("done.", flush=True)
