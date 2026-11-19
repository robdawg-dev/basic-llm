"""Resumable Shakespeare training: each run continues from the checkpoint.
Adam's m/v moments and step count are saved too — resuming with fresh
moments would momentarily destabilize training, so the optimizer state is
part of the checkpoint, exactly as in real training frameworks."""
import json, os, sys, time
import numpy as np
from bpe_tokenizer_v2 import BPETokenizer
from embedding_layer import softmax
from gqa import GQALLM
from adam import AdamW, collect_params
from lr_schedule import lr_schedule
from checkpoint import save_checkpoint, load_checkpoint
from paths import data_path, out_path

CHUNK = int(sys.argv[1]) if len(sys.argv) > 1 else 500
VOCAB, D, H, KV, FF, LAYERS, SEQ = 512, 64, 4, 2, 128, 3, 64
BATCH, STEPS, WARMUP, PEAK_LR = 24, 2500, 150, 0.02

ids = np.load(data_path("shake_ids.npy"))
split = int(len(ids) * 0.95)
train_ids, val_ids = ids[:split], ids[split:]

if os.path.exists(out_path("shake_ckpt.json")):
    model, tok, extra = load_checkpoint(out_path("shake_ckpt"))
    start = extra["step"]
    opt = AdamW(collect_params(model), lr=0.0)
    od = np.load(out_path("shake_opt.npz"))
    opt.m = [od[f"m{i}"] for i in range(len(opt.m))]
    opt.v = [od[f"v{i}"] for i in range(len(opt.v))]
    opt.t = extra["opt_t"]
    print(f"resumed from step {start}")
else:
    tok = BPETokenizer()
    for i, (a, b) in enumerate(json.load(open(data_path("shake_merges.json")))):
        tok.merges[(a, b)] = 256 + i
        tok.vocab[256 + i] = tok.vocab[a] + tok.vocab[b]
    model = GQALLM(VOCAB, n_kv_heads=KV, d_model=D, n_heads=H, d_ff=FF,
                   n_layers=LAYERS, max_seq=256)
    opt = AdamW(collect_params(model), lr=0.0)
    start = 0
    n = sum(p.size for _, p, _ in collect_params(model))
    print(f"fresh start: {n:,} params, baseline ln(512)={np.log(512):.3f}")

rng = np.random.default_rng(1000 + start)   # different batches per chunk
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

vr = np.random.default_rng(99)
VAL = [vr.integers(0, len(val_ids) - SEQ - 1, size=BATCH) for _ in range(4)]

end = min(start + CHUNK, STEPS)
t0 = time.perf_counter()
for step in range(start, end):
    opt.lr = lr_schedule(step, STEPS, PEAK_LR, WARMUP)
    st = rng.integers(0, len(train_ids) - SEQ - 1, size=BATCH)
    tl = batch_loss(train_ids, st, backward=True)
    opt.step()
    if step % 250 == 0 or step == end - 1:
        vl = float(np.mean([batch_loss(val_ids, s) for s in VAL]))
        print(f"step {step:>5}  train {tl:.3f}  val {vl:.3f}  "
              f"lr {opt.lr:.4f}  [{(time.perf_counter()-t0)/60:.1f} min]",
              flush=True)

save_checkpoint(model, tok, out_path("shake_ckpt"), extra={"step": end, "opt_t": opt.t})
np.savez(out_path("shake_opt.npz"),
         **{f"m{i}": m for i, m in enumerate(opt.m)},
         **{f"v{i}": v for i, v in enumerate(opt.v)})
print(f"checkpoint saved at step {end} ({end}/{STEPS})")

if end >= STEPS:
    print("\n=== TRAINING COMPLETE — FINAL SAMPLES ===")
    def sample(prompt, n=140, temperature=0.8, seed=1):
        g = np.random.default_rng(seed)
        out = tok.encode(prompt)
        for _ in range(n):
            logits = model.forward(np.array([out[-SEQ:]]))[0, -1]
            p = softmax(logits / temperature)
            out.append(int(g.choice(len(p), p=p)))
        return tok.decode(out)
    for prompt in ["ROMEO:", "First Citizen:\n"]:
        print(f"--- prompt {prompt!r} ---")
        print(sample(prompt), "\n", flush=True)
