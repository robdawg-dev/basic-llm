"""
Learning-rate schedule: linear warmup + cosine decay to a floor.

Why every real LLM run uses this shape:

  WARMUP (first ~1-2% of steps). At step 0, Adam's m and v estimates are
  built from a handful of noisy gradients, and the model is at a random
  init where gradients are large and weird. A full-size step here can
  wreck the run. So lr ramps linearly from ~0 to peak_lr.

  COSINE DECAY (the rest). With a constant lr, minibatch noise puts a
  floor on the loss: the model orbits the minimum in a "noise ball" whose
  radius scales with lr. Decaying lr shrinks the orbit, letting the model
  settle. Cosine is the conventional shape — smooth, flat near the start
  (fast progress while loss is high) and flat again near the end (gentle
  landing). Decaying to ~10% of peak rather than 0 is common practice.

The practical consequence: a schedule makes HIGHER peak learning rates
usable. Fast early progress from the big lr, stability from the warmup,
and a low final loss from the decay — better than any constant lr.

    lr(t) = peak * (t+1)/warmup                          t < warmup
    lr(t) = floor + (peak-floor) * 0.5*(1+cos(pi*u))     u = progress in [0,1]
"""

import numpy as np

from bpe_tokenizer_v2 import BPETokenizer
from adam import AdamW, collect_params, make_model, train
from paths import out_path


def lr_schedule(step, max_steps, peak_lr, warmup_steps, floor_frac=0.1):
    floor = peak_lr * floor_frac
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return floor + (peak_lr - floor) * 0.5 * (1.0 + np.cos(np.pi * progress))


def scheduled_update(opt, max_steps, peak_lr, warmup_steps):
    def update(model):
        opt.lr = lr_schedule(opt.t, max_steps, peak_lr, warmup_steps)
        opt.step()
    return update


if __name__ == "__main__":
    corpus = (
        "the cat sat on the mat . the dog sat on the rug . "
        "the cat ran to the door . the dog ran to the gate . "
        "a cat is an animal . a dog is an animal . "
        "the cat ate the food . the dog ate the food . "
        "the king ruled the land . the queen ruled the land . "
        "the king wore a crown . the queen wore a crown . "
        "one plus one is two . two plus two is four . "
    ) * 20
    tok = BPETokenizer()
    tok.train(corpus, vocab_size=256 + 80)
    ids = np.array(tok.encode(corpus))
    vocab_size = 256 + len(tok.merges)
    STEPS, WARMUP = 1500, 100

    runs = {}

    print("=== A. Constant lr = 0.01 (yesterday's baseline) ===")
    m = make_model(vocab_size)
    opt = AdamW(collect_params(m), lr=0.01)
    runs["constant 0.01"] = train(m, ids, STEPS,
                                  update_fn=lambda mm: opt.step(),
                                  log_every=500)

    print("\n=== B. Constant lr = 0.05 (5x higher, no protection) ===")
    m = make_model(vocab_size)
    opt = AdamW(collect_params(m), lr=0.05)
    runs["constant 0.05"] = train(m, ids, STEPS,
                                  update_fn=lambda mm: opt.step(),
                                  log_every=500)

    print("\n=== C. Warmup + cosine, peak lr = 0.05 ===")
    m = make_model(vocab_size)
    opt = AdamW(collect_params(m), lr=0.0)
    runs["warmup+cosine 0.05"] = train(
        m, ids, STEPS,
        update_fn=scheduled_update(opt, STEPS, peak_lr=0.05,
                                   warmup_steps=WARMUP),
        log_every=500)

    print("\n=== Verdict (mean loss over final 100 steps) ===")
    for name, losses in runs.items():
        print(f"  {name:>20}: {np.mean(losses[-100:]):.4f}")

    for name, losses in runs.items():
        np.save(out_path(f"lrs_{name.replace(' ', '_').replace('+', '_')}.npy"),
                np.array(losses))
