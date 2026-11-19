"""
Decoding strategies — the last mile between logits and text.

The model gives us a probability distribution over 512 tokens. How we pick
from it changes the text more than most architecture decisions:

  GREEDY (argmax). Deterministic, locally optimal, globally awful: falls
  into loops, because once a phrase repeats, repeating it again is usually
  the single most likely continuation. Self-reinforcing.

  TEMPERATURE. Divide logits by T before softmax. T<1 sharpens (confident,
  conservative, drifts toward greedy), T>1 flattens (diverse, chaotic).
  A global knob: it rescales EVERYTHING, including the junk tail.

  TOP-K. Keep the k most likely tokens, renormalize. Fixed k is wrong in
  both directions: after "ROMEO" a colon is ~certain (k=40 lets in 39
  junk options); mid-sentence, dozens of words are plausible (k=5 chokes).

  TOP-P / NUCLEUS (Holtzman et al. 2019). Keep the smallest set of tokens
  whose probabilities sum to p (e.g. 0.9), renormalize. ADAPTIVE: the
  nucleus is tiny when the model is confident and wide when it isn't —
  exactly the behavior top-k can't provide.

  REPETITION PENALTY (CTRL, Keskar et al. 2019). Before sampling, penalize
  the logits of tokens already in the context: positive logits divided by
  the penalty, negative multiplied. Directly counteracts the loop-lock.

All of these are INFERENCE-TIME choices — same weights, radically
different text. This file runs them all against our trained Shakespeare
checkpoint.
"""

import numpy as np

from embedding_layer import softmax
from checkpoint import load_checkpoint
from paths import out_path


def adjust_for_repetition(logits, context_ids, penalty):
    """CTRL-style: seen tokens get their logits pushed toward 'unlikely'.
    Division/multiplication (not subtraction) so the effect scales with
    the logit's magnitude and works on both signs."""
    out = logits.copy()
    for t in set(context_ids):
        out[t] = out[t] / penalty if out[t] > 0 else out[t] * penalty
    return out


def top_k_filter(logits, k):
    out = np.full_like(logits, -np.inf)
    idx = np.argpartition(logits, -k)[-k:]
    out[idx] = logits[idx]
    return out


def top_p_filter(logits, p):
    """Keep the smallest prefix of the sorted distribution reaching mass p."""
    probs = softmax(logits)
    order = np.argsort(-probs)
    csum = np.cumsum(probs[order])
    cutoff = int(np.searchsorted(csum, p)) + 1     # include the crosser
    out = np.full_like(logits, -np.inf)
    out[order[:cutoff]] = logits[order[:cutoff]]
    return out, cutoff


def generate(model, tok, prompt, n=150, temperature=1.0, top_k=None,
             top_p=None, repetition_penalty=None, greedy=False, seed=0,
             seq_window=64, trace_nucleus=False):
    g = np.random.default_rng(seed)
    ids = tok.encode(prompt)
    nucleus_sizes = []
    for step in range(n):
        logits = model.forward(np.array([ids[-seq_window:]]))[0, -1]
        if repetition_penalty:
            logits = adjust_for_repetition(logits, ids[-seq_window:],
                                           repetition_penalty)
        if greedy:
            ids.append(int(np.argmax(logits)))
            continue
        logits = logits / temperature
        if top_k:
            logits = top_k_filter(logits, top_k)
        if top_p:
            logits, cut = top_p_filter(logits, top_p)
            nucleus_sizes.append(cut)
        p = softmax(logits)
        ids.append(int(g.choice(len(p), p=p)))
    text = tok.decode(ids)
    return (text, nucleus_sizes) if trace_nucleus else text


def repeated_4gram_frac(text):
    """Fraction of 4-grams (whitespace tokens) that already occurred —
    a crude but effective loopiness meter."""
    w = text.split()
    grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    if not grams:
        return 0.0
    seen, rep = set(), 0
    for gm in grams:
        rep += gm in seen
        seen.add(gm)
    return rep / len(grams)


if __name__ == "__main__":
    model, tok, extra = load_checkpoint(out_path("shake_ckpt"))
    print(f"loaded checkpoint from step {extra['step']}\n")
    P = "ROMEO:"

    print("=" * 62)
    print("1. GREEDY — watch it lock into a loop")
    print("=" * 62)
    t = generate(model, tok, P, n=120, greedy=True)
    print(t)
    print(f"\n[repeated 4-gram fraction: {repeated_4gram_frac(t):.2f}]\n")

    print("=" * 62)
    print("2. GREEDY + REPETITION PENALTY 1.3 — the loop breaker")
    print("=" * 62)
    t = generate(model, tok, P, n=120, greedy=True, repetition_penalty=1.3)
    print(t)
    print(f"\n[repeated 4-gram fraction: {repeated_4gram_frac(t):.2f}]\n")

    print("=" * 62)
    print("3. TEMPERATURE — one knob, three personalities")
    print("=" * 62)
    for T in (0.3, 0.8, 1.4):
        t = generate(model, tok, P, n=70, temperature=T, seed=3)
        print(f"--- T = {T} ---")
        print(t, "\n")

    print("=" * 62)
    print("4. TOP-P 0.9 — and the nucleus adapting per step")
    print("=" * 62)
    t, sizes = generate(model, tok, P, n=120, temperature=0.9, top_p=0.9,
                        seed=3, trace_nucleus=True)
    print(t)
    sizes = np.array(sizes)
    print(f"\n[nucleus size out of 512 tokens: min {sizes.min()}, "
          f"median {int(np.median(sizes))}, max {sizes.max()}]")
    print(f"[repeated 4-gram fraction: {repeated_4gram_frac(t):.2f}]\n")

    print("=" * 62)
    print("5. THE PRODUCTION COMBO — top-p 0.9, T 0.9, rep penalty 1.15")
    print("=" * 62)
    t = generate(model, tok, P, n=150, temperature=0.9, top_p=0.9,
                 repetition_penalty=1.15, seed=7)
    print(t)
    print(f"\n[repeated 4-gram fraction: {repeated_4gram_frac(t):.2f}]")
