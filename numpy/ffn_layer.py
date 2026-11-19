"""
The feed-forward network (SwiGLU variant), from scratch (NumPy).

Pipeline: ids --embedding--> vectors --attention--> context mix --FFN--> ???

What the FFN adds: attention MOVES information between positions but does
only a weighted average — a linear operation over the values. The FFN is
where nonlinear COMPUTATION on each token happens, position by position,
with no cross-token interaction at all. Two-thirds of a modern LLM's
parameters live here, and mechanistic interpretability work suggests this
is where much of the factual "knowledge" is stored as key->value patterns.

The classic 2017 design:      out = ReLU(x @ W1) @ W2
What modern models use (SwiGLU, from Shazeer 2020 "GLU Variants"):

    gate = x @ W_gate            # decides HOW MUCH signal passes
    up   = x @ W_up              # decides WHAT signal passes
    h    = silu(gate) * up       # elementwise gating
    out  = h @ W_down

silu(z) = z * sigmoid(z) — a smooth ReLU. The multiplicative gate lets the
network switch features on/off per token, which empirically beats the plain
MLP at equal parameter count.
"""

import numpy as np

from bpe_tokenizer_v2 import BPETokenizer
from embedding_layer import Embedding, sinusoidal_positions, softmax
from attention_layer import MultiHeadAttention

rng = np.random.default_rng(3)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class SwiGLU:
    def __init__(self, d_model: int, d_ff: int):
        s = 0.02
        self.Wg = rng.normal(0, s, (d_model, d_ff))    # gate projection
        self.Wu = rng.normal(0, s, (d_model, d_ff))    # up projection
        self.Wd = rng.normal(0, s, (d_ff, d_model))    # down projection
        self.grads = {k: np.zeros_like(getattr(self, k))
                      for k in ("Wg", "Wu", "Wd")}

    def forward(self, x: np.ndarray) -> np.ndarray:
        g = x @ self.Wg                    # (B,S,F)
        u = x @ self.Wu                    # (B,S,F)
        sig = sigmoid(g)
        silu_g = g * sig                   # silu(g)
        h = silu_g * u                     # the gating — elementwise
        out = h @ self.Wd                  # (B,S,D)
        self._cache = (x, g, u, sig, silu_g, h)
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        x, g, u, sig, silu_g, h = self._cache

        # out = h @ Wd
        self.grads["Wd"] += np.einsum("bsf,bsd->fd", h, d_out)
        d_h = d_out @ self.Wd.T

        # h = silu(g) * u   (elementwise product: each factor gets the
        # incoming gradient times the OTHER factor)
        d_silu_g = d_h * u
        d_u = d_h * silu_g

        # silu'(z) = sigmoid(z) * (1 + z * (1 - sigmoid(z)))
        d_g = d_silu_g * sig * (1.0 + g * (1.0 - sig))

        # g = x @ Wg ; u = x @ Wu
        self.grads["Wg"] += np.einsum("bsd,bsf->df", x, d_g)
        self.grads["Wu"] += np.einsum("bsd,bsf->df", x, d_u)
        return d_g @ self.Wg.T + d_u @ self.Wu.T

    def zero_grad(self):
        for gr in self.grads.values():
            gr[:] = 0.0

    def step(self, lr: float):
        for name, gr in self.grads.items():
            getattr(self, name).__isub__(lr * gr)


# ---------------------------------------------------------------------- #
# Gradient check
# ---------------------------------------------------------------------- #
def gradient_check():
    B, S, D, F = 2, 4, 8, 16
    ffn = SwiGLU(D, F)
    # Larger init so silu operates in its curved region, not near zero.
    for name in ("Wg", "Wu", "Wd"):
        setattr(ffn, name, rng.normal(0, 0.5, getattr(ffn, name).shape))
    x = rng.normal(0, 1, (B, S, D))
    R = rng.normal(0, 1, (B, S, D))
    loss = lambda: float((ffn.forward(x) * R).sum())

    loss()
    ffn.zero_grad()
    d_x = ffn.backward(R)

    eps, worst = 1e-5, 0.0
    for name in ("Wg", "Wu", "Wd"):
        W, G = getattr(ffn, name), ffn.grads[name]
        for _ in range(20):
            i = rng.integers(W.shape[0]); j = rng.integers(W.shape[1])
            old = W[i, j]
            W[i, j] = old + eps; up = loss()
            W[i, j] = old - eps; dn = loss()
            W[i, j] = old
            num = (up - dn) / (2 * eps)
            worst = max(worst, abs(num - G[i, j]) / (abs(num) + abs(G[i, j]) + 1e-12))
    for _ in range(20):
        b, s, d = rng.integers(B), rng.integers(S), rng.integers(D)
        old = x[b, s, d]
        x[b, s, d] = old + eps; up = loss()
        x[b, s, d] = old - eps; dn = loss()
        x[b, s, d] = old
        num = (up - dn) / (2 * eps)
        worst = max(worst, abs(num - d_x[b, s, d]) / (abs(num) + abs(d_x[b, s, d]) + 1e-12))
    print(f"gradient check: worst relative error = {worst:.2e} "
          f"({'PASS' if worst < 1e-5 else 'FAIL'})\n")


# ---------------------------------------------------------------------- #
# Nearly a full transformer block:
#   h1 = e + attention(e)
#   h2 = h1 + ffn(h1)          <- new
#   logits = h2 @ E.T
# (Still missing: normalization. Clipping remains our crutch.)
# ---------------------------------------------------------------------- #
def train_block_lm(tok, ids, d_model=32, n_heads=4, d_ff=64, seq=8,
                   batch=32, steps=3000, lr=0.15, clip=1.0):
    vocab_size = 256 + len(tok.merges)
    emb = Embedding(vocab_size, d_model)
    attn = MultiHeadAttention(d_model, n_heads)
    ffn = SwiGLU(d_model, d_ff)
    pe = sinusoidal_positions(seq, d_model)

    for step in range(steps):
        starts = rng.integers(0, len(ids) - seq - 1, size=batch)
        xb = np.stack([ids[s: s + seq] for s in starts])
        yb = np.stack([ids[s + 1: s + seq + 1] for s in starts])

        # ---- forward ----
        e = emb.forward(xb) + pe
        h1 = e + attn.forward(e)             # attention sublayer + residual
        h2 = h1 + ffn.forward(h1)            # FFN sublayer + residual
        logits = h2 @ emb.weight.T
        probs = softmax(logits)

        B, S, V = probs.shape
        bi, si = np.arange(B)[:, None], np.arange(S)[None, :]
        loss = -np.log(probs[bi, si, yb] + 1e-12).mean()

        # ---- backward ----
        d_logits = probs.copy()
        d_logits[bi, si, yb] -= 1.0
        d_logits /= B * S

        emb.zero_grad(); attn.zero_grad(); ffn.zero_grad()
        emb.grad += np.einsum("bsv,bsd->vd", d_logits, h2)
        d_h2 = d_logits @ emb.weight
        d_h1 = d_h2 + ffn.backward(d_h2)     # residual: gradient adds
        d_e = d_h1 + attn.backward(d_h1)
        emb.backward(d_e)

        all_grads = ([emb.grad] + list(attn.grads.values())
                     + list(ffn.grads.values()))
        total_norm = np.sqrt(sum((g ** 2).sum() for g in all_grads))
        if total_norm > clip:
            for g in all_grads:
                g *= clip / total_norm

        emb.weight -= lr * emb.grad
        attn.step(lr); ffn.step(lr)

        if step % (steps // 10) == 0 or step == steps - 1:
            print(f"step {step:>5}  loss {loss:.4f}")

    return emb, attn, ffn, pe


if __name__ == "__main__":
    print("=== 1. Prove the SwiGLU backward pass is correct ===")
    gradient_check()

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

    print("=== 2. Train the (norm-less) transformer block ===")
    print("history: bigram 2.39 -> +attention 0.37 -> +FFN ?\n")
    emb, attn, ffn, pe = train_block_lm(tok, ids)

    print("\n=== 3. FFN gate activity: how many features fire per token? ===")
    sample = np.array([tok.encode("the cat sat on the mat")])
    e = emb.forward(sample) + sinusoidal_positions(sample.shape[1], 32)
    h1 = e + attn.forward(e)
    ffn.forward(h1)
    g = ffn._cache[1][0]                     # gate pre-activations (S,F)
    active = (g > 0).mean(axis=1)
    toks = tok.tokens_as_strings(sample[0])
    for t, a in zip(toks, active):
        bar = "#" * int(a * 30)
        print(f"  {t!r:>8}  {a * 100:5.1f}% gates open  {bar}")
