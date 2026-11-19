"""
Masked multi-head self-attention, from scratch (NumPy) — the heart of the LLM.

Pipeline so far:
  text --tokenizer--> ids --embedding--> vectors --ATTENTION--> context-aware vectors

The forward pass, per head:
    Q = x @ Wq        "what am I looking for?"
    K = x @ Wk        "what do I contain?"
    V = x @ Wv        "what do I give to whoever attends to me?"
    scores = Q @ K.T / sqrt(d_head)         similarity of every query to every key
    scores += causal mask                    -inf above the diagonal: no peeking
    P = softmax(scores)                      each row: a probability dist over the past
    out = P @ V                              weighted mixture of past values
Heads run in parallel on slices of the model dimension, get concatenated,
and pass through one final output projection Wo.

Everything here also implements backward() by hand, and we PROVE it is
correct with a finite-difference gradient check before trusting it.
"""

import numpy as np

from bpe_tokenizer_v2 import BPETokenizer
from embedding_layer import Embedding, sinusoidal_positions, softmax

rng = np.random.default_rng(7)


# ---------------------------------------------------------------------- #
# Multi-head masked self-attention
# ---------------------------------------------------------------------- #
class MultiHeadAttention:
    def __init__(self, d_model: int, n_heads: int):
        assert d_model % n_heads == 0
        self.d, self.h = d_model, n_heads
        self.dh = d_model // n_heads
        s = 0.02
        self.Wq = rng.normal(0, s, (d_model, d_model))
        self.Wk = rng.normal(0, s, (d_model, d_model))
        self.Wv = rng.normal(0, s, (d_model, d_model))
        self.Wo = rng.normal(0, s, (d_model, d_model))
        self.grads = {k: np.zeros_like(getattr(self, k))
                      for k in ("Wq", "Wk", "Wv", "Wo")}

    def _split(self, t):   # (B,S,D) -> (B,H,S,dh)
        B, S, _ = t.shape
        return t.reshape(B, S, self.h, self.dh).transpose(0, 2, 1, 3)

    def _join(self, t):    # (B,H,S,dh) -> (B,S,D)
        B, H, S, dh = t.shape
        return t.transpose(0, 2, 1, 3).reshape(B, S, H * dh)

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, S, D = x.shape
        q = self._split(x @ self.Wq)
        k = self._split(x @ self.Wk)
        v = self._split(x @ self.Wv)

        scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(self.dh)  # (B,H,S,S)

        # Causal mask: position i may attend to positions <= i only.
        mask = np.triu(np.full((S, S), -1e9), k=1)
        scores = scores + mask

        p = softmax(scores)          # rows sum to 1; masked cells are ~0
        heads = p @ v                # (B,H,S,dh)
        out = self._join(heads) @ self.Wo

        # Cache for backward.
        self._cache = (x, q, k, v, p, heads)
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """d_out: gradient wrt forward's output. Returns gradient wrt x."""
        x, q, k, v, p, heads = self._cache
        B, S, D = x.shape

        # out = join(heads) @ Wo
        joined = self._join(heads)
        self.grads["Wo"] += np.einsum("bsd,bse->de", joined, d_out)
        d_joined = d_out @ self.Wo.T
        d_heads = self._split(d_joined)                      # (B,H,S,dh)

        # heads = p @ v
        d_p = d_heads @ v.transpose(0, 1, 3, 2)              # (B,H,S,S)
        d_v = p.transpose(0, 1, 3, 2) @ d_heads              # (B,H,S,dh)

        # p = softmax(scores): dS = P * (dP - sum(dP * P, last axis))
        # The mask needs no special handling — masked cells have p ~= 0,
        # so their gradient is annihilated right here.
        d_scores = p * (d_p - (d_p * p).sum(axis=-1, keepdims=True))
        d_scores /= np.sqrt(self.dh)

        # scores = q @ k.T
        d_q = d_scores @ k                                   # (B,H,S,dh)
        d_k = d_scores.transpose(0, 1, 3, 2) @ q             # (B,H,S,dh)

        # Back through the three input projections.
        d_q, d_k, d_v = map(self._join, (d_q, d_k, d_v))     # (B,S,D)
        self.grads["Wq"] += np.einsum("bsd,bse->de", x, d_q)
        self.grads["Wk"] += np.einsum("bsd,bse->de", x, d_k)
        self.grads["Wv"] += np.einsum("bsd,bse->de", x, d_v)
        return d_q @ self.Wq.T + d_k @ self.Wk.T + d_v @ self.Wv.T

    def zero_grad(self):
        for g in self.grads.values():
            g[:] = 0.0

    def step(self, lr: float):
        for name, g in self.grads.items():
            getattr(self, name).__isub__(lr * g)


# ---------------------------------------------------------------------- #
# Gradient check: analytic backward vs finite differences
# ---------------------------------------------------------------------- #
def gradient_check():
    B, S, D, H = 2, 4, 8, 2
    attn = MultiHeadAttention(D, H)
    x = rng.normal(0, 1, (B, S, D))
    R = rng.normal(0, 1, (B, S, D))       # fixed projection => scalar loss
    loss = lambda: float((attn.forward(x) * R).sum())

    loss()                                 # populate cache
    attn.zero_grad()
    d_x = attn.backward(R)                 # d(out*R)/d(out) = R

    eps = 1e-5
    worst = 0.0
    for name in ("Wq", "Wk", "Wv", "Wo"):
        W, G = getattr(attn, name), attn.grads[name]
        for _ in range(20):                # spot-check 20 random entries
            i, j = rng.integers(D), rng.integers(D)
            old = W[i, j]
            W[i, j] = old + eps; up = loss()
            W[i, j] = old - eps; down = loss()
            W[i, j] = old
            num = (up - down) / (2 * eps)
            worst = max(worst, abs(num - G[i, j]) / (abs(num) + abs(G[i, j]) + 1e-12))
    # and a few entries of d_x
    for _ in range(20):
        b, s, d = rng.integers(B), rng.integers(S), rng.integers(D)
        old = x[b, s, d]
        x[b, s, d] = old + eps; up = loss()
        x[b, s, d] = old - eps; down = loss()
        x[b, s, d] = old
        num = (up - down) / (2 * eps)
        worst = max(worst, abs(num - d_x[b, s, d]) / (abs(num) + abs(d_x[b, s, d]) + 1e-12))
    print(f"gradient check: worst relative error = {worst:.2e} "
          f"({'PASS' if worst < 1e-5 else 'FAIL'})\n")


# ---------------------------------------------------------------------- #
# A one-block toy LLM: embedding + attention + residual + tied head
# ---------------------------------------------------------------------- #
def train_attention_lm(tok, ids, d_model=32, n_heads=4, seq=8, batch=32,
                       steps=3000, lr=0.15, clip=1.0):
    vocab_size = 256 + len(tok.merges)
    emb = Embedding(vocab_size, d_model)
    attn = MultiHeadAttention(d_model, n_heads)
    pe = sinusoidal_positions(seq, d_model)

    for step in range(steps):
        starts = rng.integers(0, len(ids) - seq - 1, size=batch)
        xb = np.stack([ids[s: s + seq] for s in starts])
        yb = np.stack([ids[s + 1: s + seq + 1] for s in starts])

        # ---- forward ----
        e = emb.forward(xb) + pe                 # (B,S,D)
        a = attn.forward(e)
        h = e + a                                # residual connection
        logits = h @ emb.weight.T                # tied LM head
        probs = softmax(logits)

        B, S, V = probs.shape
        bi, si = np.arange(B)[:, None], np.arange(S)[None, :]
        loss = -np.log(probs[bi, si, yb] + 1e-12).mean()

        # ---- backward ----
        d_logits = probs.copy()
        d_logits[bi, si, yb] -= 1.0
        d_logits /= B * S

        emb.zero_grad(); attn.zero_grad()
        emb.grad += np.einsum("bsv,bsd->vd", d_logits, h)   # head use of E
        d_h = d_logits @ emb.weight
        d_e = d_h + attn.backward(d_h)          # residual: gradient adds
        emb.backward(d_e)                        # lookup use of E

        # Global-norm gradient clipping — the standard cure for the
        # exploding-gradient blowups attention suffers without layer norm.
        all_grads = [emb.grad] + list(attn.grads.values())
        total_norm = np.sqrt(sum((g ** 2).sum() for g in all_grads))
        if total_norm > clip:
            for g in all_grads:
                g *= clip / total_norm

        emb.weight -= lr * emb.grad
        attn.step(lr)

        if step % (steps // 10) == 0 or step == steps - 1:
            print(f"step {step:>5}  loss {loss:.4f}")

    return emb, attn, pe


def show_attention(tok, emb, attn, pe, text: str, head: int = 0):
    """Print one head's attention pattern as a text heatmap."""
    ids = np.array([tok.encode(text)])
    S = ids.shape[1]
    e = emb.forward(ids) + sinusoidal_positions(S, emb.weight.shape[1])
    attn.forward(e)
    p = attn._cache[4][0, head]                  # (S,S)
    labels = [t.replace(" ", "_") for t in tok.tokens_as_strings(ids[0])]
    w = max(len(l) for l in labels) + 1
    shades = " .:-=+*#%@"
    print(f"attention pattern, head {head} (rows attend to columns):")
    print(" " * w + "".join(l[:6].ljust(7) for l in labels))
    for i in range(S):
        row = "".join(
            (f"{p[i, j]:.2f}".lstrip('0') + " " + shades[min(int(p[i, j] * 9.99), 9)])
            .ljust(7) if j <= i else "  --   "
            for j in range(S)
        )
        print(labels[i].ljust(w) + row)


if __name__ == "__main__":
    print("=== 1. Prove the backward pass is correct ===")
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

    print("=== 2. Train: embedding + attention + residual + tied head ===")
    emb, attn, pe = train_attention_lm(tok, ids)
    print("\nbigram ceiling from last time: ~2.39 — attention should be far below\n")

    print("=== 3. What did attention learn? ===")
    show_attention(tok, emb, attn, pe, "the cat sat on the mat")
