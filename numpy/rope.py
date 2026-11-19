"""
RoPE — rotary position embeddings (Su et al. 2021), what Llama/Qwen/most
modern models use instead of additive positional encodings.

The old way (our sinusoidal PE): ADD a position vector to each embedding
once, at the bottom of the stack. Absolute positions; attention must learn
to decode "position 7 attending to position 5" separately from "position
107 attending to position 105".

RoPE instead ROTATES q and k inside every attention layer. Split each
head's dimensions into pairs; treat pair i as a 2D point; rotate it by
angle (position * theta_i), where theta_i = 10000^(-2i/d) gives each pair
its own wavelength — fast-spinning pairs resolve nearby offsets, slow ones
encode long ranges.

Why this is beautiful: a dot product between a vector rotated by angle a
and one rotated by angle b depends only on (a - b):

    R(m)q . R(n)k  =  q^T R(m)^T R(n) k  =  q^T R(n - m) k

So attention scores automatically depend on RELATIVE position m - n, at
every layer, with zero learned parameters. Values are never rotated — only
the q/k geometry that decides WHO attends to WHOM.

Bonus properties: nothing is added to the residual stream; and it composes
perfectly with the KV cache (cached keys are stored already-rotated; each
new token just gets its own angle).
"""

import numpy as np

from bpe_tokenizer_v2 import BPETokenizer
from embedding_layer import Embedding, softmax
from attention_layer import MultiHeadAttention
from full_model import TinyLLM

rng = np.random.default_rng(11)


# ---------------------------------------------------------------------- #
# The rotation itself
# ---------------------------------------------------------------------- #
def rope_tables(dh: int, max_seq: int, base: float = 10000.0):
    """cos/sin lookup tables, shape (max_seq, dh//2)."""
    inv_freq = base ** (-np.arange(0, dh, 2) / dh)          # (dh/2,)
    angles = np.arange(max_seq)[:, None] * inv_freq[None, :]
    return np.cos(angles), np.sin(angles)


def rope_apply(t: np.ndarray, cos: np.ndarray, sin: np.ndarray,
               positions: np.ndarray) -> np.ndarray:
    """Rotate t (B, H, S, dh) by each position's angles. Half-split layout:
    dims [0:dh/2] and [dh/2:dh] are the (x, y) of each 2D pair."""
    c = cos[positions][None, None]        # (1,1,S,dh/2)
    s = sin[positions][None, None]
    dh2 = t.shape[-1] // 2
    x1, x2 = t[..., :dh2], t[..., dh2:]
    return np.concatenate([x1 * c - x2 * s,
                           x1 * s + x2 * c], axis=-1)


def rope_apply_inverse(t, cos, sin, positions):
    """Rotation matrices are orthogonal: the backward pass through a
    rotation is simply the inverse rotation (transpose = rotate by -angle)."""
    c = cos[positions][None, None]
    s = sin[positions][None, None]
    dh2 = t.shape[-1] // 2
    x1, x2 = t[..., :dh2], t[..., dh2:]
    return np.concatenate([x1 * c + x2 * s,
                           -x1 * s + x2 * c], axis=-1)


# ---------------------------------------------------------------------- #
# Attention with RoPE (q and k rotated; v untouched)
# ---------------------------------------------------------------------- #
class RoPEAttention(MultiHeadAttention):
    def __init__(self, d_model: int, n_heads: int, max_seq: int = 256):
        super().__init__(d_model, n_heads)
        self.cos, self.sin = rope_tables(self.dh, max_seq)

    def forward(self, x: np.ndarray, start_pos: int = 0) -> np.ndarray:
        B, S, D = x.shape
        pos = np.arange(start_pos, start_pos + S)
        q0 = self._split(x @ self.Wq)                 # pre-rotation
        k0 = self._split(x @ self.Wk)
        v = self._split(x @ self.Wv)
        q = rope_apply(q0, self.cos, self.sin, pos)   # <- RoPE happens here
        k = rope_apply(k0, self.cos, self.sin, pos)

        scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(self.dh)
        scores = scores + np.triu(np.full((S, S), -1e9), k=1)
        p = softmax(scores)
        heads = p @ v
        out = self._join(heads) @ self.Wo
        self._cache = (x, q, k, v, p, heads, pos)
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        x, q, k, v, p, heads, pos = self._cache

        joined = self._join(heads)
        self.grads["Wo"] += np.einsum("bsd,bse->de", joined, d_out)
        d_heads = self._split(d_out @ self.Wo.T)

        d_p = d_heads @ v.transpose(0, 1, 3, 2)
        d_v = p.transpose(0, 1, 3, 2) @ d_heads
        d_scores = p * (d_p - (d_p * p).sum(axis=-1, keepdims=True))
        d_scores /= np.sqrt(self.dh)
        d_q = d_scores @ k
        d_k = d_scores.transpose(0, 1, 3, 2) @ q

        # NEW versus the parent class: undo the rotation before the
        # gradients flow into the (unrotated) projections.
        d_q = rope_apply_inverse(d_q, self.cos, self.sin, pos)
        d_k = rope_apply_inverse(d_k, self.cos, self.sin, pos)

        d_q, d_k, d_v = map(self._join, (d_q, d_k, d_v))
        self.grads["Wq"] += np.einsum("bsd,bse->de", x, d_q)
        self.grads["Wk"] += np.einsum("bsd,bse->de", x, d_k)
        self.grads["Wv"] += np.einsum("bsd,bse->de", x, d_v)
        return d_q @ self.Wq.T + d_k @ self.Wk.T + d_v @ self.Wv.T


# ---------------------------------------------------------------------- #
# Model: TinyLLM with RoPE attention and NO additive positional encoding
# ---------------------------------------------------------------------- #
class RoPELLM(TinyLLM):
    def __init__(self, vocab_size, d_model=32, n_heads=4, d_ff=64,
                 n_layers=2, max_seq=256):
        super().__init__(vocab_size, d_model, n_heads, d_ff, n_layers, max_seq)
        for blk in self.blocks:
            blk.attn = RoPEAttention(d_model, n_heads, max_seq)

    def forward(self, ids: np.ndarray) -> np.ndarray:
        x = self.emb.forward(ids)          # note: no "+ pe" — RoPE replaces it
        for blk in self.blocks:
            x = blk.forward(x)
        self._h = self.final_norm.forward(x)
        return self._h @ self.emb.weight.T
    # backward() is inherited unchanged: the additive PE was a constant, so
    # nothing else in the gradient path differs.


# ---------------------------------------------------------------------- #
# KV-cached generation with RoPE
# ---------------------------------------------------------------------- #
def attn_step_rope(attn: RoPEAttention, x_new, cache, start_pos: int):
    T_new = x_new.shape[1]
    pos = np.arange(start_pos, start_pos + T_new)
    q = rope_apply(attn._split(x_new @ attn.Wq), attn.cos, attn.sin, pos)
    k_new = rope_apply(attn._split(x_new @ attn.Wk), attn.cos, attn.sin, pos)
    v_new = attn._split(x_new @ attn.Wv)

    # Cached keys are stored ALREADY ROTATED at their own positions — each
    # key's angle never changes, so the cache stays valid forever.
    if cache["k"] is None:
        k, v = k_new, v_new
    else:
        k = np.concatenate([cache["k"], k_new], axis=2)
        v = np.concatenate([cache["v"], v_new], axis=2)
    cache["k"], cache["v"] = k, v

    scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(attn.dh)
    S_total = k.shape[2]
    if T_new > 1:
        col = np.arange(S_total)[None, :]
        row_pos = (S_total - T_new) + np.arange(T_new)[:, None]
        scores = scores + np.where(col > row_pos, -1e9, 0.0)
    return attn._join(softmax(scores) @ v) @ attn.Wo


def generate_cached(model: RoPELLM, tok, prompt, n_tokens=30,
                    temperature=0.8, greedy=False, seed=0):
    g = np.random.default_rng(seed)
    ids = tok.encode(prompt)
    caches = [{"k": None, "v": None} for _ in model.blocks]

    def step(new_ids, pos):
        x = model.emb.forward(np.array([new_ids]))
        for blk, cache in zip(model.blocks, caches):
            h = x + attn_step_rope(blk.attn, blk.norm1.forward(x), cache, pos)
            x = h + blk.ffn.forward(blk.norm2.forward(h))
        return (model.final_norm.forward(x) @ model.emb.weight.T)[0, -1]

    logits = step(ids, 0)
    for _ in range(n_tokens):
        nxt = int(np.argmax(logits)) if greedy else \
            int(g.choice(logits.size, p=softmax(logits / temperature)))
        ids.append(nxt)
        logits = step([nxt], pos=len(ids) - 1)
    return tok.decode(ids)


# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    print("=== 1. The relative-position property, numerically ===")
    dh = 8
    cos, sin = rope_tables(dh, 64)
    q = rng.normal(0, 1, (1, 1, 1, dh))
    k = rng.normal(0, 1, (1, 1, 1, dh))
    def score(m, n):
        qm = rope_apply(q, cos, sin, np.array([m]))
        kn = rope_apply(k, cos, sin, np.array([n]))
        return float((qm * kn).sum())
    print(f"  positions ( 3, 1): q.k = {score(3, 1):+.6f}")
    print(f"  positions (10, 8): q.k = {score(10, 8):+.6f}   <- same offset 2")
    print(f"  positions (40,38): q.k = {score(40, 38):+.6f}   <- same offset 2")
    print(f"  positions ( 3, 2): q.k = {score(3, 2):+.6f}   <- offset 1: differs")
    print(f"  unrotated:         q.k = {float((q * k).sum()):+.6f}   <- offset 0\n")

    print("=== 2. Gradient check RoPE attention ===")
    B, S, D, H = 2, 5, 8, 2
    attn = RoPEAttention(D, H, max_seq=16)
    for name in ("Wq", "Wk", "Wv", "Wo"):
        setattr(attn, name, rng.normal(0, 0.4, (D, D)))
    x = rng.normal(0, 1, (B, S, D))
    R = rng.normal(0, 1, (B, S, D))
    loss = lambda: float((attn.forward(x) * R).sum())
    loss(); attn.zero_grad()
    d_x = attn.backward(R)
    eps, worst = 1e-5, 0.0
    for name in ("Wq", "Wk", "Wv", "Wo"):
        W, G = getattr(attn, name), attn.grads[name]
        for _ in range(15):
            i, j = rng.integers(D), rng.integers(D)
            old = W[i, j]
            W[i, j] = old + eps; up = loss()
            W[i, j] = old - eps; dn = loss()
            W[i, j] = old
            num = (up - dn) / (2 * eps)
            worst = max(worst, abs(num - G[i, j]) / (abs(num) + abs(G[i, j]) + 1e-12))
    for _ in range(15):
        b, s, d = rng.integers(B), rng.integers(S), rng.integers(D)
        old = x[b, s, d]
        x[b, s, d] = old + eps; up = loss()
        x[b, s, d] = old - eps; dn = loss()
        x[b, s, d] = old
        num = (up - dn) / (2 * eps)
        worst = max(worst, abs(num - d_x[b, s, d]) / (abs(num) + abs(d_x[b, s, d]) + 1e-12))
    print(f"  worst relative error = {worst:.2e} "
          f"({'PASS' if worst < 1e-5 else 'FAIL'})\n")

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

    print("=== 3. Train with RoPE (sinusoidal-PE model reached ~0.10) ===")
    import embedding_layer, attention_layer, ffn_layer
    embedding_layer.rng = np.random.default_rng(42)
    attention_layer.rng = np.random.default_rng(7)
    ffn_layer.rng = np.random.default_rng(3)
    model = RoPELLM(vocab_size)
    seq, batch, steps, lr = 16, 32, 3000, 0.5
    for step_i in range(steps):
        st = rng.integers(0, len(ids) - seq - 1, size=batch)
        xb = np.stack([ids[s: s + seq] for s in st])
        yb = np.stack([ids[s + 1: s + seq + 1] for s in st])
        logits = model.forward(xb)
        probs = softmax(logits)
        bi, si = np.arange(batch)[:, None], np.arange(seq)[None, :]
        loss = -np.log(probs[bi, si, yb] + 1e-12).mean()
        d = probs.copy(); d[bi, si, yb] -= 1.0; d /= batch * seq
        model.zero_grad(); model.backward(d); model.step(lr)
        if step_i % 1000 == 0 or step_i == steps - 1:
            print(f"step {step_i:>5}  loss {loss:.4f}")

    print("\n=== 4. KV-cached generation with RoPE ===")

    def generate_uncached(prompt, n_tokens):
        ids_g = tok.encode(prompt)
        for _ in range(n_tokens):
            logits = model.forward(np.array([ids_g]))[0, -1]
            ids_g.append(int(np.argmax(logits)))
        return tok.decode(ids_g)

    for prompt in ["the cat", "the queen"]:
        a = generate_uncached(prompt, 25)
        b = generate_cached(model, tok, prompt, n_tokens=25, greedy=True)
        match = "MATCH" if a == b else "MISMATCH"
        print(f"  [{match}] {prompt!r} -> {b!r}")
