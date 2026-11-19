"""
GQA — grouped-query attention (Ainslie et al. 2023), plus its extreme
case MQA, multi-query attention (Shazeer 2019).

The observation: at inference, the KV cache dominates memory —
2 * layers * positions * n_kv_heads * dh floats per sequence — and memory
bandwidth spent reading that cache dominates decode latency. Queries are
never cached; only K and V are. So:

    MHA:  H query heads, H kv heads       (what we've built so far)
    GQA:  H query heads, G kv heads,      G < H; each kv head serves H/G
                                          query heads (a "group")
    MQA:  H query heads, 1 kv head        (all queries share one K/V)

Cache shrinks by H/G with almost no quality loss for moderate G — the
queries stay diverse (each head still asks its own questions); they just
share the library they read from. Llama-2/3 70B: 64 query heads, 8 kv
heads — an 8x cache reduction.

Implementation notes:
- Wq stays (D, D) -> H heads. Wk, Wv shrink to (D, G*dh) -> G heads.
  Fewer parameters too, as a side effect.
- At compute time the G kv heads are broadcast (np.repeat) across their
  query groups. The SAVING is in the cache/stored tensors, not FLOPs.
- Backward through a repeat = SUM the gradients over each group.
"""

import numpy as np

from bpe_tokenizer_v2 import BPETokenizer
from embedding_layer import softmax
from rope import RoPELLM, rope_tables, rope_apply, rope_apply_inverse
from adam import AdamW, collect_params, train
from lr_schedule import scheduled_update

rng = np.random.default_rng(11)


class GQAAttention:
    """Grouped-query attention with RoPE. n_kv_heads=H gives plain MHA,
    n_kv_heads=1 gives MQA."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 max_seq: int = 256):
        assert d_model % n_heads == 0 and n_heads % n_kv_heads == 0
        self.d, self.h, self.g = d_model, n_heads, n_kv_heads
        self.dh = d_model // n_heads
        self.rep = n_heads // n_kv_heads          # queries per kv head
        s = 0.02
        self.Wq = rng.normal(0, s, (d_model, d_model))
        self.Wk = rng.normal(0, s, (d_model, self.g * self.dh))   # smaller!
        self.Wv = rng.normal(0, s, (d_model, self.g * self.dh))   # smaller!
        self.Wo = rng.normal(0, s, (d_model, d_model))
        self.grads = {k: np.zeros_like(getattr(self, k))
                      for k in ("Wq", "Wk", "Wv", "Wo")}
        self.cos, self.sin = rope_tables(self.dh, max_seq)

    def _split(self, t, n):    # (B,S,n*dh) -> (B,n,S,dh)
        B, S, _ = t.shape
        return t.reshape(B, S, n, self.dh).transpose(0, 2, 1, 3)

    def _join(self, t):        # (B,n,S,dh) -> (B,S,n*dh)
        B, n, S, dh = t.shape
        return t.transpose(0, 2, 1, 3).reshape(B, S, n * dh)

    def forward(self, x: np.ndarray, start_pos: int = 0) -> np.ndarray:
        B, S, D = x.shape
        pos = np.arange(start_pos, start_pos + S)
        q = rope_apply(self._split(x @ self.Wq, self.h), self.cos, self.sin, pos)
        k = rope_apply(self._split(x @ self.Wk, self.g), self.cos, self.sin, pos)
        v = self._split(x @ self.Wv, self.g)

        # Broadcast each kv head to its group of query heads.
        k_rep = np.repeat(k, self.rep, axis=1)     # (B,H,S,dh)
        v_rep = np.repeat(v, self.rep, axis=1)

        scores = q @ k_rep.transpose(0, 1, 3, 2) / np.sqrt(self.dh)
        scores = scores + np.triu(np.full((S, S), -1e9), k=1)
        p = softmax(scores)
        heads = p @ v_rep
        out = self._join(heads) @ self.Wo
        self._cache = (x, q, k_rep, v_rep, p, heads, pos)
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        x, q, k_rep, v_rep, p, heads, pos = self._cache
        B = x.shape[0]

        joined = self._join(heads)
        self.grads["Wo"] += np.einsum("bsd,bse->de", joined, d_out)
        d_heads = self._split(d_out @ self.Wo.T, self.h)

        d_p = d_heads @ v_rep.transpose(0, 1, 3, 2)
        d_v_rep = p.transpose(0, 1, 3, 2) @ d_heads
        d_scores = p * (d_p - (d_p * p).sum(axis=-1, keepdims=True))
        d_scores /= np.sqrt(self.dh)
        d_q = d_scores @ k_rep
        d_k_rep = d_scores.transpose(0, 1, 3, 2) @ q

        # Backward through np.repeat: each kv head received copies of the
        # gradient from every query head in its group -> SUM them.
        shp = (B, self.g, self.rep) + d_k_rep.shape[2:]
        d_k = d_k_rep.reshape(shp).sum(axis=2)     # (B,G,S,dh)
        d_v = d_v_rep.reshape(shp).sum(axis=2)

        d_q = rope_apply_inverse(d_q, self.cos, self.sin, pos)
        d_k = rope_apply_inverse(d_k, self.cos, self.sin, pos)

        d_q, d_k, d_v = self._join(d_q), self._join(d_k), self._join(d_v)
        self.grads["Wq"] += np.einsum("bsd,bse->de", x, d_q)
        self.grads["Wk"] += np.einsum("bsd,bse->de", x, d_k)
        self.grads["Wv"] += np.einsum("bsd,bse->de", x, d_v)
        return d_q @ self.Wq.T + d_k @ self.Wk.T + d_v @ self.Wv.T

    def zero_grad(self):
        for g in self.grads.values():
            g[:] = 0.0

    def step(self, lr):
        for name, g in self.grads.items():
            getattr(self, name).__isub__(lr * g)


class GQALLM(RoPELLM):
    def __init__(self, vocab_size, n_kv_heads, d_model=32, n_heads=4,
                 d_ff=64, n_layers=2, max_seq=256):
        super().__init__(vocab_size, d_model, n_heads, d_ff, n_layers, max_seq)
        for blk in self.blocks:
            blk.attn = GQAAttention(d_model, n_heads, n_kv_heads, max_seq)


def kv_cache_bytes(model, positions):
    blk = model.blocks[0].attn
    g = getattr(blk, "g", blk.h)          # RoPEAttention has no .g -> full H
    return 2 * len(model.blocks) * positions * g * blk.dh * 8


if __name__ == "__main__":
    print("=== 1. Gradient check (G=2 of H=4, the interesting case) ===")
    B, S, D, H, G = 2, 5, 8, 4, 2
    attn = GQAAttention(D, H, G, max_seq=16)
    attn.Wq = rng.normal(0, 0.4, attn.Wq.shape)
    attn.Wk = rng.normal(0, 0.4, attn.Wk.shape)
    attn.Wv = rng.normal(0, 0.4, attn.Wv.shape)
    attn.Wo = rng.normal(0, 0.4, attn.Wo.shape)
    x = rng.normal(0, 1, (B, S, D))
    R = rng.normal(0, 1, (B, S, D))
    loss = lambda: float((attn.forward(x) * R).sum())
    loss(); attn.zero_grad()
    d_x = attn.backward(R)
    eps, worst = 1e-5, 0.0
    for name in ("Wq", "Wk", "Wv", "Wo"):
        W, Gr = getattr(attn, name), attn.grads[name]
        for _ in range(15):
            i = rng.integers(W.shape[0]); j = rng.integers(W.shape[1])
            old = W[i, j]
            W[i, j] = old + eps; up = loss()
            W[i, j] = old - eps; dn = loss()
            W[i, j] = old
            num = (up - dn) / (2 * eps)
            worst = max(worst, abs(num - Gr[i, j]) / (abs(num) + abs(Gr[i, j]) + 1e-12))
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
    STEPS = 1500

    def fresh(n_kv):
        import embedding_layer, attention_layer, ffn_layer
        embedding_layer.rng = np.random.default_rng(42)
        attention_layer.rng = np.random.default_rng(7)
        ffn_layer.rng = np.random.default_rng(3)
        global rng
        rng = np.random.default_rng(5)          # GQA weight init stream
        return GQALLM(vocab_size, n_kv_heads=n_kv)

    results = {}
    for n_kv, label in [(4, "MHA  (4 kv heads)"),
                        (2, "GQA  (2 kv heads)"),
                        (1, "MQA  (1 kv head) ")]:
        print(f"=== 2. Train {label}, AdamW + warmup/cosine ===")
        m = fresh(n_kv)
        opt = AdamW(collect_params(m), lr=0.0)
        losses = train(m, ids, STEPS,
                       update_fn=scheduled_update(opt, STEPS, peak_lr=0.05,
                                                  warmup_steps=100),
                       log_every=750)
        results[label] = (np.mean(losses[-100:]), kv_cache_bytes(m, 1000), m)
        print()

    print("=== 3. Verdict: quality vs cache size (per 1000 cached tokens) ===")
    for label, (loss_f, cache, _) in results.items():
        print(f"  {label}:  final loss {loss_f:.4f}   "
              f"KV cache {cache / 1024:6.1f} KB")

    print("\n=== 4. MQA still speaks ===")
    m = results["MQA  (1 kv head) "][2]
    ids_g = tok.encode("the cat")
    for _ in range(25):
        logits = m.forward(np.array([ids_g]))[0, -1]
        ids_g.append(int(np.argmax(logits)))
    print(f"  {tok.decode(ids_g)!r}")
