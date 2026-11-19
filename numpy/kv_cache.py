"""
KV-cache inference — why LLM generation is fast.

The problem: naive generation re-runs the ENTIRE forward pass over the
whole context for every new token. Step t costs O(t^2) attention; producing
N tokens costs O(N^3) total. Contexts of 100k tokens would be unusable.

The fix: at step t, the keys K and values V of tokens 0..t-1 are exactly
what they were at step t-1 — nothing upstream of them changed (causal mask
guarantees it). So we cache K and V per layer and per step only compute:

    q, k, v for the ONE new token
    append k, v to the cache
    attention = softmax(q @ K_cache.T) @ V_cache      <- one row, not a matrix

Each step drops from O(t^2) to O(t); total generation from O(N^3) to O(N^2).
No mask is needed for the new token: it is the LAST position, so it may
legally attend to everything in the cache (including itself).

The two phases every serving stack talks about:
  PREFILL — the whole prompt goes through in one batched forward pass
            (masked, parallel) to populate the cache. Compute-bound.
  DECODE  — one token at a time against the cache. Memory-bound: the cache
            for a big model is gigabytes, and reading it dominates.

The price is memory: 2 * n_layers * seq_len * d_model floats per sequence.
For our toy: 2*2*64*32 = 16 KB. For a 70B model at 100k context: tens of
GB — which is why GQA/MQA (fewer K/V heads) and paged attention exist.
"""

import time

import numpy as np

from bpe_tokenizer_v2 import BPETokenizer
from embedding_layer import Embedding, sinusoidal_positions, softmax
from full_model import TinyLLM

rng = np.random.default_rng(11)


# ---------------------------------------------------------------------- #
# Cached attention step (works on the existing MultiHeadAttention weights)
# ---------------------------------------------------------------------- #
def attn_step(attn, x_new: np.ndarray, cache: dict) -> np.ndarray:
    """Attention for T_new new tokens given cached K/V of all past tokens.

    x_new: (1, T_new, D).  T_new = prompt length during prefill, 1 during
    decode. cache holds 'k','v' of shape (1, H, S_past, dh) or None.
    """
    q = attn._split(x_new @ attn.Wq)
    k_new = attn._split(x_new @ attn.Wk)
    v_new = attn._split(x_new @ attn.Wv)

    if cache["k"] is None:
        k, v = k_new, v_new
    else:
        k = np.concatenate([cache["k"], k_new], axis=2)
        v = np.concatenate([cache["v"], v_new], axis=2)
    cache["k"], cache["v"] = k, v                    # the whole trick

    scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(attn.dh)

    T_new, S_total = q.shape[2], k.shape[2]
    if T_new > 1:
        # Prefill: new token i (global position S_total - T_new + i) may
        # attend to global positions <= its own. Decode (T_new == 1)
        # needs no mask at all — the new token is last, everything is fair.
        col = np.arange(S_total)[None, :]
        row_pos = (S_total - T_new) + np.arange(T_new)[:, None]
        scores = scores + np.where(col > row_pos, -1e9, 0.0)

    p = softmax(scores)
    return attn._join(p @ v) @ attn.Wo


def block_step(blk, x_new: np.ndarray, cache: dict) -> np.ndarray:
    """One transformer block with cached attention. RMSNorm and the FFN are
    position-wise — they never look at other tokens — so they need no cache
    and run unchanged on just the new token(s)."""
    h = x_new + attn_step(blk.attn, blk.norm1.forward(x_new), cache)
    return h + blk.ffn.forward(blk.norm2.forward(h))


def generate_cached(model: TinyLLM, tok, prompt: str, n_tokens: int = 30,
                    temperature: float = 0.8, greedy: bool = False,
                    seed: int = 0) -> str:
    g = np.random.default_rng(seed)
    ids = tok.encode(prompt)
    max_ctx = model.pe.shape[0]
    assert len(ids) + n_tokens <= max_ctx, "would exceed cache/PE capacity"

    caches = [{"k": None, "v": None} for _ in model.blocks]

    def step(new_ids: list[int], pos: int) -> np.ndarray:
        """Run only the new token(s) through the model, using the caches."""
        x = model.emb.forward(np.array([new_ids])) + model.pe[pos: pos + len(new_ids)]
        for blk, cache in zip(model.blocks, caches):
            x = block_step(blk, x, cache)
        h = model.final_norm.forward(x)
        return (h @ model.emb.weight.T)[0, -1]       # last position's logits

    logits = step(ids, 0)                            # PREFILL: whole prompt
    for _ in range(n_tokens):                        # DECODE: 1 token/step
        if greedy:
            nxt = int(np.argmax(logits))
        else:
            p = softmax(logits / temperature)
            nxt = int(g.choice(len(p), p=p))
        ids.append(nxt)
        logits = step([nxt], pos=len(ids) - 1)
    return tok.decode(ids)


# ---------------------------------------------------------------------- #
# Train the model (same recipe as full_model.py), then verify + benchmark
# ---------------------------------------------------------------------- #
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

    print("=== 1. Train (same recipe as before) ===")
    import embedding_layer, attention_layer, ffn_layer
    embedding_layer.rng = np.random.default_rng(42)
    attention_layer.rng = np.random.default_rng(7)
    ffn_layer.rng = np.random.default_rng(3)
    model = TinyLLM(vocab_size, d_model=32, n_heads=4, d_ff=64,
                    n_layers=2, max_seq=256)
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

    print("\n=== 2. Correctness: cached must equal uncached ===")
    for prompt in ["the cat", "the queen wore"]:
        a = model.generate(tok, prompt, n_tokens=25, greedy=True)
        b = generate_cached(model, tok, prompt, n_tokens=25, greedy=True)
        match = "MATCH" if a == b else "MISMATCH"
        print(f"  [{match}] {prompt!r}")
        print(f"     uncached: {a!r}")
        print(f"     cached:   {b!r}")
    assert a == b

    print("\n=== 3. Benchmark: generate 200 tokens ===")
    t0 = time.perf_counter()
    model.generate(tok, "the cat", n_tokens=200, greedy=True)
    t_naive = time.perf_counter() - t0

    t0 = time.perf_counter()
    generate_cached(model, tok, "the cat", n_tokens=200, greedy=True)
    t_cached = time.perf_counter() - t0

    print(f"  uncached (full re-forward per token): {t_naive * 1000:8.1f} ms")
    print(f"  KV-cached (one token per step):       {t_cached * 1000:8.1f} ms")
    print(f"  speedup: {t_naive / t_cached:.1f}x")

    n_layers, H, dh = len(model.blocks), model.blocks[0].attn.h, model.blocks[0].attn.dh
    ctx = len(tok.encode("the cat")) + 200
    kb = 2 * n_layers * ctx * H * dh * 8 / 1024
    print(f"  cache size at the end: 2 * {n_layers} layers * {ctx} positions "
          f"* {H * dh} dims * 8 bytes = {kb:.0f} KB")
