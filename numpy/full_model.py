"""
The full model: RMSNorm + stacked transformer blocks + generation.

This completes the architecture we diagrammed at the start:

  text -> tokenizer -> embedding(+positions)
       -> [ norm -> attention -> +residual -> norm -> FFN -> +residual ] x N
       -> final norm -> tied LM head -> softmax -> sample -> repeat

New in this file:

1. RMSNorm (Zhang & Sennrich 2019) — what Llama/Mistral use instead of
   LayerNorm. It rescales each token's vector to unit RMS, then applies a
   learned per-dimension gain:  y = g * x / sqrt(mean(x^2) + eps).
   No mean-centering, no bias — simpler and just as good.

2. PRE-NORM block layout:  x + attn(norm(x)),  h + ffn(norm(h)).
   Normalizing the INPUT of each sublayer keeps activations bounded no
   matter how deep the stack, which is what lets us DELETE the gradient
   clipping crutch and raise the learning rate. (The 2017 paper used
   post-norm and was famously hard to train.)

3. generate() — autoregressive sampling with temperature. The model
   finally talks.
"""

import numpy as np

from bpe_tokenizer_v2 import BPETokenizer
from embedding_layer import Embedding, sinusoidal_positions, softmax
from attention_layer import MultiHeadAttention
from ffn_layer import SwiGLU

rng = np.random.default_rng(11)


# ---------------------------------------------------------------------- #
# RMSNorm
# ---------------------------------------------------------------------- #
class RMSNorm:
    def __init__(self, d_model: int, eps: float = 1e-6):
        self.g = np.ones(d_model)         # learned per-dimension gain
        self.grad_g = np.zeros(d_model)
        self.eps = eps

    def forward(self, x: np.ndarray) -> np.ndarray:
        r = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + self.eps)
        xhat = x / r
        self._cache = (r, xhat)
        return xhat * self.g

    def backward(self, d_y: np.ndarray) -> np.ndarray:
        r, xhat = self._cache
        self.grad_g += (d_y * xhat).sum(axis=(0, 1))
        d_xhat = d_y * self.g
        # d_x = (d_xhat - xhat * mean(d_xhat * xhat)) / r
        return (d_xhat - xhat * (d_xhat * xhat).mean(-1, keepdims=True)) / r

    def zero_grad(self):
        self.grad_g[:] = 0.0

    def step(self, lr: float):
        self.g -= lr * self.grad_g


# ---------------------------------------------------------------------- #
# One pre-norm transformer block
# ---------------------------------------------------------------------- #
class TransformerBlock:
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = x + self.attn.forward(self.norm1.forward(x))
        return h + self.ffn.forward(self.norm2.forward(h))

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        # out = h + ffn(norm2(h)); residual gradient ADDS to sublayer path.
        d_h = d_out + self.norm2.backward(self.ffn.backward(d_out))
        return d_h + self.norm1.backward(self.attn.backward(d_h))

    def zero_grad(self):
        for m in (self.norm1, self.attn, self.norm2, self.ffn):
            m.zero_grad()

    def step(self, lr: float):
        for m in (self.norm1, self.attn, self.norm2, self.ffn):
            m.step(lr)


# ---------------------------------------------------------------------- #
# The complete model
# ---------------------------------------------------------------------- #
class TinyLLM:
    def __init__(self, vocab_size, d_model=32, n_heads=4, d_ff=64,
                 n_layers=2, max_seq=64):
        self.emb = Embedding(vocab_size, d_model)
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff)
                       for _ in range(n_layers)]
        self.final_norm = RMSNorm(d_model)
        self.pe = sinusoidal_positions(max_seq, d_model)

    def forward(self, ids: np.ndarray) -> np.ndarray:
        S = ids.shape[1]
        x = self.emb.forward(ids) + self.pe[:S]
        for blk in self.blocks:
            x = blk.forward(x)
        self._h = self.final_norm.forward(x)
        return self._h @ self.emb.weight.T          # tied LM head -> logits

    def backward(self, d_logits: np.ndarray) -> None:
        # Tied embedding gets gradient from its head use...
        self.emb.grad += np.einsum("bsv,bsd->vd", d_logits, self._h)
        d_x = self.final_norm.backward(d_logits @ self.emb.weight)
        for blk in reversed(self.blocks):
            d_x = blk.backward(d_x)
        # ...and from its lookup use.
        self.emb.backward(d_x)

    def zero_grad(self):
        self.emb.zero_grad()
        for blk in self.blocks:
            blk.zero_grad()
        self.final_norm.zero_grad()

    def step(self, lr: float):
        self.emb.weight -= lr * self.emb.grad
        for blk in self.blocks:
            blk.step(lr)
        self.final_norm.step(lr)

    # ------------------------- generation --------------------------- #
    def generate(self, tok, prompt: str, n_tokens: int = 30,
                 temperature: float = 0.8, greedy: bool = False,
                 seed: int = 0) -> str:
        g = np.random.default_rng(seed)
        ids = tok.encode(prompt)
        max_ctx = self.pe.shape[0]
        for _ in range(n_tokens):
            ctx = np.array([ids[-max_ctx:]])
            logits = self.forward(ctx)[0, -1]        # last position only
            if greedy:
                nxt = int(np.argmax(logits))
            else:
                p = softmax(logits / temperature)
                nxt = int(g.choice(len(p), p=p))
            ids.append(nxt)
        return tok.decode(ids)


# ---------------------------------------------------------------------- #
# End-to-end gradient check on the WHOLE model
# ---------------------------------------------------------------------- #
def gradient_check():
    model = TinyLLM(vocab_size=50, d_model=8, n_heads=2, d_ff=16,
                    n_layers=2, max_seq=8)
    # Beef up the weights: with tiny 0.02 init, gradients at block 0 are
    # ~1e-10 and finite differences measure only float noise. A gradient
    # check needs healthy-sized gradients to be meaningful.
    for blk in model.blocks:
        for name in ("Wq", "Wk", "Wv", "Wo"):
            setattr(blk.attn, name, rng.normal(0, 0.4, (8, 8)))
        for name in ("Wg", "Wu", "Wd"):
            W = getattr(blk.ffn, name)
            setattr(blk.ffn, name, rng.normal(0, 0.4, W.shape))
        blk.norm1.g = rng.normal(1, 0.2, 8)
        blk.norm2.g = rng.normal(1, 0.2, 8)
    ids = rng.integers(0, 50, (2, 5))
    y = rng.integers(0, 50, (2, 5))
    bi, si = np.arange(2)[:, None], np.arange(5)[None, :]

    def loss_and_dlogits():
        logits = model.forward(ids)
        probs = softmax(logits)
        loss = -np.log(probs[bi, si, y] + 1e-12).mean()
        d = probs.copy(); d[bi, si, y] -= 1.0; d /= 10
        return loss, d

    _, d_logits = loss_and_dlogits()
    model.zero_grad()
    model.backward(d_logits)

    # Spot-check gradients scattered across every parameter type.
    checks = [
        ("emb.weight", model.emb.weight, model.emb.grad),
        ("block0 Wq", model.blocks[0].attn.Wq, model.blocks[0].attn.grads["Wq"]),
        ("block1 Wg", model.blocks[1].ffn.Wg, model.blocks[1].ffn.grads["Wg"]),
        ("block0 norm1.g", model.blocks[0].norm1.g, model.blocks[0].norm1.grad_g),
        ("final_norm.g", model.final_norm.g, model.final_norm.grad_g),
    ]
    eps, floor, worst = 1e-4, 1e-7, 0.0
    for name, W, G in checks:
        flat_W, flat_G = W.reshape(-1), G.reshape(-1)
        for _ in range(10):
            i = rng.integers(flat_W.size)
            old = flat_W[i]
            flat_W[i] = old + eps; up, _ = loss_and_dlogits()
            flat_W[i] = old - eps; dn, _ = loss_and_dlogits()
            flat_W[i] = old
            num = (up - dn) / (2 * eps)
            if abs(num) < floor and abs(flat_G[i]) < floor:
                continue                     # below finite-difference noise
            rel = abs(num - flat_G[i]) / (abs(num) + abs(flat_G[i]) + 1e-12)
            worst = max(worst, rel)
    print(f"end-to-end gradient check: worst relative error = {worst:.2e} "
          f"({'PASS' if worst < 1e-4 else 'FAIL'})\n")


# ---------------------------------------------------------------------- #
# Train and generate
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    print("=== 1. End-to-end gradient check (2 blocks, all layer types) ===")
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
    vocab_size = 256 + len(tok.merges)

    print("=== 2. Train 2-block model: NO gradient clipping, lr up 3x ===")
    print("history: bigram 2.39 -> +attn 0.37 -> +ffn 0.38 -> full model ?\n")
    # Fresh init streams (gradient_check consumed the module RNGs above).
    import embedding_layer, attention_layer, ffn_layer
    embedding_layer.rng = np.random.default_rng(42)
    attention_layer.rng = np.random.default_rng(7)
    ffn_layer.rng = np.random.default_rng(3)
    rng = np.random.default_rng(11)
    model = TinyLLM(vocab_size, d_model=32, n_heads=4, d_ff=64,
                    n_layers=2, max_seq=64)
    seq, batch, steps, lr = 16, 32, 3000, 0.5

    for step in range(steps):
        starts = rng.integers(0, len(ids) - seq - 1, size=batch)
        xb = np.stack([ids[s: s + seq] for s in starts])
        yb = np.stack([ids[s + 1: s + seq + 1] for s in starts])

        logits = model.forward(xb)
        probs = softmax(logits)
        B, S, V = probs.shape
        bi, si = np.arange(B)[:, None], np.arange(S)[None, :]
        loss = -np.log(probs[bi, si, yb] + 1e-12).mean()

        d_logits = probs.copy()
        d_logits[bi, si, yb] -= 1.0
        d_logits /= B * S

        model.zero_grad()
        model.backward(d_logits)
        model.step(lr)

        if step % (steps // 10) == 0 or step == steps - 1:
            print(f"step {step:>5}  loss {loss:.4f}")

    print("\n=== 3. The model speaks ===")
    for prompt in ["the cat", "the king", "one plus"]:
        out = model.generate(tok, prompt, n_tokens=25, temperature=0.7,
                             seed=1)
        print(f"  prompt {prompt!r}\n  -> {out!r}\n")
    print("greedy decoding (temperature -> 0, always argmax):")
    out = model.generate(tok, "the queen", n_tokens=25, greedy=True)
    print(f"  -> {out!r}")
