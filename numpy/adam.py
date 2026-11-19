"""
Adam / AdamW — the optimizer that actually trains LLMs.

Plain SGD moves every parameter by lr * gradient. Two problems at scale:
noisy gradients zigzag, and different parameters need wildly different
step sizes (an embedding row for a rare token vs a norm gain in layer 40).

Adam (Kingma & Ba 2014) fixes both with two exponential moving averages
kept PER PARAMETER:

    m = b1*m + (1-b1)*g          first moment  — smoothed gradient (momentum)
    v = b2*v + (1-b2)*g^2        second moment — smoothed squared magnitude
    m_hat = m / (1 - b1^t)       bias correction: m,v start at 0, so early
    v_hat = v / (1 - b2^t)       averages are biased low; this rescales them
    theta -= lr * m_hat / (sqrt(v_hat) + eps)

The division by sqrt(v_hat) is the magic: every parameter's step is
normalized by its own typical gradient size, so the EFFECTIVE step is
roughly lr for everyone — parameters with tiny, rare gradients move just
as decisively as busy ones. That's why Adam's lr (~1e-3) is so much
smaller than SGD's and yet converges far faster.

AdamW (Loshchilov & Hutter 2017): weight decay applied DIRECTLY to weights
(theta -= lr * wd * theta) instead of mixed into the gradient, where Adam's
normalization would distort it. Following standard practice (e.g. nanoGPT),
we decay 2D matrices but not 1D vectors like norm gains.

Cost: m and v are full copies of every parameter — optimizer state is 2x
model size (8 bytes/param in fp32 moments). For a 70B model that's over
half a terabyte of optimizer state, a real constraint in training clusters.
"""

import numpy as np

from bpe_tokenizer_v2 import BPETokenizer
from embedding_layer import softmax
from full_model import TinyLLM
from paths import out_path

rng = np.random.default_rng(11)


# ---------------------------------------------------------------------- #
# AdamW
# ---------------------------------------------------------------------- #
class AdamW:
    def __init__(self, params, lr=1e-2, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.01):
        """params: list of (name, param_array, grad_array). Arrays are held
        by reference and updated in place."""
        self.params = params
        self.lr, (self.b1, self.b2) = lr, betas
        self.eps, self.wd = eps, weight_decay
        self.m = [np.zeros_like(p) for _, p, _ in params]
        self.v = [np.zeros_like(p) for _, p, _ in params]
        self.t = 0

    def step(self):
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t          # bias corrections
        bc2 = 1.0 - self.b2 ** self.t
        for (name, p, g), m, v in zip(self.params, self.m, self.v):
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g * g
            update = (m / bc1) / (np.sqrt(v / bc2) + self.eps)
            if p.ndim >= 2:                    # decoupled decay, matrices only
                p -= self.lr * self.wd * p
            p -= self.lr * update


def collect_params(model: TinyLLM):
    """Walk the model and gather (name, param, grad) references."""
    params = [("emb.weight", model.emb.weight, model.emb.grad)]
    for i, blk in enumerate(model.blocks):
        for w in ("Wq", "Wk", "Wv", "Wo"):
            params.append((f"blk{i}.attn.{w}", getattr(blk.attn, w),
                           blk.attn.grads[w]))
        for w in ("Wg", "Wu", "Wd"):
            params.append((f"blk{i}.ffn.{w}", getattr(blk.ffn, w),
                           blk.ffn.grads[w]))
        params.append((f"blk{i}.norm1.g", blk.norm1.g, blk.norm1.grad_g))
        params.append((f"blk{i}.norm2.g", blk.norm2.g, blk.norm2.grad_g))
    params.append(("final_norm.g", model.final_norm.g, model.final_norm.grad_g))
    return params


# ---------------------------------------------------------------------- #
# Head-to-head: identical init, identical batches, only the update differs
# ---------------------------------------------------------------------- #
def make_model(vocab_size):
    """Fresh model with the exact same initialization every call."""
    import embedding_layer, attention_layer, ffn_layer
    embedding_layer.rng = np.random.default_rng(42)
    attention_layer.rng = np.random.default_rng(7)
    ffn_layer.rng = np.random.default_rng(3)
    return TinyLLM(vocab_size, d_model=32, n_heads=4, d_ff=64,
                   n_layers=2, max_seq=64)


def train(model, ids, steps, update_fn, seq=16, batch=32, log_every=100):
    data_rng = np.random.default_rng(123)      # same batches for both runs
    losses = []
    bi, si = np.arange(batch)[:, None], np.arange(seq)[None, :]
    for step_i in range(steps):
        st = data_rng.integers(0, len(ids) - seq - 1, size=batch)
        xb = np.stack([ids[s: s + seq] for s in st])
        yb = np.stack([ids[s + 1: s + seq + 1] for s in st])
        probs = softmax(model.forward(xb))
        loss = -np.log(probs[bi, si, yb] + 1e-12).mean()
        losses.append(float(loss))
        d = probs.copy(); d[bi, si, yb] -= 1.0; d /= batch * seq
        model.zero_grad(); model.backward(d)
        update_fn(model)
        if step_i % log_every == 0 or step_i == steps - 1:
            print(f"  step {step_i:>5}  loss {loss:.4f}")
    return losses


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
    STEPS = 1500

    print("=== SGD, lr=0.5 (our best hand-tuned setting) ===")
    model_sgd = make_model(vocab_size)
    sgd_losses = train(model_sgd, ids, STEPS,
                       update_fn=lambda m: m.step(0.5), log_every=250)

    print("\n=== AdamW, lr=0.01 ===")
    model_adam = make_model(vocab_size)
    opt = AdamW(collect_params(model_adam), lr=0.01)
    adam_losses = train(model_adam, ids, STEPS,
                        update_fn=lambda m: opt.step(), log_every=250)

    print("\n=== Verdict ===")
    def first_below(losses, thresh):
        for i, l in enumerate(losses):
            if l < thresh:
                return i
        return None
    for thresh in (2.0, 1.0, 0.5, 0.2):
        s, a = first_below(sgd_losses, thresh), first_below(adam_losses, thresh)
        print(f"  loss < {thresh}: SGD at step {s}, AdamW at step {a}")
    print(f"  final: SGD {np.mean(sgd_losses[-50:]):.4f}, "
          f"AdamW {np.mean(adam_losses[-50:]):.4f}")

    # Optimizer memory accounting
    n_params = sum(p.size for _, p, _ in collect_params(model_adam))
    print(f"\n  model parameters: {n_params:,}")
    print(f"  AdamW state (m + v): {2 * n_params:,} floats — "
          f"2x the model, before gradients")

    np.save(out_path("sgd_losses.npy"), np.array(sgd_losses))
    np.save(out_path("adam_losses.npy"), np.array(adam_losses))
