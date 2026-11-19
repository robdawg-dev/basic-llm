"""
The embedding layer, from scratch (NumPy) — the second stage of our LLM.

Pipeline so far:  text --tokenizer--> token ids --EMBEDDING--> vectors

Three ideas in this file:

1. TOKEN EMBEDDING = a lookup table.
   A (vocab_size, d_model) matrix. Forward pass is literally `W[ids]` —
   no matmul, just row indexing. Every "meaning" the model attaches to a
   token lives in its row of this matrix, and it's all learned.

2. THE BACKWARD PASS IS SCATTER-ADD.
   Only the rows that were looked up receive gradient, and a token that
   appears k times in the batch accumulates k gradient contributions.
   This is why rare tokens train slowly: their rows are rarely touched.

3. POSITIONS must be injected separately — attention is order-blind.
   We implement the original sinusoidal encodings here. (Modern models
   mostly use RoPE, which rotates query/key vectors inside attention
   instead of adding anything to embeddings — that belongs in the
   attention layer, so it's noted but not built here.)

To prove the layer works end-to-end, we train the world's smallest
language model: embeddings + tied output projection = a neural bigram
model. logits = E[ids] @ E.T. After training, tokens used in similar
contexts end up with similar vectors — the famous emergent property.
"""

import numpy as np

from bpe_tokenizer_v2 import BPETokenizer

rng = np.random.default_rng(42)


# ---------------------------------------------------------------------- #
# 1. Token embedding layer with forward + backward
# ---------------------------------------------------------------------- #
class Embedding:
    def __init__(self, vocab_size: int, d_model: int):
        # Small random init. Real models use ~N(0, 0.02) just like this.
        self.weight = rng.normal(0.0, 0.02, size=(vocab_size, d_model))
        self.grad = np.zeros_like(self.weight)
        self._last_ids: np.ndarray | None = None

    def forward(self, ids: np.ndarray) -> np.ndarray:
        """(batch, seq) int ids -> (batch, seq, d_model) vectors."""
        self._last_ids = ids
        return self.weight[ids]  # pure indexing — that's the whole layer

    def backward(self, d_out: np.ndarray) -> None:
        """Scatter-add d_out back into the rows that were looked up.

        np.add.at is the unbuffered form: if the same id appears twice,
        both gradient contributions accumulate (plain fancy-index
        assignment would silently keep only one — a classic bug).
        """
        np.add.at(self.grad, self._last_ids, d_out)

    def zero_grad(self) -> None:
        self.grad[:] = 0.0


# ---------------------------------------------------------------------- #
# 2. Sinusoidal positional encodings (Vaswani et al., 2017)
# ---------------------------------------------------------------------- #
def sinusoidal_positions(seq_len: int, d_model: int) -> np.ndarray:
    """(seq_len, d_model) matrix; row p encodes position p.

    Each dimension pair (2i, 2i+1) is a sine/cosine at a different
    wavelength, from 2*pi up to 10000*2*pi. Nearby positions get similar
    vectors; any fixed offset is a linear transform of the original —
    which is what lets attention learn relative positioning.
    """
    positions = np.arange(seq_len)[:, None]              # (seq, 1)
    dims = np.arange(0, d_model, 2)[None, :]             # (1, d/2)
    angle_rates = 1.0 / np.power(10000.0, dims / d_model)
    angles = positions * angle_rates                     # (seq, d/2)
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)
    return pe


# ---------------------------------------------------------------------- #
# 3. Demo: train embeddings as a neural bigram model
#    logits = E[ids] @ E.T   (weight tying: output head reuses E)
# ---------------------------------------------------------------------- #
def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)  # numerical stability
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def train_bigram_embeddings(
    emb: Embedding, ids: np.ndarray, steps: int, lr: float, batch: int, seq: int
):
    """Predict token[t+1] from token[t] using only the embedding table."""
    losses = []
    for step in range(steps):
        # Sample random windows from the corpus.
        starts = rng.integers(0, len(ids) - seq - 1, size=batch)
        x = np.stack([ids[s : s + seq] for s in starts])          # (B, S)
        y = np.stack([ids[s + 1 : s + seq + 1] for s in starts])  # (B, S)

        # Forward: lookup, then tied projection back onto the vocab.
        vecs = emb.forward(x)                       # (B, S, D)
        logits = vecs @ emb.weight.T                # (B, S, V)
        probs = softmax(logits)

        # Cross-entropy loss on the true next tokens.
        B, S, V = probs.shape
        p_true = probs[np.arange(B)[:, None], np.arange(S)[None, :], y]
        loss = -np.log(p_true + 1e-12).mean()
        losses.append(loss)

        # Backward. d_logits = (probs - onehot(y)) / (B*S)
        d_logits = probs.copy()
        d_logits[np.arange(B)[:, None], np.arange(S)[None, :], y] -= 1.0
        d_logits /= B * S

        emb.zero_grad()
        # Tied weights get gradient from BOTH uses:
        #   as the output head:  d_W += d_logits.T @ vecs
        emb.grad += np.einsum("bsv,bsd->vd", d_logits, vecs)
        #   as the lookup table: scatter-add of d_vecs
        d_vecs = d_logits @ emb.weight              # (B, S, D)
        emb.backward(d_vecs)

        emb.weight -= lr * emb.grad                 # plain SGD

        if step % (steps // 10) == 0 or step == steps - 1:
            print(f"step {step:>5}  loss {loss:.4f}")
    return losses


def nearest_neighbours(emb: Embedding, tok, token_str: str, k: int = 5):
    """Cosine-similarity neighbours of a token in embedding space."""
    tid = tok.encode(token_str)
    assert len(tid) == 1, f"{token_str!r} is not a single token"
    W = emb.weight
    W_norm = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    sims = W_norm @ W_norm[tid[0]]
    best = np.argsort(-sims)[1 : k + 1]  # skip self
    pretty = [(tok.vocab[i].decode("utf-8", "replace"), round(float(sims[i]), 3))
              for i in best]
    print(f"  {token_str!r:>12} -> {pretty}")


if __name__ == "__main__":
    # -- Corpus with deliberate structure so similar words share contexts --
    corpus = (
        "the cat sat on the mat . the dog sat on the rug . "
        "the cat ran to the door . the dog ran to the gate . "
        "a cat is an animal . a dog is an animal . "
        "the cat ate the food . the dog ate the food . "
        "the king ruled the land . the queen ruled the land . "
        "the king wore a crown . the queen wore a crown . "
        "one plus one is two . two plus two is four . "
    ) * 20

    print("=== 1. Tokenize (reusing our BPE tokenizer) ===")
    tok = BPETokenizer()
    tok.train(corpus, vocab_size=256 + 80)
    ids = np.array(tok.encode(corpus))
    vocab_size = 256 + len(tok.merges)
    print(f"corpus: {len(ids)} tokens, vocab size {vocab_size}\n")

    print("=== 2. Embedding lookup is just indexing ===")
    d_model = 32
    emb = Embedding(vocab_size, d_model)
    sample = np.array([tok.encode("the cat and the dog saw the cat")])
    vecs = emb.forward(sample)
    print(f"ids {sample.shape} -> vectors {vecs.shape} "
          f"(seq of {sample.shape[1]} tokens, each a {d_model}-dim row of W)\n")

    print("=== 3. Add positional information ===")
    pe = sinusoidal_positions(sample.shape[1], d_model)
    x = vecs + pe  # what actually enters transformer block 1
    seen: dict[int, int] = {}
    positions_of_the: list[int] = []
    for i, t in enumerate(sample[0]):
        if int(t) in seen:
            positions_of_the = [seen[int(t)], i]
            break
        seen[int(t)] = i
    if len(positions_of_the) >= 2:
        a, b = positions_of_the
        token_str = tok.vocab[int(sample[0, a])].decode("utf-8", "replace")
        print(f"token {token_str!r} appears at positions {a} and {b}")
        print(f"identical before positions: "
              f"{np.allclose(vecs[0, a], vecs[0, b])}")
        print(f"different after positions:  "
              f"{not np.allclose(x[0, a], x[0, b])}\n")

    print("=== 4. Train embeddings (tied-weight neural bigram) ===")
    losses = train_bigram_embeddings(emb, ids, steps=2000, lr=1.0,
                                     batch=32, seq=8)
    baseline = np.log(vocab_size)
    print(f"\nuniform-guess baseline loss = ln({vocab_size}) = {baseline:.3f}")
    print(f"final loss = {losses[-1]:.3f}\n")

    print("=== 5. Emergent structure: nearest neighbours in vector space ===")
    for w in [" cat", " king", " ran", " one"]:
        nearest_neighbours(emb, tok, w)
