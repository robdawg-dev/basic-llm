"""
A minimal byte-level BPE (byte-pair encoding) tokenizer, from scratch.

This is the same core algorithm used by GPT-2/3/4, Llama, and similar models
(minus optimizations and regex pre-splitting). The idea:

  1. Start with a base vocabulary of all 256 possible bytes.
  2. Find the most frequent adjacent pair of tokens in the training data.
  3. Merge that pair into a new token, added to the vocabulary.
  4. Repeat until the vocabulary reaches the desired size.

Encoding new text then replays those merges in the order they were learned.
Working at the byte level means the tokenizer can represent ANY string
(any language, emoji, code) with zero "unknown token" problems.
"""

from collections import Counter


class BPETokenizer:
    def __init__(self):
        # merges: maps a pair of token ids -> the new token id created for it.
        # Insertion order == training order, which matters at encode time.
        self.merges: dict[tuple[int, int], int] = {}
        # vocab: token id -> raw bytes it represents (used for decoding).
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Learn `vocab_size - 256` merges from the training text."""
        assert vocab_size >= 256, "vocab must at least cover all single bytes"
        num_merges = vocab_size - 256

        # Represent the corpus as a list of token ids (initially raw bytes).
        ids = list(text.encode("utf-8"))

        for i in range(num_merges):
            # Count all adjacent pairs.
            pair_counts = Counter(zip(ids, ids[1:]))
            if not pair_counts:
                break  # corpus too small to merge further

            # Most frequent pair gets merged into a brand-new token id.
            best_pair, count = pair_counts.most_common(1)[0]
            if count < 2:
                break  # nothing repeats; further merges are pointless

            new_id = 256 + i
            ids = self._merge(ids, best_pair, new_id)

            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            if verbose:
                print(
                    f"merge {i + 1:>4}: {best_pair} -> {new_id} "
                    f"({self.vocab[new_id]!r}, {count} occurrences)"
                )

    @staticmethod
    def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """Replace every occurrence of `pair` in `ids` with `new_id`."""
        out = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    # ------------------------------------------------------------------ #
    # Encoding / decoding
    # ------------------------------------------------------------------ #
    def encode(self, text: str) -> list[int]:
        """Text -> token ids, replaying learned merges in training order."""
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            # Among all pairs currently present, pick the one that was
            # learned EARLIEST during training (lowest merge id).
            pairs = set(zip(ids, ids[1:]))
            candidate = min(
                pairs, key=lambda p: self.merges.get(p, float("inf"))
            )
            if candidate not in self.merges:
                break  # no learned merge applies anymore
            ids = self._merge(ids, candidate, self.merges[candidate])
        return ids

    def decode(self, ids: list[int]) -> str:
        """Token ids -> text. Always lossless for anything encode() produced."""
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ #
    # Introspection helpers
    # ------------------------------------------------------------------ #
    def tokens_as_strings(self, ids: list[int]) -> list[str]:
        """Human-readable view of a token id sequence."""
        return [self.vocab[i].decode("utf-8", errors="replace") for i in ids]

    def stats(self, text: str) -> None:
        ids = self.encode(text)
        n_bytes = len(text.encode("utf-8"))
        print(f"text: {text!r}")
        print(f"tokens ({len(ids)}): {self.tokens_as_strings(ids)}")
        print(f"compression: {n_bytes} bytes -> {len(ids)} tokens "
              f"({n_bytes / len(ids):.2f} bytes/token)")
        print()


if __name__ == "__main__":
    # A small training corpus. Real tokenizers train on hundreds of GB;
    # the algorithm is identical.
    corpus = (
        "The cat sat on the mat. The cat saw the rat. "
        "A transformer is a neural network. The network learns patterns. "
        "Language models predict the next token. Tokens are subword units. "
        "The model reads tokens and predicts tokens. Learning is repeated "
        "pattern matching over the training data. The training data teaches "
        "the network which patterns matter. "
    ) * 4

    tok = BPETokenizer()
    tok.train(corpus, vocab_size=256 + 40, verbose=True)

    print("\n--- Encoding examples ---\n")
    tok.stats("The cat sat on the mat.")
    tok.stats("The network predicts the next token.")
    tok.stats("Completely unseen wörds & emoji 🚀 still work!")

    # Round-trip guarantee
    sample = "Byte-level BPE never fails to round-trip: 日本語もOK! 🎉"
    assert tok.decode(tok.encode(sample)) == sample
    print(f"round-trip OK: {sample!r}")
