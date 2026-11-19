"""
Byte-level BPE tokenizer, v2 — now with the two tricks production tokenizers use:

1. REGEX PRE-SPLITTING (GPT-2 style)
   Text is first split into chunks: words (with their leading space),
   numbers, punctuation runs, and common English contractions. Merges are
   learned and applied WITHIN chunks only — they never cross a chunk
   boundary. This prevents junk tokens like "cat s" (word fragment + space
   + next word's first letter) and keeps the vocabulary linguistically sane.

2. PRIORITY-QUEUE ENCODING
   v1 encoding rescanned the whole sequence once per merge: O(n * m).
   v2 keeps tokens in a doubly linked list and candidate merges in a
   min-heap keyed by merge rank (earlier-learned merge = lower rank = higher
   priority). Each merge is a pop + local splice + at most two pushes:
   O(n log n) overall. Stale heap entries are detected lazily and skipped.

Training also gets the classic speedup: instead of one giant id list, we
count UNIQUE chunks and merge inside each unique chunk once, weighting pair
counts by chunk frequency. "the" appearing 10,000 times is processed once.
"""

import heapq
import time
from collections import Counter

import regex  # supports \p{L} / \p{N} Unicode classes, unlike stdlib `re`

# GPT-2's actual split pattern:
#   contractions | optional-space+letters | optional-space+digits |
#   optional-space+other-symbols | trailing-space runs | whitespace
GPT2_SPLIT_PATTERN = (
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class BPETokenizer:
    def __init__(self, split_pattern: str = GPT2_SPLIT_PATTERN):
        self.pattern = regex.compile(split_pattern)
        # pair -> new token id. Ids are assigned in training order, so the
        # id itself doubles as the merge's priority rank.
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # Split into chunks; count unique chunks so repeated words are
        # processed once and merely weighted by their frequency.
        chunk_counts = Counter(self.pattern.findall(text))
        # unique chunk -> its current token-id sequence
        words: dict[str, list[int]] = {
            chunk: list(chunk.encode("utf-8")) for chunk in chunk_counts
        }

        for i in range(num_merges):
            pair_counts: Counter = Counter()
            for chunk, ids in words.items():
                weight = chunk_counts[chunk]
                for pair in zip(ids, ids[1:]):
                    pair_counts[pair] += weight

            if not pair_counts:
                break
            best_pair, count = pair_counts.most_common(1)[0]
            if count < 2:
                break

            new_id = 256 + i
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            # Apply the merge inside every unique chunk that contains it.
            for chunk, ids in words.items():
                if len(ids) >= 2:
                    words[chunk] = self._merge(ids, best_pair, new_id)

            if verbose:
                print(
                    f"merge {i + 1:>4}: {best_pair} -> {new_id} "
                    f"({self.vocab[new_id]!r}, weighted count {count})"
                )

    @staticmethod
    def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
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
    # Encoding — priority queue + doubly linked list
    # ------------------------------------------------------------------ #
    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for chunk in self.pattern.findall(text):
            out.extend(self._encode_chunk(list(chunk.encode("utf-8"))))
        return out

    def _encode_chunk(self, ids: list[int]) -> list[int]:
        n = len(ids)
        if n < 2:
            return ids

        # Doubly linked list over positions; alive[] marks live nodes.
        nxt = list(range(1, n)) + [-1]
        prv = [-1] + list(range(n - 1))
        alive = [True] * n

        # Seed the heap with every mergeable adjacent pair.
        # Entries are (rank, left_position, pair); pair lets us detect
        # stale entries after neighbours have changed.
        heap: list[tuple[int, int, tuple[int, int]]] = []
        for i in range(n - 1):
            rank = self.merges.get((ids[i], ids[i + 1]))
            if rank is not None:
                heapq.heappush(heap, (rank, i, (ids[i], ids[i + 1])))

        while heap:
            rank, i, pair = heapq.heappop(heap)
            if not alive[i]:
                continue
            j = nxt[i]
            if j == -1 or (ids[i], ids[j]) != pair:
                continue  # stale entry: a neighbour was merged since push

            # Merge: node i absorbs node j.
            ids[i] = rank  # new token id == merge rank by construction
            alive[j] = False
            nxt[i] = nxt[j]
            if nxt[j] != -1:
                prv[nxt[j]] = i

            # Only two pairs can newly exist: (left neighbour, i) and (i, right).
            p = prv[i]
            if p != -1:
                r = self.merges.get((ids[p], ids[i]))
                if r is not None:
                    heapq.heappush(heap, (r, p, (ids[p], ids[i])))
            m = nxt[i]
            if m != -1:
                r = self.merges.get((ids[i], ids[m]))
                if r is not None:
                    heapq.heappush(heap, (r, i, (ids[i], ids[m])))

        return [ids[k] for k in range(n) if alive[k]]

    # v1's rescan-everything encoder, kept for benchmarking.
    def _encode_chunk_naive(self, ids: list[int]) -> list[int]:
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:]))
            candidate = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if candidate not in self.merges:
                break
            ids = self._merge(ids, candidate, self.merges[candidate])
        return ids

    def encode_naive(self, text: str) -> list[int]:
        out: list[int] = []
        for chunk in self.pattern.findall(text):
            out.extend(self._encode_chunk_naive(list(chunk.encode("utf-8"))))
        return out

    # ------------------------------------------------------------------ #
    # Decoding & helpers
    # ------------------------------------------------------------------ #
    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def tokens_as_strings(self, ids: list[int]) -> list[str]:
        return [self.vocab[i].decode("utf-8", errors="replace") for i in ids]

    def stats(self, text: str) -> None:
        ids = self.encode(text)
        n_bytes = len(text.encode("utf-8"))
        print(f"text: {text!r}")
        print(f"tokens ({len(ids)}): {self.tokens_as_strings(ids)}")
        print(f"compression: {n_bytes} bytes -> {len(ids)} tokens "
              f"({n_bytes / len(ids):.2f} bytes/token)\n")


if __name__ == "__main__":
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

    # Correctness: heap encoder must agree with the naive encoder.
    tests = [
        "The cat sat on the mat.",
        "Patterns aren't tokens; they're learned. It's 2026!",
        "unseen wörds & emoji 🚀 日本語もOK",
        corpus[:500],
    ]
    for t in tests:
        assert tok.encode(t) == tok.encode_naive(t), f"mismatch on {t!r}"
        assert tok.decode(tok.encode(t)) == t
    print("heap encoder == naive encoder on all tests; round-trips OK\n")

    # Benchmark: naive rescan vs priority queue on a longer document.
    doc = corpus * 50  # ~90 KB
    t0 = time.perf_counter()
    a = tok.encode_naive(doc)
    t_naive = time.perf_counter() - t0
    t0 = time.perf_counter()
    b = tok.encode(doc)
    t_heap = time.perf_counter() - t0
    assert a == b
    print(f"benchmark on {len(doc.encode('utf-8')):,} bytes:")
    print(f"  naive rescan encoder: {t_naive * 1000:7.1f} ms")
    print(f"  priority-queue encoder: {t_heap * 1000:7.1f} ms "
          f"({t_naive / t_heap:.1f}x faster)")
