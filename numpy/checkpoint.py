"""
Checkpointing: serialize the model + tokenizer, restore them exactly.

Weights go in a single .npz (numpy's zip-of-arrays format), keyed by the
same names collect_params() assigns. Everything non-array — model config
and the tokenizer's learned merges — goes in a .json sidecar. The
tokenizer must ship with the model: weights are meaningless under a
different token <-> id mapping.

The only subtlety is restoring the tokenizer: vocab is DERIVED state.
We save the merge list in training order and replay it, rebuilding each
new token's bytes from its parents — same as training did.
"""

import json

import numpy as np

from bpe_tokenizer_v2 import BPETokenizer
from adam import collect_params
from gqa import GQALLM
from paths import out_path


def save_checkpoint(model: GQALLM, tok: BPETokenizer, prefix: str,
                    extra: dict | None = None):
    np.savez(prefix + ".npz",
             **{name: p for name, p, _ in collect_params(model)})
    blk = model.blocks[0]
    meta = {
        "config": {
            "vocab_size": model.emb.weight.shape[0],
            "d_model": model.emb.weight.shape[1],
            "n_heads": blk.attn.h,
            "n_kv_heads": blk.attn.g,
            "d_ff": blk.ffn.Wg.shape[1],
            "n_layers": len(model.blocks),
            "max_seq": blk.attn.cos.shape[0],
        },
        # merges in training order; ids are implicit (256, 257, ...)
        "merges": [[int(a), int(b)] for (a, b) in tok.merges],
        "extra": extra or {},
    }
    with open(prefix + ".json", "w") as f:
        json.dump(meta, f)


def load_checkpoint(prefix: str) -> tuple[GQALLM, BPETokenizer, dict]:
    with open(prefix + ".json") as f:
        meta = json.load(f)

    tok = BPETokenizer()
    for i, (a, b) in enumerate(meta["merges"]):
        new_id = 256 + i
        tok.merges[(a, b)] = new_id
        tok.vocab[new_id] = tok.vocab[a] + tok.vocab[b]   # replay derivation

    model = GQALLM(**meta["config"])
    data = np.load(prefix + ".npz")
    for name, p, _ in collect_params(model):
        p[:] = data[name]                                  # in-place restore
    return model, tok, meta["extra"]


if __name__ == "__main__":
    # Self-test: train briefly, save, load, demand EXACT agreement.
    corpus = ("the cat sat on the mat . the dog sat on the rug . " * 40)
    tok = BPETokenizer()
    tok.train(corpus, vocab_size=256 + 20)
    model = GQALLM(256 + len(tok.merges), n_kv_heads=2)

    ids = np.array([tok.encode("the cat sat on the")])
    logits_before = model.forward(ids)

    save_checkpoint(model, tok, out_path("selftest"),
                    extra={"note": "checkpoint self-test"})
    model2, tok2, extra = load_checkpoint(out_path("selftest"))

    logits_after = model2.forward(np.array([tok2.encode("the cat sat on the")]))
    exact = np.array_equal(logits_before, logits_after)
    same_tok = tok2.encode(corpus[:200]) == tok.encode(corpus[:200])
    print(f"logits byte-identical after reload: {exact}")
    print(f"tokenizer identical after reload:   {same_tok}")
    assert exact and same_tok
    print("checkpoint self-test PASSED")
