"""
train_gpt.py — the PyTorch port of our NumPy stack, sized for a 12 GB GPU.

Every component here is one we hand-built and gradient-checked in NumPy;
PyTorch replaces our hand-written backward passes with autograd and our
matmuls with CUDA kernels. The mapping:

    NumPy file            ->  here
    ------------------------------------------------------------------
    embedding_layer.py    ->  nn.Embedding + weight tying
    rope.py               ->  RoPE buffers + rope_apply (same half-split)
    gqa.py                ->  Attention (F.scaled_dot_product_attention
                              + repeat_interleave for kv groups)
    ffn_layer.py          ->  SwiGLU (three nn.Linear, silu gate)
    full_model.py         ->  RMSNorm, pre-norm Block, GPT
    adam.py               ->  torch.optim.AdamW (decay 2D params only)
    lr_schedule.py        ->  same warmup + cosine function
    checkpoint.py         ->  torch.save/load incl. optimizer state
    sampling.py           ->  generate() with temp / top-p / rep penalty
    shake_resume.py       ->  the training loop, memmap data, val split

Usage:
    python prepare_data.py                    # writes train.bin / val.bin
    python train_gpt.py --preset debug        # ~11M params, sanity run
    python train_gpt.py --preset gpt2ish      # ~101M params, the real run
    python train_gpt.py --preset gpt2ish --resume   # continue a run
    python train_gpt.py --preset gpt2ish --sample_only --resume

VRAM plan for 12 GB, gpt2ish preset (~101M params):
    fp32 params 0.4 GB + fp32 grads 0.4 GB + Adam moments 0.8 GB = 1.6 GB
    persistent; bf16 activations at micro_batch 12 x block 1024 fit with
    several GB to spare (SDPA uses memory-efficient attention kernels).
    Raise micro_batch until you OOM, then back off one notch.
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------- #
# Config
# ---------------------------------------------------------------------- #
@dataclass
class Config:
    vocab_size: int = 32768
    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: int = 4          # GQA: 12 query heads share 4 kv heads
    d_ff: int = 2048             # SwiGLU sizing ~ (2/3)*4*d_model
    n_layers: int = 12
    block_size: int = 1024       # context length
    dropout: float = 0.0         # 0 for ~1-epoch pretraining; ~0.1 if
                                 # looping a small corpus many times
    # training
    micro_batch: int = 12        # sequences per fwd/bwd on the GPU
    grad_accum: int = 40         # micro-steps per optimizer step
                                 # -> 12*1024*40 = 491k tokens/step
    max_steps: int = 5000
    warmup_steps: int = 200
    peak_lr: float = 6e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_batches: int = 20
    ckpt_every: int = 500


PRESETS = {
    # ~0.9M params: CPU-runnable end-to-end verification of the pipeline
    "tiny": dict(vocab_size=512, d_model=128, n_heads=4, n_kv_heads=2,
                 d_ff=384, n_layers=4, block_size=128, micro_batch=16,
                 grad_accum=2, max_steps=300, warmup_steps=30,
                 eval_every=50, eval_batches=8, ckpt_every=150,
                 peak_lr=1e-3),
    # ~11M params: shapes and pipeline sanity, minutes to see learning
    "debug": dict(vocab_size=512, d_model=256, n_heads=8, n_kv_heads=4,
                  d_ff=768, n_layers=6, block_size=256, micro_batch=32,
                  grad_accum=4, max_steps=2000, warmup_steps=100,
                  eval_every=200, peak_lr=1e-3),
    # ~45M params: overnight-quality runs, fast iteration
    "small": dict(vocab_size=32768, d_model=512, n_heads=8, n_kv_heads=4,
                  d_ff=1408, n_layers=8, block_size=512, micro_batch=24,
                  grad_accum=20),
    # ~101M params: the 12 GB flagship (GPT-2-small class)
    "gpt2ish": dict(),
    # gpt2ish, but sized for prepare_fineweb.py's default tokenizer
    # (SmolLM2's, 49,152 vocab) instead of the 32,768-vocab tokenizers
    # prepare_data.py / prepare_real_data.py train. 49,152 = 384x128 is
    # already a GPU-friendly size — no padding needed. (A non-power-of-2,
    # non-128-multiple vocab_size, e.g. GPT-2's raw 50,257, reproducibly
    # hung torch.compile on this setup; pad to the next multiple of 64 if
    # you ever switch to a tokenizer with an awkward vocab size — see
    # RUNBOOK.md.)
    "fineweb": dict(vocab_size=49152),
}


# ---------------------------------------------------------------------- #
# Model — every module has a NumPy twin you have already gradient-checked
# ---------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.eps = eps
        self.d = d

    def forward(self, x):
        # F.rms_norm (native ATen op, torch>=2.4) instead of the decomposed
        # pow/mean/rsqrt/mul chain — same math, real fused kernel instead of
        # several small ones. Parameter kept named `g` (not swapped for
        # nn.RMSNorm outright) so existing checkpoints still load: their
        # state_dict keys are `...norm1.g`, not `...norm1.weight`.
        return F.rms_norm(x, (self.d,), self.g, self.eps)


def rope_tables(dh, max_seq, base=10000.0, device="cpu"):
    inv = base ** (-torch.arange(0, dh, 2, device=device).float() / dh)
    ang = torch.arange(max_seq, device=device).float()[:, None] * inv[None, :]
    return torch.cos(ang), torch.sin(ang)             # (max_seq, dh/2)


def rope_apply(t, cos, sin, start_pos=0):
    # t: (B, H, S, dh), half-split pairing — identical to our rope.py
    S = t.size(2)
    c = cos[start_pos:start_pos + S][None, None]      # (1,1,S,dh/2)
    s = sin[start_pos:start_pos + S][None, None]
    x1, x2 = t.chunk(2, dim=-1)
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.h, self.g = cfg.n_heads, cfg.n_kv_heads
        self.dh = cfg.d_model // cfg.n_heads
        self.rep = self.h // self.g
        self.wq = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wk = nn.Linear(cfg.d_model, self.g * self.dh, bias=False)
        self.wv = nn.Linear(cfg.d_model, self.g * self.dh, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout
        cos, sin = rope_tables(self.dh, cfg.block_size)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x, cache=None, start_pos=0):
        """Training/eval: cache=None, full sequence, causal mask.
        Generation: pass a per-layer cache dict {'k': ..., 'v': ...}.
        Cached keys/values are stored PRE-broadcast (G heads, already
        rotated at their positions) — GQA's cache saving carries through."""
        B, S, D = x.shape
        q = self.wq(x).view(B, S, self.h, self.dh).transpose(1, 2)
        k = self.wk(x).view(B, S, self.g, self.dh).transpose(1, 2)
        v = self.wv(x).view(B, S, self.g, self.dh).transpose(1, 2)
        q = rope_apply(q, self.cos, self.sin, start_pos)
        k = rope_apply(k, self.cos, self.sin, start_pos)

        if cache is not None:
            if cache["k"] is not None:
                # Appending to a non-empty cache: decode is one token at a
                # time. (SDPA's is_causal aligns top-left, which would be
                # wrong for multi-token appends against a cache.)
                assert S == 1, "cached decode appends one token at a time"
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v

        # broadcast kv groups to query heads (our np.repeat / sum pair —
        # autograd derives the sum side for us now)
        k = k.repeat_interleave(self.rep, dim=1)
        v = v.repeat_interleave(self.rep, dim=1)
        # Prefill / training (S > 1): causal mask over the square block.
        # Single-token decode: the new token is last — everything in the
        # cache is legally visible, no mask needed. Same asymmetry as our
        # NumPy kv_cache.py.
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=(S > 1),
            dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, S, D)
        return self.wo(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.wg = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.wu = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.wd = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.wd(F.silu(self.wg(x)) * self.wu(x))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1, self.attn = RMSNorm(cfg.d_model), Attention(cfg)
        self.norm2, self.ffn = RMSNorm(cfg.d_model), SwiGLU(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cache=None, start_pos=0):
        x = x + self.drop(self.attn(self.norm1(x), cache, start_pos))
        return x + self.drop(self.ffn(self.norm2(x)))


class GPT(nn.Module):
    def __init__(self, cfg: Config, fused_ce: bool = False):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight        # weight tying

        # torch's native fused linear+cross-entropy (>=2.13): computes the
        # lm_head projection and the loss together, chunked over the vocab
        # dimension, so the full [B, T, vocab_size] logits tensor is never
        # materialized — this matters once vocab_size gets large (e.g. 50k+):
        # that tensor (and its softmax/gradient buffers) becomes a memory-
        # bandwidth-bound bottleneck of its own, separate from the
        # transformer's matmuls. Also just works under --compile, unlike
        # liger-kernel's equivalent (used here previously — its custom
        # autograd.Function crashed Dynamo's fake-tensor tracing and needed
        # a @torch.compiler.disable wrapper; native needs no such thing).
        self.fused_ce = fused_ce
        self._native_ce = None
        if fused_ce:
            if not hasattr(nn, "LinearCrossEntropyLoss"):
                raise SystemExit("--fused_ce requires torch>=2.13 (this "
                                 "build has no nn.LinearCrossEntropyLoss)")
            self._native_ce = nn.LinearCrossEntropyLoss(
                cfg.d_model, cfg.vocab_size, bias=False)
            self._native_ce.linear.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init)
        # residual-branch projections get scaled-down init (GPT-2 trick):
        # N residual adds would otherwise grow the stream like sqrt(2N)
        for name, p in self.named_parameters():
            if name.endswith(("wo.weight", "wd.weight")):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layers))

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, ids, targets=None):
        x = self.drop(self.tok_emb(ids))
        for blk in self.blocks:
            x = blk(x)
        x = self.final_norm(x)
        if targets is None:
            return self.lm_head(x[:, -1:]), None         # inference: last pos
        if self.fused_ce:
            # No logits tensor: the fused kernel goes straight from hidden
            # states to a scalar loss. Nothing downstream uses the logits
            # returned from a targets-provided call, so None is safe here.
            loss = self._native_ce(x.reshape(-1, x.size(-1)),
                                   targets.reshape(-1))
            return None, loss
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, ids, n_tokens, temperature=0.9, top_p=0.9,
                 repetition_penalty=1.15, greedy=False, stop_token=None):
        """stop_token: end generation when the model emits this id (the
        EOT written by prepare_data.encode_documents). The stop token is
        not included in the returned sequence."""
        assert ids.size(0) == 1, "generate() handles one sequence at a time"
        assert ids.size(1) + n_tokens <= self.cfg.block_size, (
            "prompt + n_tokens exceeds block_size (KV cache and RoPE "
            "tables are sized to block_size)")
        was_training = self.training
        self.eval()
        caches = [{"k": None, "v": None} for _ in self.blocks]

        def step(toks, start_pos):
            """Run only the new token(s) through the model via the caches."""
            x = self.drop(self.tok_emb(toks))
            for blk, c in zip(self.blocks, caches):
                x = blk(x, cache=c, start_pos=start_pos)
            return self.lm_head(self.final_norm(x))[0, -1]

        logits = step(ids, 0)                        # PREFILL: whole prompt
        for _ in range(n_tokens):                    # DECODE: 1 token/step
            if repetition_penalty and repetition_penalty != 1.0:
                seen = torch.unique(ids)
                pos = logits[seen] > 0
                logits[seen] = torch.where(pos,
                                           logits[seen] / repetition_penalty,
                                           logits[seen] * repetition_penalty)
            if greedy:
                nxt = torch.argmax(logits)
            else:
                logits = logits / temperature
                if top_p and top_p < 1.0:
                    probs = F.softmax(logits, dim=-1)
                    sp, si = torch.sort(probs, descending=True)
                    keep = torch.cumsum(sp, dim=-1) - sp < top_p
                    keep[0] = True                        # always keep top-1
                    logits[si[~keep]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1).squeeze()
            if stop_token is not None and nxt.item() == stop_token:
                break
            ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
            logits = step(ids[:, -1:], ids.size(1) - 1)
        if was_training:
            self.train()
        return ids


# ---------------------------------------------------------------------- #
# Data: memmapped token ids — RAM holds only the pages being read
# ---------------------------------------------------------------------- #
def get_batch(split, cfg, device, data_dir, generator=None):
    # Re-opening the memmap per call is deliberate (the nanoGPT pattern):
    # it avoids a slow memory leak from long-lived memmaps.
    data = np.memmap(os.path.join(data_dir, f"{split}.bin"),
                     dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - cfg.block_size - 1, (cfg.micro_batch,),
                       generator=generator)
    x = torch.stack([torch.from_numpy(
        data[i: i + cfg.block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(
        data[i + 1: i + 1 + cfg.block_size].astype(np.int64)) for i in ix])
    if device.startswith("cuda"):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def lr_schedule(step, cfg):                # same shape as lr_schedule.py
    if step < cfg.warmup_steps:
        return cfg.peak_lr * (step + 1) / cfg.warmup_steps
    prog = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    floor = 0.1 * cfg.peak_lr
    return floor + (cfg.peak_lr - floor) * 0.5 * (1 + math.cos(math.pi * prog))


# ---------------------------------------------------------------------- #
# Minimal BPE encode/decode from a merges .json — lets --prompt/--merges
# work without importing the training tokenizer. Encoding replays merges
# lowest-rank-first, exactly like training did; prompts are short, so the
# simple O(n^2) loop is fine.
# ---------------------------------------------------------------------- #
def load_bpe(merges_path):
    merges, vocab = {}, {i: bytes([i]) for i in range(256)}
    for i, (a, b) in enumerate(json.load(open(merges_path))):
        merges[(a, b)] = 256 + i
        vocab[256 + i] = vocab[a] + vocab[b]
    return merges, vocab


def bpe_encode(text, merges):
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        pairs = set(zip(ids, ids[1:]))
        best = min(pairs, key=lambda p: merges.get(p, float("inf")))
        if best not in merges:
            break
        new, out, i = merges[best], [], 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best:
                out.append(new); i += 2
            else:
                out.append(ids[i]); i += 1
        ids = out
    return ids


def bpe_decode(ids, vocab):
    return b"".join(vocab[t] for t in ids).decode("utf-8", "replace")


# ---------------------------------------------------------------------- #
# Training
# ---------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="gpt2ish", choices=PRESETS)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init_from", default=None,
                    help="load model weights from another checkpoint, with "
                         "a fresh optimizer/schedule/step count — for "
                         "continuing training as a new run (e.g. a second "
                         "epoch), as opposed to --resume's exact-state "
                         "continuation")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--compile_mode", default=None,
                    help="torch.compile mode, e.g. reduce-overhead "
                         "(CUDA graphs) — riskier, test before trusting "
                         "on a long run")
    ap.add_argument("--fused_ce", action="store_true",
                    help="fused lm_head+cross_entropy (torch>=2.13's "
                         "nn.LinearCrossEntropyLoss) — helps large-"
                         "vocab_size runs (see the fineweb preset)")
    ap.add_argument("--sample_only", action="store_true")
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--warmup_steps", type=int, default=None)
    ap.add_argument("--eval_every", type=int, default=None)
    ap.add_argument("--ckpt_every", type=int, default=None)
    ap.add_argument("--micro_batch", type=int, default=None,
                    help="sequences per fwd/bwd; raise for more GPU "
                         "utilization until VRAM runs out (pair with "
                         "--grad_accum to hold tokens/step constant)")
    ap.add_argument("--grad_accum", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default=None, help="force 'cpu' or 'cuda'")
    ap.add_argument("--merges", default=None,
                    help="BPE merges .json (bpe_tokenizer_v2 format) for "
                         "encoding/decoding samples")
    ap.add_argument("--tokenizer_json", default=None,
                    help="HF tokenizers tokenizer.json (from "
                         "prepare_real_data.py) for encoding/decoding "
                         "samples; use instead of --merges for real-data "
                         "checkpoints")
    ap.add_argument("--prompt", default=None,
                    help="prompt text for --sample_only (needs --merges or "
                         "--tokenizer_json)")
    ap.add_argument("--stop_token", type=int, default=None,
                    help="EOT id: sampling stops when the model emits it")
    ap.add_argument("--n_tokens", type=int, default=None,
                    help="sample length (default: fill the block)")
    ap.add_argument("--log", default=None,
                    help="JSONL metrics file (default: <out>_log.jsonl)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    cfg = Config(**PRESETS[args.preset])
    if args.max_steps:
        cfg.max_steps = args.max_steps
    if args.warmup_steps:
        cfg.warmup_steps = args.warmup_steps
    if args.eval_every:
        cfg.eval_every = args.eval_every
    if args.ckpt_every:
        cfg.ckpt_every = args.ckpt_every
    if args.micro_batch:
        cfg.micro_batch = args.micro_batch
    if args.grad_accum:
        cfg.grad_accum = args.grad_accum

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    torch.set_float32_matmul_precision("high")           # TF32 matmuls
    print(f"device {device} | amp {amp_dtype if device == 'cuda' else 'off'}"
          f"{' | fused_ce' if args.fused_ce else ''}")

    model = GPT(cfg, fused_ce=args.fused_ce).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"preset {args.preset}: {n_params / 1e6:.1f}M params | "
          f"{cfg.micro_batch * cfg.block_size * cfg.grad_accum:,} tokens/step")

    # AdamW with our decay rule: 2D matrices decay, norms/1D don't
    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=cfg.peak_lr, betas=(0.9, 0.95),
        fused=(device == "cuda"))
    scaler = torch.amp.GradScaler(device,
                                  enabled=(device == "cuda" and not use_bf16))

    def check_arch(ck_config, ck_path):
        # Architecture must match the checkpoint — fail loudly, not with a
        # cryptic shape error deep inside load_state_dict.
        arch_keys = ("vocab_size", "d_model", "n_heads", "n_kv_heads",
                     "d_ff", "n_layers", "block_size")
        mismatch = {k: (ck_config[k], getattr(cfg, k)) for k in arch_keys
                    if ck_config[k] != getattr(cfg, k)}
        if mismatch:
            raise SystemExit(f"checkpoint/preset architecture mismatch: "
                             f"{mismatch} — use the preset {ck_path} "
                             f"was trained with")

    def load_model_state(sd):
        # A checkpoint saved with --fused_ce carries an extra
        # "_native_ce.linear.weight" key — tied to tok_emb.weight (same
        # tensor, same values), never independent information. Drop it so
        # loading doesn't require --fused_ce to match between the run that
        # saved the checkpoint and whatever loads it later (--resume,
        # --init_from, or --sample_only, which never even invokes
        # _native_ce, since generation always goes through targets=None).
        sd = {k: v for k, v in sd.items() if k != "_native_ce.linear.weight"}
        model.load_state_dict(sd)

    start_step, best_val = 0, float("inf")
    if args.resume and os.path.exists(args.out):
        ck = torch.load(args.out, map_location=device)
        check_arch(ck["config"], args.out)
        load_model_state(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"]
        best_val = ck.get("best_val", float("inf"))
        print(f"resumed from step {start_step}")
    elif args.init_from:
        ck = torch.load(args.init_from, map_location=device)
        check_arch(ck["config"], args.init_from)
        load_model_state(ck["model"])
        print(f"initialized weights from {args.init_from} "
              f"(fresh optimizer, schedule, step count)")

    # Keep a handle to the uncompiled model: torch.compile wraps it and
    # prefixes state_dict keys with `_orig_mod.`, which would poison
    # checkpoints. Always save/load through raw_model.
    raw_model = model
    if args.compile:
        model = torch.compile(model, mode=args.compile_mode)

    def save_ckpt(step, vl):
        payload = {"model": raw_model.state_dict(), "opt": opt.state_dict(),
                   "step": step, "best_val": best_val,
                   "config": asdict(cfg)}
        tmp = args.out + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, args.out)          # atomic: no corrupt checkpoints

    if args.sample_only:
        if args.prompt and not (args.merges or args.tokenizer_json):
            raise SystemExit("--prompt requires --merges or --tokenizer_json "
                             "to encode it")
        encode_fn = decode_fn = None
        if args.tokenizer_json:
            from tokenizers import Tokenizer
            hf_tok = Tokenizer.from_file(args.tokenizer_json)
            encode_fn = lambda t: hf_tok.encode(t).ids
            decode_fn = lambda ids: hf_tok.decode(ids)
        elif args.merges:
            merges, vocab = load_bpe(args.merges)
            encode_fn = lambda t: bpe_encode(t, merges)
            decode_fn = lambda ids: bpe_decode(ids, vocab)
        prompt_ids = (encode_fn(args.prompt) if args.prompt
                      else [10])                     # default seed: '\n'
        assert len(prompt_ids) < cfg.block_size, "prompt exceeds block_size"
        ids = torch.tensor([prompt_ids], device=device)
        n = min(args.n_tokens or cfg.block_size,
                cfg.block_size - ids.size(1))        # KV cache is block-sized
        out = raw_model.generate(ids, n,
                                 stop_token=args.stop_token)[0].tolist()
        print(decode_fn(out) if decode_fn else out)
        return

    @torch.no_grad()
    def val_loss():
        model.eval()
        # Same seed every eval -> same val batches -> the val curve
        # measures the model, not batch sampling noise.
        g = torch.Generator().manual_seed(42)
        losses = []
        for _ in range(cfg.eval_batches):
            x, y = get_batch("val", cfg, device, args.data_dir, generator=g)
            with torch.autocast(device_type=device, dtype=amp_dtype,
                                enabled=(device == "cuda")):
                _, loss = model(x, y)
            losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    log_path = args.log or args.out.replace(".pt", "") + "_log.jsonl"

    model.train()
    t0, tokens_seen = time.time(), 0
    for step in range(start_step, cfg.max_steps):
        lr = lr_schedule(step, cfg)
        for group in opt.param_groups:
            group["lr"] = lr

        opt.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(cfg.grad_accum):
            x, y = get_batch("train", cfg, device, args.data_dir)
            with torch.autocast(device_type=device, dtype=amp_dtype,
                                enabled=(device == "cuda")):
                _, loss = model(x, y)
            scaler.scale(loss / cfg.grad_accum).backward()
            loss_accum += loss.item() / cfg.grad_accum   # mean over window
            tokens_seen += x.numel()
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   cfg.grad_clip)
        scaler.step(opt)
        scaler.update()

        if step % cfg.eval_every == 0 or step == cfg.max_steps - 1:
            vl = val_loss()
            if vl < best_val:
                best_val = vl
                torch.save({"model": raw_model.state_dict(), "step": step + 1,
                            "config": asdict(cfg)},
                           args.out.replace(".pt", "_best.pt"))
            tps = tokens_seen / max(1e-9, time.time() - t0)
            print(f"step {step:>6}  train {loss_accum:.3f}  val {vl:.3f}  "
                  f"lr {lr:.2e}  gnorm {grad_norm:.2f}  "
                  f"{tps / 1e3:.0f}k tok/s", flush=True)
            # Append-only JSONL: crash-safe, resume-friendly, trivially
            # plottable (pandas.read_json(path, lines=True)).
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "step": step, "train": round(loss_accum, 4),
                    "val": round(vl, 4), "lr": lr,
                    "gnorm": round(float(grad_norm), 3),
                    "tok_s": round(tps),
                    "tokens": (step + 1) * cfg.micro_batch
                              * cfg.block_size * cfg.grad_accum,
                    "wall_s": round(time.time() - t0, 1),
                }) + "\n")
        if step > 0 and (step % cfg.ckpt_every == 0
                         or step == cfg.max_steps - 1):
            save_ckpt(step + 1, best_val)
            print(f"  checkpoint -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
