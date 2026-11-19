# Getting Started

A walkthrough from "clone the repo" to training a real language model on
real data, in three escalating steps: Shakespeare (minutes, proves the
pipeline works), TinyStories (hours, your first real trained model),
FineWeb-Edu (many hours, the real thing). Each step reuses the same code
— `train_gpt.py` — against progressively bigger data and vocabularies.

For exact expected numbers at each phase, deep calibration detail, and a
long troubleshooting list, see **RUNBOOK.md**. This guide is the shorter
path through the same territory.

## 1. Install

Pick one. All three end up running the same `train_gpt.py`.

### Linux (native)

```bash
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

PyPI's default `torch` wheel bundles CUDA and pulls in a real `triton`
wheel automatically — `--compile` needs no extra setup here.

### Windows (native)

```bash
uv sync
```

PyPI serves **CPU-only** `torch` on Windows, so this project pins
PyTorch's own index in `pyproject.toml` (currently `cu132`). If the check
above prints `False`, run `nvidia-smi` and compare its reported "CUDA
Version" against the pin — if yours is lower, swap both `cu132`s in
`pyproject.toml` for a matching `cuXXX`, then `uv lock && uv sync`.

`--compile` needs extra setup here — Triton has no official Windows
wheel. `uv sync --extra compile` installs a community build
(`triton-windows`), but it also needs a real C compiler (Visual Studio
Build Tools) on PATH. See RUNBOOK.md's **"Windows: enabling --compile"**
for the full one-time setup — worth doing, `--compile` is a genuine
2-3x speedup once vocab-size is handled right (see step 3).

### Docker

Builds a Linux environment on top of Windows/WSL2 — same official
`torch`/`triton` wheels as native Linux, none of the Windows-specific
workarounds above. Doesn't run any faster on the same GPU; it's for
lower-friction, more reliable long unattended runs.

```bash
docker build -t llm-fineweb .
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi   # passthrough sanity check
```

Then run anything by mounting the project directory in:

```bash
docker run --rm --gpus all -v "/path/to/pytorch:/workspace" -w /workspace \
    llm-fineweb python train_gpt.py --preset tiny
```

(From Git Bash on Windows: prefix with `MSYS_NO_PATHCONV=1` and use a
`//c/Users/...` double-slash path, or MSYS mangles the volume mount.)

## 2. Prove it works

Three quick checks, cheapest first — each isolates a different thing
that could be wrong before you spend real GPU time.

```bash
# 1. GPU visible? (see install section above if this prints False on Windows)
uv run python -c "import torch; print(torch.cuda.is_available())"

# 2. Code correct, independent of GPU/driver? (~1 min on CPU)
uv run train-gpt --preset tiny --device cpu --max_steps 100

# 3. GPU path correct? Header should say `device cuda | amp torch.bfloat16`
uv run train-gpt --preset tiny --max_steps 200
```

If all three pass, the pipeline works end to end. RUNBOOK.md's Phases
0-3 cover this in more depth (expected loss numbers, a kill+resume drill,
a `--compile` drill) if you want to be thorough before committing to a
long run.

## 3. The three-corpus journey

**Match the preset's `vocab_size` to whichever data you last wrote** —
this is the single most common mistake. Each corpus below has its own
tokenizer and its own preset; mixing them (e.g. sampling `fineweb`
against Shakespeare's bins) silently corrupts output in eager mode, or
crashes loudly under `--compile` (see RUNBOOK's troubleshooting).

| Corpus | prep script | vocab | preset |
|---|---|---|---|
| Shakespeare | `prepare_data.py` | 512 | `tiny` / `debug` |
| TinyStories | `prepare_real_data.py` | 32,768 | `small` / `gpt2ish` |
| FineWeb-Edu | `prepare_fineweb.py` | 49,152 | `fineweb` |

### Step 1 — Shakespeare (minutes)

Proves the whole loop — data, model, training, checkpointing, sampling —
works, on a corpus small enough that a bug shows up in seconds, not hours.

```bash
uv run prepare-data                        # writes data/{train,val}.bin
uv run train-gpt --preset tiny             # a couple minutes, CPU or any GPU
uv run train-gpt --preset tiny --resume --sample_only \
    --merges shake_merges.json --prompt "ROMEO:" --n_tokens 60
```

Expect pseudo-Shakespeare gibberish with correct speaker-name structure
(`ROMEO:`, line breaks) — the point isn't quality, it's confirming
sampling actually works before you wait hours for something worth
sampling.

### Step 2 — TinyStories (hours, your first real model)

```bash
uv sync --extra data
uv run python prepare_real_data.py --n_stories 2200000
# smaller --n_stories first (e.g. 300000) if you want a faster pipeline
# check before committing to the full ~2.1M-story corpus
```

Prints the actual token count when done — use it to compute the launch
schedule (steps = train tokens ÷ tokens-per-step, where tokens-per-step
= `micro_batch × block_size × grad_accum` = 491,520 for `gpt2ish`'s
defaults):

```bash
uv run train-gpt --preset gpt2ish --out ckpt_full.pt \
    --max_steps 893 --warmup_steps 36 --eval_every 75 --ckpt_every 150
    # ^ the numbers this repo measured for the full 2.1M-story corpus
    #   (439M train tokens) — recompute for your own token count
```

Add `--compile` if you did the Windows Triton setup (or you're on
Linux/Docker) — expect roughly 2-2.5x over eager mode. Sample the same
way, pointing at the tokenizer this run trained:

```bash
uv run train-gpt --preset gpt2ish --out ckpt_full.pt --resume --sample_only \
    --tokenizer_json tinystories_tokenizer.json \
    --prompt "Once upon a time" --stop_token 0
```

Expect real (if simple) short stories — coherent grammar, consistent
characters, the TinyStories moral-at-the-end style.

### Step 3 — FineWeb-Edu (many hours, the real thing)

```bash
uv sync --extra data
uv run python prepare_fineweb.py --n_tokens 2000000000
# pick a token budget deliberately — this corpus is large enough that
# training less than one full epoch is normal, unlike the two steps
# above; see RUNBOOK Phase 6 for time/disk estimates at a few sizes
```

Default tokenizer here is SmolLM2's (49,152 vocab) — SmolLM2 was itself
trained substantially on FineWeb-Edu, so its vocabulary is shaped by this
exact corpus, unlike GPT-2's tokenizer (still available via `--tokenizer
gpt2`, but 50,257 isn't a GPU-friendly size and hung `--compile`
indefinitely on this repo's setup — see RUNBOOK's troubleshooting).

`--fused_ce` matters here in a way it didn't for the smaller vocabs
above — it fuses the final projection with the loss (torch's native
`nn.LinearCrossEntropyLoss`, needs torch>=2.13) so the
`[batch, seq, vocab_size]` logits tensor is never materialized, which is
a real bottleneck at this vocab size. Combined with `--compile`, this
repo measured **~4.3x over plain eager** (56-61k vs 13k tok/s):

```bash
uv run train-gpt --preset fineweb --data_dir data_fineweb --out ckpt_fineweb.pt \
    --max_steps 3865 --warmup_steps 150 --eval_every 200 --ckpt_every 200 \
    --compile --fused_ce
    # ^ the numbers this repo measured for a 2B-token budget — recompute
    #   for your own (see the formula in Step 2)
```

Sample the same way as Step 2, pointing at `smollm2_tokenizer.json` —
EOT id is **0** here (same convention as TinyStories' custom tokenizer):

```bash
uv run train-gpt --preset fineweb --out ckpt_fineweb.pt --resume --sample_only \
    --tokenizer_json smollm2_tokenizer.json \
    --prompt "The" --stop_token 0
```

Expect noticeably more coherent, varied prose than the TinyStories run —
real web text instead of a narrow children's-story domain, and
(depending on your token budget) more total training.

### If something's slow or wrong

RUNBOOK.md has the deep version of all of this: exact expected numbers
per phase, a VRAM-calibration drill, and a troubleshooting list covering
(among other things) the specific bugs this repo hit building the
FineWeb-Edu path — a `--compile` hang traced to a non-power-of-2
`vocab_size`, a Dynamo crash tracing into a third-party fused-CE kernel,
why `mode="reduce-overhead"` doesn't work with this codebase's training
loop, and a Docker gotcha where stopping the shell that launched a
foreground `docker run` doesn't actually stop the container. Worth a
read before assuming a slowdown or crash is new.
