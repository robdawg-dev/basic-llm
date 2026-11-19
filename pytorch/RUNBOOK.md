# Test-Drive Runbook — llm-from-scratch on a 12 GB GPU

Escalation ladder: each phase is cheap and validates the next one's
assumptions. Don't skip — every phase has "expected" numbers from verified
runs, so a mismatch tells you *where* something differs on your machine.

## Phase 0 — Setup (10 min)

    unzip llm-from-scratch.zip && cd llm-from-scratch
    uv sync

`uv sync` creates `.venv/` and installs torch + numpy exactly as pinned in
`uv.lock`. Verify the GPU is visible:

    uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

Expected: `True` and your GPU's name. If `False` (ALWAYS the case on
Windows first try — PyPI serves CPU-only torch there): run `nvidia-smi`,
note the CUDA Version, uncomment the index block at the bottom of
`pyproject.toml` with a cuXXX at or below it, then `uv lock && uv sync`.

    uv run prepare-data

Expected output:
    data/train.bin: 550,017 tokens
    data/val.bin:   28,949 tokens

## Phase 1 — CPU parity check (5 min)

Proves the code behaves identically to the verified reference, isolating
"my GPU/driver" from "the code" before the GPU ever runs.

    uv run train-gpt --preset tiny --device cpu --max_steps 100

Expected (default seed 1337; tiny BLAS differences may nudge digits):
    step      0  train 6.219  val 6.202  ...  gnorm 1.33
    step     50  train ~4.50  val ~4.43
    step     99  train ~4.03  val ~4.01

Checkpoint `ckpt.pt` and `ckpt_log.jsonl` should exist afterward.

## Phase 2 — GPU smoke test (2 min)

    rm ckpt.pt ckpt_best.pt ckpt_log.jsonl        # fresh start
    uv run train-gpt --preset tiny --max_steps 200

Check three things:
1. Header says `device cuda | amp torch.bfloat16` (Ampere/RTX 30xx or
   newer). Older cards show float16 — fine, the GradScaler handles it,
   but watch for NaN losses (see Troubleshooting).
2. Loss trajectory matches Phase 1's shape (~3.6 by step 200).
3. Sampling works end to end:

    uv run train-gpt --preset tiny --resume --sample_only \
        --merges shake_merges.json --prompt "ROMEO:" --n_tokens 60

Expected: pseudo-Shakespeare gibberish with speaker-name structure.

## Phase 3 — Debug preset + resilience drills (15–25 min)

The 4.9M-parameter model, and the phase where you rehearse everything a
multi-day run depends on.

    rm ckpt.pt ckpt_best.pt ckpt_log.jsonl
    uv run train-gpt --preset debug

Drills, in order:
- **Kill + resume**: Ctrl-C mid-run, then
  `uv run train-gpt --preset debug --resume` — loss must continue
  from where it stopped (checkpoints land every 500 steps; you lose at
  most the current window).
- **Compile**: restart with `--compile` added. First ~2 steps are slow
  (compilation), then throughput should rise 20–40% vs. the log's
  earlier tok/s rows. On Windows, `--compile` needs Triton, which has no
  official Windows wheel — see "Windows: enabling --compile" below before
  trying this drill.
- **Watch VRAM**: `nvidia-smi -l 2` in a second terminal. debug should
  use only ~1–2 GB.

Expected end state — READ THIS, it looks alarming and isn't: val bottoms
out around 2.8–2.9 within a few hundred steps, then RISES while train
keeps falling toward zero. That's overfitting by design: 2000 steps here
is ~65M tokens, ~119 epochs over a 550k-token corpus, and a 5M-param
model memorizes it. This is the "train falls, val rises" troubleshooting
row demonstrated live — and note ckpt_best.pt quietly saved the model at
val's minimum. Real runs (one epoch over a large corpus) don't do this.
Samples from ckpt_best.pt should read as clearly better Shakespeare than
tiny's; samples from the final ckpt.pt will be memorized corpus chunks. Plot the run
(`uv sync --extra plot` once, first):

    uv run python -c "import pandas as pd; df = pd.read_json('ckpt_log.jsonl', lines=True); print(df[['step','train','val','gnorm','tok_s']])"

## Phase 4 — Size the flagship (15 min)

Calibrate gpt2ish (~101M) on Shakespeare data before real data exists.
Note: gpt2ish expects vocab 32768 but Shakespeare bins are vocab-512 —
that's fine for a throughput test (ids just occupy the bottom of the
embedding); the loss numbers are meaningless here, only tok/s and VRAM
matter.

    uv run train-gpt --preset gpt2ish --out size_test.pt --max_steps 30

- Read steady-state tok/s from the log (~50–150k expected).
- Watch `nvidia-smi`: if comfortably under ~10 GB, edit `micro_batch`
  upward in the preset (16, then 20...) until OOM, back off one notch.
  When raising micro_batch, lower grad_accum to keep
  micro_batch × grad_accum ≈ 480 (constant tokens/step).
- Budget the real run: total_tokens / tok_s = seconds.
  E.g. 1B tokens at 80k tok/s ≈ 3.5 hours/epoch of pure compute.
- Delete size_test.pt when done.

## Phase 5 — Real data + launch

    uv sync --extra data       # adds tokenizers + datasets

Use `prepare_real_data.py`, not the smoke-test path in prepare_data.py's
docstring: it trains a fast Rust-backed `tokenizers.ByteLevelBPETokenizer`
(32768 vocab, `<|endoftext|>` at id 0) instead of `bpe_tokenizer_v2.py`'s
hand-built one — that trainer recomputes pair counts from scratch on
every merge, fine for Shakespeare's 256-merge tokenizer but the wrong
algorithmic shape for a 32k-vocab tokenizer over a real corpus (hours-to-
days, not minutes).

    uv run python prepare_real_data.py --n_stories 2200000
    # --n_stories caps how many TinyStories examples to stream (the full
    # train split has 2,119,719); pass a smaller number for a quick
    # end-to-end pipeline check before committing to the full corpus.

Writes `data/train.bin` / `data/val.bin` and a `tokenizer.json`-format
file (default `tinystories_tokenizer.json`). Sanity checks before burning
GPU-hours:
- token count printed matches expectation (TinyStories full corpus ≈
  440M train-split tokens; a subset scales roughly linearly with
  `--n_stories`)
- first tokens decode to a real story opening and EOT count ≈ story
  count:

      uv run python -c "
      import numpy as np
      from tokenizers import Tokenizer
      tok = Tokenizer.from_file('tinystories_tokenizer.json')
      train = np.memmap('data/train.bin', dtype=np.uint16, mode='r')
      print(tok.decode(train[:50].tolist()))
      print('EOT count:', int((train == 0).sum()))"

Because this tokenizer is retrained per run, a checkpoint's token ids are
only meaningful under the `tokenizer.json` that produced them — never mix
a checkpoint with a tokenizer file from a different prepare_real_data.py
run (rename/move one of them aside if you keep both around, the way
`ckpt_subset300k*` / `tinystories_tokenizer_subset300k.json` are kept
separate from the full-corpus `ckpt_full*` / `tinystories_tokenizer.json`
in this repo's history).

Set `max_steps` for your token budget FIRST (steps = train.bin tokens ÷
491,520 at default settings) — the cosine schedule is baked to
`max_steps`, and extending it mid-run distorts the decay (you saw this in
testing). `--warmup_steps`, `--eval_every`, and `--ckpt_every` are also
CLI overrides (same pattern as `--max_steps`), so a run's cadence can be
rescaled to its step count without hand-editing `PRESETS`:
- `warmup_steps` ≈ 3–5% of `max_steps`
- `ckpt_every` sized so checkpoints land under an hour apart:
  `ckpt_every ≈ 3600 / seconds_per_step`
- `eval_every` ≈ `max_steps / 10`–`max_steps / 20` for a readable curve

Launch under tmux/screen (survives SSH drops; checkpoints survive
everything else):

    tmux new -s train
    uv run train-gpt --preset gpt2ish --out ckpt_full.pt \
        --max_steps <computed> --warmup_steps <computed> \
        --eval_every <computed> --ckpt_every <computed> --compile
    # detach: Ctrl-B D    monitor: tail -f ckpt_full_log.jsonl

Sample mid-run anytime from another terminal (reads the checkpoint,
doesn't disturb training):

    uv run train-gpt --preset gpt2ish --out ckpt_full.pt --resume \
        --sample_only --tokenizer_json tinystories_tokenizer.json \
        --prompt "Once upon a time" --stop_token 0

(`--merges` still works, but only for checkpoints trained on the
`bpe_tokenizer_v2.py` format, e.g. `shake_merges.json` — use
`--tokenizer_json` for anything from `prepare_real_data.py`.)

## Phase 6 — FineWeb-Edu (bigger vocab, real speed tuning)

    uv sync --extra data

Use `prepare_fineweb.py`, not `prepare_real_data.py`: FineWeb-Edu's smallest
pre-shuffled sample (`sample-10BT`) is still billions of tokens, so it
streams+encodes in small batches straight to `train.bin`/`val.bin` (bounded
memory, unlike `prepare_real_data.py`'s "collect everything into a Python
list" approach) against a fixed `--n_tokens` budget decided up front:

    uv run python prepare_fineweb.py --n_tokens 2000000000
    # --config sample-10BT (default) / sample-100BT / sample-350BT / default
    # --tokenizer smollm2 (default, reuses SmolLM2's 49,152-vocab tokenizer —
    #   SmolLM2 was itself trained substantially on FineWeb-Edu, so its
    #   vocab is shaped by this exact corpus, unlike GPT-2's 2019 WebText-
    #   derived one, still available via --tokenizer gpt2) or --tokenizer
    #   train (fits a fresh BPE on a bounded document sample, not the whole
    #   budget)

Use the **`fineweb`** preset with this data, not `gpt2ish` — same
architecture, `vocab_size=49152` instead of 32768. 49,152 = 384x128 is
already a GPU-friendly size (see Troubleshooting for why that matters —
GPT-2's raw 50,257 reproducibly **hung `--compile` indefinitely** on this
setup; a tokenizer with an awkward vocab size needs padding to the next
multiple of 64, the nanoGPT trick, to avoid it).

**`--fused_ce`** fuses the lm_head projection with the cross-entropy loss
(torch's native `nn.LinearCrossEntropyLoss`, added in 2.13) so the full
`[batch, seq, vocab_size]` logits tensor is never materialized. This
matters once `vocab_size` is large: that tensor becomes its own memory-
bandwidth-bound bottleneck, separate from (and on top of) the
transformer's matmuls. Measured on this repo's setup (`fineweb` preset,
RTX 4070 SUPER, torch 2.14.0+cu132):

| Config | tok/s | 3,865-step run |
|---|---|---|
| eager | 13k | ~40.6h |
| `--compile` | 28k | ~18.8h |
| `--fused_ce` | 34k | ~15.5h |
| `--compile --fused_ce` | **56-61k** | **~8.7-9.4h** |

`--compile --fused_ce` needs no extra flags to work together, and no
extra dependency — this project used `liger-kernel` for fused CE before
torch 2.13 shipped a native version; liger's equivalent needed
`@torch.compiler.disable` to avoid a Dynamo/fake-tensor crash tracing its
custom autograd.Function under `--compile` (still true if you ever
reach for a third-party fused-CE kernel again), but torch's native one
needs no such workaround and measured faster besides — see git history
if you want the liger-based numbers (44k tok/s) for comparison.

`--micro_batch`/`--grad_accum` are CLI overrides (pair them to hold
`micro_batch × grad_accum` constant = constant tokens/step) for
re-calibrating VRAM headroom without hand-editing `PRESETS` — worth
re-checking after `--fused_ce`, since it frees the VRAM the logits tensor
used to occupy. On this setup, raising `micro_batch` past its `gpt2ish`
default (12) landed within noise of no change (already compute-bound:
power draw sits at ~200-216W of the card's 220W cap either way) — recheck
on your own hardware rather than assuming either result.

**Docker**: `Dockerfile`/`.dockerignore` in this repo build a Linux
environment (official `torch`/`triton` wheels, no Windows workarounds)
for exactly this kind of comparison testing:

    docker build -t llm-fineweb .
    docker run --rm --gpus all -v "//c/path/to/pytorch:/workspace" -w /workspace \
        llm-fineweb python train_gpt.py --preset fineweb --data_dir data_fineweb \
        --out ckpt_fineweb.pt --compile --fused_ce ...

(from Git Bash: prefix with `MSYS_NO_PATHCONV=1` and use the `//c/...`
double-slash form, or MSYS mangles the volume-mount path. Prefer
`docker run -d` + `docker logs -f <name>` for anything long — stopping
the *shell* that launched a foreground `docker run` does not stop the
*container*; it keeps running and holding VRAM until you `docker stop`
it by name.) Measured identical throughput to native Windows (56k tok/s,
same power draw) once `build-essential` is in the image — Docker doesn't
make this workload *faster* on the same GPU, it trades zero speed for
meaningfully less fragile plumbing (no `triton-windows` version pinning,
no MSVC/`cl.exe` PATH dance) ahead of a many-hour unattended run.

## Windows: enabling --compile

Triton has no official Windows wheel, so `--compile` fails with
`TritonMissing` out of the box. Fix, once per machine:

1. Install a matching community build. Pin it to the exact `triton`
   version torch's own `requires-dist` pins (check with
   `python -c "import importlib.metadata as m; [print(r) for r in m.requires('torch') if 'triton' in r.lower()]"`
   against a non-Windows `torch` metadata, or just check PyPI) — this
   repo pins `triton-windows` in `pyproject.toml`'s `compile` extra:

       uv sync --extra compile

2. Triton JIT-compiles kernels through `cl.exe` (MSVC), which isn't
   bundled. Install Visual Studio Build Tools with the C++ workload if
   `where cl.exe` finds nothing:

       winget install --id Microsoft.VisualStudio.2022.BuildTools --silent \
           --accept-package-agreements --accept-source-agreements \
           --override "--wait --quiet --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"

3. `cl.exe` isn't added to PATH by the installer (by design — MSVC
   supports multiple side-by-side versions). Prepend its directory before
   any `--compile` run, once per shell session:

       # find it once:
       $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
       & $vswhere -latest -property installationPath
       # then (adjust the MSVC version folder to what's actually there):
       $env:PATH = "C:\Program Files (x86)\Microsoft Visual Studio\<ver>\BuildTools\VC\Tools\MSVC\<msvc-ver>\bin\Hostx64\x64;$env:PATH"

   (bash/git-bash equivalent: `export PATH="/c/Program Files (x86)/.../Hostx64/x64:$PATH"`)

Verified working on this repo's setup (RTX 4070 SUPER, torch 2.11.0+cu128,
triton-windows 3.6.0.post26, Python 3.14): `gpt2ish` throughput rose from
~9k tok/s at step 0 (compilation) to ~49k tok/s by step 20 — roughly
1.7–1.9x over eager mode's ~26–29k tok/s.

## Troubleshooting

- **CUDA OOM**: lower `micro_batch`, raise `grad_accum` to compensate.
  If it OOMs only with `--compile`, try without.
- **torch installed CPU-only**: uncomment the CUDA index block in
  `pyproject.toml`, then `uv lock && uv sync`.
- **NaN loss on an fp16 card**: first restarts are the GradScaler
  calibrating; persistent NaN → halve `peak_lr`.
- **gnorm pinned at 1.00 for hundreds of steps**: gradients constantly
  clipped → `peak_lr` too hot for this data; drop 2–3x.
- **train falls, val flat/rising**: overfitting — more data, or set
  `dropout=0.1` in the preset (only worth it if looping a small corpus).
- **Slow first steps with --compile**: normal (kernel compilation).
- **`TritonMissing` with --compile on Windows**: see "Windows: enabling
  --compile" above.
- **`device-side assert triggered` / `index out of bounds` with
  --compile**: this is usually not a Triton bug — it means the preset's
  `vocab_size` doesn't match the ids actually in `data/`. Eager mode can
  silently read garbage on an out-of-range embedding index; a compiled
  Triton kernel adds an explicit bounds check and crashes loudly instead.
  Whichever data you last wrote with `prepare_data.py` (vocab 512) vs.
  `prepare_real_data.py` (vocab 32768) must match the preset you launch
  with (`tiny`/`debug` vs. `small`/`gpt2ish`).
- **Resume refuses with "architecture mismatch"**: you changed presets;
  use the one the checkpoint was trained with (this guard is a feature).
- **Throughput sags mid-run**: check `nvidia-smi` for thermal throttling
  or a competing process; the tok_s column in the log timestamps it.
- **`--compile` hangs indefinitely (GPU shows activity, but no step
  progresses for far longer than the loss values would take)**: check
  `nvidia-smi` power draw over a few samples. A real hang reads flat and
  low (~60W on a card whose TDP is 220W+) — the SMs are spinning, not
  computing. Fluctuating power (compute bursts alternating with idle) just
  means `--compile` is still autotuning candidate kernels, which can
  legitimately take 10+ minutes before the first step prints; let it run
  longer before concluding it's actually stuck. A genuine flat-power hang
  on a non-power-of-2 `vocab_size` (e.g. GPT-2's real 50,257) is fixed by
  padding to the next multiple of 64 — see Phase 6.
- **A third-party fused-CE kernel crashes with a `FakeTensor`/`addmm`
  `TypeError` during Dynamo tracing under `--compile`**: this repo hit
  this with `liger-kernel` before switching to torch's own native
  `nn.LinearCrossEntropyLoss` (2.13+), which doesn't have the problem.
  Dynamo was trying to trace *inside* liger's custom autograd.Function,
  which it couldn't do correctly for that op. If you ever reach for
  another third-party fused kernel and hit this: wrap the call in
  `@torch.compiler.disable` so Dynamo treats it as an opaque call instead
  of tracing into it.
- **`--compile_mode reduce-overhead` crashes with "accessing tensor output
  of CUDAGraphs that has been overwritten by a subsequent run"**: CUDA
  graphs reuse the same memory buffers across replays, which conflicts
  with this codebase's grad-accumulation loop (many compiled calls per
  optimizer step, each holding a reference to the previous call's output
  for backward). Fixable in principle with
  `torch.compiler.cudagraph_mark_step_begin()` before each call in that
  loop, but untested here — `reduce-overhead` mainly helps kernel-launch-
  overhead-bound workloads, and this one is already compute-bound (see
  Phase 6's power-draw numbers), so the payoff is uncertain even if fixed.
- **VRAM stays high / shared GPU memory balloons after stopping a Docker
  test, or `nvidia-smi` shows load with no `--compile` progress**: check
  `docker ps -a` — stopping the *shell* that ran a foreground `docker run`
  does not stop the *container*; it keeps running (and holding VRAM,
  competing with whatever you launch next) until you `docker stop
  <name>` it explicitly. Launch anything long-running with `docker run -d
  --name <name>` so it's easy to find and stop later.
