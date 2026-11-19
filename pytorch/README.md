# llm-from-scratch

A modern decoder-only transformer — RoPE, grouped-query attention, SwiGLU,
RMSNorm, pre-norm blocks, tied embeddings — with a complete training loop
(AdamW, warmup+cosine, grad accumulation, bf16 autocast, atomic resumable
checkpoints, JSONL metrics) and KV-cached sampling (temperature, top-p,
repetition penalty, EOT stopping). Single file, sized for a 12 GB GPU.

Grown out of a from-scratch NumPy implementation of every layer, each with
hand-derived, numerically verified gradients; this is the PyTorch port.

## Quickstart

```bash
uv sync                                  # creates .venv, installs torch+numpy
uv run prepare-data                      # writes data/{train,val}.bin (Shakespeare)
uv run train-gpt --preset tiny           # 0.9M params, minutes on CPU/any GPU
uv run train-gpt --preset tiny --resume --sample_only \
    --merges shake_merges.json --prompt "ROMEO:"
```

Then follow **RUNBOOK.md** — a staged test-drive from CPU parity check to
the 101M-parameter `gpt2ish` run, with expected numbers at each phase.
(Its `python train_gpt.py ...` commands are equivalent to
`uv run train-gpt ...` here.)

## Presets

| preset  | params | context | for                                    |
|---------|--------|---------|----------------------------------------|
| tiny    | 0.9M   | 128     | pipeline verification, runs on CPU     |
| debug   | 4.9M   | 256     | fast iteration, resume/compile drills  |
| small   | 40M    | 512     | overnight-quality runs                 |
| gpt2ish | 101M   | 1024    | the 12 GB flagship (GPT-2-small class) |

## Real data

```bash
uv sync --extra data       # tokenizers + datasets
```

See the executable snippet in `prepare_data.py` — train a 32k BPE with an
`<|endoftext|>` special token, run documents through `encode_documents`,
write bins. Sample with `--stop_token` for clean story endings.

## Plotting a run

```bash
uv sync --extra plot
uv run python -c "import pandas as pd; pd.read_json('ckpt_log.jsonl', lines=True).plot(x='tokens', y=['train','val']); import matplotlib.pyplot as plt; plt.show()"
```

## GPU wheel pinning

On Linux, PyPI's default `torch` wheel bundles CUDA 12 and works out of
the box. **On Windows, PyPI serves CPU-only torch** — you must pin the
PyTorch index. To pin a specific CUDA release or force CPU-only wheels,
uncomment the index block at the bottom of `pyproject.toml`.
