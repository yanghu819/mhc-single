# mhc-single

Single-folder, minimal reimplementation to compare:

- `residual` (baseline)
- `hc`
- `mhc`
- `mhc_lite`

on **FineWeb10B GPT-2 token shards** (same shard naming/layout used by `modded-nanogpt/train_gpt.py`).

This folder is self-contained: `run.py` includes the model, methods, dataloader, and training loop.

## Install

```bash
pip install -r mhc-single/requirements.txt
```

## Data Layout (local files)

By default `run.py` expects:

- train: `DATA_ROOT/data/fineweb10B/fineweb_train_*.bin`
- val:   `DATA_ROOT/data/fineweb10B/fineweb_val_*.bin`

Each `.bin` is a flat `uint16` array of GPT-2 token IDs.

You can override with `--train_glob` / `--val_glob`.

### Optional: Download From ModelScope

If your server can access ModelScope but not HuggingFace/GitHub, you can upload the shards to a ModelScope *Dataset*
repo and run:

```bash
python mhc-single/run.py --ms_dataset_id owner/dataset_repo
```

`run.py` can auto-detect common layouts (default `--auto_glob 1`), e.g. shards placed at:

- repo root: `fineweb_train_*.bin` / `fineweb_val_*.bin`
- `data/fineweb10B/`
- any nested subdir (it will search `**/fineweb_train_*.bin`)

## One-Click Runs

### Single GPU (or CPU)

Full matrix (methods x S/M/L), 10k steps each (this is long):

```bash
python mhc-single/run.py --data_root /path/to/DATA_ROOT
```

Quick smoke test:

```bash
python mhc-single/run.py --data_root /path/to/DATA_ROOT \
  --scales small --methods residual --steps 20 --eval_every 10 --eval_iters 2
```

Note: `--steps` matches `mhc-lite/train.py`'s `max_iters` semantics (the loop runs `steps + 1` updates, counting from iter 0).

Note (reproducibility): `--seed` seeds both `torch` and Python `random` so HC init choices are repeatable.

Note (dtype default): on CPU it defaults to `float32`; on CUDA it defaults to `bfloat16` if supported else `float16`.

### Multi-GPU (DDP)

```bash
torchrun --standalone --nproc_per_node=8 mhc-single/run.py \
  --data_root /path/to/DATA_ROOT
```

If `torchrun` is not available, use:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 mhc-single/run.py \
  --data_root /path/to/DATA_ROOT
```

Quick DDP smoke test:

```bash
torchrun --standalone --nproc_per_node=2 mhc-single/run.py \
  --data_root /path/to/DATA_ROOT \
  --scales small --methods residual --steps 20 --eval_every 10 --eval_iters 2
```

Fallback smoke test:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 mhc-single/run.py \
  --data_root /path/to/DATA_ROOT \
  --scales small --methods residual --steps 20 --eval_every 10 --eval_iters 2
```

## Outputs

Creates `--out_dir` (default: `runs/<timestamp>`):

- `results.jsonl`: step-level metrics (rank0 only)
- `summary.json`: per-(method,scale) summary (rank0 only)
