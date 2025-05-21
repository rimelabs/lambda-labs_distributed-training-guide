# Benchmark data loading

## Single GPU

Just loading data averages ~90k tokens/second with a single process.

```bash
python packed_dataset.py

> ntokens=417792, lapsed_time=5.06s, toks_per_sec=82529
> ntokens=827392, lapsed_time=9.34s, toks_per_sec=88575
> ntokens=1236992, lapsed_time=13.58s, toks_per_sec=91065
```

## 8 GPUs

Using 8 GPUs/processes, we can load in about 600k tokens per second.

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=8 packed_dataset.py

> ntokens=3342336, lapsed_time=6.11s, toks_per_sec=546910
> ntokens=6619136, lapsed_time=11.18s, toks_per_sec=591895
> ntokens=9895936, lapsed_time=16.35s, toks_per_sec=605101
```
