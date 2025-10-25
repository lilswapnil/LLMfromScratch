# LLM From Scratch — 125M Starter

This repo is a **step-by-step scaffold** to build and train a small decoder-only LLM (≈125M params) from scratch.

## What you’ll do (copy/paste friendly)

1) **Create & activate env (recommended)**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2) **Put text data** (UTF-8 plain text, one doc per line or a large concatenated file) in:
```
data/raw/corpus.txt
```
You can start with tiny shakespeare or any text you have permission to use.

3) **Train tokenizer (SentencePiece, 32k vocab)**
```bash
python3 scripts/train_tokenizer.py   --input data/raw/corpus.txt   --model_prefix artifacts/tokenizer/llm_spm   --vocab_size 32000
```
or
```bash
python3 scripts/train_tokenizer.py   --input data/raw/corpus.txt   --model_prefix artifacts/tokenizer/llm_spm   --vocab_size 11027
```

4) **Shard & pack data to fixed-length token sequences (2k)**
```bash
python scripts/prepare_data.py   --input data/raw/corpus.txt   --tok artifacts/tokenizer/llm_spm.model   --seq_len 2048   --shard_tokens 200_000_000   --out_dir data/shards
```

5) **Train the 125M model**
```bash
python src/train.py --config configs/125m.json
```

6) **Evaluate perplexity on a held-out file**
```bash
python src/eval.py   --tok artifacts/tokenizer/llm_spm.model   --ckpt artifacts/checkpoints/latest.pt   --input data/raw/val.txt
```

7) **Generate text (quick sanity check)**
```bash
python scripts/generate.py   --tok artifacts/tokenizer/llm_spm.model   --ckpt artifacts/checkpoints/latest.pt   --prompt "Once upon a time"   --max_new_tokens 100
```

## Scaling up
- To move toward 1.3B: increase `n_layers`, `d_model`, `n_heads` in `configs/125m.json`, use gradient checkpointing (already enabled), and increase data via more shards.
- For multi-GPU: this minimal scaffold uses PyTorch DDP if `WORLD_SIZE>1` is set (see notes in `src/train.py`).

**Note:** This is an educational baseline, not an industrial training system. See the comments in code for where to plug in FSDP/DeepSpeed, Flash-Attn, and better data pipelines.
