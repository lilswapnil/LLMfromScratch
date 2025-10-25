# LLM From Scratch — 125M Starter

This repo is a **step-by-step scaffold** to build and train a small decoder-only LLM (≈125M params) from scratch.

## What you'll do (copy/paste friendly)

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

## Troubleshooting

### Error: "Permission denied: No such file or directory" during tokenizer training
**Problem:** The `artifacts/tokenizer/` directory doesn't exist, so SentencePiece can't save the model files.

**Fix:**
```bash
mkdir -p artifacts/tokenizer
```

**Why:** SentencePiece requires the output directory to exist before writing files. The `-p` flag creates intermediate directories if needed.

---

### Error: Vocabulary size mismatch (e.g., "vocab_size 32000 but found 277 tokens")
**Problem:** Your corpus is too small. SentencePiece extracts subword units from your data; if you only have ~43 bytes of text, you can't build a 32k vocabulary.

**Fix:** Reduce vocab_size to match your data:
```bash
python3 scripts/train_tokenizer.py   --input data/raw/corpus.txt   --model_prefix artifacts/tokenizer/llm_spm   --vocab_size 256
```

Then gradually increase as you add more data to `data/raw/corpus.txt`.

**Why:** SentencePiece needs sufficient unique character sequences to build a meaningful vocabulary. Tiny datasets should use smaller vocab sizes (256–2048). Production models use 32k–100k with megabytes of text.

---

### Error: "RuntimeError: shape mismatch" in attention mask
**Problem:** The attention mask has the wrong shape when broadcasting to attention scores. This happens if the mask isn't properly reshaped from `(T, T)` to `(1, 1, T, T)`.

**Fix:** Ensure in `src/model.py` GPT.forward():
```python
attn_mask = self.mask[:T, :T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
```

**Why:** Attention scores have shape `(B, H, T, T)` where B=batch, H=heads, T=sequence length. The mask must broadcast correctly: `(1, 1, T, T)` broadcasts to `(B, H, T, T)`. Using `.unsqueeze()` twice adds the batch and head dimensions.

---

### Error: "qkv indexing fails" in multi-head attention
**Problem:** The permute order is wrong, creating shape `(B, 3, H, T, D)` instead of `(B, H, T, 3, D)`, so indexing `[..., 0, :]` fails.

**Fix:** In `src/model.py` MHA.forward():
```python
qkv = qkv.view(B, T, self.n_heads, 3, self.d_head)  # (B, T, H, 3, D)
qkv = qkv.permute(0, 2, 1, 3, 4)  # (B, H, T, 3, D) ← correct order
q, k, v = qkv[..., 0, :], qkv[..., 1, :], qkv[..., 2, :]
```

**Why:** After `.view()`, dimensions are `[0:B, 1:T, 2:H, 3:3, 4:D]`. Using `.permute(0, 2, 1, 3, 4)` reorders to `[B, H, T, 3, D]`. Then `qkv[..., 0, :]` extracts Q (shape `B, H, T, D`), `[..., 1, :]` extracts K, and `[..., 2, :]` extracts V. Any other permute order will fail.

**Note:** This is an educational baseline, not an industrial training system. See the comments in code for where to plug in FSDP/DeepSpeed, Flash-Attn, and better data pipelines.
