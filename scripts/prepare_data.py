import argparse, numpy as np, sentencepiece as spm, os, tqdm

def iter_tokens(sp, text):
    # Returns a list of ids with BOS/EOS around documents
    ids = [sp.bos_id()] + sp.EncodeAsIds(text) + [sp.eos_id()]
    return ids

def write_shards(ids, seq_len, shard_tokens, out_dir, split):
    os.makedirs(out_dir, exist_ok=True)
    total = len(ids)
    shard_id = 0
    start = 0
    while start + seq_len + 1 <= total:
        end = min(start + shard_tokens, total)
        chunk = ids[start:end]
        # pack into fixed sequences
        n_seq = (len(chunk) - 1) // seq_len
        if n_seq <= 0:
            break
        x = np.stack([chunk[i*seq_len:(i+1)*seq_len] for i in range(n_seq)], dtype=np.int32)
        y = np.stack([chunk[i*seq_len+1:(i+1)*seq_len+1] for i in range(n_seq)], dtype=np.int32)
        np.savez_compressed(os.path.join(out_dir, f"{split}_{shard_id:04d}.npz"), x=x, y=y)
        shard_id += 1
        start = end

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--shard_tokens", type=int, default=200_000_000)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--val_ratio", type=float, default=0.01)
    args = ap.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.tok)
    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    # simple split for val
    n = len(text)
    val_text = text[:int(n*args.val_ratio)]
    train_text = text[int(n*args.val_ratio):]

    train_ids = iter_tokens(sp, train_text)
    val_ids = iter_tokens(sp, val_text)

    write_shards(train_ids, args.seq_len, args.shard_tokens, args.out_dir, "train")
    write_shards(val_ids, args.seq_len, args.shard_tokens, args.out_dir, "val")

    print("Wrote shards to", args.out_dir)

if __name__ == "__main__":
    main()
