import argparse, torch, sentencepiece as spm
from model import GPT

def sample(model, idx, max_new_tokens, temperature=1.0, top_k=50):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.seq_len:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / max(1e-6, temperature)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=100)
    args = ap.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.tok)
    ck = torch.load(args.ckpt, map_location="cpu"); C = ck["config"]
    model = GPT(C["vocab_size"], C["d_model"], C["n_layers"], C["n_heads"], C["d_ff"], C["seq_len"], C["dropout"])
    model.load_state_dict(ck["model"]); model.eval()

    ids = [sp.bos_id()] + sp.EncodeAsIds(args.prompt)
    x = torch.tensor(ids).unsqueeze(0)
    out = sample(model, x, args.max_new_tokens, temperature=0.8, top_k=50)[0].tolist()
    txt = sp.DecodeIds(out)
    print(txt)

if __name__ == "__main__":
    main()
