import argparse, sentencepiece as spm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model_prefix", required=True)
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--character_coverage", type=float, default=0.9995)
    args = ap.parse_args()

    spm.SentencePieceTrainer.Train(
        input=args.input,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        model_type="unigram",
        byte_fallback=True,
        unk_id=0, bos_id=1, eos_id=2, pad_id=3
    )
    print("Tokenizer trained:", args.model_prefix + ".model")

if __name__ == "__main__":
    main()
