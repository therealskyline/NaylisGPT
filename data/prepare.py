import argparse
import os
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

TOKENIZER_ID    = "Qwen/Qwen2.5-0.5B"
TOKENIZER_LOCAL = Path("tokenizer")

# uint32 obligatoire : vocab Qwen2.5 = 151 667 > 65 535 (limite uint16)
DTYPE = np.uint32


def load_tokenizer() -> AutoTokenizer:
    src = str(TOKENIZER_LOCAL) if TOKENIZER_LOCAL.exists() else TOKENIZER_ID
    print(f"Tokenizer : {src}")
    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)

    think_tokens = ["<think>", "</think>"]
    missing = [t for t in think_tokens
               if tok.convert_tokens_to_ids(t) == tok.unk_token_id]
    if missing:
        tok.add_special_tokens({"additional_special_tokens": missing})
        print(f"  + Tokens ajoutés : {missing}")
        TOKENIZER_LOCAL.mkdir(parents=True, exist_ok=True)
        tok.save_pretrained(str(TOKENIZER_LOCAL))
        print(f"  ✓ Tokenizer sauvegardé → {TOKENIZER_LOCAL}/")
    else:
        print(f"  ✓ Tokens <think></think> déjà présents")

    print(f"  vocab={len(tok):,}  eos={tok.eos_token_id}  dtype=uint32")
    return tok


def main():
    parser = argparse.ArgumentParser(description="Prépare pretrain_data.bin (Qwen2.5, uint32)")
    parser.add_argument("--dataset",  type=str, required=True)
    parser.add_argument("--output",   type=str, default="./pretrain_data.bin")
    parser.add_argument("--split",    type=str, default="train")
    parser.add_argument("--column",   type=str, default="text")
    parser.add_argument("--limit",    type=int, default=None)
    parser.add_argument("--hf-token", type=str, default=None)
    args = parser.parse_args()

    tokenizer = load_tokenizer()
    eos = tokenizer.eos_token_id

    print(f"\nDataset : {args.dataset}  split={args.split}")
    ds = load_dataset(args.dataset, split=args.split,
                      token=args.hf_token, streaming=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    all_tokens = []
    count = 0
    for sample in tqdm(ds, desc="Tokenisation"):
        if args.limit and count >= args.limit:
            break
        ids = tokenizer.encode(sample[args.column], add_special_tokens=False)
        ids.append(eos)
        all_tokens.extend(ids)
        count += 1

    tokens = np.array(all_tokens, dtype=DTYPE)
    tokens.tofile(output)
    print(f"\nSauvegardé : {output}  "
          f"({len(tokens)/1e9:.3f}B tokens  {output.stat().st_size/1e9:.1f} GB)")


if __name__ == "__main__":
    main()
