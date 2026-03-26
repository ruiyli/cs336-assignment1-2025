# cs336_basics/encode_dataset.py

import argparse
import json
import numpy as np
from cs336_basics.bpe import Tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_path", type=str, required=True)
    p.add_argument("--vocab_path", type=str, required=True)
    p.add_argument("--merges_path", type=str, required=True)
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--special_tokens", type=str, nargs="+", default=["<|endoftext|>"])
    args = p.parse_args()

    # 加载分词器
    tokenizer = Tokenizer.from_files(
        args.vocab_path, args.merges_path, args.special_tokens
    )

    print(f"编码 {args.input_path} ...")
    # 用 encode_iterable 逐行处理, 节省内存
    with open(args.input_path, "r", encoding="utf-8") as f:
        token_ids = list(tokenizer.encode_iterable(f))

    arr = np.array(token_ids, dtype=np.uint16)
    np.save(args.output_path, arr)
    print(f"共 {len(arr):,} tokens, 保存至 {args.output_path}")
    print(f"文件大小: {arr.nbytes / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
