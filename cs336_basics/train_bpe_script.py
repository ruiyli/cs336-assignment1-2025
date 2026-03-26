# cs336_basics/train_bpe_script.py

import argparse
import json
import time
from cs336_basics.bpe import train_bpe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_path", type=str, required=True)
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--special_tokens", type=str, nargs="+", default=["<|endoftext|>"])
    p.add_argument("--vocab_out", type=str, default="vocab.json")
    p.add_argument("--merges_out", type=str, default="merges.txt")
    args = p.parse_args()

    print(f"开始训练 BPE, 词表大小={args.vocab_size}, 输入={args.input_path}")
    t0 = time.time()

    vocab, merges = train_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
    )

    elapsed = time.time() - t0
    print(f"训练完成, 耗时 {elapsed:.1f}s")

    # 找最长 token(用于 writeup)
    longest = max(vocab.values(), key=len)
    print(f"词表大小: {len(vocab)}")
    print(f"最长 token: {longest} ({len(longest)} bytes)")

    # 保存 vocab(bytes 转 list 才能 JSON 序列化)
    vocab_serializable = {k: list(v) for k, v in vocab.items()}
    with open(args.vocab_out, "w") as f:
        json.dump(vocab_serializable, f)
    print(f"vocab 保存至 {args.vocab_out}")

    # 保存 merges(每行两个 hex 字符串)
    with open(args.merges_out, "w") as f:
        for a, b in merges:
            f.write(f"{a.hex()} {b.hex()}\n")
    print(f"merges 保存至 {args.merges_out}")


if __name__ == "__main__":
    main()
