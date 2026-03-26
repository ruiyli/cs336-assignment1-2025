# compare_tokenizers.py
import time
from cs336_basics.bpe import Tokenizer

# TinyStories 分词器
ts_tok = Tokenizer.from_files(
    "tinystories_vocab.json", "tinystories_merges.txt",
    special_tokens=["<|endoftext|>"]
)

# OWT 分词器（BPE 完成后才能用）
owt_tok = Tokenizer.from_files(
    "owt_vocab.json", "owt_merges.txt",
    special_tokens=["<|endoftext|>"]
)

# 各取 10 个文档样本
with open("data/TinyStoriesV2-GPT4-train.txt") as f:
    ts_sample = f.read(50000)

with open("data/owt_train.txt") as f:
    owt_sample = f.read(50000)

for name, tok, text in [
    ("TS tok on TS data", ts_tok, ts_sample),
    ("OWT tok on OWT data", owt_tok, owt_sample),
    ("TS tok on OWT data", ts_tok, owt_sample),  # 跨域
]:
    ids = tok.encode(text)
    b = len(text.encode("utf-8"))
    print(f"{name}: {b/len(ids):.2f} bytes/token")