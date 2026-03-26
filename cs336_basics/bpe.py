# cs336_basics/bpe.py

import regex as re
import multiprocessing as mp
from collections import defaultdict
from typing import Iterator, Iterable
import json


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _get_chunk_boundaries(data: bytes, num_chunks: int, special_token: bytes) -> list[int]:
    size = len(data)
    boundaries = [0]
    for i in range(1, num_chunks):
        start = i * size // num_chunks
        idx = data.find(special_token, start)
        if idx == -1:
            boundaries.append(size)
            break
        boundaries.append(idx)
    boundaries.append(size)
    return boundaries


def _process_chunk(args):
    chunk_text, special_tokens = args
    word_freqs = defaultdict(int)

    if special_tokens:
        special_pattern = "|".join(re.escape(tok) for tok in special_tokens)
        parts = re.split(special_pattern, chunk_text)
    else:
        parts = [chunk_text]

    for part in parts:
        for match in re.finditer(PAT, part):
            word = match.group()
            byte_tuple = tuple(bytes([b]) for b in word.encode("utf-8"))
            word_freqs[byte_tuple] += 1

    return dict(word_freqs)


def _pretokenize(input_path: str, special_tokens: list[str]) -> dict[tuple, int]:
    with open(input_path, "rb") as f:
        data = f.read()

    MULTIPROCESS_THRESHOLD = 10 * 1024 * 1024  # 10MB
    num_workers = max(1, mp.cpu_count() - 1)

    if len(data) < MULTIPROCESS_THRESHOLD or num_workers == 1:
        text = data.decode("utf-8", errors="replace")
        return _process_chunk((text, special_tokens))

    # Linux 用 fork
    ctx = mp.get_context("fork")

    special_token_bytes = special_tokens[0].encode("utf-8") if special_tokens else b""
    boundaries = _get_chunk_boundaries(data, num_workers, special_token_bytes)
    chunks = [
        (data[boundaries[i]:boundaries[i+1]].decode("utf-8", errors="replace"), special_tokens)
        for i in range(len(boundaries) - 1)
    ]
    print(f"预分词: {len(chunks)} chunks, {num_workers} 进程", flush=True)

    # with ctx.Pool(num_workers) as pool:
    #     results = pool.map(_process_chunk, chunks)
    pool = ctx.Pool(num_workers)
    try:
        results = pool.map(_process_chunk, chunks)
    finally:
        pool.terminate()  # 比 close()+join() 快
        pool.join()

    total_freqs: dict[tuple, int] = defaultdict(int)
    for freq_dict in results:
        for word, count in freq_dict.items():
            total_freqs[word] += count

    print(f"预分词完成, 共 {len(total_freqs)} 个词", flush=True)
    return dict(total_freqs)


def _count_pairs(word_freqs: dict[tuple, int]) -> dict[tuple, int]:
    pair_freqs: dict[tuple, int] = defaultdict(int)
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_freqs[pair] += freq
    return dict(pair_freqs)


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    vocab: dict[int, bytes] = {}
    for i, tok in enumerate(special_tokens):
        vocab[i] = tok.encode("utf-8")
    for b in range(256):
        vocab[len(special_tokens) + b] = bytes([b])

    word_freqs = _pretokenize(input_path, special_tokens)
    num_merges = vocab_size - len(vocab)

    words = [(list(word), freq) for word, freq in word_freqs.items()]
    pair_counts = defaultdict(int)
    pair_to_words = defaultdict(set)

    for idx, (tokens, freq) in enumerate(words):
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i+1])
            pair_counts[pair] += freq
            pair_to_words[pair].add(idx)

    merges = []

    for _ in range(num_merges):
        if not pair_counts:
            break

        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        if pair_counts[best_pair] <= 0:
            break

        merges.append(best_pair)
        merged = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = merged

        affected = list(pair_to_words[best_pair])
        for idx in affected:
            tokens, freq = words[idx]

            for i in range(len(tokens) - 1):
                p = (tokens[i], tokens[i+1])
                pair_counts[p] -= freq
                pair_to_words[p].discard(idx)

            new_tokens = []
            i = 0
            while i < len(tokens):
                if (i < len(tokens) - 1
                        and tokens[i] == best_pair[0]
                        and tokens[i+1] == best_pair[1]):
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            for i in range(len(new_tokens) - 1):
                p = (new_tokens[i], new_tokens[i+1])
                pair_counts[p] += freq
                pair_to_words[p].add(idx)

            words[idx] = (new_tokens, freq)

        del pair_counts[best_pair]
        del pair_to_words[best_pair]

    return vocab, merges


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.vocab_inv = {v: k for k, v in vocab.items()}
        self.merges = {pair: rank for rank, pair in enumerate(merges)}
        self.special_tokens: list[str] = special_tokens or []
        for tok in self.special_tokens:
            tok_bytes = tok.encode("utf-8")
            if tok_bytes not in self.vocab_inv:
                new_id = len(self.vocab)
                self.vocab[new_id] = tok_bytes
                self.vocab_inv[tok_bytes] = new_id

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_raw = json.load(f)
        vocab = {int(k): bytes(v) for k, v in vocab_raw.items()}

        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                a = bytes.fromhex(parts[0])
                b = bytes.fromhex(parts[1])
                merges.append((a, b))

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        ids = []
        if self.special_tokens:
            special_pattern = "(" + "|".join(
                re.escape(tok) for tok in sorted(
                    self.special_tokens, key=len, reverse=True)
            ) + ")"
            parts = re.split(special_pattern, text)
        else:
            parts = [text]

        for part in parts:
            if part in self.special_tokens:
                ids.append(self.vocab_inv[part.encode("utf-8")])
            else:
                for match in re.finditer(PAT, part):
                    word = match.group()
                    tokens = [bytes([b]) for b in word.encode("utf-8")]
                    tokens = self._apply_merges(tokens)
                    ids.extend(self.vocab_inv[tok] for tok in tokens)
        return ids

    def _apply_merges(self, tokens: list[bytes]) -> list[bytes]:
        while len(tokens) >= 2:
            best_rank = float("inf")
            best_idx = -1
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merges.get(pair, float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_idx = i
            if best_idx == -1:
                break
            merged = tokens[best_idx] + tokens[best_idx + 1]
            tokens = tokens[:best_idx] + [merged] + tokens[best_idx + 2:]
        return tokens

    def encode_iterable(self, iterable: Iterable[str]):
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        byte_seq = b"".join(self.vocab[i] for i in ids)
        return byte_seq.decode("utf-8", errors="replace")
