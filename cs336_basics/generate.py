# cs336_basics/generate.py

import argparse
import json
import torch
import numpy as np
from cs336_basics.transformer import TransformerLM, softmax
from cs336_basics.bpe import Tokenizer
from cs336_basics.training import load_checkpoint
from cs336_basics.optimizer import AdamW


def generate(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: str = "cpu",
    context_length: int = 256,
) -> str:
    model.eval()
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    eot_id = tokenizer.vocab_inv.get("<|endoftext|>".encode("utf-8"), None)
    generated = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 正确截断到 context_length
            ctx = input_tensor[:, -context_length:]

            logits = model(ctx)
            next_logits = logits[0, -1, :].float()  # 转 float32 防止 NaN

            # Temperature scaling
            next_logits = next_logits / max(temperature, 1e-8)

            # 数值稳定的 softmax
            next_logits = next_logits - next_logits.max()
            probs = torch.exp(next_logits)
            probs = probs / probs.sum()

            # Top-p sampling
            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=0)
                cutoff = (cumsum - sorted_probs) >= top_p
                sorted_probs[cutoff] = 0.0
                probs = torch.zeros_like(probs).scatter_(0, sorted_idx, sorted_probs)
                probs = probs / probs.sum()

            # 确保没有负值或 NaN
            probs = probs.clamp(min=0)
            probs = probs / probs.sum()

            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)

            if next_token == eot_id:
                break

            input_tensor = torch.cat([
                input_tensor,
                torch.tensor([[next_token]], device=device)
            ], dim=1)

    return tokenizer.decode(generated)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--vocab_path", type=str, required=True)
    p.add_argument("--merges_path", type=str, required=True)
    p.add_argument("--prompt", type=str, default="Once upon a time")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    # 模型结构参数（需与训练时一致）
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--d_ff", type=int, default=1344)
    p.add_argument("--rope_theta", type=float, default=10000.0)
    p.add_argument("--device", type=str, default="cpu")
    
    args = p.parse_args()

    # 加载分词器
    tokenizer = Tokenizer.from_files(
        args.vocab_path, args.merges_path,
        special_tokens=["<|endoftext|>"]
    )

    # 加载模型
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(args.device)

    optimizer = AdamW(model.parameters())
    load_checkpoint(args.checkpoint, model, optimizer)
    print(f"模型加载自 {args.checkpoint}")

    # 生成
    result = generate(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
        context_length=args.context_length,
    )
    print(f"\n--- 生成结果 ---\n{args.prompt}{result}\n")


if __name__ == "__main__":
    main()