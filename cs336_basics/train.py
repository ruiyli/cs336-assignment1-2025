# cs336_basics/train.py

import argparse
import os
import time
import numpy as np
import torch
from cs336_basics.transformer import TransformerLM, cross_entropy
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule, clip_gradient
from cs336_basics.training import get_batch, save_checkpoint, load_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    # 数据
    p.add_argument("--train_path", type=str, required=True)
    p.add_argument("--val_path", type=str, required=True)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    # 模型
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--d_ff", type=int, default=1344)
    p.add_argument("--rope_theta", type=float, default=10000.0)
    # 训练
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--total_steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_min", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--use_norm", action="store_true", default=True)
    p.add_argument("--no_norm", dest="use_norm", action="store_false")
    p.add_argument("--post_norm", action="store_true", default=False)
    p.add_argument("--no_rope", dest="use_rope", action="store_false", default=True)
    p.add_argument("--ffn_type", type=str, default="swiglu", choices=["swiglu", "silu"])
    # 日志
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--val_interval", type=int, default=500)
    p.add_argument("--val_steps", type=int, default=20)
    p.add_argument("--checkpoint_interval", type=int, default=1000)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def estimate_val_loss(model, val_data, batch_size, context_length, device, val_steps):
    model.eval()
    losses = []
    for _ in range(val_steps):
        x, y = get_batch(val_data, batch_size, context_length, device)
        logits = model(x)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 加载数据（memmap 模式,不把整个文件读入内存）
    train_data = np.load(args.train_path, mmap_mode="r")
    val_data = np.load(args.val_path, mmap_mode="r")

    # 构建模型
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        use_norm=args.use_norm,
        post_norm=args.post_norm,
        use_rope=args.use_rope,
        ffn_type=args.ffn_type,
    ).to(args.device)

    # 参数量统计
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params/1e6:.1f}M")

    # 优化器
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    # 恢复 checkpoint
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer)
        print(f"从 step {start_step} 恢复训练")

    # 训练循环
    model.train()
    t0 = time.time()

    for step in range(start_step, args.total_steps):
        # 更新学习率
        lr = get_lr_cosine_schedule(
            step, args.lr, args.lr_min,
            args.warmup_steps, args.total_steps,
        )
        for g in optimizer.param_groups:
            g["lr"] = lr

        # 前向 + 反向
        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        logits = model(x)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        clip_gradient(model.parameters(), args.grad_clip)
        optimizer.step()

        # 日志
        if step % args.log_interval == 0:
            elapsed = time.time() - t0
            print(f"step {step:6d} | loss {loss.item():.4f} | lr {lr:.2e} | {elapsed:.1f}s")

        # 验证
        if step % args.val_interval == 0:
            val_loss = estimate_val_loss(
                model, val_data, args.batch_size,
                args.context_length, args.device, args.val_steps,
            )
            print(f"step {step:6d} | val_loss {val_loss:.4f}")

        # 保存 checkpoint
        if step % args.checkpoint_interval == 0 and step > 0:
            path = os.path.join(args.checkpoint_dir, f"ckpt_{step:06d}.pt")
            save_checkpoint(model, optimizer, step, path)
            print(f"checkpoint 保存至 {path}")

    # 最终保存
    save_checkpoint(model, optimizer, args.total_steps,
                    os.path.join(args.checkpoint_dir, "ckpt_final.pt"))
    print("训练完成！")


if __name__ == "__main__":
    main()