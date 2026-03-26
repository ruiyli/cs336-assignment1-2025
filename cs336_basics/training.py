# cs336_basics/training.py

import os
import torch
import numpy as np
from typing import BinaryIO, IO


def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """随机采样一个 batch"""
    # 随机选起始位置, 确保 x[i:i+context_length+1] 不越界
    # np.random.randint(low, high) 范围是 [low, high)
    # 最大合法 start = len(dataset) - context_length - 1
    # 所以 high = len(dataset) - context_length + 1
    max_start = len(dataset) - context_length
    starts = np.random.randint(0, max_start, size=batch_size)

    x = torch.stack([
        torch.from_numpy(dataset[s: s + context_length].astype(np.int64))
        for s in starts
    ]).to(device)

    y = torch.stack([
        torch.from_numpy(dataset[s + 1: s + context_length + 1].astype(np.int64))
        for s in starts
    ]).to(device)

    return x, y


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out,):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(src, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    checkpoint = torch.load(src, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['iteration']
