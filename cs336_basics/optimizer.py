# cs336_basics/optimizer.py

import torch
import math
from typing import Callable, Optional


def get_lr_cosine_schedule(it: int, 
                           max_learning_rate: float, 
                           min_learning_rate: float, 
                           warmup_iters: int,
                           cosine_cycle_iters: int) -> float:
    # Warmup 阶段
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    
    # Post-annealing 阶段
    if it > cosine_cycle_iters:
        return min_learning_rate
    
    # Cosine annealing 阶段
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    cosine = 0.5 * (1 + math.cos(progress * math.pi))
    return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine


def clip_gradient(parameters, max_l2_norm: float, eps: float = 1e-6):
    # 收集所有有梯度的参数
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    
    # 计算全局 L2 norm
    total_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))

    # 超过阈值才裁剪
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for g in grads:
            g.mul_(scale)


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
            
                grad = p.grad.data
                state = self.state[p]

                # 初始化状态
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                state["t"] += 1
                t = state["t"]
                m, v = state["m"], state["v"]

                # 更新一阶, 二阶矩估计
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 偏差修正后的学习率
                alpha_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

                # 参数更新
                p.data.addcdiv_(m, v.sqrt().add_(eps), value=-alpha_t)

                # 权重衰减 (独立于梯度更新)
                p.data.mul_(1 - weight_decay * lr)
    
        return loss
    