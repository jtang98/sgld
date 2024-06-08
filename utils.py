import math
import torch


def sgld_sample(to_grad_fn, step_size, n_steps):
    x = torch.rand(1, requires_grad=True)
    for _ in range(n_steps):
        L = to_grad_fn(x)
        L.backward()
        x = x + step_size / 2 * x.grad + math.sqrt(step_size) * torch.randn(1)
        x = x.detach()
        x.requires_grad = True
    return x.item()