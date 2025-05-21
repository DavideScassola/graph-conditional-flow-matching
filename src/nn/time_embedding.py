import math

import torch
import torch.nn.functional as F


def sin_cos_embedding(t: torch.Tensor) -> torch.Tensor:
    x = t.reshape(-1, 1) * 2 * torch.pi
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


def circle_and_line_embedding(t: torch.Tensor) -> torch.Tensor:
    x = t.reshape(-1, 1) * 2 * torch.pi
    return torch.cat([torch.cos(x), torch.sin(x), t.reshape(-1, 1)], dim=-1)


def int2digits(x: torch.Tensor, *, base: int, digits: int | None) -> torch.Tensor:
    assert (x >= 0).all(), "All values must be positive"
    max_value = x.max().item()
    digits = math.ceil(math.log(max_value + 1, base)) if digits is None else digits
    powers = base ** torch.arange(
        digits - 1, -1, -1, dtype=torch.int64, device=x.device
    )
    return (x.unsqueeze(-1) // powers) % base


def digits_embedding(t: torch.Tensor, *, base: int, digits: int) -> torch.Tensor:
    return torch.cat(
        [
            t.reshape(-1, 1),
            int2digits(torch.round(t * (base**digits)), base=base, digits=digits)
            .float()
            .reshape(-1, digits),
        ],
        dim=1,
    )


def binary_embedding(t: torch.Tensor, digits: int = 10) -> torch.Tensor:
    return digits_embedding(t, base=2, digits=digits)


def bisecting_waves_embedding(t: torch.Tensor, n_waves=10) -> torch.Tensor:
    x = (t.reshape(-1, 1) - 0.25) * 2 * torch.pi
    f = torch.arange(1, n_waves + 1) ** 2
    return torch.sin(x * f / 2 - 0.25 * 2 * torch.pi)


def classic_embedding(t, embedding_dim=20, max_positions=10000) -> torch.Tensor:
    timesteps = t.flatten() * 1000
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb
    )
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode="constant")
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def tabddpm_embedding(timesteps, dim=128, max_period=10000):
    """
    From Dhariwal et al. (2020), "Diffusion Models Beat GANs on Image Synthesis"

    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
