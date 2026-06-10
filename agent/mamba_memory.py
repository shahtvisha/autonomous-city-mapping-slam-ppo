"""
Mamba-based Spatial Memory for Active SLAM
------------------------------------------
Base components imported by mamba_memory_fast.py:
  - MambaConfig
  - MambaBlock
  - MambaSpatialEncoder
  - NeuralMemoryBank
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass


# ── Config ────────────────────────────────────────────────────────────────────

class MambaConfig:
    def __init__(self,
                 d_model:   int   = 128,
                 d_state:   int   = 16,
                 d_conv:    int   = 4,
                 expand:    int   = 2,
                 dt_rank:   Optional[int] = None,
                 dt_min:    float = 0.001,
                 dt_max:    float = 0.1,
                 dt_init:   str   = 'random',
                 dt_scale:  float = 1.0,
                 bias:      bool  = False,
                 conv_bias: bool  = True):
        self.d_model   = d_model
        self.d_state   = d_state
        self.d_conv    = d_conv
        self.expand    = expand
        self.dt_rank   = dt_rank if dt_rank is not None else math.ceil(d_model / 16)
        self.dt_min    = dt_min
        self.dt_max    = dt_max
        self.dt_init   = dt_init
        self.dt_scale  = dt_scale
        self.bias      = bias
        self.conv_bias = conv_bias


# ── MambaBlock ────────────────────────────────────────────────────────────────

class MambaBlock(nn.Module):
    """
    Single Mamba SSM block.

    Shapes (from checkpoint):
      d_model=128, d_inner=256, d_state=16, d_conv=4, dt_rank=8
    """

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config

        d_inner      = config.expand * config.d_model
        dt_init_std  = config.dt_rank ** -0.5 * config.dt_scale

        # Input projection: d_model → 2*d_inner (x + residual gating)
        self.in_proj  = nn.Linear(config.d_model, d_inner * 2, bias=config.bias)

        # Causal depthwise conv over the sequence dimension
        self.conv1d   = nn.Conv1d(
            d_inner, d_inner,
            kernel_size=config.d_conv,
            groups=d_inner,
            padding=config.d_conv - 1,
            bias=config.conv_bias,
        )

        # SSM input projection: d_inner → dt_rank + 2*d_state
        self.x_proj   = nn.Linear(d_inner, config.dt_rank + 2 * config.d_state, bias=False)

        # dt projection: dt_rank → d_inner
        self.dt_proj  = nn.Linear(config.dt_rank, d_inner, bias=True)
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # SSM parameters
        # A_log: log of the state-decay matrix, shape (d_inner, d_state)
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32) \
                  .unsqueeze(0).repeat(d_inner, 1)   # (d_inner, d_state)
        self.A_log = nn.Parameter(torch.log(A))

        # D: skip-connection scalar per feature
        self.D = nn.Parameter(torch.ones(d_inner))

        # Output projection: d_inner → d_model
        self.out_proj = nn.Linear(d_inner, config.d_model, bias=config.bias)

    # ── SSM selective scan ─────────────────────────────────────────────────

    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Selective state-space scan.
        x: (B, L, d_inner)
        Returns: (B, L, d_inner)
        """
        B, L, d_inner = x.shape
        d_state  = self.config.d_state
        dt_rank  = self.config.dt_rank

        # Project to SSM inputs
        x_dbl = self.x_proj(x)                                 # (B, L, dt_rank+2*d_state)
        delta  = x_dbl[..., :dt_rank]                          # (B, L, dt_rank)
        B_proj = x_dbl[..., dt_rank:dt_rank + d_state]         # (B, L, d_state)
        C      = x_dbl[..., dt_rank + d_state:]                # (B, L, d_state)

        delta = F.softplus(self.dt_proj(delta))                 # (B, L, d_inner)

        A = -torch.exp(self.A_log.float())                      # (d_inner, d_state)

        # Recurrent scan — O(L), no CUDA kernel needed
        h = torch.zeros(B, d_inner, d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            d_t = delta[:, t]       # (B, d_inner)
            b_t = B_proj[:, t]      # (B, d_state)
            c_t = C[:, t]           # (B, d_state)
            x_t = x[:, t]          # (B, d_inner)

            # Discretised A and B
            dA = torch.exp(d_t.unsqueeze(-1) * A.unsqueeze(0))     # (B, d_inner, d_state)
            dB = d_t.unsqueeze(-1) * b_t.unsqueeze(1)              # (B, d_inner, d_state)

            h  = dA * h + dB * x_t.unsqueeze(-1)                   # (B, d_inner, d_state)
            y_t = (h * c_t.unsqueeze(1)).sum(-1) + self.D * x_t    # (B, d_inner)
            ys.append(y_t)

        return torch.stack(ys, dim=1)   # (B, L, d_inner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D) where L is sequence length (trajectory length)
        Returns: (B, L, D)
        """
        B, L, D = x.shape
        d_inner = self.config.expand * self.config.d_model

        # Split into content and gating residual
        x_and_res = self.in_proj(x)                             # (B, L, d_inner*2)
        x_in, res = x_and_res.split(d_inner, dim=-1)           # each (B, L, d_inner)

        # Causal depthwise conv (operate on length dimension)
        x_in = x_in.transpose(1, 2)                            # (B, d_inner, L)
        x_in = self.conv1d(x_in)[..., :L]                      # trim causal padding
        x_in = x_in.transpose(1, 2)                            # (B, L, d_inner)
        x_in = F.silu(x_in)

        # SSM scan
        y = self.ssm(x_in)                                      # (B, L, d_inner)

        # Gated output
        output = self.out_proj(y * F.silu(res))                 # (B, L, d_model)
        return output


# ── MambaSpatialEncoder ───────────────────────────────────────────────────────

class MambaSpatialEncoder(nn.Module):
    """
    Stacks n_layers MambaBlocks followed by a LayerNorm.
    x: (B, L, d_model) → (B, L, d_model)
    """

    def __init__(self, d_model: int, n_layers: int, d_state: int = 16):
        super().__init__()
        config = MambaConfig(
            d_model=d_model,
            d_state=d_state,
            d_conv=4,
            expand=2,
        )
        self.layers = nn.ModuleList([MambaBlock(config) for _ in range(n_layers)])
        self.norm   = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                return_all_layers: bool = False):
        """
        x: (B, L, D) - sequence of spatial observations
        Returns: (B, L, D) or List[(B, L, D)] if return_all_layers
        """
        outputs = []
        for layer in self.layers:
            x = x + layer(x)       # residual connection
            outputs.append(x)
        x = self.norm(x)
        if return_all_layers:
            return outputs
        return x


# ── NeuralMemoryBank (original, slower) ──────────────────────────────────────

class NeuralMemoryBank(nn.Module):
    """
    Original memory bank — 1000 slots, key_dim=128.
    Kept for API compatibility; training uses FastNeuralMemoryBank instead.
    """

    def __init__(self,
                 memory_size: int = 1000,
                 key_dim:     int = 128,
                 value_dim:   int = 256,
                 num_heads:   int = 8):
        super().__init__()
        self.memory_size = memory_size
        self.key_dim     = key_dim
        self.value_dim   = value_dim
        self.num_heads   = num_heads

        self.register_buffer('memory_keys',  torch.zeros(memory_size, key_dim))
        self.register_buffer('memory_values', torch.zeros(memory_size, value_dim))
        self.register_buffer('memory_age',   torch.zeros(memory_size))
        self.register_buffer('memory_usage', torch.zeros(memory_size))
        self.register_buffer('write_ptr',    torch.tensor(0, dtype=torch.long))

        self.query_proj = nn.Linear(value_dim, key_dim)
        self.key_proj   = nn.Linear(value_dim, key_dim)
        self.value_proj = nn.Linear(value_dim, value_dim)
        self.multihead_attn = nn.MultiheadAttention(key_dim, num_heads, batch_first=True)

    def write(self, keys: torch.Tensor, values: torch.Tensor):
        """
        Write new landmarks to memory.

        keys: (B, key_dim) - spatial descriptors (e.g., local map features)
        values: (B, value_dim) - full landmark information
        """
        B = keys.shape[0]
        for i in range(B):
            ptr = int(self.write_ptr.item())
            self.memory_keys[ptr]   = keys[i].detach()
            self.memory_values[ptr] = values[i].detach()
            self.memory_age[ptr]    = 0
            self.memory_usage[ptr]  = 0
            self.write_ptr = (self.write_ptr + 1) % self.memory_size
        self.memory_age += 1

    def read(self, query: torch.Tensor,
             top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve similar landmarks from memory.

        query: (B, value_dim) - current observation
        Returns:
            retrieved: (B, top_k, value_dim) - retrieved landmarks
            similarities: (B, top_k) - similarity scores
        """
        B = query.shape[0]
        query_key  = self.query_proj(query)                          # (B, key_dim)
        query_norm = F.normalize(query_key, dim=-1)
        memory_norm = F.normalize(self.memory_keys, dim=-1)         # (M, key_dim)
        similarities = torch.matmul(query_norm, memory_norm.t())    # (B, M)
        top_k_sims, top_k_indices = torch.topk(similarities, k=top_k, dim=-1)

        retrieved = torch.stack(
            [self.memory_values[top_k_indices[i]] for i in range(B)], dim=0
        )   # (B, top_k, value_dim)

        self.memory_usage[top_k_indices.reshape(-1)] += 1
        return retrieved, top_k_sims


# ── Utilities ─────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_mamba_models():
    print("Testing MambaSpatialEncoder ...")
    enc = MambaSpatialEncoder(d_model=128, n_layers=2, d_state=16)
    x   = torch.randn(2, 32, 128)
    out = enc(x)
    assert out.shape == (2, 32, 128), f"unexpected shape {out.shape}"
    print(f"  OK — output shape {out.shape}")

    print("Testing NeuralMemoryBank ...")
    mb  = NeuralMemoryBank(memory_size=100, key_dim=32, value_dim=64)
    mb.write(torch.randn(4, 32), torch.randn(4, 64))
    ret, sim = mb.read(torch.randn(2, 64), top_k=5)
    assert ret.shape == (2, 5, 64)
    print(f"  OK — retrieved shape {ret.shape}")


if __name__ == '__main__':
    test_mamba_models()
