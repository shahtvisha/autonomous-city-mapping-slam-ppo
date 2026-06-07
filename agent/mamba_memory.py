"""
Mamba-based Spatial Memory for Active SLAM
-------------------------------------------
Implements two architectures:
1. Pure Mamba: State Space Model for sequential spatial reasoning
2. Mamba + Memory Bank: Hybrid with explicit landmark storage

Mamba advantages over Transformers:
- O(n) complexity vs O(n²) 
- Better long-range dependencies
- Efficient for infinite-length trajectories
- Selective state space allows focusing on important spatial features

References:
- Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)
- Structured State Space Models (S4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
import math


# ══════════════════════════════════════════════════════════════════
# Mamba Core Components
# ══════════════════════════════════════════════════════════════════

@dataclass
class MambaConfig:
    """Configuration for Mamba block"""
    d_model: int = 256          # Model dimension
    d_state: int = 16           # SSM state dimension
    d_conv: int = 4             # Convolution kernel size
    expand: int = 2             # Expansion factor
    dt_rank: str = "auto"       # Rank of Δ projection
    dt_min: float = 0.001       # Min Δ value
    dt_max: float = 0.1         # Max Δ value
    dt_init: str = "random"     # Δ initialization
    dt_scale: float = 1.0       # Δ scaling
    bias: bool = False          # Use bias in linear layers
    conv_bias: bool = True      # Use bias in conv layers
    
    def __post_init__(self):
        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)


class MambaBlock(nn.Module):
    """
    Single Mamba block with selective SSM.
    
    Key innovation: Selection mechanism allows model to filter out
    irrelevant information (e.g., revisited areas) and focus on
    novel spatial information (frontiers, unexplored regions).
    """
    
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        
        d_inner = config.expand * config.d_model
        
        # Input projection
        self.in_proj = nn.Linear(config.d_model, d_inner * 2, bias=config.bias)
        
        # Depthwise convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=config.d_conv,
            groups=d_inner,
            padding=config.d_conv - 1,
            bias=config.conv_bias
        )
        
        # SSM parameters - these are the "selective" part
        self.x_proj = nn.Linear(d_inner, config.dt_rank + config.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(config.dt_rank, d_inner, bias=True)
        
        # Initialize dt_proj to preserve gradients
        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        # SSM state matrices (A, D)
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))  # Log for numerical stability
        self.D = nn.Parameter(torch.ones(d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(d_inner, config.d_model, bias=config.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D) where L is sequence length (trajectory length)
        Returns: (B, L, D)
        """
        B, L, D = x.shape
        
        # Input projection and split
        x_and_res = self.in_proj(x)  # (B, L, 2*d_inner)
        x, res = x_and_res.split(self.config.expand * self.config.d_model, dim=-1)
        
        # Convolution for local spatial context
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)[:, :, :L]  # Trim to original length
        x = x.transpose(1, 2)  # (B, L, d_inner)
        
        # Activation
        x = F.silu(x)
        
        # SSM (the core selective state space model)
        y = self.ssm(x)
        
        # Gating with residual
        y = y * F.silu(res)
        
        # Output projection
        output = self.out_proj(y)
        
        return output
    
    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Selective State Space Model.
        
        The selection mechanism allows the model to:
        - Remember important landmarks (high Δ)
        - Forget revisited areas (low Δ)
        - Focus on frontiers and unexplored regions
        """
        B, L, D = x.shape
        
        # Compute selection parameters (Δ, B, C)
        x_proj = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        
        delta, B_ssm, C_ssm = torch.split(
            x_proj,
            [self.config.dt_rank, self.config.d_state, self.config.d_state],
            dim=-1
        )
        
        # Compute Δ (discretization step size)
        delta = F.softplus(self.dt_proj(delta))  # (B, L, d_inner)
        
        # Get A matrix
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # Discretize continuous parameters
        # This is where the "selection" happens - different Δ for each position
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, d_inner, d_state)
        deltaB = delta.unsqueeze(-1) * B_ssm.unsqueeze(2)  # (B, L, d_inner, d_state)
        
        # Selective scan (the core SSM computation)
        # This maintains a hidden state that selectively remembers spatial information
        y = self.selective_scan(x, deltaA, deltaB, C_ssm, self.D)
        
        return y
    
    def selective_scan(self, x, deltaA, deltaB, C, D):
        """
        Efficient selective scan implementation.
        
        Maintains hidden state h that evolves as:
        h_t = A * h_{t-1} + B * x_t
        y_t = C * h_t + D * x_t
        
        The selection mechanism (via Δ) allows focusing on important
        spatial features while forgetting irrelevant ones.
        """
        B, L, D_inner = x.shape
        _, _, _, d_state = deltaA.shape
        
        # Initialize hidden state
        h = torch.zeros(B, D_inner, d_state, device=x.device, dtype=x.dtype)
        
        # Scan through sequence
        ys = []
        for i in range(L):
            # Update hidden state (selective memory update)
            h = deltaA[:, i] * h + deltaB[:, i] * x[:, i].unsqueeze(-1)
            
            # Compute output
            y = torch.einsum('bdn,bn->bd', h, C[:, i]) + D * x[:, i]
            ys.append(y)
        
        y = torch.stack(ys, dim=1)  # (B, L, d_inner)
        return y


class MambaSpatialEncoder(nn.Module):
    """
    Stack of Mamba blocks for encoding spatial trajectories.
    
    Processes agent's trajectory through the environment, maintaining
    long-term spatial context efficiently.
    """
    
    def __init__(self, d_model: int = 256, n_layers: int = 4, d_state: int = 16):
        super().__init__()
        
        config = MambaConfig(
            d_model=d_model,
            d_state=d_state,
            d_conv=4,
            expand=2
        )
        
        self.layers = nn.ModuleList([
            MambaBlock(config) for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, return_all_layers: bool = False) -> torch.Tensor:
        """
        x: (B, L, D) - sequence of spatial observations
        Returns: (B, L, D) or List[(B, L, D)] if return_all_layers
        """
        if return_all_layers:
            outputs = []
        
        for layer in self.layers:
            x = x + layer(x)  # Residual connection
            if return_all_layers:
                outputs.append(x)
        
        x = self.norm(x)
        
        return outputs if return_all_layers else x


# ══════════════════════════════════════════════════════════════════
# Neural Memory Bank for Landmark Storage
# ══════════════════════════════════════════════════════════════════

class NeuralMemoryBank(nn.Module):
    """
    External memory bank for storing and retrieving spatial landmarks.
    
    Key features:
    - Stores landmark descriptors (local map patches)
    - Efficient similarity-based retrieval
    - Supports loop closure detection
    - Differentiable read/write operations
    """
    
    def __init__(self, 
                 memory_size: int = 1000,
                 key_dim: int = 128,
                 value_dim: int = 256,
                 num_heads: int = 4):
        super().__init__()
        
        self.memory_size = memory_size
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        
        # Memory slots
        self.register_buffer('memory_keys', torch.zeros(memory_size, key_dim))
        self.register_buffer('memory_values', torch.zeros(memory_size, value_dim))
        self.register_buffer('memory_age', torch.zeros(memory_size))
        self.register_buffer('memory_usage', torch.zeros(memory_size))
        self.register_buffer('write_ptr', torch.tensor(0, dtype=torch.long))
        
        # Query/Key/Value projections for attention-based retrieval
        self.query_proj = nn.Linear(value_dim, key_dim)
        self.key_proj = nn.Linear(key_dim, key_dim)
        self.value_proj = nn.Linear(value_dim, value_dim)
        
        # Multi-head attention for retrieval
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=value_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
    def write(self, keys: torch.Tensor, values: torch.Tensor):
        """
        Write new landmarks to memory.
        
        keys: (B, key_dim) - spatial descriptors (e.g., local map features)
        values: (B, value_dim) - full landmark information
        """
        B = keys.shape[0]
        
        for i in range(B):
            # Write to current position
            ptr = int(self.write_ptr.item())
            self.memory_keys[ptr] = keys[i].detach()
            self.memory_values[ptr] = values[i].detach()
            self.memory_age[ptr] = 0
            self.memory_usage[ptr] = 0
            
            # Advance write pointer (circular buffer)
            self.write_ptr = (self.write_ptr + 1) % self.memory_size
        
        # Age all memories
        self.memory_age += 1
    
    def read(self, query: torch.Tensor, top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve similar landmarks from memory.
        
        query: (B, value_dim) - current observation
        Returns: 
            retrieved: (B, top_k, value_dim) - retrieved landmarks
            similarities: (B, top_k) - similarity scores
        """
        B = query.shape[0]
        
        # Project query to key space
        query_key = self.query_proj(query)  # (B, key_dim)
        
        # Compute similarities with all memory keys
        # Cosine similarity for better generalization
        query_norm = F.normalize(query_key, dim=-1)
        memory_norm = F.normalize(self.memory_keys, dim=-1)
        
        similarities = torch.matmul(query_norm, memory_norm.t())  # (B, memory_size)
        
        # Get top-k most similar
        top_k_sims, top_k_indices = torch.topk(similarities, k=top_k, dim=-1)
        
        # Retrieve corresponding values
        retrieved = torch.stack([
            self.memory_values[top_k_indices[i]] for i in range(B)
        ])  # (B, top_k, value_dim)
        
        # Update usage statistics
        for i in range(B):
            self.memory_usage[top_k_indices[i]] += 1
        
        return retrieved, top_k_sims
    
    def get_loop_closure_candidates(self, query: torch.Tensor, 
                                    threshold: float = 0.85) -> List[int]:
        """
        Find potential loop closures (revisited locations).
        
        query: (1, value_dim) - current observation
        threshold: similarity threshold for loop closure
        Returns: list of memory indices that might be loop closures
        """
        query_key = self.query_proj(query)
        query_norm = F.normalize(query_key, dim=-1)
        memory_norm = F.normalize(self.memory_keys, dim=-1)
        
        similarities = torch.matmul(query_norm, memory_norm.t()).squeeze(0)
        
        # Find high-similarity memories (excluding very recent ones)
        candidates = []
        for i, sim in enumerate(similarities):
            if sim > threshold and self.memory_age[i] > 50:  # Not too recent
                candidates.append(i)
        
        return candidates


# ══════════════════════════════════════════════════════════════════
# Complete Policy Networks
# ══════════════════════════════════════════════════════════════════

class MambaSLAMPolicy(nn.Module):
    """
    Pure Mamba-based policy for Active SLAM.
    
    Architecture:
    1. Encode current observation
    2. Process with Mamba (maintains spatial context)
    3. Output action and value
    """
    
    def __init__(self, 
                 obs_dim: int,
                 n_actions: int,
                 d_model: int = 256,
                 n_layers: int = 4,
                 max_seq_len: int = 1000):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Observation encoder - handles variable input dimensions
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Mamba encoder for spatial reasoning
        self.mamba = MambaSpatialEncoder(
            d_model=d_model,
            n_layers=n_layers,
            d_state=16
        )
        
        # Policy and value heads
        self.policy_head = nn.Linear(d_model, n_actions)
        self.value_head = nn.Linear(d_model, 1)
        
        # Sequence buffer (stores recent trajectory)
        self.register_buffer('seq_buffer', torch.zeros(1, max_seq_len, d_model))
        self.register_buffer('seq_len', torch.tensor(0, dtype=torch.long))
        
    def forward(self, obs: torch.Tensor, update_buffer: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        obs: (B, obs_dim)
        Returns: logits (B, n_actions), value (B,)
        """
        B = obs.shape[0]
        
        # Encode observation
        obs_encoded = self.obs_encoder(obs)  # (B, d_model)
        
        if update_buffer and B == 1:  # Only update buffer during rollout
            # Add to sequence buffer
            ptr = int(self.seq_len.item()) % self.max_seq_len
            self.seq_buffer[0, ptr] = obs_encoded[0].detach()
            self.seq_len.add_(1).clamp_(max=self.max_seq_len)
            
            # Get current sequence
            seq_len = int(self.seq_len.item())
            if seq_len < self.max_seq_len:
                sequence = self.seq_buffer[:, :seq_len]
            else:
                # Circular buffer - reorder
                sequence = torch.cat([
                    self.seq_buffer[:, ptr+1:],
                    self.seq_buffer[:, :ptr+1]
                ], dim=1)
        else:
            # Batch mode - treat each as independent sequence
            sequence = obs_encoded.unsqueeze(1)  # (B, 1, d_model)
        
        # Process with Mamba
        spatial_context = self.mamba(sequence)  # (B, L, d_model)
        
        # Use last position for action/value
        current_state = spatial_context[:, -1]  # (B, d_model)
        
        # Compute policy and value
        logits = self.policy_head(current_state)
        value = self.value_head(current_state).squeeze(-1)
        
        return logits, value
    
    def reset_sequence(self):
        """Reset sequence buffer (call at episode start)"""
        self.seq_buffer.zero_()
        self.seq_len.zero_()


class MambaMemorySLAMPolicy(nn.Module):
    """
    Hybrid Mamba + Memory Bank policy for Active SLAM.
    
    Architecture:
    1. Encode current observation
    2. Retrieve similar landmarks from memory bank
    3. Process with Mamba (trajectory + retrieved memories)
    4. Update memory bank with current observation
    5. Output action and value
    
    Advantages:
    - Mamba handles sequential reasoning
    - Memory bank provides explicit landmark storage
    - Loop closure detection via memory retrieval
    - Best of both worlds: efficiency + explicit memory
    """
    
    def __init__(self,
                 obs_dim: int,
                 n_actions: int,
                 d_model: int = 256,
                 n_layers: int = 4,
                 memory_size: int = 1000,
                 max_seq_len: int = 500):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Memory bank for landmark storage
        self.memory_bank = NeuralMemoryBank(
            memory_size=memory_size,
            key_dim=128,
            value_dim=d_model,
            num_heads=4
        )
        
        # Mamba encoder for spatial reasoning
        self.mamba = MambaSpatialEncoder(
            d_model=d_model,
            n_layers=n_layers,
            d_state=16
        )
        
        # Fusion layer (combine current state + retrieved memories)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Policy and value heads
        self.policy_head = nn.Linear(d_model, n_actions)
        self.value_head = nn.Linear(d_model, 1)
        
        # Loop closure detection head
        self.loop_closure_head = nn.Linear(d_model, 1)
        
        # Sequence buffer
        self.register_buffer('seq_buffer', torch.zeros(1, max_seq_len, d_model))
        self.register_buffer('seq_len', torch.tensor(0, dtype=torch.long))
        
        # Memory write frequency (don't write every step)
        self.write_counter = 0
        self.write_frequency = 5  # Write every N steps
        
    def forward(self, obs: torch.Tensor, 
                update_buffer: bool = True,
                update_memory: bool = True) -> Dict[str, torch.Tensor]:
        """
        obs: (B, obs_dim)
        Returns: dict with logits, value, loop_closure_prob
        """
        B = obs.shape[0]
        
        # Encode observation
        obs_encoded = self.obs_encoder(obs)  # (B, d_model)
        
        # Retrieve similar landmarks from memory
        retrieved_memories, similarities = self.memory_bank.read(
            obs_encoded, top_k=10
        )  # (B, 10, d_model), (B, 10)
        
        # Aggregate retrieved memories (weighted by similarity)
        memory_context = torch.sum(
            retrieved_memories * similarities.unsqueeze(-1),
            dim=1
        )  # (B, d_model)
        
        # Update sequence buffer
        if update_buffer and B == 1:
            ptr = int(self.seq_len.item()) % self.max_seq_len
            self.seq_buffer[0, ptr] = obs_encoded[0].detach()
            self.seq_len.add_(1).clamp_(max=self.max_seq_len)
            
            seq_len = int(self.seq_len.item())
            if seq_len < self.max_seq_len:
                sequence = self.seq_buffer[:, :seq_len]
            else:
                sequence = torch.cat([
                    self.seq_buffer[:, ptr+1:],
                    self.seq_buffer[:, :ptr+1]
                ], dim=1)
        else:
            sequence = obs_encoded.unsqueeze(1)
        
        # Process with Mamba
        spatial_context = self.mamba(sequence)
        current_state = spatial_context[:, -1]  # (B, d_model)
        
        # Fuse current state with memory context
        fused_state = self.fusion(
            torch.cat([current_state, memory_context], dim=-1)
        )  # (B, d_model)
        
        # Compute outputs
        logits = self.policy_head(fused_state)
        value = self.value_head(fused_state).squeeze(-1)
        loop_closure_prob = torch.sigmoid(
            self.loop_closure_head(fused_state)
        ).squeeze(-1)
        
        # Update memory bank (periodically)
        if update_memory and B == 1:
            self.write_counter += 1
            if self.write_counter >= self.write_frequency:
                # Write current observation to memory
                key = obs_encoded[:, :128]  # Use first 128 dims as key
                self.memory_bank.write(key, obs_encoded)
                self.write_counter = 0
        
        return {
            'logits': logits,
            'value': value,
            'loop_closure_prob': loop_closure_prob,
            'retrieved_similarities': similarities,
            'memory_context': memory_context
        }
    
    def reset_sequence(self):
        """Reset sequence buffer (call at episode start)"""
        self.seq_buffer.zero_()
        self.seq_len.zero_()
        self.write_counter = 0
    
    def get_loop_closures(self, obs: torch.Tensor, threshold: float = 0.85) -> List[int]:
        """
        Detect potential loop closures.
        
        obs: (1, obs_dim)
        Returns: list of memory indices that are potential loop closures
        """
        obs_encoded = self.obs_encoder(obs)
        return self.memory_bank.get_loop_closure_candidates(obs_encoded, threshold)


# ══════════════════════════════════════════════════════════════════
# Utility Functions
# ══════════════════════════════════════════════════════════════════

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_mamba_models():
    """Test both Mamba architectures"""
    print("Testing Mamba-based SLAM policies...\n")
    
    obs_dim = 500  # Example observation dimension
    n_actions = 8
    batch_size = 4
    
    # Test Pure Mamba
    print("1. Pure Mamba Policy")
    print("-" * 50)
    mamba_policy = MambaSLAMPolicy(obs_dim, n_actions)
    print(f"Parameters: {count_parameters(mamba_policy):,}")
    
    obs = torch.randn(batch_size, obs_dim)
    logits, value = mamba_policy(obs, update_buffer=False)
    print(f"Input shape: {obs.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Value shape: {value.shape}")
    print()
    
    # Test Mamba + Memory Bank
    print("2. Mamba + Memory Bank Policy")
    print("-" * 50)
    hybrid_policy = MambaMemorySLAMPolicy(obs_dim, n_actions)
    print(f"Parameters: {count_parameters(hybrid_policy):,}")
    
    outputs = hybrid_policy(obs, update_buffer=False, update_memory=False)
    print(f"Input shape: {obs.shape}")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Value shape: {outputs['value'].shape}")
    print(f"Loop closure prob shape: {outputs['loop_closure_prob'].shape}")
    print(f"Retrieved similarities shape: {outputs['retrieved_similarities'].shape}")
    print()
    
    print("✓ Both models initialized successfully!")


if __name__ == '__main__':
    test_mamba_models()
