"""
Fast PPO Trainer for Optimized Mamba Policies
----------------------------------------------
Supports:
1. FastMambaMemorySLAMPolicy (10-30 FPS)
2. UltraFastMambaSLAMPolicy (30-100 FPS)

Optimizations:
- Reduced batch processing overhead
- Cached computations
- Efficient memory management
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List
from torch.distributions import Categorical

from agent.mamba_memory_fast import FastMambaMemorySLAMPolicy, UltraFastMambaSLAMPolicy


class FastMambaPPOTrainer:
    """
    Optimized PPO trainer for fast Mamba policies.
    
    Target: 10-50 FPS (vs 2 FPS baseline)
    """
    
    def __init__(self,
                 obs_dim: int,
                 n_actions: int,
                 policy_type: str = 'fast_hybrid',  # 'fast_hybrid' or 'ultra_fast'
                 d_model: int = 128,
                 n_layers: int = 2,
                 memory_size: int = 500,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_eps: float = 0.2,
                 vf_coef: float = 0.5,
                 ent_coef: float = 0.01,
                 max_grad: float = 0.5,
                 n_epochs: int = 10,
                 batch_size: int = 64,
                 seq_len: int = 32,
                 device: str = 'auto'):
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Create policy network
        self.policy_type = policy_type
        if policy_type == 'fast_hybrid':
            self.net = FastMambaMemorySLAMPolicy(
                obs_dim=obs_dim,
                n_actions=n_actions,
                d_model=d_model,
                n_layers=n_layers,
                memory_size=memory_size
            ).to(self.device)
        elif policy_type == 'ultra_fast':
            self.net = UltraFastMambaSLAMPolicy(
                obs_dim=obs_dim,
                n_actions=n_actions,
                d_model=d_model,
                n_layers=n_layers
            ).to(self.device)
        else:
            raise ValueError(f"Unknown policy_type: {policy_type}")
        
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=1000
        )
        
        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad = max_grad
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.seq_len = seq_len
        
        # Rollout buffer
        self._obs: List = []
        self._actions: List = []
        self._log_probs: List = []
        self._values: List = []
        self._rewards: List = []
        self._dones: List = []
        
        # Additional metrics
        if policy_type == 'fast_hybrid':
            self._loop_closures: List = []
        
        self.train_steps = 0
        self.total_env_steps = 0
        self.losses: Dict = {}
    
    def collect(self, obs: np.ndarray, action: int, log_prob: float,
                value: float, reward: float, done: bool, 
                loop_closure: Optional[float] = None):
        """Collect experience"""
        self._obs.append(obs)
        self._actions.append(action)
        self._log_probs.append(log_prob)
        self._values.append(value)
        self._rewards.append(reward)
        self._dones.append(done)
        
        if self.policy_type == 'fast_hybrid' and loop_closure is not None:
            self._loop_closures.append(loop_closure)
    
    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> Tuple:
        """Select action with NaN detection"""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        if self.policy_type == 'ultra_fast':
            logits, value = self.net(obs_t, update_buffer=True)
            
            # Check for NaN/Inf
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("WARNING: NaN/Inf detected in logits, using uniform distribution")
                logits = torch.zeros_like(logits)
            
            if torch.isnan(value).any() or torch.isinf(value).any():
                print("WARNING: NaN/Inf detected in value, using 0.0")
                value = torch.zeros_like(value)
            
            dist = Categorical(logits=logits)
            action = logits.argmax(-1) if deterministic else dist.sample()
            log_prob = dist.log_prob(action)
            
            return (
                int(action.item()),
                float(log_prob.item()),
                float(value.item())
            )
        
        else:  # fast_hybrid
            outputs = self.net(obs_t, update_buffer=True, update_memory=True)
            logits = outputs['logits']
            value = outputs['value']
            loop_closure_prob = outputs['loop_closure_prob']

            # Check for NaN/Inf
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("WARNING: NaN/Inf detected in logits, using uniform distribution")
                logits = torch.zeros_like(logits)

            if torch.isnan(value).any() or torch.isinf(value).any():
                print("WARNING: NaN/Inf detected in value, using 0.0")
                value = torch.zeros_like(value)

            if torch.isnan(loop_closure_prob).any() or torch.isinf(loop_closure_prob).any():
                print("WARNING: NaN/Inf detected in loop_closure_prob, using 0.5")
                loop_closure_prob = torch.full_like(loop_closure_prob, 0.5)
            
            dist = Categorical(logits=logits)
            action = logits.argmax(-1) if deterministic else dist.sample()
            log_prob = dist.log_prob(action)
            
            return (
                int(action.item()),
                float(log_prob.item()),
                float(value.item()),
                float(loop_closure_prob.item())
            )
    
    def reset_episode(self):
        """Reset policy state"""
        self.net.reset_sequence()
    
    def train(self, last_obs: np.ndarray, last_done: bool) -> Dict:
        """PPO update with sequence-aware mini-batches for Mamba temporal learning."""
        N = len(self._obs)
        if N < self.seq_len * 2:
            return {}

        # Bootstrap value
        with torch.no_grad():
            obs_t = torch.FloatTensor(last_obs).unsqueeze(0).to(self.device)
            if self.policy_type == 'ultra_fast':
                _, last_val = self.net(obs_t, update_buffer=False)
            else:
                out = self.net(obs_t, update_buffer=False, update_memory=False)
                last_val = out['value']
            last_val = float(last_val.item()) * (1.0 - float(last_done))

        advantages = self._compute_gae(last_val)
        returns    = advantages + np.array(self._values, dtype=np.float32)

        # Flat numpy arrays for indexing
        obs_arr = np.array(self._obs,       dtype=np.float32)   # [N, obs_dim]
        act_arr = np.array(self._actions,   dtype=np.int64)
        lp_arr  = np.array(self._log_probs, dtype=np.float32)
        val_arr = np.array(self._values,    dtype=np.float32)
        adv_arr = advantages.astype(np.float32)
        ret_arr = returns.astype(np.float32)

        # Normalize advantages over the full rollout
        adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)

        # Fixed-length non-overlapping sequence chunks
        T         = self.seq_len
        n_seqs    = N // T          # discard the tail (< T steps)
        seq_starts = np.arange(n_seqs) * T   # [n_seqs]

        if n_seqs == 0:
            self._clear_buffer()
            return {}

        # Number of sequences per mini-batch (keeps token count ≈ batch_size)
        seqs_per_batch = max(1, self.batch_size // T)

        total_pl = total_vl = total_el = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            perm = np.random.permutation(n_seqs)

            for chunk in range(0, n_seqs, seqs_per_batch):
                idxs = perm[chunk:chunk + seqs_per_batch]
                if len(idxs) == 0:
                    continue

                B  = len(idxs)
                BT = B * T
                s  = seq_starts[idxs]   # start indices for chosen sequences

                # Build [B, T, ...] sequence batches (vectorized slicing)
                obs_seq = np.stack([obs_arr[si:si+T] for si in s])  # [B, T, obs_dim]
                act_bt  = np.concatenate([act_arr[si:si+T] for si in s])  # [BT]
                lp_bt   = np.concatenate([lp_arr[si:si+T]  for si in s])
                adv_bt  = np.concatenate([adv_arr[si:si+T] for si in s])
                ret_bt  = np.concatenate([ret_arr[si:si+T] for si in s])
                val_bt  = np.concatenate([val_arr[si:si+T] for si in s])

                obs_t = torch.FloatTensor(obs_seq).to(self.device)  # [B, T, obs_dim]
                act_t = torch.LongTensor(act_bt).to(self.device)
                lp_t  = torch.FloatTensor(lp_bt).to(self.device)
                adv_t = torch.FloatTensor(adv_bt).to(self.device)
                ret_t = torch.FloatTensor(ret_bt).to(self.device)
                val_t = torch.FloatTensor(val_bt).to(self.device)

                # Sequence forward — gradients flow through all T Mamba steps
                if self.policy_type == 'ultra_fast':
                    logits, value = self.net.forward_sequence(obs_t)  # [BT, ...], [BT]
                else:
                    out    = self.net.forward_sequence(obs_t)
                    logits = out['logits']   # [BT, n_actions]
                    value  = out['value']    # [BT]

                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    continue
                if torch.isnan(value).any() or torch.isinf(value).any():
                    continue

                dist      = Categorical(logits=logits)
                log_prob  = dist.log_prob(act_t)
                entropy   = dist.entropy()

                ratio = torch.exp(log_prob - lp_t)
                pl1   = -ratio * adv_t
                pl2   = -torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t
                policy_loss = torch.max(pl1, pl2).mean()

                v_clipped  = val_t + torch.clamp(value - val_t, -self.clip_eps, self.clip_eps)
                value_loss = torch.max(F.mse_loss(value, ret_t),
                                       F.mse_loss(v_clipped, ret_t))

                entropy_loss = -entropy.mean()
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                self.optimizer.zero_grad()
                loss.backward()

                has_nan_grad = any(
                    p.grad is not None and
                    (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                    for p in self.net.parameters()
                )
                if has_nan_grad:
                    self.optimizer.zero_grad()
                    continue

                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad)
                self.optimizer.step()

                total_pl += policy_loss.item()
                total_vl += value_loss.item()
                total_el += -entropy_loss.item()
                n_updates += 1

        self.scheduler.step()
        self.train_steps += 1

        if n_updates == 0:
            self._clear_buffer()
            return {}

        self.losses = {
            'policy_loss': round(total_pl / n_updates, 4),
            'value_loss':  round(total_vl / n_updates, 4),
            'entropy':     round(total_el / n_updates, 4),
            'lr':          round(self.scheduler.get_last_lr()[0], 6),
        }

        if self.policy_type == 'fast_hybrid' and self._loop_closures:
            self.losses['mean_loop_closure'] = round(np.mean(self._loop_closures), 4)

        self._clear_buffer()
        return self.losses
    
    def _compute_gae(self, last_val: float) -> np.ndarray:
        """Compute GAE"""
        rewards = np.array(self._rewards, dtype=np.float32)
        values = np.array(self._values, dtype=np.float32)
        dones = np.array(self._dones, dtype=np.float32)
        
        advantages = np.zeros_like(rewards)
        gae = 0.0
        next_val = last_val
        
        for t in reversed(range(len(rewards))):
            delta = (rewards[t] + 
                    self.gamma * next_val * (1 - dones[t]) - 
                    values[t])
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            next_val = values[t]
        
        return advantages
    
    def _clear_buffer(self):
        """Clear rollout buffer"""
        self._obs.clear()
        self._actions.clear()
        self._log_probs.clear()
        self._values.clear()
        self._rewards.clear()
        self._dones.clear()
        if self.policy_type == 'fast_hybrid':
            self._loop_closures.clear()
    
    def save(self, path: str):
        """Save checkpoint"""
        checkpoint = {
            'net_state': self.net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'train_steps': self.train_steps,
            'total_env_steps': self.total_env_steps,
            'policy_type': self.policy_type,
        }
        
        if self.policy_type == 'fast_hybrid':
            checkpoint['memory_bank'] = {
                'memory_keys': self.net.memory_bank.memory_keys,
                'memory_values': self.net.memory_bank.memory_values,
                'memory_age': self.net.memory_bank.memory_age,
                'memory_usage': self.net.memory_bank.memory_usage,
                'write_ptr': self.net.memory_bank.write_ptr,
                'memory_keys_norm': self.net.memory_bank.memory_keys_norm,
                'cache_valid': self.net.memory_bank.cache_valid,
            }
        
        torch.save(checkpoint, path)
        print(f"Saved {self.policy_type} model → {path}")
    
    def load(self, path: str):
        """Load checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint['net_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.train_steps = checkpoint.get('train_steps', 0)
        self.total_env_steps = checkpoint.get('total_env_steps', 0)
        
        if self.policy_type == 'fast_hybrid' and 'memory_bank' in checkpoint:
            mb = checkpoint['memory_bank']
            self.net.memory_bank.memory_keys.copy_(mb['memory_keys'])
            self.net.memory_bank.memory_values.copy_(mb['memory_values'])
            self.net.memory_bank.memory_age.copy_(mb['memory_age'])
            self.net.memory_bank.memory_usage.copy_(mb['memory_usage'])
            self.net.memory_bank.write_ptr.copy_(mb['write_ptr'])
            if 'memory_keys_norm' in mb:
                self.net.memory_bank.memory_keys_norm.copy_(mb['memory_keys_norm'])
                self.net.memory_bank.cache_valid.copy_(mb['cache_valid'])
        
        print(f"Loaded {self.policy_type} model ← {path} (step {self.train_steps})")
    
    def get_memory_stats(self) -> Dict:
        """Get memory stats (fast_hybrid only)"""
        if self.policy_type != 'fast_hybrid':
            return {}
        
        mb = self.net.memory_bank
        return {
            'memory_utilization': float((mb.cache_valid).sum() / mb.memory_size),
            'mean_memory_age': float(mb.memory_age.mean()),
            'max_memory_age': float(mb.memory_age.max()),
            'mean_memory_usage': float(mb.memory_usage.mean()),
            'write_ptr': int(mb.write_ptr.item()),
        }
