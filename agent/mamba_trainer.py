"""
PPO Trainer for Mamba-based SLAM Policies
------------------------------------------
Supports both:
1. Pure Mamba policy
2. Mamba + Memory Bank hybrid

Key differences from standard PPO:
- Handles sequential state (trajectory buffer)
- Memory bank updates during training
- Loop closure detection metrics
- Efficient batching for Mamba's sequential nature
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List
from torch.distributions import Categorical

from agent.mamba_memory import MambaSLAMPolicy, MambaMemorySLAMPolicy


class MambaPPOTrainer:
    """
    PPO trainer adapted for Mamba-based policies.
    
    Handles:
    - Sequential state management
    - Memory bank updates
    - Loop closure detection
    - Efficient training with long sequences
    """
    
    def __init__(self,
                 obs_dim: int,
                 n_actions: int,
                 policy_type: str = 'hybrid',  # 'pure' or 'hybrid'
                 d_model: int = 256,
                 n_layers: int = 4,
                 memory_size: int = 1000,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_eps: float = 0.2,
                 vf_coef: float = 0.5,
                 ent_coef: float = 0.01,
                 max_grad: float = 0.5,
                 n_epochs: int = 10,
                 batch_size: int = 64,
                 device: str = 'auto'):
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Create policy network
        self.policy_type = policy_type
        if policy_type == 'pure':
            self.net = MambaSLAMPolicy(
                obs_dim=obs_dim,
                n_actions=n_actions,
                d_model=d_model,
                n_layers=n_layers
            ).to(self.device)
        elif policy_type == 'hybrid':
            self.net = MambaMemorySLAMPolicy(
                obs_dim=obs_dim,
                n_actions=n_actions,
                d_model=d_model,
                n_layers=n_layers,
                memory_size=memory_size
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
        
        # Rollout buffer
        self._obs: List = []
        self._actions: List = []
        self._log_probs: List = []
        self._values: List = []
        self._rewards: List = []
        self._dones: List = []
        
        # Additional metrics for hybrid policy
        if policy_type == 'hybrid':
            self._loop_closures: List = []
            self._memory_similarities: List = []
        
        self.train_steps = 0
        self.losses: Dict = {}
    
    # ── Rollout collection ─────────────────────────────────────────
    
    def collect(self, obs: np.ndarray, action: int, log_prob: float,
                value: float, reward: float, done: bool, 
                loop_closure: Optional[float] = None):
        """Collect experience for training"""
        self._obs.append(obs)
        self._actions.append(action)
        self._log_probs.append(log_prob)
        self._values.append(value)
        self._rewards.append(reward)
        self._dones.append(done)
        
        if self.policy_type == 'hybrid' and loop_closure is not None:
            self._loop_closures.append(loop_closure)
    
    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> Tuple:
        """
        Select action using current policy.
        
        Returns:
            action: int
            log_prob: float
            value: float
            (optional) loop_closure_prob: float (for hybrid policy)
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        if self.policy_type == 'pure':
            logits, value = self.net(obs_t, update_buffer=True)
            dist = Categorical(logits=logits)
            action = logits.argmax(-1) if deterministic else dist.sample()
            log_prob = dist.log_prob(action)
            
            return (
                int(action.item()),
                float(log_prob.item()),
                float(value.item())
            )
        
        else:  # hybrid
            outputs = self.net(obs_t, update_buffer=True, update_memory=True)
            logits = outputs['logits']
            value = outputs['value']
            loop_closure_prob = outputs['loop_closure_prob']
            
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
        """Reset policy state at episode start"""
        self.net.reset_sequence()
    
    # ── PPO update ─────────────────────────────────────────────────
    
    def train(self, last_obs: np.ndarray, last_done: bool) -> Dict:
        """Run PPO update on collected rollout"""
        if len(self._obs) < self.batch_size:
            return {}
        
        # Bootstrap value
        with torch.no_grad():
            obs_t = torch.FloatTensor(last_obs).unsqueeze(0).to(self.device)
            if self.policy_type == 'pure':
                _, last_val = self.net(obs_t, update_buffer=False)
            else:
                outputs = self.net(obs_t, update_buffer=False, update_memory=False)
                last_val = outputs['value']
            last_val = float(last_val.item()) * (1.0 - float(last_done))
        
        # Compute advantages with GAE
        advantages = self._compute_gae(last_val)
        returns = advantages + np.array(self._values, dtype=np.float32)
        
        # Convert to tensors
        obs_t = torch.FloatTensor(np.array(self._obs)).to(self.device)
        act_t = torch.LongTensor(self._actions).to(self.device)
        lp_t = torch.FloatTensor(self._log_probs).to(self.device)
        adv_t = torch.FloatTensor(advantages).to(self.device)
        ret_t = torch.FloatTensor(returns).to(self.device)
        val_t = torch.FloatTensor(self._values).to(self.device)
        
        # Normalize advantages
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        
        N = len(self._obs)
        total_pl = total_vl = total_el = 0.0
        n_updates = 0
        
        # PPO epochs
        for _ in range(self.n_epochs):
            idxs = np.random.permutation(N)
            
            for start in range(0, N, self.batch_size):
                b = idxs[start:start + self.batch_size]
                
                # Forward pass
                if self.policy_type == 'pure':
                    logits, value = self.net(obs_t[b], update_buffer=False)
                    dist = Categorical(logits=logits)
                    log_prob = dist.log_prob(act_t[b])
                    entropy = dist.entropy()
                else:
                    outputs = self.net(obs_t[b], update_buffer=False, update_memory=False)
                    logits = outputs['logits']
                    value = outputs['value']
                    dist = Categorical(logits=logits)
                    log_prob = dist.log_prob(act_t[b])
                    entropy = dist.entropy()
                
                # Policy loss (clipped surrogate)
                ratio = torch.exp(log_prob - lp_t[b])
                adv_b = adv_t[b]
                
                pl1 = -ratio * adv_b
                pl2 = -torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                policy_loss = torch.max(pl1, pl2).mean()
                
                # Value loss (clipped)
                v_clipped = val_t[b] + torch.clamp(
                    value - val_t[b], -self.clip_eps, self.clip_eps
                )
                vl1 = F.mse_loss(value, ret_t[b])
                vl2 = F.mse_loss(v_clipped, ret_t[b])
                value_loss = torch.max(vl1, vl2)
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (policy_loss + 
                       self.vf_coef * value_loss + 
                       self.ent_coef * entropy_loss)
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad)
                self.optimizer.step()
                
                total_pl += policy_loss.item()
                total_vl += value_loss.item()
                total_el += (-entropy_loss.item())
                n_updates += 1
        
        self.scheduler.step()
        self.train_steps += 1
        
        # Compute metrics
        self.losses = {
            'policy_loss': round(total_pl / n_updates, 4),
            'value_loss': round(total_vl / n_updates, 4),
            'entropy': round(total_el / n_updates, 4),
            'lr': round(self.scheduler.get_last_lr()[0], 6),
        }
        
        # Add hybrid-specific metrics
        if self.policy_type == 'hybrid' and self._loop_closures:
            self.losses['mean_loop_closure'] = round(
                np.mean(self._loop_closures), 4
            )
        
        self._clear_buffer()
        return self.losses
    
    def _compute_gae(self, last_val: float) -> np.ndarray:
        """Compute Generalized Advantage Estimation"""
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
        if self.policy_type == 'hybrid':
            self._loop_closures.clear()
            self._memory_similarities.clear()
    
    # ── Save/Load ──────────────────────────────────────────────────
    
    def save(self, path: str):
        """Save model checkpoint"""
        checkpoint = {
            'net_state': self.net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'train_steps': self.train_steps,
            'policy_type': self.policy_type,
        }
        
        # Save memory bank state for hybrid policy
        if self.policy_type == 'hybrid':
            checkpoint['memory_bank'] = {
                'memory_keys': self.net.memory_bank.memory_keys,
                'memory_values': self.net.memory_bank.memory_values,
                'memory_age': self.net.memory_bank.memory_age,
                'memory_usage': self.net.memory_bank.memory_usage,
                'write_ptr': self.net.memory_bank.write_ptr,
            }
        
        torch.save(checkpoint, path)
        print(f"Saved {self.policy_type} model → {path}")
    
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint['net_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.train_steps = checkpoint.get('train_steps', 0)
        
        # Restore memory bank for hybrid policy
        if self.policy_type == 'hybrid' and 'memory_bank' in checkpoint:
            mb = checkpoint['memory_bank']
            self.net.memory_bank.memory_keys.copy_(mb['memory_keys'])
            self.net.memory_bank.memory_values.copy_(mb['memory_values'])
            self.net.memory_bank.memory_age.copy_(mb['memory_age'])
            self.net.memory_bank.memory_usage.copy_(mb['memory_usage'])
            self.net.memory_bank.write_ptr.copy_(mb['write_ptr'])
        
        print(f"Loaded {self.policy_type} model ← {path} (step {self.train_steps})")
    
    # ── Analysis ───────────────────────────────────────────────────
    
    def get_memory_stats(self) -> Dict:
        """Get memory bank statistics (hybrid policy only)"""
        if self.policy_type != 'hybrid':
            return {}
        
        mb = self.net.memory_bank
        return {
            'memory_utilization': float((mb.memory_usage > 0).sum() / mb.memory_size),
            'mean_memory_age': float(mb.memory_age.mean()),
            'max_memory_age': float(mb.memory_age.max()),
            'mean_memory_usage': float(mb.memory_usage.mean()),
            'write_ptr': int(mb.write_ptr.item()),
        }
    
    def analyze_loop_closures(self, obs: np.ndarray, threshold: float = 0.85) -> List[int]:
        """Detect loop closures for current observation"""
        if self.policy_type != 'hybrid':
            return []
        
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        return self.net.get_loop_closures(obs_t, threshold)


def compare_policies(obs_dim: int, n_actions: int):
    """Compare Pure Mamba vs Hybrid architectures"""
    print("\n" + "="*60)
    print("Mamba Policy Comparison")
    print("="*60 + "\n")
    
    # Pure Mamba
    print("1. Pure Mamba Policy")
    print("-" * 60)
    pure_trainer = MambaPPOTrainer(
        obs_dim=obs_dim,
        n_actions=n_actions,
        policy_type='pure',
        d_model=256,
        n_layers=4
    )
    
    n_params_pure = sum(p.numel() for p in pure_trainer.net.parameters())
    print(f"Parameters: {n_params_pure:,}")
    print(f"Device: {pure_trainer.device}")
    print()
    
    # Hybrid Mamba + Memory
    print("2. Hybrid Mamba + Memory Bank Policy")
    print("-" * 60)
    hybrid_trainer = MambaPPOTrainer(
        obs_dim=obs_dim,
        n_actions=n_actions,
        policy_type='hybrid',
        d_model=256,
        n_layers=4,
        memory_size=1000
    )
    
    n_params_hybrid = sum(p.numel() for p in hybrid_trainer.net.parameters())
    print(f"Parameters: {n_params_hybrid:,}")
    print(f"Memory Bank Size: 1000 landmarks")
    print(f"Device: {hybrid_trainer.device}")
    print()
    
    # Comparison
    print("Comparison")
    print("-" * 60)
    print(f"Parameter Overhead: +{n_params_hybrid - n_params_pure:,} "
          f"({(n_params_hybrid/n_params_pure - 1)*100:.1f}%)")
    print()
    
    print("Advantages of Pure Mamba:")
    print("  ✓ Simpler architecture")
    print("  ✓ Fewer parameters")
    print("  ✓ Faster inference")
    print()
    
    print("Advantages of Hybrid (Mamba + Memory):")
    print("  ✓ Explicit landmark storage")
    print("  ✓ Loop closure detection")
    print("  ✓ Better long-term spatial memory")
    print("  ✓ Interpretable memory retrieval")
    print()
    
    print("Recommendation for City Mapping:")
    print("  → Use Hybrid for 90%+ accuracy goal")
    print("  → Memory bank crucial for loop closure")
    print("  → Explicit landmarks improve consistency")
    print()


if __name__ == '__main__':
    # Test with typical SLAM observation dimension
    obs_dim = 500  # Local patch + global map + metadata
    n_actions = 8  # 8-directional movement
    
    compare_policies(obs_dim, n_actions)
