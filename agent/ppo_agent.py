"""
PPO Agent with SLAM-aware Policy Network
-----------------------------------------
Architecture designed specifically for Active SLAM:

  Input observation splits into 3 streams:
    1. Local CNN  — processes the nearby SLAM patch spatially
    2. Global MLP — processes downsampled full-map view
    3. Meta MLP   — processes pose, coverage, frontier info

  Streams are concatenated and fed to shared actor-critic trunk.
  
  This multi-stream architecture is key for real-world deployment:
  the agent simultaneously understands local geometry, global progress,
  and high-level task state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from typing import Tuple, Optional


class LocalCNN(nn.Module):
    """
    Processes the local SLAM patch (P×P) as a spatial feature extractor.
    Mimics how a real robot would process a local occupancy grid.
    """
    def __init__(self, patch_size: int, out_features: int = 64):
        super().__init__()
        P = patch_size
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, out_features),
            nn.ReLU(),
        )
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, P*P) → reshape to (B, 1, P, P)
        B = x.shape[0]
        P = int(round(x.shape[1] ** 0.5))
        x = x.view(B, 1, P, P)
        return self.net(x)


class GlobalMLP(nn.Module):
    """Processes the downsampled global map."""
    def __init__(self, global_cells: int, out_features: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_cells, 128),
            nn.ReLU(),
            nn.Linear(128, out_features),
            nn.ReLU(),
        )
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MetaMLP(nn.Module):
    """Processes pose, coverage, frontier, and step info."""
    def __init__(self, meta_dim: int, out_features: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(meta_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_features),
            nn.ReLU(),
        )
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SLAMPolicyNet(nn.Module):
    """Multi-stream policy for local map, global map, and task metadata."""
    def __init__(self, obs_dim, n_actions, patch_r=5,
                 global_size=16, hidden_size=128):
        super().__init__()
        self.local_dim  = (2*patch_r+1)**2
        self.global_dim = global_size**2
        self.meta_dim = obs_dim - self.local_dim - self.global_dim
        if self.meta_dim <= 0:
            raise ValueError(
                f"obs_dim={obs_dim} is too small for local/global inputs")

        self.local_net = LocalCNN(2*patch_r+1, out_features=64)
        self.global_net = GlobalMLP(self.global_dim, out_features=64)
        self.meta_net = MetaMLP(self.meta_dim, out_features=32)

        fused_dim = (
            self.local_net.out_features +
            self.global_net.out_features +
            self.meta_net.out_features
        )
        self.trunk = nn.Sequential(
            nn.Linear(fused_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_size, n_actions)
        self.value_head  = nn.Linear(hidden_size, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight,  gain=1.0)

    def forward(self, obs):
        local = obs[:, :self.local_dim]
        global_map = obs[:, self.local_dim:self.local_dim + self.global_dim]
        meta = obs[:, self.local_dim + self.global_dim:]
        h = torch.cat([
            self.local_net(local),
            self.global_net(global_map),
            self.meta_net(meta),
        ], dim=-1)
        h = self.trunk(h)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    def act(self, obs, deterministic=False):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = logits.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(self, obs, actions):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), value, dist.entropy()
    
    
class PPOTrainer:
    """
    PPO implementation with:
      - Generalised Advantage Estimation (GAE)
      - Value function clipping
      - Entropy bonus for exploration
      - Gradient clipping
      - Learning rate annealing
    """

    def __init__(self,
                 obs_dim:      int,
                 n_actions:    int,
                 patch_r:      int   = 5,
                 global_size:  int   = 16,
                 lr:           float = 3e-4,
                 gamma:        float = 0.99,
                 gae_lambda:   float = 0.95,
                 clip_eps:     float = 0.2,
                 vf_coef:      float = 0.5,
                 ent_coef:     float = 0.01,
                 max_grad:     float = 0.5,
                 n_epochs:     int   = 10,
                 batch_size:   int   = 64,
                 device:       str   = 'auto'):

        if device == 'auto':
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.net = SLAMPolicyNet(
            obs_dim, n_actions, patch_r, global_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr,
                                          eps=1e-5)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1.0, end_factor=0.1,
            total_iters=1000)

        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps   = clip_eps
        self.vf_coef    = vf_coef
        self.ent_coef   = ent_coef
        self.max_grad   = max_grad
        self.n_epochs   = n_epochs
        self.batch_size = batch_size

        # rollout buffer
        self._obs:      list = []
        self._actions:  list = []
        self._log_probs:list = []
        self._values:   list = []
        self._rewards:  list = []
        self._dones:    list = []

        self.train_steps = 0
        self.losses: dict = {}

    # ── Rollout collection ─────────────────────────────────────────

    def collect(self, obs: np.ndarray, action: int, log_prob: float,
                value: float, reward: float, done: bool):
        self._obs.append(obs)
        self._actions.append(action)
        self._log_probs.append(log_prob)
        self._values.append(value)
        self._rewards.append(reward)
        self._dones.append(done)

    @torch.no_grad()
    def act(self, obs: np.ndarray,
            deterministic: bool = False) -> Tuple[int, float, float]:
        t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action, log_prob, value = self.net.act(t, deterministic)
        return (int(action.item()),
                float(log_prob.item()),
                float(value.item()))

    # ── PPO update ─────────────────────────────────────────────────

    def train(self, last_obs: np.ndarray, last_done: bool) -> dict:
        """Run PPO update on the collected rollout."""
        if len(self._obs) < self.batch_size:
            return {}

        # bootstrap value
        with torch.no_grad():
            t = torch.FloatTensor(last_obs).unsqueeze(0).to(self.device)
            _, _, last_val = self.net.act(t)
            last_val = float(last_val) * (1.0 - float(last_done))

        # GAE
        advantages = self._compute_gae(last_val)
        returns = advantages + np.array(self._values, dtype=np.float32)

        # to tensors
        obs_t   = torch.FloatTensor(np.array(self._obs)).to(self.device)
        act_t   = torch.LongTensor(self._actions).to(self.device)
        lp_t    = torch.FloatTensor(self._log_probs).to(self.device)
        adv_t   = torch.FloatTensor(advantages).to(self.device)
        ret_t   = torch.FloatTensor(returns).to(self.device)
        val_t   = torch.FloatTensor(self._values).to(self.device)

        # normalise advantages
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        N = len(self._obs)
        total_pl = total_vl = total_el = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            idxs = np.random.permutation(N)
            for start in range(0, N, self.batch_size):
                b = idxs[start:start + self.batch_size]
                log_prob, value, entropy = self.net.evaluate(
                    obs_t[b], act_t[b])

                ratio = torch.exp(log_prob - lp_t[b])
                adv_b = adv_t[b]

                # policy loss (clipped surrogate)
                pl1 = -ratio * adv_b
                pl2 = -torch.clamp(ratio, 1 - self.clip_eps,
                                   1 + self.clip_eps) * adv_b
                policy_loss = torch.max(pl1, pl2).mean()

                # value loss (clipped)
                v_clipped = val_t[b] + torch.clamp(
                    value - val_t[b], -self.clip_eps, self.clip_eps)
                vl1 = F.mse_loss(value, ret_t[b])
                vl2 = F.mse_loss(v_clipped, ret_t[b])
                value_loss = torch.max(vl1, vl2)

                entropy_loss = -entropy.mean()
                loss = (policy_loss
                        + self.vf_coef * value_loss
                        + self.ent_coef * entropy_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.net.parameters(), self.max_grad)
                self.optimizer.step()

                total_pl += policy_loss.item()
                total_vl += value_loss.item()
                total_el += (-entropy_loss.item())
                n_updates += 1

        self.scheduler.step()
        self.train_steps += 1

        self.losses = {
            'policy_loss':  round(total_pl / n_updates, 4),
            'value_loss':   round(total_vl / n_updates, 4),
            'entropy':      round(total_el / n_updates, 4),
            'lr':           round(self.scheduler.get_last_lr()[0], 6),
        }

        self._clear_buffer()
        return self.losses

    def _compute_gae(self, last_val: float) -> np.ndarray:
        rewards  = np.array(self._rewards,  dtype=np.float32)
        values   = np.array(self._values,   dtype=np.float32)
        dones    = np.array(self._dones,    dtype=np.float32)

        advantages = np.zeros_like(rewards)
        gae = 0.0
        next_val = last_val

        for t in reversed(range(len(rewards))):
            delta = (rewards[t]
                     + self.gamma * next_val * (1 - dones[t])
                     - values[t])
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            next_val = values[t]

        return advantages

    def _clear_buffer(self):
        self._obs.clear()
        self._actions.clear()
        self._log_probs.clear()
        self._values.clear()
        self._rewards.clear()
        self._dones.clear()

    def save(self, path: str):
        torch.save({
            'net_state':       self.net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'train_steps':     self.train_steps,
        }, path)
        print(f"Saved model → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt['net_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.train_steps = ckpt.get('train_steps', 0)
        print(f"Loaded model ← {path} (step {self.train_steps})")
