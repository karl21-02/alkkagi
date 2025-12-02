"""
=================================================================
Al-Kka-Gi RL Agent - PPO (Complete Implementation)
=================================================================

PPO Optimizations (12):
1. GAE (Generalized Advantage Estimation)
2. Value Clipping (PPO2 style)
3. Policy Clipping
4. Entropy Bonus
5. Advantage Normalization
6. Gradient Clipping
7. Critic Extra Training (5x)
8. Learning Rate Scheduling
9. Reward Normalization
10. Observation Normalization
11. Orthogonal Initialization
12. Mixed Precision Training (AMP)

Self-Play Strategy:
- ELO-based Matching
- Rule-based Opponent (diversity)
- Curriculum Learning

=================================================================
"""

import gymnasium as gym
import kymnasium as kym
import numpy as np
import os
import glob
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.amp import autocast, GradScaler
from gymnasium.vector import AsyncVectorEnv
from typing import Any, Dict, Optional, Tuple, List

# ==========================================
# 1. Hyperparameters
# ==========================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

if torch.cuda.is_available():
    NUM_ENVS = 32
    BATCH_SIZE = 512
    T_HORIZON = 256
else:
    NUM_ENVS = 8
    BATCH_SIZE = 256
    T_HORIZON = 128

GAMMA = 0.99
GAE_LAMBDA = 0.95
EPS_CLIP = 0.2
K_EPOCHS = 3
CRITIC_EXTRA_EPOCHS = 4

LR_START = 3e-4
LR_END = 3e-5
MAX_UPDATES = 10000

ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5

BOARD_SIZE = 600.0
POWER_MIN = 300.0
POWER_MAX = 2500.0
ANGLE_RANGE = 45.0

SAVE_INTERVAL = 50
SWAP_INTERVAL = 20
MAX_POOL_SIZE = 20

SAVE_PATH = "my_alkkagi_agent.pkl"
HISTORY_DIR = "history_models"
ELO_FILE = "elo_ratings.json"


# ==========================================
# 2. Utilities
# ==========================================

class RunningMeanStd:
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)

    def state_dict(self):
        return {'mean': self.mean, 'var': self.var, 'count': self.count}

    def load_state_dict(self, state):
        self.mean = state['mean']
        self.var = state['var']
        self.count = state['count']


class ELOSystem:
    def __init__(self, k: float = 32, initial: float = 1500):
        self.k = k
        self.initial = initial
        self.ratings: Dict[str, float] = {}

    def get_rating(self, player: str) -> float:
        return self.ratings.get(player, self.initial)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update(self, winner: str, loser: str):
        ra = self.get_rating(winner)
        rb = self.get_rating(loser)
        ea = self.expected_score(ra, rb)
        eb = self.expected_score(rb, ra)
        self.ratings[winner] = ra + self.k * (1 - ea)
        self.ratings[loser] = rb + self.k * (0 - eb)

    def get_similar_opponent(self, my_rating: float, tolerance: float = 200) -> Optional[str]:
        candidates = [p for p, r in self.ratings.items() if abs(r - my_rating) < tolerance]
        return random.choice(candidates) if candidates else None

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.ratings, f)

    def load(self, path: str):
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.ratings = json.load(f)


# ==========================================
# 3. Neural Network
# ==========================================

def orthogonal_init(layer, gain=np.sqrt(2)):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)


class HybridActorCritic(nn.Module):
    def __init__(self, state_dim: int = 24):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.Tanh()
        )

        self.stone_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 3)
        )

        self.action_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )

        self.log_std = nn.Parameter(torch.zeros(2))

        self.critic_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for module in [self.shared, self.critic_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    orthogonal_init(layer, gain=np.sqrt(2))

        for layer in self.stone_head:
            if isinstance(layer, nn.Linear):
                orthogonal_init(layer, gain=0.01)

        for layer in self.action_head:
            if isinstance(layer, nn.Linear):
                orthogonal_init(layer, gain=0.01)

    def forward(self, x, stone_mask=None):
        features = self.shared(x)
        stone_logits = self.stone_head(features)
        if stone_mask is not None:
            stone_logits = stone_logits.masked_fill(stone_mask == 0, -1e9)
        action_mu = self.action_head(features)
        action_std = self.log_std.exp().expand_as(action_mu)
        value = self.critic_head(features)
        return stone_logits, action_mu, action_std, value

    def get_value(self, x):
        features = self.shared(x)
        return self.critic_head(features)


# ==========================================
# 4. Rule-based Agent
# ==========================================

class RuleBasedAgent:
    def __init__(self, my_turn: int):
        self.my_turn = my_turn

    def act(self, obs: Dict) -> Dict:
        my_stones = obs['black'] if self.my_turn == 0 else obs['white']
        enemy_stones = obs['white'] if self.my_turn == 0 else obs['black']

        my_alive = [(i, s) for i, s in enumerate(my_stones) if s[2] == 1]
        enemy_alive = [(i, s) for i, s in enumerate(enemy_stones) if s[2] == 1]

        if not my_alive or not enemy_alive:
            return {"turn": self.my_turn, "index": 0, "power": 500.0, "angle": 0.0}

        best_dist = float('inf')
        best_my_idx = 0
        best_angle = 0.0
        best_power = 500.0

        for my_idx, my_stone in my_alive:
            my_pos = np.array(my_stone[:2])
            for _, enemy_stone in enemy_alive:
                enemy_pos = np.array(enemy_stone[:2])
                dist = np.linalg.norm(enemy_pos - my_pos)
                if dist < best_dist:
                    best_dist = dist
                    best_my_idx = my_idx
                    dx = enemy_pos[0] - my_pos[0]
                    dy = enemy_pos[1] - my_pos[1]
                    best_angle = np.degrees(np.arctan2(dy, dx))
                    best_power = np.clip(500 + dist * 2 + random.uniform(-100, 100), POWER_MIN, POWER_MAX)

        return {"turn": self.my_turn, "index": best_my_idx, "power": float(best_power), "angle": float(best_angle)}


# ==========================================
# 5. Base Agent
# ==========================================

class BaseAgent(kym.Agent):
    def __init__(self, my_turn: int):
        super().__init__()
        self.my_turn = my_turn
        self.state_dim = 24
        self.model = HybridActorCritic(self.state_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR_START)
        self.obs_rms = RunningMeanStd(shape=(self.state_dim,))
        self.reward_rms = RunningMeanStd(shape=())
        self.scaler = GradScaler() if USE_AMP else None
        self.data = []

    def _process_obs(self, obs: Dict, override_turn: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        batch_size = len(obs['black'])
        if override_turn is not None:
            turns = np.full(batch_size, override_turn)
        else:
            turns = obs['turn'].flatten()

        processed = np.zeros((batch_size, self.state_dim), dtype=np.float32)
        stone_mask = np.zeros((batch_size, 3), dtype=np.float32)

        for b in range(batch_size):
            turn = turns[b]
            if turn == 0:
                my_stones = obs['black'][b]
                enemy_stones = obs['white'][b]
            else:
                my_stones = obs['white'][b]
                enemy_stones = obs['black'][b]

            for i in range(3):
                processed[b, i*3 + 0] = my_stones[i, 0] / BOARD_SIZE
                processed[b, i*3 + 1] = my_stones[i, 1] / BOARD_SIZE
                processed[b, i*3 + 2] = my_stones[i, 2]
                stone_mask[b, i] = my_stones[i, 2]

            for i in range(3):
                processed[b, 9 + i*3 + 0] = enemy_stones[i, 0] / BOARD_SIZE
                processed[b, 9 + i*3 + 1] = enemy_stones[i, 1] / BOARD_SIZE
                processed[b, 9 + i*3 + 2] = enemy_stones[i, 2]

            my_alive = np.sum(my_stones[:, 2])
            enemy_alive = np.sum(enemy_stones[:, 2])
            processed[b, 18] = my_alive / 3.0
            processed[b, 19] = enemy_alive / 3.0

            obstacles = obs.get('obstacles', None)
            if obstacles is not None and len(obstacles) > b and len(obstacles[b]) > 0:
                wall = obstacles[b][0]
                processed[b, 20] = 1.0
                processed[b, 21] = wall[0] / BOARD_SIZE
                processed[b, 22] = wall[1] / BOARD_SIZE
                processed[b, 23] = max(wall[2], wall[3]) / BOARD_SIZE
            else:
                processed[b, 20:24] = 0.0

        return processed, stone_mask

    def _normalize_obs(self, obs: np.ndarray, update: bool = True) -> np.ndarray:
        if update:
            self.obs_rms.update(obs)
        return self.obs_rms.normalize(obs)

    def get_action(self, obs_np: np.ndarray, stone_mask: np.ndarray,
                   deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obs_tensor = torch.tensor(obs_np, dtype=torch.float32).to(DEVICE)
        mask_tensor = torch.tensor(stone_mask, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            stone_logits, action_mu, action_std, _ = self.model(obs_tensor, mask_tensor)

        stone_dist = Categorical(logits=stone_logits)
        if deterministic:
            stone_idx = torch.argmax(stone_logits, dim=1)
        else:
            stone_idx = stone_dist.sample()
        log_prob_stone = stone_dist.log_prob(stone_idx)

        action_dist = Normal(action_mu, action_std)
        if deterministic:
            action = action_mu
        else:
            action = action_dist.rsample()
        log_prob_action = action_dist.log_prob(action).sum(dim=1)

        return (stone_idx.cpu().numpy(), action.cpu().numpy(),
                log_prob_stone.cpu().numpy(), log_prob_action.cpu().numpy())

    def decode_action(self, stone_idx: np.ndarray, action: np.ndarray,
                      obs: Dict, my_turn: int) -> List[Dict]:
        actions = []
        batch_size = len(stone_idx)
        my_stones = obs['black'] if my_turn == 0 else obs['white']
        enemy_stones = obs['white'] if my_turn == 0 else obs['black']

        for b in range(batch_size):
            idx = int(stone_idx[b])
            my_pos = my_stones[b, idx, :2]

            enemy_alive = enemy_stones[b, :, 2] == 1
            if np.any(enemy_alive):
                enemy_positions = enemy_stones[b, enemy_alive, :2]
                dists = np.linalg.norm(enemy_positions - my_pos, axis=1)
                nearest_enemy = enemy_positions[np.argmin(dists)]
            else:
                nearest_enemy = np.array([BOARD_SIZE/2, BOARD_SIZE/2])

            dx = nearest_enemy[0] - my_pos[0]
            dy = nearest_enemy[1] - my_pos[1]
            base_angle = np.degrees(np.arctan2(dy, dx))

            angle_offset = np.tanh(action[b, 0]) * ANGLE_RANGE
            final_angle = base_angle + angle_offset

            power_ratio = (np.tanh(action[b, 1]) + 1) / 2
            power = POWER_MIN + power_ratio * (POWER_MAX - POWER_MIN)

            actions.append({"turn": my_turn, "index": idx, "power": float(power), "angle": float(final_angle)})

        return actions

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        states, stone_indices, actions, rewards = [], [], [], []
        next_states, log_probs_stone, log_probs_action = [], [], []
        dones, stone_masks, old_values = [], [], []

        for item in self.data:
            s, si, a, r, ns, lps, lpa, d, sm, v = item
            states.append(s)
            stone_indices.append(si)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            log_probs_stone.append(lps)
            log_probs_action.append(lpa)
            dones.append(d)
            stone_masks.append(sm)
            old_values.append(v)

        self.data = []

        return (
            torch.tensor(np.array(states), dtype=torch.float32).to(DEVICE),
            torch.tensor(np.array(stone_indices), dtype=torch.long).to(DEVICE),
            torch.tensor(np.array(actions), dtype=torch.float32).to(DEVICE),
            torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1).to(DEVICE),
            torch.tensor(np.array(next_states), dtype=torch.float32).to(DEVICE),
            torch.tensor(np.array(log_probs_stone), dtype=torch.float32).to(DEVICE),
            torch.tensor(np.array(log_probs_action), dtype=torch.float32).to(DEVICE),
            torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1).to(DEVICE),
            torch.tensor(np.array(stone_masks), dtype=torch.float32).to(DEVICE),
            torch.tensor(np.array(old_values), dtype=torch.float32).unsqueeze(1).to(DEVICE)
        )

    def compute_gae(self, rewards, values, next_values, dones):
        batch_size = rewards.size(0)
        advantages = torch.zeros(batch_size, 1, device=DEVICE)
        last_gae = 0

        for t in reversed(range(batch_size)):
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + GAMMA * next_values[t] * non_terminal - values[t]
            last_gae = delta + GAMMA * GAE_LAMBDA * non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def train_net(self):
        if len(self.data) < BATCH_SIZE:
            return

        (states, stone_indices, actions, rewards, next_states,
         old_log_probs_stone, old_log_probs_action, dones,
         stone_masks, old_values) = self.make_batch()

        rewards_np = rewards.cpu().numpy().flatten()
        self.reward_rms.update(rewards_np)
        reward_std = np.sqrt(self.reward_rms.var + 1e-8)
        rewards = rewards / reward_std

        with torch.no_grad():
            _, _, _, values = self.model(states, stone_masks)
            _, _, _, next_values = self.model(next_states, stone_masks)

        advantages, returns = self.compute_gae(rewards, values, next_values, dones)

        total_samples = states.size(0)
        indices = np.arange(total_samples)

        for _ in range(K_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, total_samples, BATCH_SIZE):
                idx = indices[start:start + BATCH_SIZE]
                if USE_AMP:
                    with autocast(device_type='cuda', dtype=torch.bfloat16):
                        loss = self._compute_loss(states[idx], stone_indices[idx], actions[idx],
                                                  old_log_probs_stone[idx], old_log_probs_action[idx],
                                                  advantages[idx], returns[idx], old_values[idx], stone_masks[idx])
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss = self._compute_loss(states[idx], stone_indices[idx], actions[idx],
                                              old_log_probs_stone[idx], old_log_probs_action[idx],
                                              advantages[idx], returns[idx], old_values[idx], stone_masks[idx])
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                    self.optimizer.step()

        for _ in range(CRITIC_EXTRA_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, total_samples, BATCH_SIZE):
                idx = indices[start:start + BATCH_SIZE]
                if USE_AMP:
                    with autocast(device_type='cuda', dtype=torch.bfloat16):
                        value_loss = self._compute_value_loss(states[idx], returns[idx], old_values[idx])
                    self.optimizer.zero_grad()
                    self.scaler.scale(value_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    value_loss = self._compute_value_loss(states[idx], returns[idx], old_values[idx])
                    self.optimizer.zero_grad()
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                    self.optimizer.step()

    def _compute_loss(self, states, stone_indices, actions, old_log_probs_stone, old_log_probs_action,
                      advantages, returns, old_values, stone_masks):
        stone_logits, action_mu, action_std, values = self.model(states, stone_masks)

        stone_dist = Categorical(logits=stone_logits)
        new_log_probs_stone = stone_dist.log_prob(stone_indices)

        action_dist = Normal(action_mu, action_std)
        new_log_probs_action = action_dist.log_prob(actions).sum(dim=1)

        old_log_probs = old_log_probs_stone + old_log_probs_action
        new_log_probs = new_log_probs_stone + new_log_probs_action
        ratio = torch.exp(new_log_probs - old_log_probs)

        surr1 = ratio * advantages.squeeze()
        surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages.squeeze()
        policy_loss = -torch.min(surr1, surr2).mean()

        values_clipped = old_values + torch.clamp(values - old_values, -EPS_CLIP, EPS_CLIP)
        value_loss_1 = (values - returns) ** 2
        value_loss_2 = (values_clipped - returns) ** 2
        value_loss = 0.5 * torch.max(value_loss_1, value_loss_2).mean()

        entropy = stone_dist.entropy().mean() + action_dist.entropy().sum(dim=1).mean()

        loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
        return loss

    def _compute_value_loss(self, states, returns, old_values):
        _, _, _, values = self.model(states, None)
        values_clipped = old_values + torch.clamp(values - old_values, -EPS_CLIP, EPS_CLIP)
        value_loss_1 = (values - returns) ** 2
        value_loss_2 = (values_clipped - returns) ** 2
        value_loss = 0.5 * torch.max(value_loss_1, value_loss_2).mean()
        return value_loss

    def act(self, observation: Any, info: Dict) -> Dict:
        batch_obs = {
            'black': np.array([observation['black']]),
            'white': np.array([observation['white']]),
            'turn': np.array([observation['turn']]),
            'obstacles': np.array([observation.get('obstacles', [])])
        }
        obs_np, stone_mask = self._process_obs(batch_obs, override_turn=self.my_turn)
        obs_np = self._normalize_obs(obs_np, update=False)
        stone_idx, action, _, _ = self.get_action(obs_np, stone_mask, deterministic=True)
        return self.decode_action(stone_idx, action, batch_obs, self.my_turn)[0]


# ==========================================
# 6. Agent Wrappers
# ==========================================

class YourBlackAgent(BaseAgent):
    def __init__(self):
        super().__init__(my_turn=0)

    @classmethod
    def load(cls, path: str) -> 'YourBlackAgent':
        agent = cls()
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=DEVICE)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                agent.model.load_state_dict(checkpoint['model_state_dict'])
                if 'obs_rms' in checkpoint:
                    agent.obs_rms.load_state_dict(checkpoint['obs_rms'])
                if 'reward_rms' in checkpoint:
                    agent.reward_rms.load_state_dict(checkpoint['reward_rms'])
            else:
                agent.model.load_state_dict(checkpoint)
        return agent

    def save(self, path: str, update_step: int = 0):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'obs_rms': self.obs_rms.state_dict(),
            'reward_rms': self.reward_rms.state_dict(),
            'update_step': update_step
        }
        torch.save(checkpoint, path)


class YourWhiteAgent(BaseAgent):
    def __init__(self):
        super().__init__(my_turn=1)

    @classmethod
    def load(cls, path: str) -> 'YourWhiteAgent':
        agent = cls()
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=DEVICE)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                agent.model.load_state_dict(checkpoint['model_state_dict'])
                if 'obs_rms' in checkpoint:
                    agent.obs_rms.load_state_dict(checkpoint['obs_rms'])
                if 'reward_rms' in checkpoint:
                    agent.reward_rms.load_state_dict(checkpoint['reward_rms'])
            else:
                agent.model.load_state_dict(checkpoint)
        return agent

    def save(self, path: str, update_step: int = 0):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'obs_rms': self.obs_rms.state_dict(),
            'reward_rms': self.reward_rms.state_dict(),
            'update_step': update_step
        }
        torch.save(checkpoint, path)


# ==========================================
# 7. Opponent Manager
# ==========================================

class OpponentManager:
    def __init__(self):
        self.save_dir = HISTORY_DIR
        os.makedirs(self.save_dir, exist_ok=True)
        self.elo = ELOSystem()
        self.elo.load(os.path.join(self.save_dir, ELO_FILE))
        self.pool = glob.glob(os.path.join(self.save_dir, "model_*.pkl"))
        self.rule_based = RuleBasedAgent(my_turn=1)
        self.current_elo = 1500.0
        self.current_opponent_path = None

    def save_model(self, model, step: int):
        path = os.path.join(self.save_dir, f"model_{step}.pkl")
        torch.save(model.state_dict(), path)
        if path not in self.pool:
            self.pool.append(path)
            self.elo.ratings[path] = self.current_elo
        while len(self.pool) > MAX_POOL_SIZE:
            old_path = self.pool.pop(0)
            if old_path in self.elo.ratings:
                del self.elo.ratings[old_path]
            if os.path.exists(old_path):
                os.remove(old_path)
        self.elo.save(os.path.join(self.save_dir, ELO_FILE))

    def get_opponent(self, progress: float) -> Tuple[Any, str]:
        if progress < 0.3:
            rule_prob = 0.8
        elif progress < 0.7:
            rule_prob = 0.5
        else:
            rule_prob = 0.2

        if random.random() < rule_prob or len(self.pool) == 0:
            self.current_opponent_path = None
            return self.rule_based, "RuleBot"

        opponent_path = self.elo.get_similar_opponent(self.current_elo, tolerance=200)
        if opponent_path is None:
            opponent_path = random.choice(self.pool)

        self.current_opponent_path = opponent_path

        opponent = YourBlackAgent()
        opponent.my_turn = 1
        try:
            opponent.model.load_state_dict(torch.load(opponent_path, map_location=DEVICE))
            opponent.model.eval()
            rating = self.elo.get_rating(opponent_path)
            return opponent, f"ELO:{rating:.0f}"
        except:
            return self.rule_based, "RuleBot(fallback)"

    def update_elo(self, won: bool):
        if self.current_opponent_path is None:
            return
        opponent_elo = self.elo.get_rating(self.current_opponent_path)
        expected = self.elo.expected_score(self.current_elo, opponent_elo)
        if won:
            self.current_elo = self.current_elo + self.elo.k * (1 - expected)
            self.elo.ratings[self.current_opponent_path] = opponent_elo - self.elo.k * expected
        else:
            self.current_elo = self.current_elo + self.elo.k * (0 - expected)
            self.elo.ratings[self.current_opponent_path] = opponent_elo + self.elo.k * expected


# ==========================================
# 8. Training Loop
# ==========================================

def make_env():
    return gym.make(id='kymnasium/AlKkaGi-3x3-v0', render_mode=None, bgm=False, obs_type='custom')


def compute_reward(obs, next_obs, my_idx, term, trunc):
    batch_size = len(my_idx)
    rewards = np.zeros(batch_size, dtype=np.float32)

    for i, env_i in enumerate(my_idx):
        my_before = obs['black'][env_i]
        my_after = next_obs['black'][env_i]
        enemy_before = obs['white'][env_i]
        enemy_after = next_obs['white'][env_i]

        my_alive_before = np.sum(my_before[:, 2])
        my_alive_after = np.sum(my_after[:, 2])
        enemy_alive_before = np.sum(enemy_before[:, 2])
        enemy_alive_after = np.sum(enemy_after[:, 2])

        kills = enemy_alive_before - enemy_alive_after
        deaths = my_alive_before - my_alive_after
        rewards[i] += kills * 50.0
        rewards[i] -= deaths * 30.0

        for j in range(3):
            if enemy_before[j, 2] == 1 and enemy_after[j, 2] == 1:
                prev_edge = min(enemy_before[j, 0], enemy_before[j, 1],
                                BOARD_SIZE - enemy_before[j, 0], BOARD_SIZE - enemy_before[j, 1])
                curr_edge = min(enemy_after[j, 0], enemy_after[j, 1],
                                BOARD_SIZE - enemy_after[j, 0], BOARD_SIZE - enemy_after[j, 1])
                push_reward = (prev_edge - curr_edge) * 0.05
                rewards[i] += max(0, push_reward)

        done = term[env_i] or trunc[env_i]
        if done:
            if enemy_alive_after == 0:
                rewards[i] += 100.0
            elif my_alive_after == 0:
                rewards[i] -= 50.0

        rewards[i] -= 0.1

    return rewards


def train():
    print("=" * 60)
    print("PPO Training for Al-Kka-Gi")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"NUM_ENVS: {NUM_ENVS}, BATCH_SIZE: {BATCH_SIZE}")
    print(f"AMP: {USE_AMP}")
    print("=" * 60)

    envs = AsyncVectorEnv([make_env for _ in range(NUM_ENVS)])

    agent = YourBlackAgent()
    start_update = 1

    if os.path.exists(SAVE_PATH):
        print(f"Loading checkpoint: {SAVE_PATH}")
        checkpoint = torch.load(SAVE_PATH, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            agent.model.load_state_dict(checkpoint['model_state_dict'])
            if 'obs_rms' in checkpoint:
                agent.obs_rms.load_state_dict(checkpoint['obs_rms'])
            if 'reward_rms' in checkpoint:
                agent.reward_rms.load_state_dict(checkpoint['reward_rms'])
            start_update = checkpoint.get('update_step', 0) + 1
        else:
            agent.model.load_state_dict(checkpoint)
        print(f"Resuming from update {start_update}")

    op_manager = OpponentManager()

    if start_update == 1:
        op_manager.save_model(agent.model, 0)

    obs, _ = envs.reset()

    score_history = []
    win_count = 0
    game_count = 0

    opponent, opponent_name = op_manager.get_opponent(0.0)

    for update in range(start_update, MAX_UPDATES + 1):
        progress = update / MAX_UPDATES

        lr = LR_START - (LR_START - LR_END) * progress
        for param_group in agent.optimizer.param_groups:
            param_group['lr'] = lr

        if update % SAVE_INTERVAL == 0:
            op_manager.save_model(agent.model, update)

        if update % SWAP_INTERVAL == 0:
            opponent, opponent_name = op_manager.get_opponent(progress)

        for _ in range(T_HORIZON):
            turns = obs['turn']

            my_idx = np.where(turns == 0)[0]
            my_actions = []

            if len(my_idx) > 0:
                obs_me = {k: v[my_idx] for k, v in obs.items()}
                obs_np, stone_mask = agent._process_obs(obs_me, override_turn=0)
                obs_np = agent._normalize_obs(obs_np, update=True)

                stone_idx, action, log_p_stone, log_p_action = agent.get_action(obs_np, stone_mask, deterministic=False)
                my_actions = agent.decode_action(stone_idx, action, obs_me, 0)

                with torch.no_grad():
                    obs_tensor = torch.tensor(obs_np, dtype=torch.float32).to(DEVICE)
                    _, _, _, values = agent.model(obs_tensor, None)
                values_np = values.cpu().numpy().flatten()

            op_idx = np.where(turns == 1)[0]
            op_actions = []

            if len(op_idx) > 0:
                obs_op = {k: v[op_idx] for k, v in obs.items()}
                if isinstance(opponent, RuleBasedAgent):
                    for i in range(len(op_idx)):
                        single_obs = {k: v[i] for k, v in obs_op.items()}
                        op_actions.append(opponent.act(single_obs))
                else:
                    obs_np_op, mask_op = opponent._process_obs(obs_op, override_turn=1)
                    obs_np_op = opponent._normalize_obs(obs_np_op, update=False)
                    si, ac, _, _ = opponent.get_action(obs_np_op, mask_op, deterministic=True)
                    op_actions = opponent.decode_action(si, ac, obs_op, 1)

            action_list = [None] * NUM_ENVS
            for i, env_i in enumerate(my_idx):
                action_list[env_i] = my_actions[i]
            for i, env_i in enumerate(op_idx):
                action_list[env_i] = op_actions[i]

            batched_action = {key: np.array([d[key] for d in action_list]) for key in action_list[0].keys()}
            next_obs, _, term, trunc, _ = envs.step(batched_action)

            if len(my_idx) > 0:
                rewards = compute_reward(obs, next_obs, my_idx, term, trunc)

                next_obs_me = {k: v[my_idx] for k, v in next_obs.items()}
                next_obs_np, _ = agent._process_obs(next_obs_me, override_turn=0)
                next_obs_np = agent._normalize_obs(next_obs_np, update=False)

                for i, env_i in enumerate(my_idx):
                    done = term[env_i] or trunc[env_i]
                    agent.put_data((obs_np[i], stone_idx[i], action[i], rewards[i], next_obs_np[i],
                                    log_p_stone[i], log_p_action[i], float(done), stone_mask[i], values_np[i]))
                    score_history.append(rewards[i])

                    if done:
                        game_count += 1
                        enemy_alive = np.sum(next_obs['white'][env_i, :, 2])
                        if enemy_alive == 0:
                            win_count += 1
                            op_manager.update_elo(won=True)
                        else:
                            op_manager.update_elo(won=False)

            obs = next_obs

        agent.train_net()

        if update % 10 == 0:
            avg_score = np.mean(score_history[-1000:]) if score_history else 0
            win_rate = (win_count / game_count * 100) if game_count > 0 else 0

            print(f"Update {update:5d} | Score: {avg_score:7.2f} | "
                  f"WinRate: {win_rate:5.1f}% ({win_count}/{game_count}) | "
                  f"ELO: {op_manager.current_elo:.0f} | VS: {opponent_name} | LR: {lr:.2e}")

            agent.save(SAVE_PATH, update)
            win_count = 0
            game_count = 0

    envs.close()
    print("Training completed!")


if __name__ == "__main__":
    train()
