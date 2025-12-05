"""
=================================================================
알까기 강화학습 에이전트 - PPO v2 (통합 개선 버전)
=================================================================

[ 개선 사항 - ppo.py + dg_2025_12_04.py 통합 ]

1. 관측 공간 확장 (24차원)
   - 내 돌 위치/생존 (9) + 상대 돌 위치/생존 (9)
   - 생존 수 (2) + 벽 정보 (4)

2. 탐험 개선
   - 엔트로피 보너스 (스케줄링)
   - 시작 돌 랜덤 선택 (학습 초반)
   - Temperature 스케줄링

3. 학습 안정화
   - GAE (Generalized Advantage Estimation)
   - Gradient Clipping
   - Learning Rate Scheduling
   - Observation Normalization

4. 보상 함수 개선
   - 스케일 재조정
   - 중간 보상 추가

5. Self-Play 개선
   - 모델 풀 확대
   - 다양한 상대 전략

=================================================================
"""

import os
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import gymnasium as gym
import kymnasium as kym
import numpy as np
import glob
import random
import signal
import sys
import time
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from gymnasium.vector import AsyncVectorEnv
from typing import Any, Dict, Optional, Tuple, List


# ==============================================================================
# 1. 하이퍼파라미터
# ==============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 환경 설정
NUM_ENVS = 8
BATCH_SIZE = 256
T_HORIZON = 128

# PPO 핵심 파라미터
GAMMA = 0.99
GAE_LAMBDA = 0.95
EPS_CLIP = 0.2
K_EPOCHS = 4
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5

# 학습률 스케줄링
LR_START = 3e-4
LR_END = 1e-5
MAX_UPDATES = 10000

# 엔트로피 스케줄링 (탐험 → 활용) - 개선: 더 천천히 감소
ENTROPY_START = 0.05      # 초반: 높은 탐험
ENTROPY_END = 0.01        # 후반: 최소 탐험 유지 (0.005 → 0.01)
ENTROPY_DECAY_UPDATES = 8000  # 감소 기간 (5000 → 8000)

# 시작 돌 랜덤 선택 (탐험 다양성) - 개선: 더 오래 유지
RANDOM_STONE_PROB_START = 0.3  # 초반: 30% 확률로 랜덤 돌 선택
RANDOM_STONE_PROB_END = 0.05   # 후반: 5% 유지 (0.0 → 0.05)
RANDOM_STONE_DECAY = 7000      # 감소 기간 (3000 → 7000)

# 게임 환경
BOARD_SIZE = 600.0
POWER_MIN = 300.0
POWER_MAX = 2500.0
ANGLE_RANGE = 60.0  # ±60도 (기존 45도에서 확장)

# Self-Play 설정 - 개선: 다양성 확보
SAVE_INTERVAL = 50
SWAP_INTERVAL = 20
MODEL_POOL_SIZE = 20   # 10 → 20 (더 많은 과거 버전 보관)
RULEBOT_PHASE = 1000   # 300 → 1000 (기본기 더 오래 학습)

SAVE_PATH = "my_alkkagi_agent_v2.pkl"
HISTORY_DIR = "history_models_v2"

# 종료 플래그
SHUTDOWN_FLAG = False


# ==============================================================================
# 2. 유틸리티 클래스
# ==============================================================================

class RunningMeanStd:
    """온라인 평균/분산 계산기 (Welford 알고리즘)"""
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


# ==============================================================================
# 3. 신경망 정의
# ==============================================================================

def orthogonal_init(layer, gain=np.sqrt(2)):
    """직교 초기화"""
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)


class HybridActorCritic(nn.Module):
    """
    하이브리드 Actor-Critic 네트워크

    입력: 24차원 상태 벡터
    출력: 돌 선택 (이산) + 각도/파워 (연속) + 가치
    """
    def __init__(self, state_dim: int = 24):
        super().__init__()

        # 공유 특징 추출
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.Tanh()
        )

        # 돌 선택 헤드 (Categorical)
        self.stone_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 3)
        )

        # 연속 행동 헤드 (Gaussian)
        self.action_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )
        self.log_std = nn.Parameter(torch.zeros(2))

        # Critic 헤드
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


# ==============================================================================
# 4. 규칙 기반 에이전트
# ==============================================================================

class RuleBasedAgent:
    """규칙 기반 상대 (학습 초기용)"""
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
                    best_power = np.clip(500 + dist * 2 + random.uniform(-100, 100),
                                        POWER_MIN, POWER_MAX)

        return {"turn": self.my_turn, "index": best_my_idx,
                "power": float(best_power), "angle": float(best_angle)}


# ==============================================================================
# 5. PPO 에이전트
# ==============================================================================

class BaseAgent(kym.Agent):
    """PPO 에이전트 기본 클래스"""

    def __init__(self, my_turn: int):
        super().__init__()
        self.my_turn = my_turn
        self.state_dim = 24

        self.model = HybridActorCritic(self.state_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR_START)

        self.obs_rms = RunningMeanStd(shape=(self.state_dim,))
        self.reward_rms = RunningMeanStd(shape=())

        self.data = []
        self._obs_buffer = None
        self._mask_buffer = None

        # 스케줄링 변수
        self.current_update = 0
        self.entropy_coef = ENTROPY_START
        self.random_stone_prob = RANDOM_STONE_PROB_START

    def update_schedules(self, update: int):
        """스케줄러 업데이트"""
        self.current_update = update
        progress = min(update / MAX_UPDATES, 1.0)

        # 학습률 스케줄링
        lr = LR_START - (LR_START - LR_END) * progress
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        # 엔트로피 스케줄링
        entropy_progress = min(update / ENTROPY_DECAY_UPDATES, 1.0)
        self.entropy_coef = ENTROPY_START - (ENTROPY_START - ENTROPY_END) * entropy_progress

        # 랜덤 돌 선택 확률 스케줄링
        random_progress = min(update / RANDOM_STONE_DECAY, 1.0)
        self.random_stone_prob = RANDOM_STONE_PROB_START - (RANDOM_STONE_PROB_START - RANDOM_STONE_PROB_END) * random_progress

    def _process_obs(self, obs: Dict, override_turn: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """관측값 전처리 (24차원)"""
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

            # 내 돌 정보 (0-8)
            for i in range(3):
                processed[b, i*3 + 0] = my_stones[i, 0] / BOARD_SIZE
                processed[b, i*3 + 1] = my_stones[i, 1] / BOARD_SIZE
                processed[b, i*3 + 2] = my_stones[i, 2]
                stone_mask[b, i] = my_stones[i, 2]

            # 상대 돌 정보 (9-17)
            for i in range(3):
                processed[b, 9 + i*3 + 0] = enemy_stones[i, 0] / BOARD_SIZE
                processed[b, 9 + i*3 + 1] = enemy_stones[i, 1] / BOARD_SIZE
                processed[b, 9 + i*3 + 2] = enemy_stones[i, 2]

            # 생존 돌 수 (18-19)
            my_alive = np.sum(my_stones[:, 2])
            enemy_alive = np.sum(enemy_stones[:, 2])
            processed[b, 18] = my_alive / 3.0
            processed[b, 19] = enemy_alive / 3.0

            # 벽 정보 (20-23)
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
        """관측값 정규화"""
        if update:
            self.obs_rms.update(obs)
        return self.obs_rms.normalize(obs)

    def get_action(self, obs_np: np.ndarray, stone_mask: np.ndarray,
                   deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """행동 선택"""
        batch_size = obs_np.shape[0]

        if self._obs_buffer is None or self._obs_buffer.shape[0] != batch_size:
            self._obs_buffer = torch.empty((batch_size, self.state_dim),
                                           dtype=torch.float32, device=DEVICE)
            self._mask_buffer = torch.empty((batch_size, 3),
                                            dtype=torch.float32, device=DEVICE)

        self._obs_buffer.copy_(torch.from_numpy(obs_np))
        self._mask_buffer.copy_(torch.from_numpy(stone_mask))

        with torch.inference_mode():
            stone_logits, action_mu, action_std, _ = self.model(self._obs_buffer, self._mask_buffer)

        # 돌 선택 (랜덤 선택 확률 적용)
        stone_dist = Categorical(logits=stone_logits)
        if deterministic:
            stone_idx = torch.argmax(stone_logits, dim=1)
        else:
            stone_idx = stone_dist.sample()

            # 랜덤 돌 선택 (탐험 다양성)
            if self.random_stone_prob > 0:
                for i in range(batch_size):
                    if random.random() < self.random_stone_prob:
                        alive_indices = torch.where(self._mask_buffer[i] == 1)[0]
                        if len(alive_indices) > 0:
                            stone_idx[i] = alive_indices[random.randint(0, len(alive_indices)-1)]

        log_prob_stone = stone_dist.log_prob(stone_idx)

        # 연속 행동
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
        """행동 디코딩"""
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

            actions.append({
                "turn": my_turn,
                "index": idx,
                "power": float(power),
                "angle": float(final_angle)
            })

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
        """GAE 계산"""
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
        """PPO 학습"""
        if len(self.data) < BATCH_SIZE:
            return

        (states, stone_indices, actions, rewards, next_states,
         old_log_probs_stone, old_log_probs_action, dones,
         stone_masks, old_values) = self.make_batch()

        # 보상 정규화
        rewards_np = rewards.cpu().numpy().flatten()
        self.reward_rms.update(rewards_np)
        reward_std = np.sqrt(self.reward_rms.var + 1e-8)
        rewards = rewards / reward_std

        # GAE 계산
        with torch.inference_mode():
            _, _, _, values = self.model(states, stone_masks)
            _, _, _, next_values = self.model(next_states, stone_masks)

        advantages, returns = self.compute_gae(rewards, values, next_values, dones)

        total_samples = states.size(0)
        indices = np.arange(total_samples)

        for _ in range(K_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, total_samples, BATCH_SIZE):
                idx = indices[start:start + BATCH_SIZE]

                loss = self._compute_loss(
                    states[idx], stone_indices[idx], actions[idx],
                    old_log_probs_stone[idx], old_log_probs_action[idx],
                    advantages[idx], returns[idx], old_values[idx], stone_masks[idx]
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

    def _compute_loss(self, states, stone_indices, actions, old_log_probs_stone, old_log_probs_action,
                      advantages, returns, old_values, stone_masks):
        """PPO 손실 계산 (엔트로피 보너스 포함)"""
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

        # 엔트로피 보너스 (스케줄링된 계수 사용)
        entropy = stone_dist.entropy().mean() + action_dist.entropy().sum(dim=1).mean()

        loss = policy_loss + VALUE_COEF * value_loss - self.entropy_coef * entropy
        return loss

    def act(self, observation: Any, info: Dict) -> Dict:
        """평가용 행동"""
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


# ==============================================================================
# 6. 에이전트 래퍼
# ==============================================================================

class YourBlackAgent(BaseAgent):
    def __init__(self):
        super().__init__(my_turn=0)

    @classmethod
    def load(cls, path: str) -> 'YourBlackAgent':
        agent = cls()
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
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
            checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
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


# ==============================================================================
# 7. 상대 관리자
# ==============================================================================

class OpponentManager:
    def __init__(self):
        self.save_dir = HISTORY_DIR
        os.makedirs(self.save_dir, exist_ok=True)
        self.pool: List[str] = []
        self._load_existing_models()
        self.rule_based = RuleBasedAgent(my_turn=1)
        self.current_opponent_name = "RuleBot"

    def _load_existing_models(self):
        existing = glob.glob(os.path.join(self.save_dir, "model_*.pkl"))
        existing.sort(key=lambda x: int(x.split("_")[-1].replace(".pkl", "")))
        self.pool = existing[-MODEL_POOL_SIZE:]

    def save_model(self, agent, step: int):
        """에이전트 저장 (모델 + obs_rms + reward_rms 포함)"""
        path = os.path.join(self.save_dir, f"model_{step}.pkl")
        torch.save({
            'model_state_dict': agent.model.state_dict(),
            'obs_rms': agent.obs_rms.state_dict(),
            'reward_rms': agent.reward_rms.state_dict()
        }, path)
        if path not in self.pool:
            self.pool.append(path)
        while len(self.pool) > MODEL_POOL_SIZE:
            old_path = self.pool.pop(0)
            if os.path.exists(old_path):
                os.remove(old_path)

    def get_opponent(self, update: int) -> Tuple[Any, str]:
        if update <= RULEBOT_PHASE or len(self.pool) == 0:
            self.current_opponent_name = "RuleBot"
            return self.rule_based, "RuleBot"

        opponent_path = random.choice(self.pool)
        opponent = YourBlackAgent()
        opponent.my_turn = 1
        try:
            checkpoint = torch.load(opponent_path, map_location=DEVICE, weights_only=False)
            # 새 형식 (model_state_dict + obs_rms + reward_rms) 또는 이전 형식 (state_dict만) 지원
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                opponent.model.load_state_dict(checkpoint['model_state_dict'])
                if 'obs_rms' in checkpoint:
                    opponent.obs_rms.load_state_dict(checkpoint['obs_rms'])
                if 'reward_rms' in checkpoint:
                    opponent.reward_rms.load_state_dict(checkpoint['reward_rms'])
            else:
                opponent.model.load_state_dict(checkpoint)
            opponent.model.eval()
            version = opponent_path.split("_")[-1].replace(".pkl", "")
            self.current_opponent_name = f"Self-v{version}"
            return opponent, self.current_opponent_name
        except Exception as e:
            print(f"[Warning] Failed to load {opponent_path}: {e}")
            self.current_opponent_name = "RuleBot"
            return self.rule_based, "RuleBot"


# ==============================================================================
# 8. 보상 함수 (개선)
# ==============================================================================

def make_env():
    return gym.make(id='kymnasium/AlKkaGi-3x3-v0', render_mode=None, bgm=False, obs_type='custom')


def compute_reward(obs, next_obs, my_idx, term, trunc):
    """
    개선된 보상 함수 v2

    - 킬: +40 (킬 보상 상향)
    - 데스: -15 (데스 페널티 완화)
    - 승리: +100
    - 패배: -50
    - 밀어내기: +0.03 * 거리
    - 스텝: -0.02 (스텝 페널티 완화)
    - 경계 접근 보너스: 적이 100 이내면 추가 보상
    """
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

        # 킬/데스 - 개선: 킬 보상 상향, 데스 페널티 완화
        kills = enemy_alive_before - enemy_alive_after
        deaths = my_alive_before - my_alive_after
        rewards[i] += kills * 40.0   # 30 → 40
        rewards[i] -= deaths * 15.0  # 20 → 15

        # 밀어내기 보상
        for j in range(3):
            if enemy_before[j, 2] == 1 and enemy_after[j, 2] == 1:
                prev_edge = min(enemy_before[j, 0], enemy_before[j, 1],
                                BOARD_SIZE - enemy_before[j, 0], BOARD_SIZE - enemy_before[j, 1])
                curr_edge = min(enemy_after[j, 0], enemy_after[j, 1],
                                BOARD_SIZE - enemy_after[j, 0], BOARD_SIZE - enemy_after[j, 1])
                push_reward = (prev_edge - curr_edge) * 0.03
                rewards[i] += max(0, push_reward)

                # 경계 접근 보너스 (적이 100 이내면 추가 보상)
                if curr_edge < 100:
                    rewards[i] += (100 - curr_edge) * 0.01

        # 게임 종료
        done = term[env_i] or trunc[env_i]
        if done:
            if enemy_alive_after == 0:
                rewards[i] += 100.0  # 승리
            elif my_alive_after == 0:
                rewards[i] -= 50.0   # 패배

        # 스텝 페널티 - 개선: 더 완화 (0.05 → 0.02)
        rewards[i] -= 0.02

    return rewards


# ==============================================================================
# 9. 메인 학습 루프
# ==============================================================================

def train():
    global SHUTDOWN_FLAG

    def _signal_handler(signum, frame):
        global SHUTDOWN_FLAG
        if SHUTDOWN_FLAG:
            print("\n\n[!!] Force exit!")
            sys.exit(1)
        print("\n\n[!] Ctrl+C detected. Saving and exiting...")
        SHUTDOWN_FLAG = True

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    print("\n" + "=" * 70)
    print("  AL-KKA-GI PPO v2 TRAINING")
    print("=" * 70)
    print(f"  Device: {DEVICE} | Envs: {NUM_ENVS} | Updates: {MAX_UPDATES}")
    print(f"  Entropy: {ENTROPY_START} -> {ENTROPY_END}")
    print(f"  Random Stone: {RANDOM_STONE_PROB_START} -> {RANDOM_STONE_PROB_END}")
    print(f"  Angle Range: ±{ANGLE_RANGE}°")
    print("=" * 70 + "\n")

    envs = AsyncVectorEnv([make_env for _ in range(NUM_ENVS)])
    agent = YourBlackAgent()
    start_update = 1

    if os.path.exists(SAVE_PATH):
        print(f"[*] Loading checkpoint: {SAVE_PATH}")
        checkpoint = torch.load(SAVE_PATH, map_location=DEVICE, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            agent.model.load_state_dict(checkpoint['model_state_dict'])
            if 'obs_rms' in checkpoint:
                agent.obs_rms.load_state_dict(checkpoint['obs_rms'])
            if 'reward_rms' in checkpoint:
                agent.reward_rms.load_state_dict(checkpoint['reward_rms'])
            start_update = checkpoint.get('update_step', 0) + 1
        else:
            agent.model.load_state_dict(checkpoint)
        print(f"[*] Resuming from update {start_update}\n")

    op_manager = OpponentManager()
    if start_update == 1:
        op_manager.save_model(agent, 0)  # agent 전체 전달 (obs_rms 포함)

    obs, _ = envs.reset()

    score_history = []
    win_history = []
    win_count = 0
    game_count = 0
    total_kills = 0
    total_deaths = 0
    best_win_rate = 0.0
    start_time = time.time()

    opponent, opponent_name = op_manager.get_opponent(start_update)

    print(f"{'Upd':>6} | {'Score':>8} | {'Win%':>6} | {'Ent':>6} | {'RndStone':>8} | {'Opp':>12}")
    print("-" * 70)

    for update in range(start_update, MAX_UPDATES + 1):
        if SHUTDOWN_FLAG:
            break

        # 스케줄 업데이트
        agent.update_schedules(update)

        if update % SAVE_INTERVAL == 0:
            op_manager.save_model(agent, update)

        if update % SWAP_INTERVAL == 0:
            opponent, opponent_name = op_manager.get_opponent(update)

        for _ in range(T_HORIZON):
            turns = obs['turn']
            my_idx = np.where(turns == 0)[0]
            my_actions = []

            if len(my_idx) > 0:
                obs_me = {k: v[my_idx] for k, v in obs.items()}
                obs_np, stone_mask = agent._process_obs(obs_me, override_turn=0)
                # obs_rms 업데이트는 별도로, 정규화 시에는 update=False로 일관성 유지
                agent.obs_rms.update(obs_np)
                obs_np = agent._normalize_obs(obs_np, update=False)

                stone_idx, action, log_p_stone, log_p_action = agent.get_action(
                    obs_np, stone_mask, deterministic=False
                )
                my_actions = agent.decode_action(stone_idx, action, obs_me, 0)

                with torch.inference_mode():
                    obs_tensor = torch.tensor(obs_np, dtype=torch.float32).to(DEVICE)
                    mask_tensor = torch.tensor(stone_mask, dtype=torch.float32).to(DEVICE)
                    _, _, _, values = agent.model(obs_tensor, mask_tensor)
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
                    # 상대방도 10% 확률로 탐험 (다양한 대전 경험)
                    op_deterministic = random.random() > 0.1
                    si, ac, _, _ = opponent.get_action(obs_np_op, mask_op, deterministic=op_deterministic)
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

                for i, env_i in enumerate(my_idx):
                    kills = int(np.sum(obs['white'][env_i, :, 2]) - np.sum(next_obs['white'][env_i, :, 2]))
                    deaths = int(np.sum(obs['black'][env_i, :, 2]) - np.sum(next_obs['black'][env_i, :, 2]))
                    if kills > 0:
                        total_kills += kills
                    if deaths > 0:
                        total_deaths += deaths

                next_obs_me = {k: v[my_idx] for k, v in next_obs.items()}
                next_obs_np, _ = agent._process_obs(next_obs_me, override_turn=0)
                next_obs_np = agent._normalize_obs(next_obs_np, update=False)

                for i, env_i in enumerate(my_idx):
                    done = term[env_i] or trunc[env_i]
                    agent.put_data((
                        obs_np[i], stone_idx[i], action[i], rewards[i], next_obs_np[i],
                        log_p_stone[i], log_p_action[i], float(done), stone_mask[i], values_np[i]
                    ))
                    score_history.append(rewards[i])

                    if done:
                        game_count += 1
                        enemy_alive = np.sum(next_obs['white'][env_i, :, 2])
                        if enemy_alive == 0:
                            win_count += 1
                            win_history.append(1)
                        else:
                            win_history.append(0)

            obs = next_obs

        agent.train_net()

        if update % 10 == 0:
            avg_score = np.mean(score_history[-1000:]) if score_history else 0
            recent_win_rate = np.mean(win_history[-100:]) * 100 if win_history else 0

            if recent_win_rate > best_win_rate:
                best_win_rate = recent_win_rate

            print(f"{update:6d} | {avg_score:8.2f} | {recent_win_rate:5.1f}% | "
                  f"{agent.entropy_coef:.4f} | {agent.random_stone_prob:.4f} | {opponent_name:>12}")

            agent.save(SAVE_PATH, update)

            if update % 100 == 0:
                elapsed = time.time() - start_time
                print("-" * 70)
                print(f"  Progress: {update/MAX_UPDATES*100:.1f}% | Best WR: {best_win_rate:.1f}% | "
                      f"K/D: {total_kills}/{total_deaths} | Time: {timedelta(seconds=int(elapsed))}")
                print("-" * 70)

            win_count = 0
            game_count = 0

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETED!" if not SHUTDOWN_FLAG else "  TRAINING INTERRUPTED!")
    print("=" * 70)
    agent.save(SAVE_PATH, update)
    print(f"  Best Win Rate: {best_win_rate:.1f}%")
    print(f"  Checkpoint: {SAVE_PATH}")
    print("=" * 70 + "\n")

    envs.close()


if __name__ == "__main__":
    train()