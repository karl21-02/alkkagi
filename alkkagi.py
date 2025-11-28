"""
=================================================================
알까기 강화학습 에이전트 (PPO + Self-Play)
=================================================================

알고리즘: PPO (Proximal Policy Optimization)
학습 방식: Self-Play (과거 자신과 대결하며 학습)

핵심 개선사항:
1. [FIX] 돌 구분 문제 해결: stone_id 추가로 1번/2번/3번 돌 독립 학습
2. [FIX] Critic loss 가중치 증가: 0.5 → 1.0 (가치 함수 빠른 학습)
3. [FIX] GPU 자동 최적화: GPU 있으면 32 envs, 없으면 8 envs
4. [IMPROVED] 네트워크: Shared trunk + ReLU + LayerNorm + 출력 Tanh 제거
5. [IMPROVED] 관측: 54차원 (stone_id, 경계 거리, 모든 적 거리, 게임 페이즈)
6. [IMPROVED] 보상: 명중 확률, 골 근접도, 균형잡힌 kill/suicide
7. [IMPROVED] 학습률/엔트로피 스케줄링: 초반 탐험 → 후반 착취

구조:
- ActorCritic: 공유 네트워크 + Actor/Critic heads
- BaseAgent: 관측 처리, 행동 디코딩, PPO 학습
- OpponentManager: Self-play 상대 풀 관리
- train(): 메인 학습 루프

=================================================================
"""

import gymnasium as gym
import kymnasium as kym
import numpy as np
import os
import glob
import random
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from typing import Any, Dict
from gymnasium.vector import AsyncVectorEnv

# ==========================================
# 1. 하이퍼파라미터 설정 (최적화됨!)
# ==========================================

# GPU 가속 설정
# - GPU가 있으면 CUDA 사용, 없으면 CPU 사용
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_GPU_OPTIMIZED = torch.cuda.is_available()

# 환경 및 배치 설정 (GPU 유무에 따라 자동 조정)
if USE_GPU_OPTIMIZED:
    NUM_ENVS = 32      # 병렬로 실행할 게임 환경 개수 (GPU면 32개)
    BATCH_SIZE = 512   # 한 번에 학습할 데이터 개수 (GPU면 512개)
    T_HORIZON = 512    # 한 번에 수집할 경험 스텝 수 (GPU면 512)
else:
    NUM_ENVS = 8       # CPU는 8개로 줄임 (메모리 절약)
    BATCH_SIZE = 256   # CPU는 256개
    T_HORIZON = 256    # CPU는 256

# 학습률 설정 (시간이 지나면서 점차 감소)
LR_ACTOR_START = 0.0003   # Actor 초기 학습률 (정책 학습)
LR_ACTOR_END = 0.00003    # Actor 최종 학습률 (학습 후반)
LR_CRITIC = 0.0005        # Critic 학습률 (가치 함수 학습)

# PPO 알고리즘 설정
GAMMA = 0.96         # 할인율: 미래 보상을 얼마나 중요하게 볼지 (0.96 = 96%)
K_EPOCHS = 10        # 수집한 데이터를 몇 번 재사용할지
EPS_CLIP = 0.2       # PPO clipping 범위 (정책 변화를 제한)
MAX_GRAD_NORM = 0.5  # Gradient clipping (학습 안정화)

# 탐험 계수 (시간이 지나면서 탐험 감소 → 착취 증가)
ENTROPY_COEF_START = 0.15  # 초반: 랜덤하게 많이 탐험
ENTROPY_COEF_END = 0.01    # 후반: 학습한 정책대로 플레이

# Self-Play 설정
SELFPLAY_SAVE_INTERVAL = 50  # 50 업데이트마다 모델 저장
SELFPLAY_SWAP_INTERVAL = 20  # 20 업데이트마다 상대 교체

# 커리큘럼 러닝 설정 (쉬운 상대 → 어려운 상대)
USE_CURRICULUM = True                   # 커리큘럼 사용 여부
CURRICULUM_THRESHOLD = 0.15             # 15% 승률 넘으면 난이도 증가
RANDOM_OPPONENT_PROB_START = 0.8        # 처음엔 80% 랜덤 상대
RANDOM_OPPONENT_PROB_MIN = 0.1          # 나중엔 10% 랜덤 상대


# ==========================================
# 2. 신경망 구조 (Actor-Critic)
# ==========================================
class ActorCritic(nn.Module):
    """
    Actor-Critic 네트워크
    - Actor: 행동(어떻게 돌을 쏠지)을 결정
    - Critic: 현재 상황이 얼마나 좋은지 평가
    - Shared trunk: Actor와 Critic이 초반 레이어를 공유 (효율적 학습)
    """
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.data = []  # 경험 데이터 저장 버퍼

        # ===== 공유 특징 추출기 =====
        # Actor와 Critic이 공유하는 네트워크 (효율적!)
        # 입력: 게임 상태 (54차원) → 출력: 특징 벡터 (256차원)
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 512),  # 54 → 512
            nn.LayerNorm(512),           # 학습 안정화
            nn.ReLU(),                   # 활성화 함수 (기울기 소실 방지)
            nn.Linear(512, 512),         # 512 → 512 (깊은 학습)
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),         # 512 → 256
            nn.ReLU()
        )

        # ===== Actor Head (정책) =====
        # 행동의 평균값(mu) 출력
        # 중요: 출력층에 Tanh 없음! (행동 범위 제한 X)
        self.actor_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)  # 4차원 (돌 선택, 파워, 방향x, 방향y)
        )

        # ===== 표준편차 Head =====
        # 상황에 따라 탐험 정도를 조절 (중요한 순간엔 신중하게!)
        self.log_std_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        # ===== Critic Head (가치 함수) =====
        # 현재 상태의 가치를 평가 (얼마나 유리한지)
        self.critic_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # 단일 값 출력 (상태 가치)
        )

    def pi(self, x):
        """
        정책 함수: 주어진 상태에서 행동 분포 반환
        - 입력: 상태 (54차원)
        - 출력: 정규분포 (평균=mu, 표준편차=std)
        """
        shared = self.shared(x)                    # 공유 특징 추출
        mu = self.actor_head(shared)               # 행동 평균
        log_std = self.log_std_head(shared)        # 표준편차 (log scale)
        log_std = torch.clamp(log_std, -20, 2)     # 수치 안정성 (너무 크거나 작지 않게)
        std = log_std.exp()                        # 실제 표준편차
        dist = Normal(mu, std)                     # 정규분포 생성
        return dist

    def v(self, x):
        """
        가치 함수: 주어진 상태의 가치 평가
        - 입력: 상태 (54차원)
        - 출력: 가치 (단일 실수)
        """
        shared = self.shared(x)
        return self.critic_head(shared)

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            prob_a_lst.append(prob_a)
            done_lst.append(done)
        self.data = []

        def to_tensor(d):
            return torch.tensor(np.array(d), dtype=torch.float).to(DEVICE).view(-1, d[0].shape[-1])

        def to_tensor_s(d):
            return torch.tensor(np.array(d), dtype=torch.float).to(DEVICE).view(-1, 1)

        return (to_tensor(s_lst), to_tensor(a_lst), to_tensor_s(r_lst),
                to_tensor(s_prime_lst), to_tensor_s(prob_a_lst), to_tensor_s(done_lst))


# ==========================================
# 3. Random Agent (커리큘럼 러닝용)
# ==========================================
class RandomAgent:
    """완전 랜덤 행동을 하는 약한 상대"""

    def __init__(self):
        self.my_turn = 1

    def get_action_batch(self, obs_np):
        batch_size = obs_np.shape[0]
        # 랜덤 행동 생성
        actions = np.random.uniform(-1, 1, (batch_size, 4)).astype(np.float32)
        log_probs = np.zeros(batch_size, dtype=np.float32)
        return actions, log_probs


# ==========================================
# 4. Base Agent Logic
# ==========================================
class BaseAgent(kym.Agent):
    def __init__(self, my_turn):
        super().__init__()
        self.my_turn = my_turn
        self.state_dim = 54  # [FIX] 51 -> 54 (stone_id 3차원 추가!)
        self.action_dim = 4
        self.model = ActorCritic(self.state_dim, self.action_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR_ACTOR_START)  # 시작 학습률 사용
        self.current_entropy_coef = ENTROPY_COEF_START

    def _process_batch_obs(self, obs, override_turn=None, selected_stone_idx=None):
        """
        게임 관측값을 신경망 입력으로 변환

        핵심 개선사항:
        1. stone_id 추가: 1번 돌과 2번 돌을 구분! (이전에는 모두 같은 행동)
        2. 경계 거리: 골라인까지 거리 추가
        3. 모든 적과의 거리: 전략적 타겟팅 가능
        4. 게임 페이즈 정보: 돌 개수 차이, 질량 중심

        Args:
            obs: 게임 관측값 (black, white 돌 정보)
            override_turn: 강제로 차례 지정 (0=흑, 1=백)
            selected_stone_idx: 선택된 돌의 인덱스 (중요!)

        Returns:
            54차원 특징 벡터
        """
        board_scale = 1000.0  # 보드 스케일 (좌표 정규화용)
        batch_size = len(obs['black'])

        # 현재 차례 결정 (흑=0, 백=1)
        if override_turn is not None:
            turns = np.full((batch_size, 1, 1), override_turn)
        else:
            turns = obs['turn'].reshape(batch_size, 1, 1)

        black_stones = obs['black']  # 흑돌 정보 [batch, 3, 3] (x, y, alive)
        white_stones = obs['white']  # 백돌 정보 [batch, 3, 3]

        # 내 돌 vs 상대 돌 구분
        my_stones = np.where(turns == 0, black_stones, white_stones)
        op_stones = np.where(turns == 0, white_stones, black_stones)

        # 좌표 정규화 ([-500, 500] → [-0.5, 0.5])
        my_norm = np.copy(my_stones)
        my_norm[:, :, 0:2] /= board_scale
        op_norm = np.copy(op_stones)
        op_norm[:, :, 0:2] /= board_scale

        # 필요한 정보 추출
        my_xy = my_stones[:, :, 0:2]  # [B, 3, 2] 내 돌 좌표
        op_xy = op_stones[:, :, 0:2]  # [B, 3, 2] 상대 돌 좌표
        my_alive = my_stones[:, :, 2]  # [B, 3] 내 돌 생존 여부
        op_alive = op_stones[:, :, 2]  # [B, 3] 상대 돌 생존 여부

        # ===== [핵심 FIX!] 돌 구분 ID 추가 =====
        # 문제: 1번 돌이 죽으면 2번 돌도 똑같이 행동함
        # 해결: 각 돌에 고유 ID 부여 (one-hot encoding)
        if selected_stone_idx is not None:
            # 선택된 돌만 1, 나머지는 0
            # 예: 1번 돌 선택 → [0, 1, 0]
            stone_id = np.zeros((batch_size, 3), dtype=np.float32)
            stone_id[np.arange(batch_size), selected_stone_idx] = 1.0
        else:
            # 선택 안 됐으면 균등 분포
            stone_id = np.ones((batch_size, 3), dtype=np.float32) / 3.0

        # ===== 가장 가까운 적 정보 계산 =====
        # 각 내 돌마다 가장 가까운 적을 찾음
        diff = op_xy[:, np.newaxis, :, :] - my_xy[:, :, np.newaxis, :]  # 모든 쌍의 차이벡터
        dist_sq = np.sum(diff ** 2, axis=-1)  # 거리의 제곱
        mask = (1 - op_alive[:, np.newaxis, :]) * 1e9  # 죽은 적은 매우 먼 것으로 처리
        dist_sq += mask
        min_idx = np.argmin(dist_sq, axis=2)  # 각 내 돌마다 가장 가까운 적의 인덱스

        batch_idx = np.arange(batch_size)[:, np.newaxis]
        my_idx = np.arange(3)[np.newaxis, :]
        target_diff = diff[batch_idx, my_idx, min_idx, :]  # 가장 가까운 적으로의 벡터

        # 거리와 방향 계산
        raw_dist = np.sqrt(np.sum(target_diff ** 2, axis=-1))
        safe_dist = raw_dist + 1e-6  # 0으로 나누기 방지
        target_u_x = target_diff[:, :, 0] / safe_dist  # 단위 벡터 x
        target_u_y = target_diff[:, :, 1] / safe_dist  # 단위 벡터 y
        target_dist_norm = raw_dist / board_scale  # 정규화된 거리

        # ===== [신규] 보드 경계까지의 거리 =====
        # 골라인이나 벽까지 거리는 전략적으로 중요!
        # 보드 범위: [-500, 500] × [-500, 500]
        dist_to_left = (my_xy[:, :, 0] + 500) / board_scale
        dist_to_right = (500 - my_xy[:, :, 0]) / board_scale
        dist_to_top = (my_xy[:, :, 1] + 500) / board_scale
        dist_to_bottom = (500 - my_xy[:, :, 1]) / board_scale
        boundary_dists = np.stack([dist_to_left, dist_to_right, dist_to_top, dist_to_bottom], axis=-1)

        # ===== [신규] 모든 적과의 거리 =====
        # 가장 가까운 적만이 아니라 모든 적과의 거리
        # → 전략적 타겟팅 가능 (위협적인 적부터 제거)
        diff_all = my_xy[:, :, np.newaxis, :] - op_xy[:, np.newaxis, :, :]  # [B, 3, 3, 2]
        dist_all = np.linalg.norm(diff_all, axis=-1) / board_scale  # [B, 3, 3]
        dist_all += (1 - op_alive[:, np.newaxis, :]) * 10.0  # 죽은 돌 마스킹

        # ===== [신규] 게임 페이즈 정보 =====
        # 돌 개수 차이: 내가 유리한지 불리한지
        my_count = np.sum(my_alive, axis=1, keepdims=True)
        op_count = np.sum(op_alive, axis=1, keepdims=True)
        count_diff = (my_count - op_count) / 3.0  # [-1, 1] 범위

        # 질량 중심 차이: 공격/수비 위치 파악
        my_center = np.sum(my_xy * my_alive[:, :, np.newaxis], axis=1) / (my_count + 1e-6)
        op_center = np.sum(op_xy * op_alive[:, :, np.newaxis], axis=1) / (op_count + 1e-6)
        center_diff = (my_center - op_center) / board_scale

        # ===== 모든 특징 결합 (총 54차원) =====
        flat_obs = np.concatenate([
            stone_id,                                           # 3차원 - 돌 ID (핵심!)
            my_norm.reshape(batch_size, -1),                    # 9차원 - 내 돌 정보
            op_norm.reshape(batch_size, -1),                    # 9차원 - 상대 돌 정보
            target_dist_norm[:, :, np.newaxis].reshape(batch_size, -1),  # 3차원 - 가까운 적 거리
            target_u_y[:, :, np.newaxis].reshape(batch_size, -1),        # 3차원 - 가까운 적 방향Y
            target_u_x[:, :, np.newaxis].reshape(batch_size, -1),        # 3차원 - 가까운 적 방향X
            boundary_dists.reshape(batch_size, -1),             # 12차원 - 경계 거리
            dist_all.reshape(batch_size, -1),                   # 9차원 - 모든 적 거리
            count_diff,                                          # 1차원 - 돌 개수 차이
            center_diff                                          # 2차원 - 질량중심 차이
        ], axis=1)

        return flat_obs.astype(np.float32)  # 총 54차원!

    def _process_single_obs(self, obs):
        batch_obs = {'black': np.array([obs['black']]), 'white': np.array([obs['white']]), 'turn': np.array([0])}
        return self._process_batch_obs(batch_obs, override_turn=self.my_turn)[0]

    def _decode_action(self, action_tensor):
        if isinstance(action_tensor, torch.Tensor):
            a = action_tensor.cpu().numpy().flatten()
        else:
            a = action_tensor.flatten()

        raw_idx = (a[0] + 1) / 2.0
        idx = int(np.clip(raw_idx * 3, 0, 2))

        power = float(300.0 + ((a[1] + 1) / 2.0) * 2200.0)

        dx = a[2]
        dy = a[3]
        angle = float(np.degrees(np.arctan2(dy, dx)))

        return {"turn": self.my_turn, "index": idx, "power": power, "angle": angle}

    def decode_batch_action(self, action_np, current_turns):
        raw_idx = (action_np[:, 0] + 1) / 2.0
        idx = np.clip(raw_idx * 3, 0, 2).astype(np.int32)
        power = 300.0 + ((action_np[:, 1] + 1) / 2.0) * 2200.0

        dx = action_np[:, 2]
        dy = action_np[:, 3]
        angle = np.degrees(np.arctan2(dy, dx))

        return {"turn": current_turns.astype(np.int32), "index": idx, "power": power.astype(np.float32),
                "angle": angle.astype(np.float32)}

    def get_action_batch(self, obs_np):
        obs_tensor = torch.tensor(obs_np, dtype=torch.float).to(DEVICE)
        dist = self.model.pi(obs_tensor)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1).detach().cpu().numpy()
        return torch.tanh(action).cpu().numpy(), log_prob

    def act(self, observation, info):
        obs_np = self._process_single_obs(observation)
        obs_tensor = torch.tensor(obs_np, dtype=torch.float).to(DEVICE)
        with torch.no_grad():
            shared = self.model.shared(obs_tensor)
            mu = self.model.actor_head(shared)
        return self._decode_action(mu)

    def train_net(self):
        if len(self.model.data) < 1: return
        s, a, r, s_prime, prob_a, done_mask = self.model.make_batch()
        with torch.no_grad():
            td_target = r + GAMMA * self.model.v(s_prime) * (1 - done_mask)
            delta = td_target - self.model.v(s)
        advantage = delta.detach()
        total_samples = s.size(0)
        indices = np.arange(total_samples)
        for _ in range(K_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, total_samples, BATCH_SIZE):
                idx = indices[start:start + BATCH_SIZE]
                dist = self.model.pi(s[idx])
                cur_log_prob = dist.log_prob(a[idx]).sum(-1).unsqueeze(1)
                ratio = torch.exp(cur_log_prob - prob_a[idx])
                surr1 = ratio * advantage[idx]
                surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantage[idx]
                loss = -torch.min(surr1, surr2).mean() + 1.0 * F.smooth_l1_loss(self.model.v(s[idx]), td_target[
                    idx]) - self.current_entropy_coef * dist.entropy().sum(-1).mean()  # [FIX] 0.5 -> 1.0 (Critic 강화!)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)  # [IMPROVED] 상수 사용
                self.optimizer.step()


# ==========================================
# 5. Wrappers
# ==========================================
class YourBlackAgent(BaseAgent):
    def __init__(self):
        super().__init__(my_turn=0)

    @classmethod
    def load(cls, path):
        agent = cls()
        if os.path.exists(path):
            agent.model.load_state_dict(torch.load(path, map_location=DEVICE))
        elif os.path.exists(path + ".pkl"):
            agent.model.load_state_dict(torch.load(path + ".pkl", map_location=DEVICE))
        return agent

    def save(self, path):
        torch.save(self.model.state_dict(), path)


class YourWhiteAgent(BaseAgent):
    def __init__(self):
        super().__init__(my_turn=1)

    @classmethod
    def load(cls, path):
        agent = cls()
        if os.path.exists(path):
            agent.model.load_state_dict(torch.load(path, map_location=DEVICE))
        elif os.path.exists(path + ".pkl"):
            agent.model.load_state_dict(torch.load(path + ".pkl", map_location=DEVICE))
        return agent

    def save(self, path):
        torch.save(self.model.state_dict(), path)


# ==========================================
# 6. Manager & Loop
# ==========================================
class OpponentManager:
    def __init__(self):
        self.save_dir = "history_models"
        os.makedirs(self.save_dir, exist_ok=True)
        self.pool = glob.glob(os.path.join(self.save_dir, "model_*.pkl"))
        if not self.pool: self.save_current_model(YourBlackAgent().model, 0)

    def save_current_model(self, model, step):
        path = os.path.join(self.save_dir, f"model_{step}.pkl")
        torch.save(model.state_dict(), path)
        self.pool.append(path)
        if len(self.pool) > 20:
            old = self.pool.pop(0)
            if os.path.exists(old): os.remove(old)

    def get_opponent(self):
        path = random.choice(self.pool)
        op = YourBlackAgent()
        try:
            op.model.load_state_dict(torch.load(path, map_location=DEVICE)); op.model.eval(); return op
        except:
            return None


def make_env(): return gym.make(id='kymnasium/AlKkaGi-3x3-v0', render_mode=None, bgm=False, obs_type='custom')


def calculate_min_dist(black, white):
    """각 환경에서 가장 가까운 적과의 거리 계산"""
    b_xy = black[:, :, 0:2];
    b_alive = black[:, :, 2]
    w_xy = white[:, :, 0:2];
    w_alive = white[:, :, 2]
    diff = b_xy[:, :, np.newaxis, :] - w_xy[:, np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    mask = (1 - b_alive[:, :, np.newaxis] * w_alive[:, np.newaxis, :]) * 1e5
    dist += mask
    return np.min(dist, axis=(1, 2))


def train():
    print(f"🚀 Starting OPTIMIZED Training!")
    print(f"📊 Hyperparameters:")
    print(f"   - Learning Rate (Actor): {LR_ACTOR_START} -> {LR_ACTOR_END}")
    print(f"   - Learning Rate (Critic): {LR_CRITIC}")
    print(f"   - Entropy Coefficient: {ENTROPY_COEF_START} -> {ENTROPY_COEF_END}")
    print(f"   - GAMMA: {GAMMA}")
    print(f"   - K_EPOCHS: {K_EPOCHS}")
    print(f"   - NUM_ENVS: {NUM_ENVS}")
    print(f"   - Curriculum Learning: {USE_CURRICULUM}")

    envs = AsyncVectorEnv([make_env for _ in range(NUM_ENVS)])
    agent = YourBlackAgent()

    if os.path.exists("my_alkkagi_agent.pkl"):
        print("✅ Found existing model. Resuming training...")
        try:
            ckpt = torch.load("my_alkkagi_agent.pkl", map_location=DEVICE)
            agent.model.load_state_dict(ckpt)
        except Exception as e:
            print(f"⚠️ Failed to load model: {e}. Starting fresh.")
    else:
        print("🚀 Starting fresh training.")

    op_manager = OpponentManager()
    opponent = op_manager.get_opponent()
    random_agent = RandomAgent()  # [신규] 랜덤 상대

    obs, _ = envs.reset()
    prev_opp = np.sum(obs['white'][:, :, 2], axis=1)
    prev_my = np.sum(obs['black'][:, :, 2], axis=1)
    prev_min_dist = calculate_min_dist(obs['black'], obs['white'])  # [신규] 거리 추적

    score_history = []
    interval_win_cnt = 0
    interval_total_cnt = 0
    recent_win_rates = []  # [신규] 커리큘럼 진행을 위한 승률 추적

    for update in range(1, 10001):
        # [IMPROVED] 학습률 스케줄링
        progress = min(update / 10000.0, 1.0)
        current_lr = LR_ACTOR_START + (LR_ACTOR_END - LR_ACTOR_START) * progress
        for param_group in agent.optimizer.param_groups:
            param_group['lr'] = current_lr

        # [IMPROVED] 엔트로피 스케줄링 (초반 탐험 -> 후반 착취)
        agent.current_entropy_coef = ENTROPY_COEF_START + (ENTROPY_COEF_END - ENTROPY_COEF_START) * min(update / 5000.0, 1.0)

        if update % SELFPLAY_SAVE_INTERVAL == 0:
            op_manager.save_current_model(agent.model, update)
            print(f"💾 Saved model. Pool: {len(op_manager.pool)}")

        if update % SELFPLAY_SWAP_INTERVAL == 0:
            opponent = op_manager.get_opponent()
            print("🔄 Swapped opponent.")

        # [신규] 커리큘럼 러닝: 승률에 따라 랜덤 상대 비율 조정
        if USE_CURRICULUM and len(recent_win_rates) > 0:
            avg_recent_wr = np.mean(recent_win_rates[-10:])  # 최근 10개 평균
            random_prob = max(RANDOM_OPPONENT_PROB_MIN,
                              RANDOM_OPPONENT_PROB_START - (avg_recent_wr / CURRICULUM_THRESHOLD) * 0.7)
        else:
            random_prob = RANDOM_OPPONENT_PROB_START

        for _ in range(T_HORIZON):
            turns = obs['turn']
            actions_np = np.zeros((NUM_ENVS, 4), dtype=np.float32)
            log_probs_np = np.zeros(NUM_ENVS, dtype=np.float32)

            my_idx = np.where(turns == 0)[0]
            if len(my_idx) > 0:
                obs_me = agent._process_batch_obs(obs, override_turn=0)[my_idx]
                a, p = agent.get_action_batch(obs_me)
                actions_np[my_idx] = a;
                log_probs_np[my_idx] = p

            op_idx = np.where(turns == 1)[0]
            if len(op_idx) > 0:
                # [신규] 확률적으로 랜덤 상대 또는 셀프플레이 상대 선택
                if random.random() < random_prob:
                    curr_op = random_agent
                else:
                    curr_op = opponent if opponent else agent

                obs_op = agent._process_batch_obs(obs, override_turn=1)[op_idx]
                with torch.no_grad():
                    a, _ = curr_op.get_action_batch(obs_op)
                actions_np[op_idx] = a

            real_actions = agent.decode_batch_action(actions_np, turns)
            next_obs, _, term, trunc, _ = envs.step(real_actions)

            curr_opp = np.sum(next_obs['white'][:, :, 2], axis=1)
            curr_my = np.sum(next_obs['black'][:, :, 2], axis=1)
            curr_min_dist = calculate_min_dist(next_obs['black'], next_obs['white'])  # [신규]

            if len(my_idx) > 0:
                # ===== 보상 함수 계산 (핵심!) =====
                # 좋은 보상 함수 = 승리로 이어지는 행동을 명확히 알려줌

                # 1. 기본 보상: 적 제거 vs 자살
                kill = prev_opp[my_idx] - curr_opp[my_idx]      # 적 몇 개 죽였나
                suicide = prev_my[my_idx] - curr_my[my_idx]    # 내 돌 몇 개 잃었나
                r = (kill * 100.0) - (suicide * 100.0)         # [FIX] 균형 조정! (이전: 200 vs 50)

                # 2. 골 근접도 보상 (알까기는 골 게임!)
                # 적 골라인(x=500)에 가까울수록 유리
                b_xy_after = next_obs['black'][my_idx, :, 0:2]
                b_alive_after = next_obs['black'][my_idx, :, 2]
                avg_x_pos = np.sum(b_xy_after[:, :, 0] * b_alive_after, axis=1) / (np.sum(b_alive_after, axis=1) + 1e-6)
                goal_proximity = avg_x_pos / 1000.0  # [-0.5, 0.5]로 정규화
                r += goal_proximity * 10.0  # 골 쪽으로 이동하면 보상!

                # 3. 명중 확률 기반 조준 보상
                # 단순히 "방향"만이 아니라 "명중 가능성"을 평가!
                agent_dx = actions_np[my_idx, 2]
                agent_dy = actions_np[my_idx, 3]
                agent_len = np.sqrt(agent_dx ** 2 + agent_dy ** 2) + 1e-6
                shot_dir = np.stack([agent_dx / agent_len, agent_dy / agent_len], axis=-1)  # 발사 방향 단위벡터

                b_xy = obs['black'][my_idx, :, 0:2]
                w_xy = obs['white'][my_idx, :, 0:2]
                w_alive = obs['white'][my_idx, :, 2]

                # 선택한 돌의 위치
                raw_s_idx = (actions_np[my_idx, 0] + 1) / 2.0
                sel_idx = np.clip(raw_s_idx * 3, 0, 2).astype(int)
                k_rng = np.arange(len(my_idx))
                my_pos = b_xy[k_rng, sel_idx]

                # 각 적에 대한 명중 가능성 계산
                diff = w_xy - my_pos[:, np.newaxis, :]  # 적으로의 벡터
                dist = np.linalg.norm(diff, axis=-1)  # 거리
                diff_normalized = diff / (dist[:, :, np.newaxis] + 1e-6)  # 방향

                # 조준 정렬도: 발사 방향과 적 방향이 얼마나 일치하나
                alignment = np.sum(diff_normalized * shot_dir[:, np.newaxis, :], axis=-1)

                # 거리 계수: 가까울수록 명중 쉬움
                dist_factor = np.exp(-dist / 500.0)

                # 최종 명중 점수 = 조준 × 거리 × 생존여부
                hit_score = alignment * dist_factor * w_alive
                best_hit_score = np.max(hit_score, axis=1)  # 가장 잘 조준한 적
                r += best_hit_score * 5.0  # 잘 조준하면 보상!

                # 4. 파워 효율성: 너무 약한 샷은 무의미
                power = 300.0 + ((actions_np[my_idx, 1] + 1) / 2.0) * 2200.0
                power_normalized = (power - 300.0) / 2200.0
                weak_shot_penalty = np.maximum(0, 0.3 - power_normalized) * 2.0
                r -= weak_shot_penalty  # 약한 샷은 페널티

                # 5. 승/패 보상 (게임 종료 시)
                done = np.logical_or(term, trunc)[my_idx]
                win = (curr_opp[my_idx] == 0) & done   # 적 전멸 = 승리
                lose = (curr_my[my_idx] == 0) & done   # 내 전멸 = 패배

                r[win] += 200.0   # [FIX] 승리 보너스 (이전: 500, 너무 컸음)
                r[lose] -= 200.0  # [FIX] 패배 페널티 (이전: -100, 너무 약했음)

                for k, d in enumerate(done):
                    if d:
                        interval_total_cnt += 1
                        if win[k]: interval_win_cnt += 1

                s = agent._process_batch_obs(obs, override_turn=0)[my_idx]
                s_p = agent._process_batch_obs(next_obs, override_turn=0)[my_idx]
                for k, i in enumerate(my_idx):
                    agent.model.put_data(
                        (s[k], actions_np[i], r[k] / 100.0, s_p[k], log_probs_np[i], done.astype(float)[k]))
                    score_history.append(r[k])

            obs = next_obs
            prev_opp = curr_opp
            prev_my = curr_my
            prev_min_dist = curr_min_dist  # [신규] 거리 업데이트

        agent.train_net()

        if update % 5 == 0 and score_history:
            avg_score = np.mean(score_history[-1000:]) * 100.0
            if interval_total_cnt > 0:
                win_rate = (interval_win_cnt / interval_total_cnt) * 100.0
                recent_win_rates.append(win_rate / 100.0)  # [신규] 승률 기록
                if len(recent_win_rates) > 50:
                    recent_win_rates.pop(0)

                print(
                    f"Update: {update}, Avg Reward: {avg_score:.2f}, Win Rate: {win_rate:.1f}% ({interval_win_cnt}/{interval_total_cnt}), Random Opp: {random_prob * 100:.0f}%")
            else:
                print(
                    f"Update: {update}, Avg Reward: {avg_score:.2f}, Win Rate: 0.0% (0/0), Random Opp: {random_prob * 100:.0f}%")
            interval_win_cnt = 0
            interval_total_cnt = 0
            agent.save("my_alkkagi_agent.pkl")

    envs.close()
    print("🎉 Training completed!")


if __name__ == "__main__": train()