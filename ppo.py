"""
=================================================================
알까기 강화학습 에이전트 - PPO (Proximal Policy Optimization)
=================================================================

[ PPO 최적화 기법 12가지 ]
1. GAE (Generalized Advantage Estimation) - 분산 감소를 위한 어드밴티지 추정
2. Value Clipping (PPO2 스타일) - Critic 업데이트 안정화
3. Policy Clipping - 정책 업데이트를 ε 범위 내로 제한
4. Entropy Bonus - 탐험 장려를 위한 엔트로피 보너스
5. Advantage Normalization - 어드밴티지 정규화
6. Gradient Clipping - 그래디언트 폭발 방지
7. Critic Extra Training (5x) - Critic 추가 학습
8. Learning Rate Scheduling - 학습률 선형 감소
9. Reward Normalization - 보상 정규화
10. Observation Normalization - 관측 정규화
11. Orthogonal Initialization - 직교 초기화
12. Mixed Precision Training (AMP) - 혼합 정밀도 학습 (GPU)

[ Self-Play 전략 ]
- 단순 Self-Play: 최근 N개 모델 풀에서 랜덤 선택
- Rule-based 상대: 초반 기본기 학습용
- Curriculum Learning: 초반 500 업데이트는 RuleBot → 이후 100% Self-Play

=================================================================
"""

# ==============================================================================
# 라이브러리 임포트
# ==============================================================================
import os
# pygame 오디오 비활성화 (ALSA 에러 방지)
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import gymnasium as gym
import kymnasium as kym
import numpy as np
import glob
import json
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
from torch.amp import autocast, GradScaler
from gymnasium.vector import AsyncVectorEnv
from typing import Any, Dict, Optional, Tuple, List

# Ctrl+C 감지를 위한 플래그
SHUTDOWN_FLAG = False


# ==============================================================================
# 1. 하이퍼파라미터 설정
# ==============================================================================

# 디바이스 설정 (GPU 있으면 CUDA, 없으면 CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()  # AMP는 GPU에서만 사용

# CUDA 최적화 설정
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True      # Convolution 자동 최적화
    torch.backends.cuda.matmul.allow_tf32 = True  # TF32 활성화 (속도↑)
    torch.backends.cudnn.allow_tf32 = True

# torch.compile 설정 (CUDAGraphs 호환성 문제로 비활성화)
USE_COMPILE = False

# 환경 및 배치 설정 (GPU/CPU에 따라 다름)
if torch.cuda.is_available():
    NUM_ENVS = 32       # 병렬 환경 수
    BATCH_SIZE = 512    # 배치 크기
    T_HORIZON = 256     # 업데이트당 스텝 수
else:
    NUM_ENVS = 8
    BATCH_SIZE = 256
    T_HORIZON = 128

# PPO 핵심 하이퍼파라미터
GAMMA = 0.99            # 할인율 (미래 보상의 중요도)
GAE_LAMBDA = 0.95       # GAE 람다 (분산-편향 트레이드오프)
EPS_CLIP = 0.2          # PPO 클리핑 범위 (정책 변화 제한)
K_EPOCHS = 3            # 데이터 재사용 횟수
CRITIC_EXTRA_EPOCHS = 4 # Critic 추가 학습 횟수

# 학습률 스케줄링
LR_START = 3e-4         # 시작 학습률
LR_END = 3e-5           # 종료 학습률
MAX_UPDATES = 10000     # 총 업데이트 횟수

# 손실 함수 계수
ENTROPY_COEF = 0.01     # 엔트로피 보너스 계수 (탐험 장려)
VALUE_COEF = 0.5        # Value loss 계수
MAX_GRAD_NORM = 0.5     # 그래디언트 클리핑 임계값

# 게임 환경 설정
BOARD_SIZE = 600.0      # 보드 크기 (정규화용)
POWER_MIN = 300.0       # 최소 파워
POWER_MAX = 2500.0      # 최대 파워
ANGLE_RANGE = 45.0      # 각도 조절 범위 (±45도)

# 저장 설정
SAVE_INTERVAL = 50      # 모델 풀에 저장하는 주기
SWAP_INTERVAL = 20      # 상대 교체 주기
MODEL_POOL_SIZE = 5     # 모델 풀 크기 (최근 N개만 유지)

# 커리큘럼 설정
RULEBOT_PHASE = 500     # 이 업데이트까지 RuleBot 사용, 이후 100% Self-Play

SAVE_PATH = "my_alkkagi_agent.pkl"  # 메인 체크포인트 경로
HISTORY_DIR = "history_models"       # 과거 모델 저장 디렉토리


# ==============================================================================
# 2. 유틸리티 클래스
# ==============================================================================

class RunningMeanStd:
    """
    온라인 평균/분산 계산기 (Welford 알고리즘)

    용도: 관측값과 보상을 정규화하기 위해 평균과 분산을 실시간으로 추적

    원리:
    - 데이터가 들어올 때마다 평균과 분산을 업데이트
    - 전체 데이터를 저장하지 않고도 통계를 유지
    """
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)  # 평균
        self.var = np.ones(shape, dtype=np.float64)    # 분산
        self.count = 1e-4  # 샘플 수 (0 나누기 방지)

    def update(self, x):
        """새 배치 데이터로 평균/분산 업데이트"""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        """Welford 알고리즘으로 통계 병합"""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        # 새 평균 계산
        self.mean = self.mean + delta * batch_count / total_count
        # 새 분산 계산 (병렬 분산 병합 공식)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, x):
        """데이터를 평균=0, 분산=1로 정규화"""
        return (x - self.mean) / np.sqrt(self.var + 1e-8)

    def state_dict(self):
        """저장용 상태 딕셔너리"""
        return {'mean': self.mean, 'var': self.var, 'count': self.count}

    def load_state_dict(self, state):
        """상태 복원"""
        self.mean = state['mean']
        self.var = state['var']
        self.count = state['count']


# ==============================================================================
# 3. 신경망 정의
# ==============================================================================

def orthogonal_init(layer, gain=np.sqrt(2)):
    """
    직교 초기화 (Orthogonal Initialization)

    효과:
    - 초기 그래디언트 흐름 개선
    - 학습 초기 안정성 향상
    - PPO 논문에서 권장하는 초기화 방법
    """
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)


class HybridActorCritic(nn.Module):
    """
    하이브리드 Actor-Critic 네트워크

    구조:
    - Shared: 공유 특징 추출 레이어 (128→128)
    - Stone Head: 어떤 돌을 칠지 선택 (이산 행동)
    - Action Head: 각도/파워 조절 (연속 행동)
    - Critic Head: 상태 가치 추정

    입력: 24차원 상태 벡터
    - 내 돌 위치/생존 (9) + 상대 돌 위치/생존 (9) + 생존 수 (2) + 벽 정보 (4)

    출력:
    - stone_logits: 돌 선택 확률 (3)
    - action_mu: 연속 행동 평균 (2) [각도, 파워]
    - action_std: 연속 행동 표준편차 (2)
    - value: 상태 가치 (1)
    """
    def __init__(self, state_dim: int = 24):
        super().__init__()

        # 공유 특징 추출 레이어
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),  # 레이어 정규화 (학습 안정화)
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.Tanh()
        )

        # 돌 선택 헤드 (Categorical Policy)
        self.stone_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 3)  # 3개 돌 중 선택
        )

        # 연속 행동 헤드 (Gaussian Policy)
        self.action_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 2)  # [각도 오프셋, 파워 비율]
        )

        # 로그 표준편차 (학습 가능한 파라미터)
        self.log_std = nn.Parameter(torch.zeros(2))

        # Critic 헤드 (Value Function)
        self.critic_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """가중치 초기화"""
        # 공유/Critic 레이어: gain=sqrt(2) (ReLU용이지만 Tanh에도 잘 동작)
        for module in [self.shared, self.critic_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    orthogonal_init(layer, gain=np.sqrt(2))

        # 정책 헤드: gain=0.01 (초기 행동을 작게 시작)
        for layer in self.stone_head:
            if isinstance(layer, nn.Linear):
                orthogonal_init(layer, gain=0.01)

        for layer in self.action_head:
            if isinstance(layer, nn.Linear):
                orthogonal_init(layer, gain=0.01)

    def forward(self, x, stone_mask=None):
        """
        순전파

        Args:
            x: 상태 텐서 (batch, 24)
            stone_mask: 살아있는 돌 마스크 (batch, 3) - 죽은 돌 선택 방지

        Returns:
            stone_logits: 돌 선택 로짓
            action_mu: 연속 행동 평균
            action_std: 연속 행동 표준편차
            value: 상태 가치
        """
        features = self.shared(x)

        # 돌 선택 (죽은 돌은 -inf로 마스킹)
        stone_logits = self.stone_head(features)
        if stone_mask is not None:
            stone_logits = stone_logits.masked_fill(stone_mask == 0, -1e9)

        # 연속 행동
        action_mu = self.action_head(features)
        action_std = self.log_std.exp().expand_as(action_mu)

        # 가치
        value = self.critic_head(features)

        return stone_logits, action_mu, action_std, value

    def get_value(self, x):
        """가치만 추출 (GAE 계산용)"""
        features = self.shared(x)
        return self.critic_head(features)


# ==============================================================================
# 4. 규칙 기반 에이전트 (학습 초기 상대)
# ==============================================================================

class RuleBasedAgent:
    """
    규칙 기반 상대 에이전트

    전략: 가장 가까운 적 돌을 향해 공격

    용도:
    - 학습 초기에 기본적인 게임 방법 학습
    - 다양한 상황 경험 제공
    """
    def __init__(self, my_turn: int):
        self.my_turn = my_turn  # 0: 흑돌, 1: 백돌

    def act(self, obs: Dict) -> Dict:
        """
        행동 결정

        로직:
        1. 내 돌과 상대 돌 중 살아있는 것 찾기
        2. 가장 가까운 적 돌을 찾기
        3. 그 방향으로 공격
        """
        # 턴에 따라 내 돌/상대 돌 구분
        my_stones = obs['black'] if self.my_turn == 0 else obs['white']
        enemy_stones = obs['white'] if self.my_turn == 0 else obs['black']

        # 살아있는 돌만 필터링
        my_alive = [(i, s) for i, s in enumerate(my_stones) if s[2] == 1]
        enemy_alive = [(i, s) for i, s in enumerate(enemy_stones) if s[2] == 1]

        # 예외 처리
        if not my_alive or not enemy_alive:
            return {"turn": self.my_turn, "index": 0, "power": 500.0, "angle": 0.0}

        # 가장 가까운 적 찾기
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
                    # 적 방향으로 각도 계산
                    dx = enemy_pos[0] - my_pos[0]
                    dy = enemy_pos[1] - my_pos[1]
                    best_angle = np.degrees(np.arctan2(dy, dx))
                    # 거리에 비례한 파워 (+ 약간의 랜덤)
                    best_power = np.clip(500 + dist * 2 + random.uniform(-100, 100),
                                        POWER_MIN, POWER_MAX)

        return {"turn": self.my_turn, "index": best_my_idx,
                "power": float(best_power), "angle": float(best_angle)}


# ==============================================================================
# 5. PPO 에이전트 기본 클래스
# ==============================================================================

class BaseAgent(kym.Agent):
    """
    PPO 에이전트 기본 클래스

    역할:
    - 관측 전처리 (_process_obs)
    - 행동 선택 (get_action)
    - 행동 디코딩 (decode_action)
    - 경험 저장 (put_data)
    - 네트워크 학습 (train_net)
    """
    def __init__(self, my_turn: int):
        super().__init__()
        self.my_turn = my_turn  # 0: 흑돌, 1: 백돌
        self.state_dim = 24     # 상태 차원

        # 신경망 생성
        self.model = HybridActorCritic(self.state_dim).to(DEVICE)

        # torch.compile 적용 (활성화 시)
        if USE_COMPILE and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
            except Exception as e:
                print(f"[Warning] torch.compile failed: {e}")

        # 옵티마이저
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR_START)

        # 정규화용 통계
        self.obs_rms = RunningMeanStd(shape=(self.state_dim,))  # 관측 정규화
        self.reward_rms = RunningMeanStd(shape=())              # 보상 정규화

        # AMP용 스케일러 (GPU만)
        self.scaler = GradScaler() if USE_AMP else None

        # 경험 버퍼
        self.data = []

        # 텐서 사전 할당 (메모리 재사용으로 속도 향상)
        self._obs_buffer = None
        self._mask_buffer = None

    def _process_obs(self, obs: Dict, override_turn: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        관측값을 신경망 입력 형태로 변환

        입력 관측:
        - obs['black']: 흑돌 정보 (3, 3) - [x, y, alive]
        - obs['white']: 백돌 정보 (3, 3)
        - obs['turn']: 현재 턴
        - obs['obstacles']: 장애물 정보

        출력 (24차원):
        - [0-8]: 내 돌 (x, y, alive) × 3
        - [9-17]: 상대 돌 (x, y, alive) × 3
        - [18]: 내 생존 돌 수 / 3
        - [19]: 상대 생존 돌 수 / 3
        - [20]: 벽 존재 여부
        - [21-23]: 벽 정보 (x, y, 크기)
        """
        batch_size = len(obs['black'])
        if override_turn is not None:
            turns = np.full(batch_size, override_turn)
        else:
            turns = obs['turn'].flatten()

        processed = np.zeros((batch_size, self.state_dim), dtype=np.float32)
        stone_mask = np.zeros((batch_size, 3), dtype=np.float32)  # 살아있는 돌 마스크

        for b in range(batch_size):
            turn = turns[b]
            # 턴에 따라 내 돌/상대 돌 구분
            if turn == 0:
                my_stones = obs['black'][b]
                enemy_stones = obs['white'][b]
            else:
                my_stones = obs['white'][b]
                enemy_stones = obs['black'][b]

            # 내 돌 정보 (0-8)
            for i in range(3):
                processed[b, i*3 + 0] = my_stones[i, 0] / BOARD_SIZE  # x 정규화
                processed[b, i*3 + 1] = my_stones[i, 1] / BOARD_SIZE  # y 정규화
                processed[b, i*3 + 2] = my_stones[i, 2]               # alive (0 or 1)
                stone_mask[b, i] = my_stones[i, 2]  # 살아있는 돌만 선택 가능

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
                processed[b, 20] = 1.0                              # 벽 존재
                processed[b, 21] = wall[0] / BOARD_SIZE             # 벽 x
                processed[b, 22] = wall[1] / BOARD_SIZE             # 벽 y
                processed[b, 23] = max(wall[2], wall[3]) / BOARD_SIZE  # 벽 크기
            else:
                processed[b, 20:24] = 0.0

        return processed, stone_mask

    def _normalize_obs(self, obs: np.ndarray, update: bool = True) -> np.ndarray:
        """관측값 정규화 (평균=0, 분산=1)"""
        if update:
            self.obs_rms.update(obs)
        return self.obs_rms.normalize(obs)

    def get_action(self, obs_np: np.ndarray, stone_mask: np.ndarray,
                   deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        신경망으로 행동 선택

        Args:
            obs_np: 전처리된 관측 (batch, 24)
            stone_mask: 살아있는 돌 마스크 (batch, 3)
            deterministic: True면 평균값 사용 (평가용)

        Returns:
            stone_idx: 선택한 돌 인덱스
            action: 연속 행동 [각도, 파워]
            log_prob_stone: 돌 선택 로그 확률
            log_prob_action: 연속 행동 로그 확률
        """
        batch_size = obs_np.shape[0]

        # 텐서 사전 할당 (재사용으로 속도 향상)
        if self._obs_buffer is None or self._obs_buffer.shape[0] != batch_size:
            self._obs_buffer = torch.empty((batch_size, self.state_dim),
                                           dtype=torch.float32, device=DEVICE)
            self._mask_buffer = torch.empty((batch_size, 3),
                                            dtype=torch.float32, device=DEVICE)

        # numpy → torch 복사 (새 텐서 생성보다 빠름)
        self._obs_buffer.copy_(torch.from_numpy(obs_np))
        self._mask_buffer.copy_(torch.from_numpy(stone_mask))

        # 추론 모드 (그래디언트 계산 안 함 → 속도↑)
        with torch.inference_mode():
            stone_logits, action_mu, action_std, _ = self.model(self._obs_buffer, self._mask_buffer)

        # 돌 선택 (Categorical 분포)
        stone_dist = Categorical(logits=stone_logits)
        if deterministic:
            stone_idx = torch.argmax(stone_logits, dim=1)
        else:
            stone_idx = stone_dist.sample()
        log_prob_stone = stone_dist.log_prob(stone_idx)

        # 연속 행동 선택 (Gaussian 분포)
        action_dist = Normal(action_mu, action_std)
        if deterministic:
            action = action_mu
        else:
            action = action_dist.rsample()  # reparameterization trick
        log_prob_action = action_dist.log_prob(action).sum(dim=1)

        return (stone_idx.cpu().numpy(), action.cpu().numpy(),
                log_prob_stone.cpu().numpy(), log_prob_action.cpu().numpy())

    def decode_action(self, stone_idx: np.ndarray, action: np.ndarray,
                      obs: Dict, my_turn: int) -> List[Dict]:
        """
        신경망 출력을 게임 행동으로 변환

        신경망 출력:
        - stone_idx: 선택한 돌 인덱스 (0, 1, 2)
        - action[0]: 각도 오프셋 (tanh → ±ANGLE_RANGE)
        - action[1]: 파워 비율 (tanh → 0~1 → POWER_MIN~POWER_MAX)

        게임 행동:
        - turn: 턴 (0 or 1)
        - index: 돌 인덱스
        - angle: 발사 각도 (degrees)
        - power: 발사 파워
        """
        actions = []
        batch_size = len(stone_idx)
        my_stones = obs['black'] if my_turn == 0 else obs['white']
        enemy_stones = obs['white'] if my_turn == 0 else obs['black']

        for b in range(batch_size):
            idx = int(stone_idx[b])
            my_pos = my_stones[b, idx, :2]  # 선택한 돌의 위치

            # 가장 가까운 적 찾기 (기준 각도 계산용)
            enemy_alive = enemy_stones[b, :, 2] == 1
            if np.any(enemy_alive):
                enemy_positions = enemy_stones[b, enemy_alive, :2]
                dists = np.linalg.norm(enemy_positions - my_pos, axis=1)
                nearest_enemy = enemy_positions[np.argmin(dists)]
            else:
                nearest_enemy = np.array([BOARD_SIZE/2, BOARD_SIZE/2])

            # 기준 각도 계산 (가장 가까운 적 방향)
            dx = nearest_enemy[0] - my_pos[0]
            dy = nearest_enemy[1] - my_pos[1]
            base_angle = np.degrees(np.arctan2(dy, dx))

            # 각도 오프셋 적용 (±ANGLE_RANGE)
            angle_offset = np.tanh(action[b, 0]) * ANGLE_RANGE
            final_angle = base_angle + angle_offset

            # 파워 계산 (POWER_MIN ~ POWER_MAX)
            power_ratio = (np.tanh(action[b, 1]) + 1) / 2  # 0~1로 변환
            power = POWER_MIN + power_ratio * (POWER_MAX - POWER_MIN)

            actions.append({
                "turn": my_turn,
                "index": idx,
                "power": float(power),
                "angle": float(final_angle)
            })

        return actions

    def put_data(self, transition):
        """경험 버퍼에 전이 저장"""
        self.data.append(transition)

    def make_batch(self):
        """
        경험 버퍼를 텐서 배치로 변환

        전이 구조: (s, si, a, r, ns, lps, lpa, d, sm, v)
        - s: 상태
        - si: 선택한 돌 인덱스
        - a: 연속 행동
        - r: 보상
        - ns: 다음 상태
        - lps: 돌 선택 로그 확률
        - lpa: 연속 행동 로그 확률
        - d: 종료 여부
        - sm: 돌 마스크
        - v: 가치 추정
        """
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

        self.data = []  # 버퍼 비우기

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
        """
        GAE (Generalized Advantage Estimation) 계산

        수식: A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
        여기서 δ_t = r_t + γV(s_{t+1}) - V(s_t) (TD error)

        효과:
        - λ=0: TD(0) - 높은 편향, 낮은 분산
        - λ=1: Monte Carlo - 낮은 편향, 높은 분산
        - λ=0.95: 적절한 균형
        """
        batch_size = rewards.size(0)
        advantages = torch.zeros(batch_size, 1, device=DEVICE)
        last_gae = 0

        # 역순으로 계산 (시간 t부터 0까지)
        for t in reversed(range(batch_size)):
            non_terminal = 1.0 - dones[t]
            # TD error 계산
            delta = rewards[t] + GAMMA * next_values[t] * non_terminal - values[t]
            # GAE 누적
            last_gae = delta + GAMMA * GAE_LAMBDA * non_terminal * last_gae
            advantages[t] = last_gae

        # Returns = Advantage + Value
        returns = advantages + values
        # Advantage 정규화 (평균=0, 분산=1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def train_net(self):
        """
        PPO 알고리즘으로 네트워크 학습

        단계:
        1. 배치 데이터 준비
        2. 보상 정규화
        3. GAE로 어드밴티지 계산
        4. K_EPOCHS만큼 정책 업데이트
        5. CRITIC_EXTRA_EPOCHS만큼 추가 Critic 학습
        """
        if len(self.data) < BATCH_SIZE:
            return

        # Step 1: 배치 준비
        (states, stone_indices, actions, rewards, next_states,
         old_log_probs_stone, old_log_probs_action, dones,
         stone_masks, old_values) = self.make_batch()

        # Step 2: 보상 정규화
        rewards_np = rewards.cpu().numpy().flatten()
        self.reward_rms.update(rewards_np)
        reward_std = np.sqrt(self.reward_rms.var + 1e-8)
        rewards = rewards / reward_std

        # Step 3: GAE 계산 (그래디언트 불필요)
        with torch.inference_mode():
            _, _, _, values = self.model(states, stone_masks)
            _, _, _, next_values = self.model(next_states, stone_masks)

        advantages, returns = self.compute_gae(rewards, values, next_values, dones)

        # Step 4: PPO 업데이트 (K_EPOCHS)
        total_samples = states.size(0)
        indices = np.arange(total_samples)

        for _ in range(K_EPOCHS):
            np.random.shuffle(indices)  # 미니배치 셔플
            for start in range(0, total_samples, BATCH_SIZE):
                idx = indices[start:start + BATCH_SIZE]

                if USE_AMP:
                    # Mixed Precision Training (GPU)
                    with autocast(device_type='cuda', dtype=torch.bfloat16):
                        loss = self._compute_loss(
                            states[idx], stone_indices[idx], actions[idx],
                            old_log_probs_stone[idx], old_log_probs_action[idx],
                            advantages[idx], returns[idx], old_values[idx], stone_masks[idx]
                        )
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # 일반 학습 (CPU)
                    loss = self._compute_loss(
                        states[idx], stone_indices[idx], actions[idx],
                        old_log_probs_stone[idx], old_log_probs_action[idx],
                        advantages[idx], returns[idx], old_values[idx], stone_masks[idx]
                    )
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                    self.optimizer.step()

        # Step 5: Critic 추가 학습
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
        """
        PPO 손실 함수 계산

        Loss = Policy Loss + Value Loss - Entropy Bonus

        Policy Loss (클리핑):
        - ratio = π_new / π_old
        - L_CLIP = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)

        Value Loss (클리핑):
        - 기존 값에서 너무 멀리 벗어나지 않도록 제한

        Entropy Bonus:
        - 탐험 장려 (확률 분포가 고르면 높음)
        """
        # 현재 정책의 출력
        stone_logits, action_mu, action_std, values = self.model(states, stone_masks)

        # 새 로그 확률 계산
        stone_dist = Categorical(logits=stone_logits)
        new_log_probs_stone = stone_dist.log_prob(stone_indices)

        action_dist = Normal(action_mu, action_std)
        new_log_probs_action = action_dist.log_prob(actions).sum(dim=1)

        # 확률 비율 계산 (중요도 샘플링)
        old_log_probs = old_log_probs_stone + old_log_probs_action
        new_log_probs = new_log_probs_stone + new_log_probs_action
        ratio = torch.exp(new_log_probs - old_log_probs)

        # PPO 클리핑 손실
        surr1 = ratio * advantages.squeeze()
        surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages.squeeze()
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value 클리핑 손실 (PPO2 스타일)
        values_clipped = old_values + torch.clamp(values - old_values, -EPS_CLIP, EPS_CLIP)
        value_loss_1 = (values - returns) ** 2
        value_loss_2 = (values_clipped - returns) ** 2
        value_loss = 0.5 * torch.max(value_loss_1, value_loss_2).mean()

        # 엔트로피 보너스 (탐험 장려)
        entropy = stone_dist.entropy().mean() + action_dist.entropy().sum(dim=1).mean()

        # 총 손실
        loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
        return loss

    def _compute_value_loss(self, states, returns, old_values):
        """Critic 추가 학습용 Value 손실"""
        _, _, _, values = self.model(states, None)
        values_clipped = old_values + torch.clamp(values - old_values, -EPS_CLIP, EPS_CLIP)
        value_loss_1 = (values - returns) ** 2
        value_loss_2 = (values_clipped - returns) ** 2
        value_loss = 0.5 * torch.max(value_loss_1, value_loss_2).mean()
        return value_loss

    def act(self, observation: Any, info: Dict) -> Dict:
        """평가/실전용 행동 결정 (deterministic)"""
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
# 6. 에이전트 래퍼 (흑/백 전용)
# ==============================================================================

class YourBlackAgent(BaseAgent):
    """흑돌(선공) 에이전트"""
    def __init__(self):
        super().__init__(my_turn=0)

    @classmethod
    def load(cls, path: str) -> 'YourBlackAgent':
        """체크포인트에서 에이전트 로드"""
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
        """에이전트를 체크포인트로 저장"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'obs_rms': self.obs_rms.state_dict(),
            'reward_rms': self.reward_rms.state_dict(),
            'update_step': update_step
        }
        torch.save(checkpoint, path)


class YourWhiteAgent(BaseAgent):
    """백돌(후공) 에이전트"""
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


# ==============================================================================
# 7. 상대 관리자 (단순 Self-Play)
# ==============================================================================

class OpponentManager:
    """
    상대 관리자 (단순 Self-Play)

    역할:
    1. 최근 N개 모델 풀 관리
    2. 커리큘럼: 초반 RULEBOT_PHASE까지 RuleBot → 이후 100% Self-Play
    3. 모델 풀에서 랜덤 선택

    Self-Play 원리:
    - "과거의 나"를 상대로 학습
    - 이기면 성장한 것 → 새 모델 저장
    - 풀에서 랜덤 선택으로 다양한 전략 경험
    """
    def __init__(self):
        self.save_dir = HISTORY_DIR
        os.makedirs(self.save_dir, exist_ok=True)

        # 과거 모델 풀 (최근 MODEL_POOL_SIZE개만 유지)
        self.pool: List[str] = []
        self._load_existing_models()

        # 규칙 기반 상대 (초반 학습용)
        self.rule_based = RuleBasedAgent(my_turn=1)

        # 현재 상대 정보
        self.current_opponent_name = "RuleBot"

    def _load_existing_models(self):
        """기존 모델 파일 로드 (최근 것만)"""
        existing = glob.glob(os.path.join(self.save_dir, "model_*.pkl"))
        # 스텝 번호로 정렬
        existing.sort(key=lambda x: int(x.split("_")[-1].replace(".pkl", "")))
        # 최근 MODEL_POOL_SIZE개만 유지
        self.pool = existing[-MODEL_POOL_SIZE:]

    def save_model(self, model, step: int):
        """
        모델을 풀에 저장

        - 새 모델 추가
        - 풀 크기 초과 시 가장 오래된 것 삭제
        """
        path = os.path.join(self.save_dir, f"model_{step}.pkl")
        torch.save(model.state_dict(), path)

        if path not in self.pool:
            self.pool.append(path)

        # 풀 크기 제한 (오래된 것 삭제)
        while len(self.pool) > MODEL_POOL_SIZE:
            old_path = self.pool.pop(0)
            if os.path.exists(old_path):
                os.remove(old_path)

    def get_opponent(self, update: int) -> Tuple[Any, str]:
        """
        상대 선택

        커리큘럼:
        - update <= RULEBOT_PHASE: 100% RuleBot (기본기 학습)
        - update > RULEBOT_PHASE: 100% Self-Play (과거의 나와 대결)

        Args:
            update: 현재 업데이트 번호

        Returns:
            opponent: 상대 에이전트
            name: 상대 이름 (로그용)
        """
        # 초반: RuleBot으로 기본기 학습
        if update <= RULEBOT_PHASE or len(self.pool) == 0:
            self.current_opponent_name = "RuleBot"
            return self.rule_based, "RuleBot"

        # 이후: Self-Play (모델 풀에서 랜덤 선택)
        opponent_path = random.choice(self.pool)

        # 과거 모델 로드
        opponent = YourBlackAgent()
        opponent.my_turn = 1  # 상대는 백돌
        try:
            opponent.model.load_state_dict(torch.load(opponent_path, map_location=DEVICE))
            opponent.model.eval()

            # 모델 버전 추출 (model_500.pkl → v500)
            version = opponent_path.split("_")[-1].replace(".pkl", "")
            self.current_opponent_name = f"Self-v{version}"
            return opponent, self.current_opponent_name
        except Exception as e:
            print(f"[Warning] Failed to load {opponent_path}: {e}")
            self.current_opponent_name = "RuleBot"
            return self.rule_based, "RuleBot"


# ==============================================================================
# 8. 보상 함수
# ==============================================================================

def make_env():
    """환경 생성 함수"""
    return gym.make(id='kymnasium/AlKkaGi-3x3-v0', render_mode=None, bgm=False, obs_type='custom')


def compute_reward(obs, next_obs, my_idx, term, trunc):
    """
    보상 계산

    보상 구성:
    - 적 처치: +50 (적 돌 하나 죽일 때마다)
    - 아군 사망: -30 (내 돌 하나 죽을 때마다)
    - 밀어내기: +0.05 * 거리 (적을 경계로 밀수록)
    - 승리: +100
    - 패배: -50
    - 스텝 페널티: -0.1 (빠른 결판 유도)
    """
    batch_size = len(my_idx)
    rewards = np.zeros(batch_size, dtype=np.float32)

    for i, env_i in enumerate(my_idx):
        my_before = obs['black'][env_i]
        my_after = next_obs['black'][env_i]
        enemy_before = obs['white'][env_i]
        enemy_after = next_obs['white'][env_i]

        # 생존 돌 수 변화
        my_alive_before = np.sum(my_before[:, 2])
        my_alive_after = np.sum(my_after[:, 2])
        enemy_alive_before = np.sum(enemy_before[:, 2])
        enemy_alive_after = np.sum(enemy_after[:, 2])

        # 처치/사망 보상
        kills = enemy_alive_before - enemy_alive_after
        deaths = my_alive_before - my_alive_after
        rewards[i] += kills * 50.0   # 적 처치 보상
        rewards[i] -= deaths * 30.0  # 아군 사망 페널티

        # 밀어내기 보상 (살아있는 적이 경계에 가까워지면)
        for j in range(3):
            if enemy_before[j, 2] == 1 and enemy_after[j, 2] == 1:
                # 경계까지의 최소 거리 계산
                prev_edge = min(enemy_before[j, 0], enemy_before[j, 1],
                                BOARD_SIZE - enemy_before[j, 0], BOARD_SIZE - enemy_before[j, 1])
                curr_edge = min(enemy_after[j, 0], enemy_after[j, 1],
                                BOARD_SIZE - enemy_after[j, 0], BOARD_SIZE - enemy_after[j, 1])
                push_reward = (prev_edge - curr_edge) * 0.05
                rewards[i] += max(0, push_reward)  # 양수만 (밀어낼 때만)

        # 게임 종료 보상
        done = term[env_i] or trunc[env_i]
        if done:
            if enemy_alive_after == 0:
                rewards[i] += 100.0  # 승리
            elif my_alive_after == 0:
                rewards[i] -= 50.0   # 패배

        # 스텝 페널티 (빠른 결판 유도)
        rewards[i] -= 0.1

    return rewards


# ==============================================================================
# 9. 메인 학습 루프
# ==============================================================================

def train():
    """
    PPO 학습 메인 루프

    흐름:
    1. 환경/에이전트 초기화
    2. 체크포인트 로드 (있으면)
    3. 학습 루프:
       a. 상대 선택 (Curriculum)
       b. T_HORIZON 스텝 경험 수집
       c. PPO 업데이트
       d. 로깅 및 저장
    4. 최종 저장
    """
    global SHUTDOWN_FLAG

    # Ctrl+C 핸들러 (메인 프로세스만)
    def _signal_handler(signum, frame):
        global SHUTDOWN_FLAG
        if SHUTDOWN_FLAG:
            print("\n\n[!!] Force exit!")
            sys.exit(1)
        print("\n\n[!] Ctrl+C detected. Saving and exiting after current update...")
        SHUTDOWN_FLAG = True

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # ==================== 초기화 ====================
    print("\n" + "=" * 70)
    print("  AL-KKA-GI PPO TRAINING")
    print("=" * 70)
    print(f"  Device      : {DEVICE}")
    print(f"  Environments: {NUM_ENVS}")
    print(f"  Batch Size  : {BATCH_SIZE}")
    print(f"  T Horizon   : {T_HORIZON}")
    print(f"  Max Updates : {MAX_UPDATES}")
    print(f"  LR          : {LR_START} -> {LR_END}")
    print("-" * 70)
    print(f"  [Optimizations]")
    print(f"  AMP (Mixed Precision) : {USE_AMP}")
    print(f"  torch.compile         : {USE_COMPILE}")
    print(f"  cudnn.benchmark       : {torch.backends.cudnn.benchmark if torch.cuda.is_available() else 'N/A (CPU)'}")
    print(f"  inference_mode        : True")
    print(f"  Tensor pre-allocation : True")
    print("-" * 70)
    print(f"  [Self-Play Curriculum]")
    print(f"  RuleBot Phase : 1 ~ {RULEBOT_PHASE} updates")
    print(f"  Self-Play     : {RULEBOT_PHASE+1} ~ {MAX_UPDATES} updates")
    print(f"  Model Pool    : {MODEL_POOL_SIZE} recent models")
    print("=" * 70 + "\n")

    # 병렬 환경 생성
    envs = AsyncVectorEnv([make_env for _ in range(NUM_ENVS)])

    # 에이전트 생성
    agent = YourBlackAgent()
    start_update = 1

    # 체크포인트 로드
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

    # 상대 관리자
    op_manager = OpponentManager()

    # 초기 모델 저장
    if start_update == 1:
        op_manager.save_model(agent.model, 0)

    # 환경 리셋
    obs, _ = envs.reset()

    # ==================== 통계 추적 ====================
    score_history = []      # 점수 히스토리
    win_history = []        # 승패 히스토리
    kill_history = []       # 킬 히스토리
    win_count = 0           # 현재 주기 승리 수
    game_count = 0          # 현재 주기 게임 수
    total_kills = 0         # 총 킬 수
    total_deaths = 0        # 총 데스 수
    best_win_rate = 0.0     # 최고 승률
    start_time = time.time()
    update_times = []       # 업데이트 소요 시간

    # 초기 상대 선택
    opponent, opponent_name = op_manager.get_opponent(start_update)

    # 로그 헤더
    print("-" * 70)
    print(f"{'Upd':>6} | {'Score':>8} | {'Win%':>6} | {'K/D':>11} | {'Opp':>15} | {'Time':>8}")
    print("-" * 70)

    # ==================== 메인 학습 루프 ====================
    for update in range(start_update, MAX_UPDATES + 1):
        if SHUTDOWN_FLAG:
            break

        update_start = time.time()
        progress = update / MAX_UPDATES

        # 학습률 스케줄링 (선형 감소)
        lr = LR_START - (LR_START - LR_END) * progress
        for param_group in agent.optimizer.param_groups:
            param_group['lr'] = lr

        # 주기적 모델 저장
        if update % SAVE_INTERVAL == 0:
            op_manager.save_model(agent.model, update)

        # 상대 교체
        if update % SWAP_INTERVAL == 0:
            opponent, opponent_name = op_manager.get_opponent(update)

        update_kills = 0
        update_deaths = 0

        # ========== 경험 수집 (T_HORIZON 스텝) ==========
        for _ in range(T_HORIZON):
            turns = obs['turn']

            # --- 내 턴인 환경 처리 ---
            my_idx = np.where(turns == 0)[0]
            my_actions = []

            if len(my_idx) > 0:
                # 관측 전처리
                obs_me = {k: v[my_idx] for k, v in obs.items()}
                obs_np, stone_mask = agent._process_obs(obs_me, override_turn=0)
                obs_np = agent._normalize_obs(obs_np, update=True)

                # 행동 선택
                stone_idx, action, log_p_stone, log_p_action = agent.get_action(
                    obs_np, stone_mask, deterministic=False
                )
                my_actions = agent.decode_action(stone_idx, action, obs_me, 0)

                # 가치 추정 (경험 저장용)
                with torch.inference_mode():
                    obs_tensor = torch.tensor(obs_np, dtype=torch.float32).to(DEVICE)
                    _, _, _, values = agent.model(obs_tensor, None)
                values_np = values.cpu().numpy().flatten()

            # --- 상대 턴인 환경 처리 ---
            op_idx = np.where(turns == 1)[0]
            op_actions = []

            if len(op_idx) > 0:
                obs_op = {k: v[op_idx] for k, v in obs.items()}
                if isinstance(opponent, RuleBasedAgent):
                    # RuleBot
                    for i in range(len(op_idx)):
                        single_obs = {k: v[i] for k, v in obs_op.items()}
                        op_actions.append(opponent.act(single_obs))
                else:
                    # Self-Play 상대
                    obs_np_op, mask_op = opponent._process_obs(obs_op, override_turn=1)
                    obs_np_op = opponent._normalize_obs(obs_np_op, update=False)
                    si, ac, _, _ = opponent.get_action(obs_np_op, mask_op, deterministic=True)
                    op_actions = opponent.decode_action(si, ac, obs_op, 1)

            # --- 행동 병합 및 환경 스텝 ---
            action_list = [None] * NUM_ENVS
            for i, env_i in enumerate(my_idx):
                action_list[env_i] = my_actions[i]
            for i, env_i in enumerate(op_idx):
                action_list[env_i] = op_actions[i]

            batched_action = {key: np.array([d[key] for d in action_list]) for key in action_list[0].keys()}
            next_obs, _, term, trunc, _ = envs.step(batched_action)

            # --- 내 턴 경험 저장 ---
            if len(my_idx) > 0:
                rewards = compute_reward(obs, next_obs, my_idx, term, trunc)

                # K/D 통계 (리셋 프레임 제외, 양수만)
                for i, env_i in enumerate(my_idx):
                    if not (term[env_i] or trunc[env_i]):
                        my_before = np.sum(obs['black'][env_i, :, 2])
                        my_after = np.sum(next_obs['black'][env_i, :, 2])
                        enemy_before = np.sum(obs['white'][env_i, :, 2])
                        enemy_after = np.sum(next_obs['white'][env_i, :, 2])
                        kills = int(enemy_before - enemy_after)
                        deaths = int(my_before - my_after)
                        if kills > 0:
                            update_kills += kills
                        if deaths > 0:
                            update_deaths += deaths

                # 다음 상태 전처리
                next_obs_me = {k: v[my_idx] for k, v in next_obs.items()}
                next_obs_np, _ = agent._process_obs(next_obs_me, override_turn=0)
                next_obs_np = agent._normalize_obs(next_obs_np, update=False)

                # 경험 저장 및 승패 기록
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

        # ========== 통계 업데이트 ==========
        total_kills += update_kills
        total_deaths += update_deaths
        kill_history.append(update_kills)
        update_time = time.time() - update_start
        update_times.append(update_time)

        # ========== PPO 학습 ==========
        agent.train_net()

        # ========== 로깅 (10 업데이트마다) ==========
        if update % 10 == 0:
            avg_score = np.mean(score_history[-1000:]) if score_history else 0
            recent_win_rate = np.mean(win_history[-100:]) * 100 if win_history else 0
            session_win_rate = (win_count / game_count * 100) if game_count > 0 else 0
            kd_ratio = total_kills / max(total_deaths, 1)
            avg_update_time = np.mean(update_times[-100:])
            elapsed = time.time() - start_time
            eta = (MAX_UPDATES - update) * avg_update_time

            if recent_win_rate > best_win_rate:
                best_win_rate = recent_win_rate

            elapsed_str = str(timedelta(seconds=int(elapsed)))
            eta_str = str(timedelta(seconds=int(eta)))

            print(f"{update:6d} | {avg_score:8.2f} | {recent_win_rate:5.1f}% | "
                  f"{total_kills:5d}/{total_deaths:<5d} | {opponent_name:>15} | {elapsed_str}")

            agent.save(SAVE_PATH, update)

            # 상세 통계 (100 업데이트마다)
            if update % 100 == 0:
                print("-" * 70)
                print(f"  [Stats] Progress: {progress*100:.1f}% | Best WR: {best_win_rate:.1f}% | "
                      f"LR: {lr:.2e} | ETA: {eta_str}")
                print("-" * 70)

            win_count = 0
            game_count = 0

    # ==================== 학습 종료 ====================
    print("\n" + "=" * 70)
    if SHUTDOWN_FLAG:
        print("  TRAINING INTERRUPTED - Saving checkpoint...")
    else:
        print("  TRAINING COMPLETED!")
    print("=" * 70)

    agent.save(SAVE_PATH, update)

    total_time = time.time() - start_time
    print(f"  Total Time    : {str(timedelta(seconds=int(total_time)))}")
    print(f"  Best Win Rate : {best_win_rate:.1f}%")
    print(f"  Total K/D     : {total_kills}/{total_deaths} ({total_kills/max(total_deaths,1):.2f})")
    print(f"  Model Pool    : {len(op_manager.pool)} models")
    print(f"  Checkpoint    : {SAVE_PATH}")
    print("=" * 70 + "\n")

    envs.close()


# ==============================================================================
# 실행
# ==============================================================================

if __name__ == "__main__":
    train()
