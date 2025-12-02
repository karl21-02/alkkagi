import os
os.environ['SDL_AUDIODRIVER'] = 'dummy'  # ALSA 경고 제거

import gymnasium as gym
import kymnasium as kym
import numpy as np
import glob
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.amp import autocast, GradScaler
from gymnasium.vector import AsyncVectorEnv

# ==========================================
# 1. Hyperparameters
# ==========================================
NUM_ENVS = 64  # GPU 최적화: 환경 병렬화 증가
LR_ACTOR = 0.0002  # 배치 증가에 따른 LR 조정
GAMMA = 0.99
K_EPOCHS = 3  # 배치 증가 보상
EPS_CLIP = 0.2
T_HORIZON = 512  # GPU 최적화: 더 많은 경험 수집
BATCH_SIZE = 2048  # GPU 최적화: 배치 크기 증가
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "my_alkkagi_agent.pkl"

# Self-Play 설정 (개선됨)
SELFPLAY_SAVE_INTERVAL = 50     # 더 자주 저장
SELFPLAY_SWAP_INTERVAL = 50     # 저장과 동일
SELFPLAY_START_WINRATE = 60.0   # 더 빨리 시작

# GAE & Entropy 설정
GAE_LAMBDA = 0.95
ENTROPY_COEF = 0.01
BOARD_SIZE = 600.0  # 실제 게임 보드 크기


# ==========================================
# MuJoCo 스타일: Running Mean/Std 정규화
# ==========================================
class RunningMeanStd:
    """Welford's online algorithm for running mean/std"""
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

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


# ==========================================
# 벽 충돌 체크 유틸리티
# ==========================================
def line_intersects_rect(p1, p2, rect):
    """
    선분(p1→p2)이 사각형(rect)과 교차하는지 확인
    rect = [x, y, width, height] (x, y는 중심 좌표)
    """
    x, y, w, h = float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3])
    # 사각형의 경계
    left = x - w / 2
    right = x + w / 2
    top = y - h / 2
    bottom = y + h / 2

    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])

    # 선분의 방향
    dx = x2 - x1
    dy = y2 - y1

    # 파라메트릭 t 범위 [0, 1]에서 교차점 찾기 (Liang-Barsky)
    t0, t1 = 0.0, 1.0

    # (edge, p, q) 튜플만 사용 - 문자열 제거
    checks = [
        (-dx, x1 - left, left - x1),      # left
        (dx, right - x1, right - x1),     # right
        (-dy, y1 - top, top - y1),        # top
        (dy, bottom - y1, bottom - y1)    # bottom
    ]

    for edge, p, q in checks:
        if edge == 0:
            if p < 0:
                return True  # 선분이 사각형 밖에 평행
        else:
            t = p / edge
            if edge < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)

    return t0 <= t1


def is_path_blocked(p1, p2, obstacles):
    """
    p1에서 p2로 가는 경로가 장애물에 막히는지 확인
    obstacles: [[x, y, w, h], ...]
    """
    for obs in obstacles:
        if line_intersects_rect(p1, p2, obs):
            return True
    return False


def get_wall_avoidance_angle(my_pos, target_pos, obstacles):
    """
    벽을 피하기 위한 각도 오프셋 계산
    반환: 권장 각도 오프셋 (라디안)
    """
    if not is_path_blocked(my_pos, target_pos, obstacles):
        return 0.0  # 막히지 않음

    # 좌우로 조금씩 각도를 틀어서 막히지 않는 경로 찾기
    base_angle = np.arctan2(target_pos[1] - my_pos[1], target_pos[0] - my_pos[0])
    dist = np.linalg.norm(target_pos - my_pos)

    for offset in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:  # 라디안
        for sign in [1, -1]:
            test_angle = base_angle + sign * offset
            test_target = my_pos + np.array([np.cos(test_angle), np.sin(test_angle)]) * dist
            if not is_path_blocked(my_pos, test_target, obstacles):
                return sign * offset

    return 0.0  # 피할 방법 없음


def is_protected_by_wall(my_pos, enemy_positions, obstacles):
    """
    내 돌이 벽으로 보호받는 정도 계산
    적이 나를 공격하려면 벽을 통과해야 하면 = 보호됨
    반환: 0.0 ~ 1.0 (보호받는 적의 비율)
    """
    if len(obstacles) == 0 or len(enemy_positions) == 0:
        return 0.0

    protected_count = 0
    for enemy_pos in enemy_positions:
        # 적 → 나 경로가 벽에 막히면 보호됨
        if is_path_blocked(enemy_pos, my_pos, obstacles):
            protected_count += 1

    return protected_count / len(enemy_positions)


def calculate_wall_collision(stones, obstacles, radius=15.0):
    """
    돌이 벽과 충돌하는지 확인
    stones: (N, 3) 배열 [x, y, alive]
    obstacles: [[x, y, w, h], ...] 벽 리스트
    radius: 돌의 반지름
    반환: (N,) 불리언 배열 (충돌 여부)
    """
    if len(obstacles) == 0:
        return np.zeros(len(stones), dtype=bool)

    collisions = np.zeros(len(stones), dtype=bool)

    for i, stone in enumerate(stones):
        if stone[2] == 0:  # 죽은 돌은 무시
            continue

        sx, sy = stone[0], stone[1]

        for obs in obstacles:
            ox, oy, ow, oh = obs[0], obs[1], obs[2], obs[3]
            # 벽 경계
            left = ox - ow / 2
            right = ox + ow / 2
            top = oy - oh / 2
            bottom = oy + oh / 2

            # 돌 중심에서 벽까지 최단 거리
            closest_x = np.clip(sx, left, right)
            closest_y = np.clip(sy, top, bottom)
            dist = np.sqrt((sx - closest_x) ** 2 + (sy - closest_y) ** 2)

            if dist < radius:
                collisions[i] = True
                break

    return collisions


# ==========================================
# 2. Neural Network (The Brain)
# ==========================================
class SniperNet(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(SniperNet, self).__init__()
        # 입력: [거리, 내_생존, 적_생존]
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh()
        )
        self.score_head = nn.Linear(64, 1)

        # Action: [Angle_Offset, Power]
        self.mu_head = nn.Linear(64, action_dim)
        # MuJoCo 스타일: State-Dependent Std (상태에 따른 적응적 탐색)
        self.log_std_head = nn.Linear(64, action_dim)
        # 초기 std를 적당한 값으로 설정 (exp(-0.5) ≈ 0.6)
        nn.init.constant_(self.log_std_head.weight, 0.0)
        nn.init.constant_(self.log_std_head.bias, -0.5)

        self.critic_head = nn.Sequential(
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x, alive_mask):
        batch_size = x.size(0)
        flat_x = x.view(-1, x.size(-1))
        features = self.encoder(flat_x)

        scores = self.score_head(features).view(batch_size, 3)
        scores = scores.masked_fill(alive_mask == 0, -1e9)

        mu = self.mu_head(features).view(batch_size, 3, -1)
        # MuJoCo 스타일: State-Dependent Std
        log_std = self.log_std_head(features).view(batch_size, 3, -1)
        log_std = torch.clamp(log_std, -20, 2)  # 안정성을 위한 클램핑
        std = log_std.exp()
        return scores, mu, std, features.view(batch_size, 3, -1)

    def get_value(self, x):
        batch_size = x.size(0)
        flat_x = x.view(-1, x.size(-1))
        features = self.encoder(flat_x).view(batch_size, 3, -1)
        global_features, _ = torch.max(features, dim=1)
        return self.critic_head(global_features)


# ==========================================
# 3. Agent Logic (공통 로직)
# ==========================================
class BaseAgent(kym.Agent):
    def __init__(self, my_turn):
        super().__init__()
        self.my_turn = my_turn  # 0: Black, 1: White
        self.data = []
        self.input_dim = 30  # 벽 정보 6차원 추가 (24 + 6)
        self.action_dim = 2
        self.model = SniperNet(self.input_dim, self.action_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR_ACTOR)
        # Learning Rate 스케줄러 (500 스텝마다 0.95배 감소)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.95)
        # MuJoCo 스타일: Observation & Reward Normalization
        self.obs_rms = RunningMeanStd(shape=(3, self.input_dim))  # 관측값 정규화
        self.reward_rms = RunningMeanStd(shape=())  # 보상 정규화
        # Mixed Precision Training (AMP)
        self.scaler = GradScaler()
        # 캐시된 인덱스 텐서 (반복 생성 방지)
        self._batch_indices_cache = torch.arange(BATCH_SIZE, device=DEVICE)

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_idx_lst, a_val_lst, r_lst, s_prime_lst, prob_idx_lst, prob_val_lst, done_lst, mask_lst = [], [], [], [], [], [], [], [], []
        for item in self.data:
            s, a_idx, a_val, r, s_p, p_idx, p_val, done, mask = item
            s_lst.append(s);
            a_idx_lst.append(a_idx);
            a_val_lst.append(a_val)
            r_lst.append([r]);
            s_prime_lst.append(s_p)
            prob_idx_lst.append(p_idx);
            prob_val_lst.append(p_val)
            done_lst.append([done]);
            mask_lst.append(mask)
        self.data = []
        s = torch.stack(s_lst);
        a_idx = torch.stack(a_idx_lst);
        a_val = torch.stack(a_val_lst)
        s_prime = torch.stack(s_prime_lst);
        prob_idx = torch.stack(prob_idx_lst);
        prob_val = torch.stack(prob_val_lst)
        mask = torch.stack(mask_lst)
        r = torch.tensor(r_lst, dtype=torch.float).to(DEVICE)
        done = torch.tensor(done_lst, dtype=torch.float).to(DEVICE)
        return s, a_idx, a_val, r, s_prime, prob_idx, prob_val, done, mask

    # [중요] 흑/백 구분 없이 '나'와 '적'으로 변환하여 인식
    def _process_obs(self, obs, override_turn=None, training=True):
        """
        확장된 관측 공간 (30차원 × 3돌) - 벽 정보 추가
        [0-5]   내 돌: 경계위험도, 위치x, 위치y, 적수, 내수, 벽거리
        [6-11]  적1: 생존, 거리, 경계근접도, 방향x, 방향y, 막힘여부
        [12-17] 적2: 생존, 거리, 경계근접도, 방향x, 방향y, 막힘여부
        [18-23] 적3: 생존, 거리, 경계근접도, 방향x, 방향y, 막힘여부
        [24-29] 벽: 존재여부, 중심x, 중심y, 너비, 높이, 내돌→벽방향
        """
        batch_size = len(obs['black'])

        if override_turn is not None:
            turns = np.full((batch_size, 1, 1), override_turn)
        else:
            turns = obs['turn'].reshape(batch_size, 1, 1)

        black = obs['black']
        white = obs['white']
        my_stones = np.where(turns == 0, black, white)
        op_stones = np.where(turns == 0, white, black)

        obstacles = obs.get('obstacles', None)

        processed = np.zeros((batch_size, 3, 30), dtype=np.float32)
        alive_mask = np.zeros((batch_size, 3), dtype=np.float32)

        op_xy = op_stones[:, :, 0:2]
        op_alive = op_stones[:, :, 2]
        my_alive = my_stones[:, :, 2]

        for b in range(batch_size):
            my_s = my_stones[b]
            alive_mask[b] = my_s[:, 2]

            my_count = np.sum(my_alive[b])
            op_count = np.sum(op_alive[b])

            batch_obstacles = obstacles[b] if obstacles is not None and len(obstacles) > b else []
            all_op_pos = op_xy[b]
            all_op_alive = op_alive[b]

            for i in range(3):
                if my_s[i, 2] == 0:
                    continue

                my_pos = my_s[i, 0:2]

                # 내 경계 위험도
                my_edge_dist = min(my_pos[0], BOARD_SIZE - my_pos[0],
                                   my_pos[1], BOARD_SIZE - my_pos[1])
                my_edge_danger = 1.0 - np.clip(my_edge_dist / (BOARD_SIZE / 2), 0, 1)

                # 벽 거리
                wall_dist = BOARD_SIZE
                if len(batch_obstacles) > 0:
                    wall_centers = np.array([[w[0], w[1]] for w in batch_obstacles])
                    wall_dists = np.sqrt(np.sum((wall_centers - my_pos) ** 2, axis=1))
                    wall_dist = np.min(wall_dists)

                # [0-5] 내 돌 정보
                processed[b, i, 0] = my_edge_danger
                processed[b, i, 1] = my_pos[0] / BOARD_SIZE
                processed[b, i, 2] = my_pos[1] / BOARD_SIZE
                processed[b, i, 3] = op_count / 3.0
                processed[b, i, 4] = my_count / 3.0
                processed[b, i, 5] = wall_dist / BOARD_SIZE

                # [6-23] 적 3명 정보 (각 6차원)
                for enemy_idx in range(3):
                    base_idx = 6 + enemy_idx * 6

                    if all_op_alive[enemy_idx] == 0:
                        processed[b, i, base_idx:base_idx+6] = 0.0
                    else:
                        enemy_pos = all_op_pos[enemy_idx]
                        dist = np.sqrt(np.sum((enemy_pos - my_pos) ** 2))

                        enemy_edge_dist = min(enemy_pos[0], BOARD_SIZE - enemy_pos[0],
                                              enemy_pos[1], BOARD_SIZE - enemy_pos[1])
                        enemy_edge = 1.0 - np.clip(enemy_edge_dist / (BOARD_SIZE / 2), 0, 1)

                        diff = enemy_pos - my_pos
                        norm = np.linalg.norm(diff) + 1e-6
                        dir_x, dir_y = diff[0] / norm, diff[1] / norm

                        is_blocked = 0.0
                        if len(batch_obstacles) > 0 and is_path_blocked(my_pos, enemy_pos, batch_obstacles):
                            is_blocked = 1.0

                        processed[b, i, base_idx + 0] = 1.0                    # 생존 여부
                        processed[b, i, base_idx + 1] = dist / BOARD_SIZE     # 거리
                        processed[b, i, base_idx + 2] = enemy_edge            # 경계 근접도
                        processed[b, i, base_idx + 3] = dir_x                 # 방향 x
                        processed[b, i, base_idx + 4] = dir_y                 # 방향 y
                        processed[b, i, base_idx + 5] = is_blocked            # 막힘 여부

                # [24-29] 벽 정보 (6차원)
                if len(batch_obstacles) > 0:
                    # 가장 가까운 벽 선택
                    wall_centers = np.array([[w[0], w[1]] for w in batch_obstacles])
                    wall_dists = np.sqrt(np.sum((wall_centers - my_pos) ** 2, axis=1))
                    nearest_idx = np.argmin(wall_dists)
                    nearest_wall = batch_obstacles[nearest_idx]

                    wall_x, wall_y = nearest_wall[0], nearest_wall[1]
                    wall_w, wall_h = nearest_wall[2], nearest_wall[3]

                    # 내 돌에서 벽 방향
                    wall_diff = np.array([wall_x, wall_y]) - my_pos
                    wall_norm = np.linalg.norm(wall_diff) + 1e-6
                    wall_dir = wall_diff[0] / wall_norm  # x 방향만 (단순화)

                    processed[b, i, 24] = 1.0                          # 벽 존재
                    processed[b, i, 25] = wall_x / BOARD_SIZE          # 벽 중심 x
                    processed[b, i, 26] = wall_y / BOARD_SIZE          # 벽 중심 y
                    processed[b, i, 27] = wall_w / BOARD_SIZE          # 벽 너비
                    processed[b, i, 28] = wall_h / BOARD_SIZE          # 벽 높이
                    processed[b, i, 29] = wall_dir                     # 내돌→벽 방향
                else:
                    processed[b, i, 24:30] = 0.0  # 벽 없음

        # MuJoCo 스타일: Observation Normalization
        if training:
            self.obs_rms.update(processed)
        processed = self.obs_rms.normalize(processed)

        return torch.tensor(processed, dtype=torch.float32).to(DEVICE), torch.tensor(alive_mask, dtype=torch.float32).to(DEVICE)

    def get_action(self, s, mask, deterministic=False):
        scores, mu, std, _ = self.model(s, mask)
        dist_idx = Categorical(logits=scores)
        if deterministic:
            idx = torch.argmax(scores, dim=1)
        else:
            idx = dist_idx.sample()
        log_prob_idx = dist_idx.log_prob(idx)

        batch_indices = torch.arange(s.size(0), device=DEVICE)
        sel_mu = mu[batch_indices, idx]
        sel_std = std[batch_indices, idx]
        dist_val = Normal(sel_mu, sel_std)
        if deterministic:
            val = sel_mu
        else:
            val = dist_val.rsample()
        log_prob_val = dist_val.log_prob(val).sum(dim=1)
        return idx, val, log_prob_idx, log_prob_val

    # [중요] 보정 사격 로직 - 벽 회피 포함
    def decode_action_with_assist(self, idx, val, obs, my_turn):
        actions = []
        idx_np = idx.cpu().numpy()
        val_np = torch.tanh(val).detach().cpu().numpy()

        my_stones = obs['black'] if my_turn == 0 else obs['white']
        op_stones = obs['white'] if my_turn == 0 else obs['black']
        obstacles = obs.get('obstacles', None)

        for i in range(len(idx_np)):
            stone_idx = int(idx_np[i])
            my_pos = my_stones[i, stone_idx, 0:2]

            op_xy = op_stones[i, :, 0:2]
            op_alive = op_stones[i, :, 2]
            valid_ops = op_xy[op_alive == 1]
            if len(valid_ops) == 0:
                valid_ops = np.array([[BOARD_SIZE / 2, BOARD_SIZE / 2]])

            # 벽 정보 가져오기
            batch_obstacles = obstacles[i] if obstacles is not None and len(obstacles) > i else []

            # 벽에 막히지 않은 적 찾기
            unblocked_targets = []
            blocked_targets = []

            for op in valid_ops:
                if len(batch_obstacles) > 0 and is_path_blocked(my_pos, op, batch_obstacles):
                    blocked_targets.append(op)
                else:
                    unblocked_targets.append(op)

            # 우선순위: 막히지 않은 적 > 막힌 적
            if len(unblocked_targets) > 0:
                unblocked_targets = np.array(unblocked_targets)
                dists = np.sum((unblocked_targets - my_pos) ** 2, axis=1)
                target_pos = unblocked_targets[np.argmin(dists)]
                use_avoidance = False
            else:
                # 모든 적이 막혀있음 → 가장 가까운 적 + 회피 각도 적용
                dists = np.sum((valid_ops - my_pos) ** 2, axis=1)
                target_pos = valid_ops[np.argmin(dists)]
                use_avoidance = True

            dx = target_pos[0] - my_pos[0]
            dy = target_pos[1] - my_pos[1]
            ideal_angle = np.degrees(np.arctan2(dy, dx))

            # 벽 회피 각도 계산
            avoidance_offset = 0.0
            if use_avoidance and len(batch_obstacles) > 0:
                avoidance_offset = get_wall_avoidance_angle(my_pos, target_pos, batch_obstacles)
                avoidance_offset = np.degrees(avoidance_offset)  # 라디안 → 도

            # AI의 각도 조절 + 벽 회피 (확장: ±60도)
            angle_offset = val_np[i, 0] * 60.0 + avoidance_offset
            final_angle = ideal_angle + angle_offset

            # 파워 계산: 적→경계 거리 고려 (멀리 밀어내야 할수록 강하게)
            dist = np.sqrt(dx**2 + dy**2)

            # 적이 경계에서 얼마나 먼지 계산
            enemy_to_edge = min(target_pos[0], BOARD_SIZE - target_pos[0],
                                target_pos[1], BOARD_SIZE - target_pos[1])
            # 적이 경계에서 멀수록 더 강하게 (최대 1.8배)
            edge_factor = 1.0 + (enemy_to_edge / BOARD_SIZE) * 0.8

            # 기본 파워 상향 + 경계 팩터 적용
            base_power = (1000 + (dist / BOARD_SIZE) * 1000) * edge_factor
            # 조정 범위 비대칭: 감소 -30%, 증가 +70% (강타 유도)
            adj_val = val_np[i, 1]
            adjustment = adj_val * 0.7 if adj_val > 0 else adj_val * 0.3
            power = np.clip(base_power * (1 + adjustment), 600, 2800)

            actions.append({
                "turn": my_turn, "index": stone_idx,
                "power": float(power), "angle": float(final_angle)
            })
        return actions

    # 실제 게임에서 호출되는 함수
    def act(self, obs, info):
        # 단일 관측 -> 배치 변환
        batch_obs = {
            'black': np.array([obs['black']]),
            'white': np.array([obs['white']]),
            'turn': np.array([obs['turn']])  # 실제 턴 사용
        }
        # self.my_turn을 사용하여 내 관점에서 처리 (추론 시 training=False)
        s, mask = self._process_obs(batch_obs, override_turn=self.my_turn, training=False)
        with torch.no_grad():
            idx, val, _, _ = self.get_action(s, mask, deterministic=True)

        return self.decode_action_with_assist(idx, val, batch_obs, self.my_turn)[0]

    def train_net(self):
        if len(self.data) < 1: return
        s, a_idx, a_val, r, s_p, p_idx, p_val, done, mask = self.make_batch()

        # === MuJoCo 스타일: Reward Normalization (스케일링만, 평균 빼지 않음) ===
        r_np = r.cpu().numpy()
        self.reward_rms.update(r_np.flatten())
        reward_std = np.sqrt(self.reward_rms.var + 1e-8)
        r = r / reward_std  # 텐서 연산 유지

        # === GAE 계산 (GPU 최적화) ===
        with torch.no_grad():
            values = self.model.get_value(s)
            next_values = self.model.get_value(s_p)

            # 델타 한번에 계산 (벡터화)
            next_non_terminal = 1.0 - done
            deltas = r + GAMMA * next_values * next_non_terminal - values

            # GAE 역방향 누적 (GPU 텐서 연산)
            batch_size = s.size(0)
            advantages = torch.zeros(batch_size, 1, device=DEVICE)
            last_gae = torch.zeros(1, device=DEVICE)

            for t in reversed(range(batch_size)):
                last_gae = deltas[t] + GAMMA * GAE_LAMBDA * next_non_terminal[t] * last_gae
                advantages[t] = last_gae

            td_target = advantages + values
            # Advantage 정규화 (학습 안정화)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_samples = s.size(0)
        indices = np.arange(total_samples)

        for _ in range(K_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, total_samples, BATCH_SIZE):
                i = indices[start:start + BATCH_SIZE]

                # Mixed Precision: autocast로 forward pass
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    scores, mu, std, _ = self.model(s[i], mask[i])

                    # Categorical distribution (돌 선택)
                    dist_idx = Categorical(logits=scores)
                    cur_p_idx = dist_idx.log_prob(a_idx[i])
                    ratio_idx = torch.exp(cur_p_idx - p_idx[i])

                    # Continuous distribution (각도/파워)
                    batch_idx_range = self._batch_indices_cache[:len(i)]
                    sel_mu = mu[batch_idx_range, a_idx[i]]
                    sel_std = std[batch_idx_range, a_idx[i]]
                    dist_val = Normal(sel_mu, sel_std)
                    cur_p_val = dist_val.log_prob(a_val[i]).sum(dim=1)
                    ratio_val = torch.exp(cur_p_val - p_val[i])

                    # PPO Clipping
                    ratio = ratio_idx * ratio_val
                    surr1 = ratio * advantages[i].squeeze()
                    surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages[i].squeeze()
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value Loss
                    value_loss = F.smooth_l1_loss(self.model.get_value(s[i]), td_target[i])

                    # Entropy Bonus (탐험 장려)
                    entropy = dist_idx.entropy().mean() + dist_val.entropy().sum(dim=1).mean()

                    # Total Loss
                    loss = policy_loss + 0.5 * value_loss - ENTROPY_COEF * entropy

                # Mixed Precision: scaled backward + optimizer step
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.scaler.step(self.optimizer)
                self.scaler.update()

        # === Critic 추가 학습 (5배 = 기존 1배 + 추가 4배) ===
        for _ in range(4):
            np.random.shuffle(indices)
            for start in range(0, total_samples, BATCH_SIZE):
                i = indices[start:start + BATCH_SIZE]

                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    value_loss = F.smooth_l1_loss(self.model.get_value(s[i]), td_target[i])

                self.optimizer.zero_grad()
                self.scaler.scale(value_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.scaler.step(self.optimizer)
                self.scaler.update()

        # Learning Rate 스케줄러 업데이트
        self.scheduler.step()


# ==========================================
# 4. Black & White Wrappers (Submission Class)
# ==========================================
class YourBlackAgent(BaseAgent):
    def __init__(self):
        # 흑돌은 my_turn=0
        super().__init__(my_turn=0)

    @classmethod
    def load(cls, path):
        agent = cls()
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=DEVICE)
            # 새 형식 (dict) vs 구 형식 (state_dict만)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                agent.model.load_state_dict(checkpoint['model_state_dict'])
                # Normalization 통계 복원
                if 'obs_rms' in checkpoint:
                    agent.obs_rms.mean = checkpoint['obs_rms']['mean']
                    agent.obs_rms.var = checkpoint['obs_rms']['var']
                    agent.obs_rms.count = checkpoint['obs_rms']['count']
                if 'reward_rms' in checkpoint:
                    agent.reward_rms.mean = checkpoint['reward_rms']['mean']
                    agent.reward_rms.var = checkpoint['reward_rms']['var']
                    agent.reward_rms.count = checkpoint['reward_rms']['count']
            else:
                # 구 형식 호환
                agent.model.load_state_dict(checkpoint)
        return agent

    def save(self, path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'obs_rms': {
                'mean': self.obs_rms.mean,
                'var': self.obs_rms.var,
                'count': self.obs_rms.count
            },
            'reward_rms': {
                'mean': self.reward_rms.mean,
                'var': self.reward_rms.var,
                'count': self.reward_rms.count
            }
        }
        torch.save(checkpoint, path)


class YourWhiteAgent(BaseAgent):
    def __init__(self):
        # 백돌은 my_turn=1
        # 로직은 BaseAgent에서 알아서 뒤집어서 처리함 (Mirror Logic)
        super().__init__(my_turn=1)

    @classmethod
    def load(cls, path):
        agent = cls()
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=DEVICE)
            # 새 형식 (dict) vs 구 형식 (state_dict만)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                agent.model.load_state_dict(checkpoint['model_state_dict'])
                # Normalization 통계 복원
                if 'obs_rms' in checkpoint:
                    agent.obs_rms.mean = checkpoint['obs_rms']['mean']
                    agent.obs_rms.var = checkpoint['obs_rms']['var']
                    agent.obs_rms.count = checkpoint['obs_rms']['count']
                if 'reward_rms' in checkpoint:
                    agent.reward_rms.mean = checkpoint['reward_rms']['mean']
                    agent.reward_rms.var = checkpoint['reward_rms']['var']
                    agent.reward_rms.count = checkpoint['reward_rms']['count']
            else:
                # 구 형식 호환 (Black이 학습한 모델)
                agent.model.load_state_dict(checkpoint)
        return agent

    def save(self, path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'obs_rms': {
                'mean': self.obs_rms.mean,
                'var': self.obs_rms.var,
                'count': self.obs_rms.count
            },
            'reward_rms': {
                'mean': self.reward_rms.mean,
                'var': self.reward_rms.var,
                'count': self.reward_rms.count
            }
        }
        torch.save(checkpoint, path)


# ==========================================
# 5. Helpers & Utils
# ==========================================
class OpponentManager:
    def __init__(self):
        self.save_dir = "history_models"
        os.makedirs(self.save_dir, exist_ok=True)
        self.pool = glob.glob(os.path.join(self.save_dir, "model_*.pkl"))

    def save_current_model(self, model, step):
        path = os.path.join(self.save_dir, f"model_{step}.pkl")
        torch.save(model.state_dict(), path)
        self.pool.append(path)
        if len(self.pool) > 20:
            old = self.pool.pop(0)
            if os.path.exists(old): os.remove(old)

    def get_opponent(self):
        if not self.pool: return None, "RandomBot"
        path = random.choice(self.pool)
        filename = os.path.basename(path)
        op = YourBlackAgent()
        try:
            op.model.load_state_dict(torch.load(path, map_location=DEVICE))
            op.model.eval()
            return op, filename
        except:
            return None, "RandomBot"


def make_env(): return gym.make(id='kymnasium/AlKkaGi-3x3-v0', render_mode=None, bgm=False, obs_type='custom')


def calculate_distance_from_center(stones):
    xy = stones[:, :, 0:2]
    center = np.array([BOARD_SIZE / 2, BOARD_SIZE / 2])
    diff = xy - center
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    return dist


# ==========================================
# 6. Main Loop (Training)
# ==========================================
def train():
    print("🚀 Starting SNIPER TRAINING (Shared Brain)")

    if os.path.exists(SAVE_PATH):
        print(f"🔄 Resuming: {SAVE_PATH}")
        agent = YourBlackAgent.load(SAVE_PATH)
    else:
        print("🆕 Starting New")
        agent = YourBlackAgent()

    envs = AsyncVectorEnv([make_env for _ in range(NUM_ENVS)])
    op_manager = OpponentManager()

    obs, _ = envs.reset()
    prev_opp = np.sum(obs['white'][:, :, 2], axis=1)
    prev_my = np.sum(obs['black'][:, :, 2], axis=1)
    prev_opp_center_dist = calculate_distance_from_center(obs['white'])

    score_history = []
    recent_win_rates = []
    interval_win_cnt = 0;
    interval_total_cnt = 0
    kill_count = 0

    opponent_agent = None
    opponent_name = "RandomBot"

    for update in range(1, 100001):

        # 커리큘럼 러닝: 학습 진행에 따라 RandomBot 비율 감소
        progress = min(update / 5000.0, 1.0)  # 0~1 (5000 업데이트까지)
        random_prob = max(0.1, 0.8 - progress * 0.7)  # 80% → 10%

        # 모델 저장 (더 자주)
        if update % SELFPLAY_SAVE_INTERVAL == 0:
            op_manager.save_current_model(agent.model, update)

        # 상대 선택 (커리큘럼 적용)
        if len(op_manager.pool) > 0 and random.random() > random_prob:
            # 과거 모델 사용
            if update % SELFPLAY_SWAP_INTERVAL == 0 or opponent_agent is None:
                opponent_agent, opponent_name = op_manager.get_opponent()
                if opponent_agent is None:
                    opponent_agent = None
                    opponent_name = "RandomBot"
        else:
            # RandomBot 사용
            opponent_agent = None
            opponent_name = f"RandomBot({random_prob*100:.0f}%)"

        for _ in range(T_HORIZON):
            turns = obs['turn']
            my_idx = np.where(turns == 0)[0]
            decoded_me = []

            # [중요] 학습은 항상 '내 차례'일 때만 진행
            if len(my_idx) > 0:
                obs_me = {k: v[my_idx] for k, v in obs.items()}
                # Black 입장에서 처리 (override_turn=0)
                s_me, mask_me = agent._process_obs(obs_me, override_turn=0)
                idx_me, val_me, p_idx_me, p_val_me = agent.get_action(s_me, mask_me)
                decoded_me = agent.decode_action_with_assist(idx_me, val_me, obs_me, 0)

            op_idx = np.where(turns == 1)[0]
            decoded_op = []
            if len(op_idx) > 0:
                if opponent_agent is not None:
                    obs_op = {k: v[op_idx] for k, v in obs.items()}
                    # 상대(White) 입장에서 처리 (override_turn=1)
                    s_op, mask_op = opponent_agent._process_obs(obs_op, override_turn=1)
                    with torch.no_grad():
                        idx_op, val_op, _, _ = opponent_agent.get_action(s_op, mask_op, deterministic=True)
                    # 상대도 보정 사격 사용
                    decoded_op = opponent_agent.decode_action_with_assist(idx_op, val_op, obs_op, 1)
                else:
                    for k in range(len(op_idx)):
                        decoded_op.append({
                            "turn": 1, "index": random.randint(0, 2),
                            "power": random.uniform(300, 2000), "angle": random.uniform(-180, 180)
                        })

            action_list = [None] * NUM_ENVS
            if len(my_idx) > 0:
                for i, env_i in enumerate(my_idx): action_list[env_i] = decoded_me[i]
            if len(op_idx) > 0:
                for i, env_i in enumerate(op_idx): action_list[env_i] = decoded_op[i]

            batched_action = {key: np.array([d[key] for d in action_list]) for key in action_list[0].keys()}
            next_obs, _, term, trunc, _ = envs.step(batched_action)

            curr_opp = np.sum(next_obs['white'][:, :, 2], axis=1)
            curr_my = np.sum(next_obs['black'][:, :, 2], axis=1)
            curr_opp_center_dist = calculate_distance_from_center(next_obs['white'])

            # --- Reward Logic ---
            if len(my_idx) > 0:
                opp_alive_mask = next_obs['white'][my_idx, :, 2]
                push_diff = curr_opp_center_dist[my_idx] - prev_opp_center_dist[my_idx]
                valid_push = (push_diff > 1.0) & (push_diff < 500.0) & (opp_alive_mask == 1)
                push_reward = np.sum(np.where(valid_push, push_diff * 0.1, 0.0), axis=1)

                raw_k_r = prev_opp[my_idx] - curr_opp[my_idx]
                raw_s_r = prev_my[my_idx] - curr_my[my_idx]
                k_r = np.maximum(raw_k_r, 0)
                s_r = np.maximum(raw_s_r, 0)
                if np.sum(k_r) > 0: kill_count += np.sum(k_r)

                suicide_penalty = np.zeros(len(my_idx))
                suicide_penalty[(s_r > 0) & (k_r == 0)] = -5.0

                # === 벽 뒤 방어 보너스 & 벽 충돌 페널티 ===
                wall_defense_bonus = np.zeros(len(my_idx))
                wall_collision_penalty = np.zeros(len(my_idx))
                obstacles = next_obs.get('obstacles', None)
                for i, env_i in enumerate(my_idx):
                    my_stones = next_obs['black'][env_i]
                    op_stones = next_obs['white'][env_i]
                    batch_obs = obstacles[env_i] if obstacles is not None and len(obstacles) > env_i else []

                    if len(batch_obs) > 0:
                        # 살아있는 적 돌 위치
                        op_alive = op_stones[:, 2]
                        valid_enemies = op_stones[op_alive == 1, 0:2]

                        if len(valid_enemies) > 0:
                            # 내 살아있는 돌들의 평균 보호율 계산
                            total_protection = 0.0
                            alive_count = 0
                            for stone in my_stones:
                                if stone[2] == 1:  # 살아있는 돌
                                    protection = is_protected_by_wall(stone[0:2], valid_enemies, batch_obs)
                                    total_protection += protection
                                    alive_count += 1
                            if alive_count > 0:
                                wall_defense_bonus[i] = (total_protection / alive_count) * 0.15

                        # === 벽 충돌 페널티 ===
                        # 내 돌이 벽에 부딪힌 경우 페널티
                        my_collisions = calculate_wall_collision(my_stones, batch_obs)
                        wall_collision_penalty[i] = -np.sum(my_collisions) * 2.0  # 충돌당 -2.0

                r = (k_r * 50.0) - (s_r * 5.0) + push_reward + suicide_penalty + wall_defense_bonus + wall_collision_penalty - 0.1

                done = np.logical_or(term, trunc)[my_idx]
                win = (curr_opp[my_idx] == 0) & done
                lose = (curr_my[my_idx] == 0) & done

                r[win] += 20.0
                r[lose] -= 10.0

                next_obs_dict = {k: v[my_idx] for k, v in next_obs.items()}
                s_prime_me, _ = agent._process_obs(next_obs_dict, override_turn=0)

                for i in range(len(my_idx)):
                    agent.put_data((
                        s_me[i], idx_me[i].detach(), val_me[i].detach(), r[i], s_prime_me[i],
                        p_idx_me[i].detach(), p_val_me[i].detach(), done.astype(float)[i], mask_me[i]
                    ))
                    score_history.append(r[i])
                    if done[i]:
                        interval_total_cnt += 1
                        if win[i]: interval_win_cnt += 1

            obs = next_obs;
            prev_opp = curr_opp;
            prev_my = curr_my
            prev_opp_center_dist = curr_opp_center_dist

        agent.train_net()

        if update % 10 == 0:
            if interval_total_cnt > 0:
                current_win_rate = (interval_win_cnt / interval_total_cnt) * 100.0
            else:
                current_win_rate = 0.0

            recent_win_rates.append(current_win_rate)
            if len(recent_win_rates) > 20: recent_win_rates.pop(0)
            avg_win = np.mean(recent_win_rates)

            avg_score = np.mean(score_history[-1000:]) if score_history else 0
            print(
                f"📊 Upd {update:4d} | Score: {avg_score:6.2f} | Win: {current_win_rate:5.1f}% (Avg: {avg_win:5.1f}%) | Kills: {int(kill_count):3d} | VS: {opponent_name}")

            kill_count = 0;
            interval_win_cnt = 0;
            interval_total_cnt = 0
            agent.save(SAVE_PATH)

    envs.close()


if __name__ == "__main__": train()