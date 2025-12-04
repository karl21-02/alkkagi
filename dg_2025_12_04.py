import gymnasium as gym
import kymnasium as kym
import numpy as np
import os
import glob
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from gymnasium.vector import AsyncVectorEnv

# ==========================================
# 1. Hyperparameters
# ==========================================
NUM_ENVS = 8
LR_ACTOR = 0.0003
LR_CRITIC = 0.001
GAMMA = 0.99
K_EPOCHS = 4
EPS_CLIP = 0.2
T_HORIZON = 128
BATCH_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "my_alkkagi_agent.pkl"

# 맵 사이즈 정보
BOARD_SIZE = 600.0
CENTER_POS = 300.0

SELFPLAY_SAVE_INTERVAL = 100
SELFPLAY_SWAP_INTERVAL = 50
SELFPLAY_START_WINRATE = 80.0


# ==========================================
# 2. Neural Network
# ==========================================
class SniperNet(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(SniperNet, self).__init__()
        # 입력: [거리, 내_생존, 적_생존, 벽_막힘] -> Dim=4
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh()
        )
        self.score_head = nn.Linear(64, 1)

        # Action: [Angle_Offset, Power]
        self.mu_head = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

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
        std = self.log_std.exp().expand_as(mu)
        return scores, mu, std, features.view(batch_size, 3, -1)

    def get_value(self, x):
        batch_size = x.size(0)
        flat_x = x.view(-1, x.size(-1))
        features = self.encoder(flat_x).view(batch_size, 3, -1)
        global_features, _ = torch.max(features, dim=1)
        return self.critic_head(global_features)


# ==========================================
# 3. Agent Logic
# ==========================================
class BaseAgent(kym.Agent):
    def __init__(self, my_turn):
        super().__init__()
        self.my_turn = my_turn
        self.data = []
        # [수정] 입력 차원 4 (벽 정보 포함)
        self.input_dim = 4
        self.action_dim = 2
        self.model = SniperNet(self.input_dim, self.action_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR_ACTOR)

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

    def load(self, path):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=DEVICE))
            print(f"✅ Model loaded from {path}")
        else:
            print(f"⚠️ No model found at {path}")

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    # [수정됨] 직사각형(AABB) 충돌 체크 로직
    def _is_wall_blocking(self, my_pos, target_pos, box_center, box_size):
        """
        경로(Line) 위에 점을 여러 개 찍어,
        직사각형(Wall) 영역 안에 들어가는 점이 있는지 확인하는 샘플링 방식
        """
        # 1. 벽의 영역 (직사각형 경계) 정의
        # 돌의 크기(반지름 약 15)만큼 벽을 뚱뚱하게 인식해야 스치지 않고 피해감
        stone_radius = 15.0

        # box_size = [width, height]
        half_w = box_size[0] / 2 + stone_radius
        half_h = box_size[1] / 2 + stone_radius

        x_min = box_center[0] - half_w
        x_max = box_center[0] + half_w
        y_min = box_center[1] - half_h
        y_max = box_center[1] + half_h

        # 2. 경로 샘플링 (내 위치 -> 타겟 위치)
        # 시작점과 끝점 사이를 10등분 해서 확인 (이 정도면 충분히 정확함)
        num_steps = 10

        # 벡터 계산 최적화를 위해 반복문 밖에서 차이 계산
        diff = target_pos - my_pos

        for i in range(num_steps + 1):
            t = i / num_steps  # 0.0 ~ 1.0

            # 현재 검사할 점의 위치
            check_x = my_pos[0] + t * diff[0]
            check_y = my_pos[1] + t * diff[1]

            # 3. 점이 직사각형 범위 안에 있는지 체크 (AABB Check)
            if x_min <= check_x <= x_max and y_min <= check_y <= y_max:
                return 1.0  # 막힘 (벽 내부 혹은 경계)

        return 0.0  # 통과 (벽에 걸리는 지점 없음)

    def _process_obs(self, obs, override_turn=None):
        board_scale = BOARD_SIZE
        batch_size = len(obs['black'])
        if override_turn is not None:
            turns = np.full((batch_size, 1, 1), override_turn)
        else:
            turns = obs['turn'].reshape(batch_size, 1, 1)

        black = obs['black'];
        white = obs['white']
        my_stones = np.where(turns == 0, black, white)
        op_stones = np.where(turns == 0, white, black)

        # [수정] 입력 차원 4
        processed = np.zeros((batch_size, 3, 4), dtype=np.float32)
        alive_mask = np.zeros((batch_size, 3), dtype=np.float32)

        op_xy = op_stones[:, :, 0:2]
        op_alive = op_stones[:, :, 2]

        # 장애물 정보
        obstacles = obs['obstacles']

        for b in range(batch_size):
            my_s = my_stones[b]
            alive_mask[b] = my_s[:, 2]

            valid_op_idx = np.where(op_alive[b] == 1)[0]
            if len(valid_op_idx) == 0:
                valid_op_pos = np.array([[CENTER_POS, CENTER_POS]])
            else:
                valid_op_pos = op_xy[b][valid_op_idx]

            for i in range(3):
                if my_s[i, 2] == 0: continue
                my_pos = my_s[i, 0:2]

                # 타겟 선정
                dists = np.sum((valid_op_pos - my_pos) ** 2, axis=1)
                nearest_idx = np.argmin(dists)
                dist = np.sqrt(dists[nearest_idx])
                target_pos = valid_op_pos[nearest_idx]

                # [New] 벽 감지 로직 호출
                is_blocked = 0.0
                # 배치 내의 장애물 루프 (보통 3개)
                for obs_idx in range(len(obstacles[b])):
                    obs_center = obstacles[b, obs_idx, 0:2]
                    obs_size = obstacles[b, obs_idx, 2:4]  # w, h (여기선 중심거리만 씀)

                    if self._is_wall_blocking(my_pos, target_pos, obs_center, obs_size) > 0.5:
                        is_blocked = 1.0
                        break

                processed[b, i, 0] = dist / board_scale
                processed[b, i, 1] = 1.0
                processed[b, i, 2] = 1.0
                processed[b, i, 3] = is_blocked  # 벽 정보 추가

        return torch.tensor(processed).to(DEVICE), torch.tensor(alive_mask).to(DEVICE)

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

    def decode_action_with_assist(self, idx, val, obs, my_turn):
        actions = []
        idx_np = idx.cpu().numpy()
        val_np = torch.tanh(val).detach().cpu().numpy()

        my_stones = obs['black'] if my_turn == 0 else obs['white']
        op_stones = obs['white'] if my_turn == 0 else obs['black']

        for i in range(len(idx_np)):
            stone_idx = int(idx_np[i])
            my_pos = my_stones[i, stone_idx, 0:2]
            op_xy = op_stones[i, :, 0:2]
            op_alive = op_stones[i, :, 2]
            valid_ops = op_xy[op_alive == 1]
            if len(valid_ops) == 0: valid_ops = np.array([[CENTER_POS, CENTER_POS]])

            dists = np.sum((valid_ops - my_pos) ** 2, axis=1)
            target_pos = valid_ops[np.argmin(dists)]

            dx = target_pos[0] - my_pos[0]
            dy = target_pos[1] - my_pos[1]
            ideal_angle = np.degrees(np.arctan2(dy, dx))

            angle_offset = val_np[i, 0] * 30.0
            final_angle = ideal_angle + angle_offset
            power = 300.0 + ((val_np[i, 1] + 1) / 2.0) * 2200.0

            actions.append({
                "turn": my_turn, "index": stone_idx,
                "power": float(power), "angle": float(final_angle)
            })
        return actions

    def act(self, obs, info):
        batch_obs = {
            'black': np.array([obs['black']]),
            'white': np.array([obs['white']]),
            'turn': np.array([obs['turn']]),
            'obstacles': np.array([obs['obstacles']])
        }
        s, mask = self._process_obs(batch_obs, override_turn=self.my_turn)
        with torch.no_grad():
            idx, val, _, _ = self.get_action(s, mask, deterministic=True)
        return self.decode_action_with_assist(idx, val, batch_obs, self.my_turn)[0]

    def train_net(self):
        if len(self.data) < 1: return
        s, a_idx, a_val, r, s_p, p_idx, p_val, done, mask = self.make_batch()
        with torch.no_grad():
            td_target = r + GAMMA * self.model.get_value(s_p) * (1 - done)
            delta = td_target - self.model.get_value(s)
        advantage = delta.detach()
        total_samples = s.size(0)
        indices = np.arange(total_samples)
        for _ in range(K_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, total_samples, BATCH_SIZE):
                i = indices[start:start + BATCH_SIZE]
                scores, mu, std, _ = self.model(s[i], mask[i])
                dist_idx = Categorical(logits=scores)
                cur_p_idx = dist_idx.log_prob(a_idx[i])
                ratio_idx = torch.exp(cur_p_idx - p_idx[i])
                batch_idx_range = torch.arange(len(i), device=DEVICE)
                sel_mu = mu[batch_idx_range, a_idx[i]]
                sel_std = std[batch_idx_range, a_idx[i]]
                dist_val = Normal(sel_mu, sel_std)
                cur_p_val = dist_val.log_prob(a_val[i]).sum(dim=1)
                ratio_val = torch.exp(cur_p_val - p_val[i])
                ratio = ratio_idx * ratio_val
                surr1 = ratio * advantage[i]
                surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantage[i]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.smooth_l1_loss(self.model.get_value(s[i]), td_target[i])
                loss = policy_loss + 0.5 * value_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


# ==========================================
# 4. Helpers & Opponent Manager
# ==========================================
class YourBlackAgent(BaseAgent):
    def __init__(self): super().__init__(my_turn=0)

    @classmethod
    def load(cls, path):
        agent = cls()
        if os.path.exists(path): agent.model.load_state_dict(torch.load(path, map_location=DEVICE))
        return agent

    def save(self, path): torch.save(self.model.state_dict(), path)


class YourWhiteAgent(BaseAgent):
    def __init__(self): super().__init__(my_turn=1)

    @classmethod
    def load(cls, path):
        agent = cls()
        if os.path.exists(path): agent.model.load_state_dict(torch.load(path, map_location=DEVICE))
        return agent

    def save(self, path): torch.save(self.model.state_dict(), path)


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


class RandomAgent:
    def __init__(self): pass

    def get_action(self, obs_tensor, alive_mask, deterministic=False):
        batch_size = obs_tensor.size(0)
        indices = torch.zeros(batch_size, dtype=torch.long, device=DEVICE)
        actions = torch.rand(batch_size, 2, device=DEVICE) * 2 - 1
        dummy_probs = torch.zeros(batch_size, device=DEVICE)
        return indices, actions, dummy_probs, dummy_probs


def make_env(): return gym.make(id='kymnasium/AlKkaGi-3x3-v0', render_mode=None, bgm=False, obs_type='custom')


def calculate_distance_from_center(stones):
    xy = stones[:, :, 0:2]
    center = np.array([CENTER_POS, CENTER_POS])
    diff = xy - center
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    return dist


def calculate_wall_collision(stones, obstacles, radius=15.0):
    stone_xy = stones[:, :, 0:2]
    stone_radius = radius

    if isinstance(obstacles, list): obstacles = np.array(obstacles)
    obs_x = obstacles[..., 0]
    obs_y = obstacles[..., 1]
    obs_w = obstacles[..., 2]
    obs_h = obstacles[..., 3]

    dx = np.abs(stone_xy[:, :, 0, np.newaxis] - obs_x[:, np.newaxis, :])
    dy = np.abs(stone_xy[:, :, 1, np.newaxis] - obs_y[:, np.newaxis, :])

    collision_x = dx < (obs_w[:, np.newaxis, :] / 2 + stone_radius)
    collision_y = dy < (obs_h[:, np.newaxis, :] / 2 + stone_radius)

    is_collided = collision_x & collision_y
    return np.any(is_collided, axis=2)


# ==========================================
# 6. Main Loop
# ==========================================
# ==========================================
# 6. Main Loop (Fixed for White/Black Balance)
# ==========================================
def train():
    print("🚀 Starting Training: Chapter 2 (Balanced Training)")

    envs = AsyncVectorEnv([make_env for _ in range(NUM_ENVS)])
    stone_radius = 15.0

    op_manager = OpponentManager()

    # [수정 1] 초기 에이전트 설정 (팀은 동적으로 바뀜)
    agent = BaseAgent(my_turn=0)

    obs, _ = envs.reset()

    # [수정 2] 이전 상태 기록 변수들을 딕셔너리로 관리 (흑/백 모두 추적)
    prev_counts = {
        'black': np.sum(obs['black'][:, :, 2], axis=1),
        'white': np.sum(obs['white'][:, :, 2], axis=1)
    }
    prev_center_dist = {
        'black': calculate_distance_from_center(obs['black']),
        'white': calculate_distance_from_center(obs['white'])
    }

    score_history = []
    recent_win_rates = []
    interval_win_cnt = 0
    interval_total_cnt = 0
    kill_count = 0

    opponent_agent = None
    opponent_name = "RandomBot"

    # [수정 3] 현재 메인 에이전트가 맡을 팀 (0: 흑, 1: 백)
    current_agent_team = 0

    for update in range(1, 100001):
        # --- 주기적으로 팀 교체 (흑/백 밸런스 학습) ---
        if update % SELFPLAY_SWAP_INTERVAL == 0:
            current_agent_team = 1 - current_agent_team  # 0 <-> 1 토글
            agent.my_turn = current_agent_team  # 에이전트 내부 인식 변경

            # 상대방 변경 로직도 여기서 수행
            if len(recent_win_rates) > 0 and np.mean(recent_win_rates) >= SELFPLAY_START_WINRATE:
                opponent_agent, opponent_name = op_manager.get_opponent()
                if opponent_agent is not None:
                    # 상대방은 내가 흑이면 백, 내가 백이면 흑을 맡아야 함
                    opponent_agent.my_turn = 1 - current_agent_team
            else:
                opponent_agent = None
                opponent_name = "RandomBot"

        if update % SELFPLAY_SAVE_INTERVAL == 0:
            op_manager.save_current_model(agent.model, update)

        # -------------------------------------------------

        for _ in range(T_HORIZON):
            turns = obs['turn']

            # [수정 4] 동적 인덱싱: 현재 에이전트 팀에 따라 my_idx 결정
            if current_agent_team == 0:  # 내가 흑돌이면
                my_idx = np.where(turns == 0)[0]
                op_idx = np.where(turns == 1)[0]
                my_color, op_color = 'black', 'white'
            else:  # 내가 백돌이면
                my_idx = np.where(turns == 1)[0]
                op_idx = np.where(turns == 0)[0]
                my_color, op_color = 'white', 'black'

            # 1. 내 행동 (Main Agent)
            decoded_me = []
            if len(my_idx) > 0:
                obs_me = {k: v[my_idx] for k, v in obs.items()}
                # override_turn을 현재 내 팀으로 설정
                s_me, mask_me = agent._process_obs(obs_me, override_turn=current_agent_team)
                idx_me, val_me, p_idx_me, p_val_me = agent.get_action(s_me, mask_me)
                decoded_me = agent.decode_action_with_assist(idx_me, val_me, obs_me, current_agent_team)

            # 2. 상대 행동 (Opponent / Random)
            decoded_op = []
            if len(op_idx) > 0:
                if opponent_agent is not None:
                    obs_op = {k: v[op_idx] for k, v in obs.items()}
                    # 상대방 턴 넣어주기
                    s_op, mask_op = opponent_agent._process_obs(obs_op, override_turn=1 - current_agent_team)
                    with torch.no_grad():
                        idx_op, val_op, _, _ = opponent_agent.get_action(s_op, mask_op, deterministic=True)
                    decoded_op = opponent_agent.decode_action_with_assist(idx_op, val_op, obs_op,
                                                                          1 - current_agent_team)
                else:
                    for k in range(len(op_idx)):
                        decoded_op.append({
                            "turn": 1 - current_agent_team,  # 상대방 턴
                            "index": random.randint(0, 2),
                            "power": random.uniform(300, 2000), "angle": random.uniform(-180, 180)
                        })

            # 3. 행동 합치기
            action_list = [None] * NUM_ENVS
            if len(my_idx) > 0:
                for i, env_i in enumerate(my_idx): action_list[env_i] = decoded_me[i]
            if len(op_idx) > 0:
                for i, env_i in enumerate(op_idx): action_list[env_i] = decoded_op[i]

            batched_action = {key: np.array([d[key] for d in action_list]) for key in action_list[0].keys()}
            next_obs, _, term, trunc, _ = envs.step(batched_action)

            # [수정 5] 보상 계산을 위한 상태 업데이트 (동적 키 사용)
            curr_counts = {
                'black': np.sum(next_obs['black'][:, :, 2], axis=1),
                'white': np.sum(next_obs['white'][:, :, 2], axis=1)
            }
            curr_center_dist = {
                'black': calculate_distance_from_center(next_obs['black']),
                'white': calculate_distance_from_center(next_obs['white'])
            }

            # 벽 충돌 체크 (내 돌에 대해서만)
            # next_obs[my_color]를 써야 함
            my_wall_hits = calculate_wall_collision(next_obs[my_color], next_obs['obstacles'], radius=stone_radius)

            # --- Reward Logic (Generalized) ---
            if len(my_idx) > 0:
                # (1) Push Reward (상대가 중심에서 멀어졌는가?)
                # 상대방 돌의 생존 마스크
                opp_alive_mask = next_obs[op_color][my_idx, :, 2]

                # 거리 차이: (현재 상대 거리 - 이전 상대 거리) -> 양수면 밀려난 것
                push_diff = curr_center_dist[op_color][my_idx] - prev_center_dist[op_color][my_idx]
                valid_push = (push_diff > 1.0) & (push_diff < 500.0) & (opp_alive_mask == 1)
                push_reward = np.sum(np.where(valid_push, push_diff * 0.05, 0.0), axis=1)

                # (2) Smart Kill & Suicide
                # 내 팀과 상대 팀의 돌 개수 변화량
                raw_k_r = prev_counts[op_color][my_idx] - curr_counts[op_color][my_idx]  # 적 죽음
                raw_s_r = prev_counts[my_color][my_idx] - curr_counts[my_color][my_idx]  # 나 죽음

                k_r = np.maximum(raw_k_r, 0)
                s_r = np.maximum(raw_s_r, 0)
                if np.sum(k_r) > 0: kill_count += np.sum(k_r)

                kill_reward = np.zeros(len(my_idx))
                kill_mask = k_r > 0
                trade_mask = kill_mask & (s_r > 0)
                kill_reward[trade_mask] = 10.0
                perfect_mask = kill_mask & (s_r == 0)
                kill_reward[perfect_mask] = 50.0

                suicide_penalty = np.zeros(len(my_idx))
                suicide_penalty[(s_r > 0) & (k_r == 0)] = -10.0

                # (3) Wall Penalty
                wall_penalty = np.sum(my_wall_hits[my_idx], axis=1) * -10.0

                # (4) Total Reward
                r = kill_reward - (s_r * 10.0) + push_reward + suicide_penalty + wall_penalty - 0.3

                done = np.logical_or(term, trunc)[my_idx]

                # 승리 조건: 적 돌이 0개 & 게임 끝
                win = (curr_counts[op_color][my_idx] == 0) & done
                # 패배 조건: 내 돌이 0개 & 게임 끝
                lose = (curr_counts[my_color][my_idx] == 0) & done

                r[win] += 20.0
                r[lose] -= 10.0

                # 데이터 저장
                next_obs_dict = {k: v[my_idx] for k, v in next_obs.items()}
                s_prime_me, _ = agent._process_obs(next_obs_dict, override_turn=current_agent_team)

                for i in range(len(my_idx)):
                    agent.put_data((
                        s_me[i], idx_me[i].detach(), val_me[i].detach(), r[i], s_prime_me[i],
                        p_idx_me[i].detach(), p_val_me[i].detach(), done.astype(float)[i], mask_me[i]
                    ))
                    score_history.append(r[i])
                    if done[i]:
                        interval_total_cnt += 1
                        if win[i]: interval_win_cnt += 1

            obs = next_obs
            prev_counts = curr_counts
            prev_center_dist = curr_center_dist

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

            # 로그에 현재 팀 정보(B/W) 추가
            team_str = "Black" if current_agent_team == 0 else "White"
            print(
                f"📊 Upd {update:4d} | Team: {team_str} | Score: {avg_score:6.2f} | Win: {current_win_rate:5.1f}% | VS: {opponent_name}")

            kill_count = 0;
            interval_win_cnt = 0;
            interval_total_cnt = 0
            agent.save(SAVE_PATH)

    envs.close()


if __name__ == "__main__": train()