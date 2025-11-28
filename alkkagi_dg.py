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

# Self-Play 설정
SELFPLAY_SAVE_INTERVAL = 100
SELFPLAY_SWAP_INTERVAL = 50
SELFPLAY_START_WINRATE = 80.0

# GAE & Entropy 설정
GAE_LAMBDA = 0.95
ENTROPY_COEF = 0.01
BOARD_SIZE = 600.0  # 실제 게임 보드 크기


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
# 3. Agent Logic (공통 로직)
# ==========================================
class BaseAgent(kym.Agent):
    def __init__(self, my_turn):
        super().__init__()
        self.my_turn = my_turn  # 0: Black, 1: White
        self.data = []
        self.input_dim = 8  # 확장된 전략적 관측 공간
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

    # [중요] 흑/백 구분 없이 '나'와 '적'으로 변환하여 인식
    def _process_obs(self, obs, override_turn=None):
        """
        전략적 관측 공간 (8차원 × 3돌)
        [0] 가장 가까운 적까지 거리
        [1] 그 적의 경계 근접도 (높을수록 밀어내기 쉬움)
        [2] 내 돌의 경계 근접도 (높을수록 위험)
        [3] 적 방향 x (-1~1)
        [4] 적 방향 y (-1~1)
        [5] 내 돌 생존 여부
        [6] 적 돌 수 (정규화)
        [7] 내 돌 수 (정규화)
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

        processed = np.zeros((batch_size, 3, 8), dtype=np.float32)
        alive_mask = np.zeros((batch_size, 3), dtype=np.float32)

        op_xy = op_stones[:, :, 0:2]
        op_alive = op_stones[:, :, 2]
        my_alive = my_stones[:, :, 2]

        for b in range(batch_size):
            my_s = my_stones[b]
            alive_mask[b] = my_s[:, 2]

            # 돌 수 계산
            my_count = np.sum(my_alive[b])
            op_count = np.sum(op_alive[b])

            valid_op_idx = np.where(op_alive[b] == 1)[0]
            if len(valid_op_idx) == 0:
                valid_op_pos = np.array([[BOARD_SIZE / 2, BOARD_SIZE / 2]])
                valid_op_edge = np.array([0.0])
            else:
                valid_op_pos = op_xy[b][valid_op_idx]
                # 각 적의 경계 근접도 계산
                edge_dists = np.minimum(
                    np.minimum(valid_op_pos[:, 0], BOARD_SIZE - valid_op_pos[:, 0]),
                    np.minimum(valid_op_pos[:, 1], BOARD_SIZE - valid_op_pos[:, 1])
                )
                valid_op_edge = 1.0 - np.clip(edge_dists / (BOARD_SIZE / 2), 0, 1)

            for i in range(3):
                if my_s[i, 2] == 0:
                    continue

                my_pos = my_s[i, 0:2]

                # 가장 가까운 적 찾기
                dists = np.sqrt(np.sum((valid_op_pos - my_pos) ** 2, axis=1))
                nearest_idx = np.argmin(dists)
                dist = dists[nearest_idx]
                target_pos = valid_op_pos[nearest_idx]

                # 방향 벡터 계산
                diff = target_pos - my_pos
                norm = np.linalg.norm(diff) + 1e-6
                direction = diff / norm

                # 내 돌의 경계 근접도
                my_edge_dist = min(my_pos[0], BOARD_SIZE - my_pos[0],
                                   my_pos[1], BOARD_SIZE - my_pos[1])
                my_edge_danger = 1.0 - np.clip(my_edge_dist / (BOARD_SIZE / 2), 0, 1)

                processed[b, i, 0] = dist / BOARD_SIZE                    # 거리
                processed[b, i, 1] = valid_op_edge[nearest_idx]           # 적 경계 근접도
                processed[b, i, 2] = my_edge_danger                       # 내 경계 위험도
                processed[b, i, 3] = direction[0]                         # 적 방향 x
                processed[b, i, 4] = direction[1]                         # 적 방향 y
                processed[b, i, 5] = 1.0                                  # 내 돌 생존
                processed[b, i, 6] = op_count / 3.0                       # 적 돌 수
                processed[b, i, 7] = my_count / 3.0                       # 내 돌 수

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

    # [중요] 보정 사격 로직도 흑/백 구분 처리
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
            if len(valid_ops) == 0: valid_ops = np.array([[BOARD_SIZE / 2, BOARD_SIZE / 2]])

            dists = np.sum((valid_ops - my_pos) ** 2, axis=1)
            target_pos = valid_ops[np.argmin(dists)]

            dx = target_pos[0] - my_pos[0]
            dy = target_pos[1] - my_pos[1]
            ideal_angle = np.degrees(np.arctan2(dy, dx))

            angle_offset = val_np[i, 0] * 30.0
            final_angle = ideal_angle + angle_offset

            # 거리 기반 파워 자동 조절
            dist = np.sqrt(dx**2 + dy**2)
            base_power = 800 + (dist / BOARD_SIZE) * 1200  # 가까우면 800, 멀면 2000
            adjustment = val_np[i, 1] * 0.3  # AI는 ±30%만 미세 조정
            power = np.clip(base_power * (1 + adjustment), 500, 2500)

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
        # self.my_turn을 사용하여 내 관점에서 처리
        s, mask = self._process_obs(batch_obs, override_turn=self.my_turn)
        with torch.no_grad():
            idx, val, _, _ = self.get_action(s, mask, deterministic=True)

        return self.decode_action_with_assist(idx, val, batch_obs, self.my_turn)[0]

    def train_net(self):
        if len(self.data) < 1: return
        s, a_idx, a_val, r, s_p, p_idx, p_val, done, mask = self.make_batch()

        # === GAE 계산 ===
        with torch.no_grad():
            values = self.model.get_value(s)
            next_values = self.model.get_value(s_p)

            # GAE: A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
            batch_size = s.size(0)
            advantages = torch.zeros(batch_size, 1).to(DEVICE)
            last_gae = 0

            for t in reversed(range(batch_size)):
                next_non_terminal = 1.0 - done[t]
                delta = r[t] + GAMMA * next_values[t] * next_non_terminal - values[t]
                advantages[t] = last_gae = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae

            td_target = advantages + values
            # Advantage 정규화 (학습 안정화)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_samples = s.size(0)
        indices = np.arange(total_samples)

        for _ in range(K_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, total_samples, BATCH_SIZE):
                i = indices[start:start + BATCH_SIZE]
                scores, mu, std, _ = self.model(s[i], mask[i])

                # Categorical distribution (돌 선택)
                dist_idx = Categorical(logits=scores)
                cur_p_idx = dist_idx.log_prob(a_idx[i])
                ratio_idx = torch.exp(cur_p_idx - p_idx[i])

                # Continuous distribution (각도/파워)
                batch_idx_range = torch.arange(len(i), device=DEVICE)
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

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


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
            agent.model.load_state_dict(torch.load(path, map_location=DEVICE))
        return agent

    def save(self, path):
        torch.save(self.model.state_dict(), path)


class YourWhiteAgent(BaseAgent):
    def __init__(self):
        # 백돌은 my_turn=1
        # 로직은 BaseAgent에서 알아서 뒤집어서 처리함 (Mirror Logic)
        super().__init__(my_turn=1)

    @classmethod
    def load(cls, path):
        agent = cls()
        if os.path.exists(path):
            # Black이 학습한 모델을 그대로 불러옴
            agent.model.load_state_dict(torch.load(path, map_location=DEVICE))
        return agent

    def save(self, path):
        torch.save(self.model.state_dict(), path)


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


class RandomAgent:
    def __init__(self): pass

    def get_action(self, obs_tensor, alive_mask, deterministic=False):
        batch_size = obs_tensor.size(0)
        indices = torch.zeros(batch_size, dtype=torch.long, device=DEVICE)
        actions = torch.rand(batch_size, 2, device=DEVICE) * 2 - 1
        dummy_probs = torch.zeros(batch_size, device=DEVICE)
        return indices, actions, dummy_probs, dummy_probs


def make_env(): return gym.make(id='kymnasium/AlKkaGi-3x3-v0', render_mode=None, bgm=False, obs_type='custom')


def calculate_stone_movement(prev_stones, curr_stones):
    prev_xy = prev_stones[:, :, 0:2];
    curr_xy = curr_stones[:, :, 0:2]
    prev_alive = prev_stones[:, :, 2];
    curr_alive = curr_stones[:, :, 2]
    valid_mask = prev_alive * curr_alive
    diff = curr_xy - prev_xy
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    return np.sum(dist * valid_mask, axis=1)


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
    prev_opp_stones = obs['white'].copy()
    prev_opp_center_dist = calculate_distance_from_center(obs['white'])

    score_history = []
    recent_win_rates = []
    interval_win_cnt = 0;
    interval_total_cnt = 0
    kill_count = 0

    opponent_agent = None
    opponent_name = "RandomBot"

    for update in range(1, 100001):

        # --- Self-Play Logic ---
        if len(recent_win_rates) > 0:
            avg_win_rate = np.mean(recent_win_rates)
        else:
            avg_win_rate = 0.0

        if avg_win_rate >= SELFPLAY_START_WINRATE and len(op_manager.pool) > 0:
            if update % SELFPLAY_SWAP_INTERVAL == 0 or opponent_agent is None:
                opponent_agent, opponent_name = op_manager.get_opponent()
                if opponent_agent is None: opponent_name = "RandomBot"
        else:
            opponent_agent = None
            opponent_name = "RandomBot"

        if update % SELFPLAY_SAVE_INTERVAL == 0:
            op_manager.save_current_model(agent.model, update)

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

                r = (k_r * 50.0) - (s_r * 5.0) + push_reward + suicide_penalty - 0.1

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
            prev_opp_stones = obs['white'].copy()
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