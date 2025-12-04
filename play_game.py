import gymnasium as gym
import torch
import numpy as np
import os
import time

# 각 모델에 맞는 에이전트 클래스 가져오기
# kg: ppo_without_elo.py (HybridActorCritic 구조)
# dg: dg_2025_12_04.py (SniperNet 구조)
from ppo_without_elo import YourBlackAgent as KGBlackAgent, YourWhiteAgent as KGWhiteAgent
from dg_2025_12_04 import YourBlackAgent as DGBlackAgent, YourWhiteAgent as DGWhiteAgent

# ==========================================
# 대결 설정: 두 에이전트 파일 경로
# ==========================================
BLACK_MODEL = "my_alkkagi_agent_dg.pkl"  # 흑돌 (선공) - dg (dg_2025_12_04.py)
WHITE_MODEL = "my_alkkagi_agent_kg.pkl"  # 백돌 (후공) - kg (ppo_without_elo.py)


def evaluate():
    # 1. 환경 생성
    env = gym.make(
        id='kymnasium/AlKkaGi-3x3-v0',
        render_mode='human',
        bgm=True,
        obs_type='custom'
    )

    # 2. 각 모델에 맞는 에이전트 생성
    # Black: dg 모델 → DGBlackAgent (dg_2025_12_04.py 구조)
    black_agent = DGBlackAgent()
    # White: kg 모델 → KGWhiteAgent (ppo_without_elo.py 구조)
    white_agent = KGWhiteAgent()

    # 3. 각 에이전트에 모델 로드
    # Black (dg) 모델 로드
    if os.path.exists(BLACK_MODEL):
        print(f"Loading Black (dg) from {BLACK_MODEL}...")
        try:
            ckpt = torch.load(BLACK_MODEL, map_location='cpu', weights_only=False)
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                black_agent.model.load_state_dict(ckpt['model_state_dict'])
            else:
                black_agent.model.load_state_dict(ckpt)
            black_agent.model.eval()
            print(f"  Black (dg) loaded!")
        except Exception as e:
            print(f"  Failed to load Black model: {e}")
            return
    else:
        print(f"Black model not found: {BLACK_MODEL}")
        return

    # White (kg) 모델 로드
    if os.path.exists(WHITE_MODEL):
        print(f"Loading White (kg) from {WHITE_MODEL}...")
        try:
            ckpt = torch.load(WHITE_MODEL, map_location='cpu', weights_only=False)
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                white_agent.model.load_state_dict(ckpt['model_state_dict'])
            else:
                white_agent.model.load_state_dict(ckpt)
            white_agent.model.eval()
            print(f"  White (kg) loaded!")
        except Exception as e:
            print(f"  Failed to load White model: {e}")
            return
    else:
        print(f"White model not found: {WHITE_MODEL}")
        return

    # 이름 추출
    black_name = "dg"
    white_name = "kg"
    print(f"\nReady to fight! [{black_name}] vs [{white_name}]")
    print("=" * 40)

    # 통계
    black_wins = 0
    white_wins = 0
    draws = 0
    total_games = 0

    # 4. 게임 루프
    obs, _ = env.reset()
    done = False

    while not done:
        # 흑돌 턴 (kg)
        if obs['turn'] == 0:
            with torch.no_grad():
                action = black_agent.act(obs, {})
        # 백돌 턴 (dg)
        else:
            with torch.no_grad():
                action = white_agent.act(obs, {})

        # 행동 수행
        try:
            if action is not None:
                if 'index' in action: action['index'] = int(action['index'])
                if 'power' in action: action['power'] = float(action['power'])
                if 'angle' in action: action['angle'] = float(action['angle'])

                obs, reward, term, trunc, info = env.step(action)
                done = term or trunc
            else:
                print("No valid action possible.")
                done = True
        except Exception as e:
            print(f"Error during step: {e}")
            done = True

        if done:
            total_games += 1
            black_cnt = len([s for s in obs['black'] if s[2] == 1])
            white_cnt = len([s for s in obs['white'] if s[2] == 1])

            print("=" * 40)
            if white_cnt == 0:
                print(f"BLACK [{black_name}] WINS!")
                black_wins += 1
            elif black_cnt == 0:
                print(f"WHITE [{white_name}] WINS!")
                white_wins += 1
            else:
                print("DRAW")
                draws += 1

            print(f"[Stats] {black_name}: {black_wins} | {white_name}: {white_wins} | Draw: {draws}")
            print("=" * 40)

            time.sleep(3)
            obs, _ = env.reset()
            done = False

    env.close()


if __name__ == "__main__":
    evaluate()
