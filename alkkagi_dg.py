import gymnasium as gym
import torch
import numpy as np
import os
import time
import random

# alkkagi.py에서 클래스 임포트
from alkkagi import YourBlackAgent, YourWhiteAgent

# ==========================================
# ⚙️ 설정 (Configuration)
# ==========================================
MODEL_PATH = "my_alkkagi_agent.pkl"

# 상대방 설정: "Random" 또는 "AI"
OPPONENT_TYPE = "AI"

# 내 진영 설정: "Black", "White", "Random"
MY_SIDE = "BLack"

# 렌더링 속도 (초 단위, 0이면 최대 속도)
RENDER_DELAY = 0.1


def evaluate():
    # 1. 환경 생성
    env = gym.make(
        id='kymnasium/AlKkaGi-3x3-v0',
        render_mode='human',
        bgm=True,
        obs_type='custom'
    )

    # 2. 내 진영 결정
    final_my_side = MY_SIDE
    if final_my_side == "Random":
        final_my_side = random.choice(["Black", "White"])

    print(f"\n🎮 Player Side: {final_my_side} | Opponent: {OPPONENT_TYPE}")

    # 3. 에이전트 로드 (ggagi.py의 load 메서드 활용)
    black_agent = None
    white_agent = None

    if os.path.exists(MODEL_PATH):
        print(f"📂 Loading model from {MODEL_PATH}...")

        # 흑돌 에이전트 설정
        if final_my_side == "Black":
            black_agent, _ = YourBlackAgent.load(MODEL_PATH)  # 내 모델
            if OPPONENT_TYPE == "AI":
                white_agent = YourWhiteAgent.load(MODEL_PATH)  # 상대(AI)도 같은 모델
        else:  # 내가 White
            white_agent = YourWhiteAgent.load(MODEL_PATH)  # 내 모델
            if OPPONENT_TYPE == "AI":
                black_agent, _ = YourBlackAgent.load(MODEL_PATH)  # 상대(AI)도 같은 모델
    else:
        print("❌ Model file not found. Playing with Random Agents.")

    # 만약 로드에 실패했거나 파일이 없으면 RandomBot 대응을 위해 None으로 유지
    # (YourBlackAgent()로 생성하면 깡통 뇌가 되어 랜덤보다 못할 수 있음)

    print(f"✅ Ready to fight!\n")

    obs, _ = env.reset()
    done = False

    while not done:
        if RENDER_DELAY > 0: time.sleep(RENDER_DELAY)

        current_turn = obs['turn']
        action = None

        # ---------------------------------------
        # 1. 흑돌 차례 (Turn 0)
        # ---------------------------------------
        if current_turn == 0:
            # 흑돌이 AI(나 or 상대AI)인 경우
            if black_agent is not None:
                with torch.no_grad():
                    # ggagi.py의 act는 내부에서 배치 처리하므로 obs를 그대로 넘김
                    action = black_agent.act(obs, {})

        # ---------------------------------------
        # 2. 백돌 차례 (Turn 1)
        # ---------------------------------------
        else:
            # 백돌이 AI(나 or 상대AI)인 경우
            if white_agent is not None:
                with torch.no_grad():
                    action = white_agent.act(obs, {})

        # ---------------------------------------
        # 3. 행동 실행
        # ---------------------------------------
        try:
            if action is not None:
                # 데이터 타입 보정 (필수)
                if 'index' in action: action['index'] = int(action['index'])
                if 'power' in action: action['power'] = float(action['power'])
                if 'angle' in action: action['angle'] = float(action['angle'])

                obs, reward, term, trunc, info = env.step(action)
                done = term or trunc
            else:
                print("⚠️ No valid action possible (All stones dead?)")
                done = True
        except Exception as e:
            print(f"❌ Error during step: {e}")
            done = True

        # ---------------------------------------
        # 4. 게임 종료 처리
        # ---------------------------------------
        if done:
            black_cnt = len([s for s in obs['black'] if s[2] == 1])
            white_cnt = len([s for s in obs['white'] if s[2] == 1])

            winner = "None"
            if white_cnt == 0:
                winner = "Black"
            elif black_cnt == 0:
                winner = "White"

            print("=" * 40)
            if winner == final_my_side:
                print(f"🏆 YOU ({winner}) WIN! 🎉")
            elif winner == "None":
                print("🤝 DRAW (Timeout)")
            else:
                print(f"💀 YOU ({final_my_side}) LOSE... (Winner: {winner})")
            print("=" * 40)

            time.sleep(2)

            # 게임 재설정
            obs, _ = env.reset()
            done = False

            # 랜덤 진영 모드일 경우 다음 판 진영 변경
            if MY_SIDE == "Random":
                final_my_side = random.choice(["Black", "White"])
                print(f"\n🔄 Switching Sides! You are now: {final_my_side}")

                # 에이전트 포지션 재배치
                # (모델은 이미 로드되어 있으므로 변수만 스왑하면 됨)
                # 주의: 단순히 변수만 바꾸면 안되고, '나'와 '상대'의 정체성에 맞게 다시 할당해야 함
                # 하지만 가장 쉬운 방법은 위 로직을 그대로 다시 타는 것

                # 간단히 다시 로드 로직 수행 (메모리 낭비 거의 없음)
                if os.path.exists(MODEL_PATH):
                    if final_my_side == "Black":
                        black_agent = YourBlackAgent.load(MODEL_PATH)
                        white_agent = YourWhiteAgent.load(MODEL_PATH) if OPPONENT_TYPE == "AI" else None
                    else:
                        white_agent = YourWhiteAgent.load(MODEL_PATH)
                        black_agent = YourBlackAgent.load(MODEL_PATH) if OPPONENT_TYPE == "AI" else None

    env.close()


if __name__ == "__main__":
    evaluate()