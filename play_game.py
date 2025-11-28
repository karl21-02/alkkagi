import gymnasium as gym
import torch
import numpy as np
import os
import time
import random

# ggagi.py 파일에서 클래스들을 가져옵니다.
from alkkagi_dg import YourBlackAgent, YourWhiteAgent

# ==========================================
# 설정: 누구랑 싸울 것인가?
# "Random": 랜덤 봇 (샌드백)
# "RuleBased": 규칙 기반 봇 (선생님)
# "AI": 학습된 모델 (미러전)
# ==========================================
#OPPONENT_TYPE = "Random" 
#OPPONENT_TYPE = "RuleBased"
OPPONENT_TYPE = "AI" 

def get_random_action(obs):
    """평가용 랜덤 행동 생성기"""
    alive_indices = [i for i, s in enumerate(obs['white']) if s[2] == 1]
    if not alive_indices: return None
    
    return {
        "turn": 1,
        "index": random.choice(alive_indices),
        "power": random.uniform(300, 2000),
        "angle": random.uniform(-180, 180)
    }

def evaluate():
    # 1. 환경 생성
    env = gym.make(
        id='kymnasium/AlKkaGi-3x3-v0',
        render_mode='human',
        bgm=True,
        obs_type='custom'
    )

    # 2. 나(Black) 생성 - 학습된 AI
    black_agent = YourBlackAgent()
    
    # 3. 상대(White) 생성 - 설정에 따라 다름
    if OPPONENT_TYPE == "AI":
        white_agent = YourWhiteAgent() # [중요] WhiteAgent로 생성해야 함!
    else:
        white_agent = None # Random은 함수로 처리

    # 4. 모델 불러오기
    model_path = "my_alkkagi_agent.pkl"
    
    if os.path.exists(model_path):
        print(f"📂 Loading model from {model_path}...")
        try:
            ckpt = torch.load(model_path, map_location='cpu')
            
            # Black Agent 로드
            black_agent.model.load_state_dict(ckpt)
            black_agent.model.eval()
            
            # 상대가 AI라면 상대에게도 모델 로드
            if OPPONENT_TYPE == "AI":
                white_agent.model.load_state_dict(ckpt)
                white_agent.model.eval()
                
            print(f"✅ Ready to fight! (VS {OPPONENT_TYPE})")
        except Exception as e:
            print(f"⚠️ Load failed: {e}")
            return
    else:
        print("❌ Model file not found. Train first!")
        return

    # 5. 게임 루프
    obs, _ = env.reset()
    done = False

    while not done:
        # 흑돌 (나 - AI)
        if obs['turn'] == 0: 
            with torch.no_grad():
                # act 메서드 내부에서 obs 처리 및 action 반환
                action = black_agent.act(obs, {})

        # 백돌 (상대)
        else: 
            if OPPONENT_TYPE == "AI":
                with torch.no_grad():
                    action = white_agent.act(obs, {})
            
            elif OPPONENT_TYPE == "RuleBased":
                # RuleBasedAgent는 딕셔너리를 바로 반환함
                action = white_agent.get_action(obs, 1)
            
            else: # Random
                action = get_random_action(obs)

        # 행동 수행
        # (가끔 죽은 돌을 선택하는 경우 에러 방지용 try-except)
        try:
            if action is not None:
                # numpy 타입 등을 float/int로 변환 (안전장치)
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
            black_cnt = len([s for s in obs['black'] if s[2] == 1])
            white_cnt = len([s for s in obs['white'] if s[2] == 1])
            
            print("="*30)
            if white_cnt == 0:
                print("🏆 BLACK (AI) WINS!")
            elif black_cnt == 0:
                print("💀 WHITE (Opponent) WINS!")
            else:
                print("🤝 DRAW")
            print("="*30)
            
            time.sleep(3)
            obs, _ = env.reset()
            done = False

    env.close()

if __name__ == "__main__":
    evaluate()