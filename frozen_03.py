import numpy as np
import gym
import random
import time
from IPython.display import clear_output
 
# FrozenLake-v0을 gym 환경에서 불러온다
env = gym.make("FrozenLake-v1")
 
action_space_size = env.action_space.n
state_space_size = env.observation_space.n
q_table = np.zeros((state_space_size, action_space_size))
# 최초 q table
# Row : states
# Column : actions
print(q_table)
 
# Hyperparameter 설정
debug = False
num_episodes = 10000
max_steps_per_episode = 100
learning_rate = 0.1
discount_rate = 0.99
 
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001
 
rewards_all_episodes = []
 
# Q-learning 학습
for episode in range(num_episodes):
    # 새로운 에피소드 초기화
    state = env.reset()
    done = False
    rewards_current_episode = 0
    for step in range(max_steps_per_episode):
        # Explore vs Exploit
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state, :])
        else:
            action = env.action_space.sample()
        if debug:
            print("action : ", action)
            print("state : ", state)
        # 새로운 action 취하기
        new_state, reward, done, info = env.step(action)
        # 새로운 action에 대한 결과를 반영하여 q_table 작성
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
          learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))
        # 새로운 상태로의 변경
        state = new_state
        # reward를 업데이트 한다
        rewards_current_episode += reward
        # 게임이 끝났으면 for 문을 종료한다.
        if done == True:
            break
    # Explore할 확률을 지수적으로 감소시킨다.
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
    # 현재 episode에서의 reward를 전체 episode의 reward 리스트에 넣는다
    rewards_all_episodes.append(rewards_current_episode)
  
 
# 1000번의 에피소드당 평균 성공 확률을 구한다.
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes/1000)
count = 1000
 
print("********1000 에피소드당 평균 reward ********\n")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000
 
 
# 업데이트된 q_table을 출력한다.
print("\n\n********Q_table********\n")
print(q_table)
 
# 업데이트한 q_table을 바탕으로 플레이 한다.
 
 
for episode in range(10):
    # 각 에피소드의 변수를 초기화한다.
    state = env.reset()
    done = False
    print("*****에피소드 ", episode+1, "*****\n\n\n\n")
    time.sleep(1)
    for step in range(max_steps_per_episode):
        # 현재 상태를 그려 본다.
        clear_output(wait=True)
        env.render()
        time.sleep(0.3)
        # 현재 상태에서의 q값(보상)이 가장 큰 action을 취한다.
        action = np.argmax(q_table[state, :]) 
        # 새로운 action을 취한다
        new_state, reward, done, info = env.step(action)
        if done:
            if reward == 1:
                # 만약에 Goal에 도착하여 reward가 1이라면
                print("****목표에 도달하였습니다.!****")
                time.sleep(3)
            else:
                # Goal에 도달하지 못했다면
                print("****Hole에 빠지고 말았습니다.****")
                time.sleep(3)
                clear_output(wait=True)            
            break
        
        # 새로운 상태를 설정한다.
        state = new_state
 
env.close()
