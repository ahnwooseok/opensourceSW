# import gym
# env = gym.make('FrozenLake-v1')
# env.reset()
# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample()) # take a random action
# env.close()


import gym
env = gym.make('FrozenLake-v1',is_slippery=False)
for i_episode in range(1000):
    observation = env.reset()
    for t in range(100):
        env.render()
        #print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        #print(observation,reward,done,info)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()