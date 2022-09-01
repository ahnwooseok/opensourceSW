import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()


# import gym
# env = gym.make('CartPole-v0')
# for i_episode in range(100):
#     # observation = env.reset()
#     for tac in range(10000):
#         env.render()
#         print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(tac+1))
#             break