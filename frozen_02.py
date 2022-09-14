import gym
env = gym.make('FrozenLake-v1',is_slippery=False)
for i_episode in range(1):
    observation = env.reset()
    for t in range(100):
        env.render()
        #action = env.action_space.sample()
        action = int(input("Enter a #:left-0 down-1 right-2 up-3:"))
        print(action)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()