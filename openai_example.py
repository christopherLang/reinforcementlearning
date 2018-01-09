# This is example code from OpenAI tutorial at:
# https://gym.openai.com/docs/
#
# January 8, 2018
import gym

env = gym.make('CartPole-v0')

for i_episode in range(20):
    observation = env.reset()

    for t in range(100):
        env.render()

        print(observation)

        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        if done:
            print("Episode finished after {} timesteps".format(t + 1))

            break

env.close()