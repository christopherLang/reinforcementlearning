import gym
import sys
import os
import time
from tqdm import tqdm

os.chdir('/home/chlang/Desktop/reinforcementlearning')
sys.path.append('lib')

from reilearn import STDL

env = gym.make('Taxi-v2')

agent = STDL(env.observation_space.n, env.action_space.n, lrate=0.80,
             drate=0.2)

max_epi = 1000000
for i_episode in tqdm(range(max_epi), ncols=80):
    agent.new_episode()

    obs_1 = env.reset()
    action_num = agent.action(obs_1)

    done = False

    while done is not True:
        obs_2, reward, done, info = env.step(action_num)

        agent.update_q(obs_1, obs_2, action_num, reward)

        obs_1 = obs_2

        action_num = agent.action(obs_1)

        agent.update_performance(reward)

env.close()


# visually verify
agent.reset_episodes()

for i_episode in range(100):
    agent.new_episode()

    obs = env.reset()

    done = False

    while done is not True:
        env.render()
        action_num = agent.action(obs)

        obs, reward, done, info = env.step(action_num)

        agent.update_performance(reward)

        time.sleep(0.15)

env.close()
