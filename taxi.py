import gym
import sys
import os
import time
from tqdm import tqdm

os.chdir('/home/chlang/Desktop/reinforcementlearning')
sys.path.append('lib')

from reilearn import DiscreteLearn

rlenv = gym.make('Taxi-v2')

agent = DiscreteLearn(rlenv.observation_space.n, rlenv.action_space.n)

max_epi = 1000000
for i_episode in tqdm(range(max_epi), ncols=80):
    obs1 = rlenv.reset()
    action_num = agent.action(obs1)

    done = False
    while done is not True:
        obs2, reward, done, info = rlenv.step(action_num)

        agent.update_q(obs1, obs2, action_num, reward)

        obs1 = obs2

        action_num = agent.action(obs1)

        agent.update_performance(reward)

rlenv.close()


# visually verify
agent.reset_episodes()

for i_episode in range(100):
    agent.new_episode()

    obs = rlenv.reset()

    done = False

    while done is not True:
        rlenv.render()
        action_num = agent.action(obs)

        obs, reward, done, info = rlenv.step(action_num)

        agent.update_performance(reward)

        time.sleep(0.15)

rlenv.close()
