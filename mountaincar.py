import gym
import sys
import os
from tqdm import tqdm

os.chdir('/home/chlang/Desktop/reinforcementlearning')
sys.path.append('lib')

from reilearn import SDTDL, DiscreteParameter

env = gym.make('MountainCar-v0')

vel_param = DiscreteParameter(env.observation_space.low[0],
                              env.observation_space.high[0],
                              20)

agent = SDTDL(vel_param, env.action_space.n, lrate=0.80, drate=0.2)

max_epi = 100000
for i_episode in tqdm(range(max_epi), ncols=80):
    agent.new_episode()

    obs_1 = env.reset()[0]
    action_num = agent.action(obs_1)

    done = False

    while done is not True:
        obs_2, reward, done, info = env.step(action_num)

        agent.update_q(obs_1, obs_2[0], action_num, reward)

        obs_1 = obs_2[0]

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

        action_num = agent.action(obs[1])

        obs, reward, done, info = env.step(action_num)

        agent.update_performance(reward)

        # time.sleep(0.15)

env.close()
