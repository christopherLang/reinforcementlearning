import gym
import numpy as np

env = gym.make('MountainCar-v0')

# Possible states:
# [0] position: -1.2 - 0.6
# [1] velicity: -0.07 - 0.07

# To create a set of discrete possible states, split position/velocity into
# categorical, and create pair-wise combinations
nsteps = 5
pos_brks = np.round(np.linspace(-1.2, 0.6, nsteps), 2)

q = (
    np.array(np.meshgrid(pos_brks, [0, 1, 2], [0])).T.
    reshape(nsteps * 3, 3)
)

gamma = 0.9

def update_q(position, action, reward, gamma):
    # for current state
    position = np.round(position, 2)
    # velocity = np.round(velocity, 2)

    cat_pos = np.digitize(position, pos_brks) - 1
    # vel_pos = np.digitize(velocity, vel_brks) - 1

    cat_pos = pos_brks[cat_pos]
    # vel_pos = vel_brks[vel_pos]

    update_index = list()

    for i in range(len(q)):
        if q[i][0] == cat_pos and q[i][2] == action:
            j = np.where(q[i][0] == pos_brks)[0][0]

            if j == 0:
                next_state = pos_brks[:2]
            elif j == (len(pos_brks) - 1):
                next_state = pos_brks[-2:]
            else:
                next_state = pos_brks[(j - 1):(j + 1)]

            max_q = np.max([j[2] for j in q if j[0] in next_state])

            r = reward + gamma * max_q

            q[i][2] = r

            update_index.append(i)

    return(update_index)


def get_action(position):
    position = np.round(position, 2)
    # velocity = np.round(velocity, 2)

    cat_pos = np.digitize(position, pos_brks) - 1
    # vel_pos = np.digitize(velocity, vel_brks) - 1

    cat_pos = pos_brks[cat_pos]
    # vel_pos = vel_brks[vel_pos]

    next_actions = list()

    for a_state in q:
        if a_state[0] == cat_pos:
            next_actions.append(a_state)

    max_q = np.max([i[2] for i in next_actions])

    next_actions = [i[1] for i in next_actions if i[2] == max_q]

    if len(next_actions) > 1:
        result = np.random.choice(next_actions)
    else:
        result = next_actions[0]

    return int(result)

for i_episode in range(20):
    observation = env.reset()

    position = observation[0]
    # velocity = observation[1]

    reward = 0
    action = get_action(position)

    for t in range(100):
        env.render()

        print(observation)

        observation, reward, done, info = env.step(action)

        position = observation[0]
        # velocity = observation[1]

        update_q(position, action, reward, gamma)

        # action = env.action_space.sample()

        if done:
            print("Episode finished after {} timesteps".format(t+1))

            break

        action = get_action(position)

env.close()