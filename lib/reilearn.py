import numpy as np
import copy


class STDL(object):
    """Simple Temporal Difference Learning Algorithm

    This reinforcement learning algorithm permits a simple discrete state,
    discrete action learning algorithm using Bellman's equation to update the
    Q-values

    Internally, the Q matrix is a 2D matrix, with states represented as rows
    and actions represented as columns

    Please see docstring for method update_q for the formula used to estimate
    new Q-values

    Attributes:
        nstates (int): Number of discrete states available
        nactions (int): Number of descrete actions available
        lrate (float): Learning rate. Affects exploration vs. exploitation
        drate (float): Discount rate. Weight given to newly estimated Q-values
        q (numpy.ndarray): A ndarray of shape (nstates, nactions)
        episodes (list): A list of dicts, containing episode performance
        episode_i (int): A counter keeping track of number of episodes
        timesteps (int): A counter keeping track of timestep for an episode
        cumulative_reward (float): Total reward received for an episode

    """

    def __init__(self, nstates, nactions, lrate=0.618, drate=0.10):
        """Instantiate SDTL class

        Args:
            nstates (int): Number of distinct states
            nactions (int): Number of distinct actions
            lrate (int): Learning rate. Affects exploration vs. exploitation.
                         Defaults to 0.618
            drate (int): Discount rate. Weight given to new Q-values. Defaults
                         to 0.10
        """

        self._nstates = nstates
        self._nactions = nactions
        self._lrate = lrate
        self._drate = drate

        self.q = np.zeros((self._nstates, self._nactions), dtype=np.float64)

        self.episodes = list()
        self.episode_i = 1
        self.timesteps = 0
        self.cumulative_reward = 0

    @property
    def lrate(self):
        return self._lrate

    @property
    def drate(self):
        return self._drate

    @lrate.setter
    def lrate(self, value):
        if value < 0 or value > 1:
            raise ValueError("lrate must be between 0 and 1, inclusive")

        self._lrate = value

    @drate.setter
    def drate(self, value):
        if value < 0 or value > 1:
            raise ValueError("drate must be between 0 and 1, inclusive")

        self._drate = value

    @lrate.getter
    def lrate(self):
        """:obj:`float`: Learning rate. Affects exploration vs. exploitation

        Learning rate values must be [0, 1]
        """
        return self._lrate

    @drate.getter
    def drate(self):
        """:obj:`float`: Discount rate. Weight given to newly estimated Q-values

        Discount rate values must be [0, 1]
        """
        return self._drate

    def _learn_q(self, obs1, obs2, action, reward):
        old_q = self.q[obs1, action]
        new_q = np.max(self.q[obs2, :])

        updated_old_q = old_q
        updated_old_q += self._lrate * (reward + self._drate * new_q - old_q)

        return updated_old_q

    def update_q(self, obs1, obs2, action, reward):
        """Update internal Q matrix with new Q-values

        For current state, a new Q-value is computed using the bellman's
        equation. The formula is:

        Q_1 = Q_1 + lrate * (reward + drate * Q_2 - Q_1)

        where...
        Q_1 : old Q-value for current state (obs1)
        Q_2 : new Q-value for next state (obs2)
        lrate : Learning rate
        drate : Discount rate
        reward : reward value

        Args:
            obs1 (int): Current state's index
            obs2 (int): Next state's index
            action (int): Taken action's index
            reward (float): Reward value offered for transitioning from current
                            to new next state (obs1 -> obs2)
        """
        self.q[obs1, action] = self._learn_q(obs1, obs2, action, reward)

    def action(self, obs):
        """Which action to take to transition to next state

        For selecting the next action, SDTL observes the Q-values for all
        actions for the current state (obs1) and selects the action with the
        largest Q-value

        If multiple actions are found, they are randomly sampled

        Args:
            obs (int): The current state's index value

        Returns:
            (int) the next action's index value
        """
        possible_actions = np.where(self.q[obs, :] == np.max(self.q[obs, :]))
        possible_actions = possible_actions[0]

        if len(possible_actions) > 1:
            action = np.random.choice(possible_actions)
        else:
            action = possible_actions[0]

        return action

    def new_episode(self):
        """Reset internal storage for tracking timesteps and cumulative rewards

        SDTL maintains the running count of the total timesteps for the
        current episode as well as cumulative rewards received

        Calling this method resets those values:
            (episode_i = 1, timesteps = 0, cumulative reward = 0)
        """
        self.episodes.append((self.episode_i, self.cumulative_reward,
                              self.timesteps))

        self.episode_i += 1
        self.timesteps = 0
        self.cumulative_reward = 0

    def update_performance(self, reward):
        """Updates internal counter for timestep and cumulative reward

        Calling this method will increment timestep by 1 and add reward to
        the cumulative reward received for a given episode

        Args:
            reward (float): Received reward for given transition
        """
        self.timesteps += 1
        self.cumulative_reward += reward

    def reset_episodes(self):
        """Reset all episode counters

        Similar to new_episode method, but episode list is also reset
        """
        self.episode_i = 1
        self.timesteps = 0
        self.cumulative_reward = 0

        self.episodes = list()

    def top_episodes(self, n=10):
        """Retrieve top n episodes by cumulative rewards

        Args:
            n (int): top n results to return

        Returns:
            :obj:`list` of :obj:`list`: Each internal list returns:
                [episode index, cumulative reward, timesteps]
        """
        self.episodes.sort(key=lambda x: x[1], reverse=True)

        if len(self.episodes) > n:
            result = self.episodes[:n]
        else:
            result = copy.deepcopy(self.episodes)

        self.episodes.sort(key=lambda x: x[0])

        return result

    def episode_performance(self):
        """Retrieve summary statistcs for episodes

        Returns:
            :obj:`dict`: With the following key/values:
                nepisodes (int): number of episodes
                mean_timesteps (float): Mean timesteps
                mean_reward (float): Mean reward across all episodes
                lrate (float): learning rate
                drate (float): discount rate
        """
        result = dict()
        result['nepisodes'] = len(self.episodes)
        result['mean_timesteps'] = np.mean([i[2] for i in self.episodes])
        result['mean_reward'] = np.mean([i[1] for i in self.episodes])
        result['std_reward'] = np.std([i[1] for i in self.episodes])
        result['lrate'] = self._lrate
        result['drate'] = self._drate

        return result


class SDTDL(STDL):
    def __init__(self, statespec, actionspec, lrate=0.618, drate=0.10):
        if isinstance(statespec, DiscreteParameter):
            self._state_is_continuous = True
            self._state_min = statespec.num_range[0]
            self._state_max = statespec.num_range[1]
            self._statespec = statespec

            nstate = statespec.nclasses
        elif isinstance(statespec, int):
            self._state_is_continuous = False
            self._state_min = None
            self._state_max = None
            self._statespec = None

            nstate = statespec
        else:
            raise TypeError('statespec not DiscreteParameter class, or int')

        if isinstance(actionspec, DiscreteParameter):
            self._action_is_continuous = True
            self._action_min = actionspec.num_range[0]
            self._action_max = actionspec.num_range[1]
            self._actionspec = actionspec

            naction = actionspec.nclasses
        elif isinstance(actionspec, int):
            self._action_is_continuous = False
            self._action_min = None
            self._action_max = None
            self._actionspec = None

            naction = actionspec
        else:
            raise TypeError('actionspec not DiscreteParameter class, or int')

        super(SDTDL, self).__init__(nstate, naction, lrate, drate)

    @property
    def state_is_continuous(self):
        return self._state_is_continuous

    @property
    def state_min(self):
        return self._state_min

    @property
    def state_max(self):
        return self._state_max

    @property
    def statespec(self):
        return self._statespec

    @property
    def action_is_continuous(self):
        return self._action_is_continuous

    @property
    def action_min(self):
        return self._action_min

    @property
    def action_max(self):
        return self._action_max

    @property
    def actionspec(self):
        return self._actionspec

    def update_q(self, obs1, obs2, action, reward):
        """Update internal Q matrix with new Q-values

        For current state, a new Q-value is computed using the bellman's
        equation. The formula is:

        Q_1 = Q_1 + lrate * (reward + drate * Q_2 - Q_1)

        where...
        Q_1 : old Q-value for current state (obs1)
        Q_2 : new Q-value for next state (obs2)
        lrate : Learning rate
        drate : Discount rate
        reward : reward value

        Args:
            obs1 (int): Current state's index
            obs2 (int): Next state's index
            action (int): Taken action's index
            reward (float): Reward value offered for transitioning from current
                            to new next state (obs1 -> obs2)
        """
        if self._state_is_continuous is True:
            obs1 = self._statespec.classify(obs1)['class']
            obs2 = self._statespec.classify(obs2)['class']

        if self._action_is_continuous is True:
            action = self._actionspec.classify(action)['class']

        super(SDTDL, self).update_q(obs1, obs2, action, reward)

    def action(self, obs):
        if self._state_is_continuous is True:
            obs = self._statespec.classify(obs)['class']

        return super(SDTDL, self).action(obs)


class DiscreteParameter(object):
    def __init__(self, minimum, maximum, n, discretizer=None):
        if discretizer is None:
            self.discretizer = discretizer_equal

        self._numbreaks = self.discretizer(minimum, maximum, n)
        self._nclasses = len(self._numbreaks) - 1
        self._break_index = np.array(range(self._nclasses))
        self._numrange = np.array((minimum, maximum))

    @property
    def numbreaks(self):
        return self._numbreaks

    @property
    def nclasses(self):
        return self._nclasses

    @property
    def break_index(self):
        return self._break_index

    @property
    def num_range(self):
        return self._numrange

    def classify(self, value):
        left_i = np.where(self._numbreaks <= value)[0][-1]
        right_i = np.where(value < self._numbreaks)[0]

        if len(right_i) == 0:
            right_i = len(self._numbreaks) - 1
            left_i -= 1
        else:
            right_i = right_i[0]

        og_intervals = (self._numbreaks[left_i], self._numbreaks[right_i])

        return {'intervals': og_intervals, 'class': self.break_index[left_i]}


def discretizer_equal(minimum, maximum, n):
    return np.linspace(minimum, maximum, n + 1, dtype=np.float64)
