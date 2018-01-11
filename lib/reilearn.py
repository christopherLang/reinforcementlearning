import numpy as np


class DiscreteQLearn(object):
    """Simple Temporal Difference Learning Algorithm

    This reinforcement learning algorithm permits a simple discrete state(s),
    discrete action(s) learning algorithm using Bellman's equation to update
    the Q-values

    Internally, the Q matrix is a numpy.ndarray, with the first n-dimensions
    representing the states, and the second m-dimensions representing actions.
    Hence this class supports multidimensional states and actions

    Please see docstring for method update_q for the formula used to estimate
    new Q-values

    Attributes:
        lrate : float
            Learning rate. Affects exploration vs. exploitation
        drate : float
            Discount rate. Weight given to newly estimated Q-values
        nstates : int
            Number of discrete states available
        nactions : int
            Number of descrete actions available
        qmatshape : tuple of int
            The shape of `qmat`, the Q-Matrix
        qmat : numpy.ndarray
            The Q-matrix, a `numpy.ndarray` storing the Q-values
    """

    def __init__(self, nstates, nactions, lrate=0.618, drate=0.10):
        """Construct DiscreteQLearn object

        Args:
            nstates : int, iterable of int, (required)
                Specifies the dimension of states and possible values.
                If `int` is provided, it is assumed to be a 1D state with
                `nstates` possible states. If a `tuple` of `int` is provided,
                the length of the tuple` specifies the number of dimensions
                there is, and the integers specify the number of states per
                dimension
            nactions : int, iterable of int, (required)
                Similar to `nstates`, but for actions
            lrate : float, (default to 0.618)
                Learning rate. Affects exploration vs. exploitation
            drate : float (default to 0.10)
                Discount rate. Weight given to newly estimated Q-values
        """
        if isinstance(nstates, int):
            nstates = (nstates,)
        else:
            nstates = tuple(nstates)

        if isinstance(nactions, int):
            nactions = (nactions,)
        else:
            nactions = tuple(nactions)

        self._nstates = len(nstates)
        self._nactions = len(nactions)
        self._lrate = lrate
        self._drate = drate

        qmat_shape = self._nstates + self._nactions
        self._qmat = np.zeros(qmat_shape, dtype=np.float64)
        self._qmatshape = self._qmat.shape

    @property
    def lrate(self):
        """(float) Learning rate. Affects exploration vs. exploitation

        Learning rate values must be [0, 1]
        """
        return self._lrate

    @property
    def drate(self):
        """(float) Discount rate. Weight given to newly estimated Q-values

        Discount rate values must be [0, 1]
        """
        return self._drate

    @property
    def nstates(self):
        """(int) Count of dimensions of states"""
        return self._nstates

    @property
    def nactions(self):
        """(int) Count of dimensions of actions"""
        return self._nactions

    @property
    def qmatshape(self):
        """(tuple of int) The shape of the Q-matrix"""
        return self._qmatshape

    @property
    def qmat(self):
        """(numpy.ndarray) The Q-matrix"""
        return self._qmat

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

    def _learn_q(self, obs1, obs2, action, reward):
        old_q = self._qmat[obs1 + action]
        new_q = np.max(self._qmat[obs2])

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
            obs1 : int, iterable of int, (required)
                The current state's index. If `int`, than the value itself is
                the index for 1-dimension. If `tuple`, than values inside is
                the indices for the corresponding dimension. The length should
                equal the dimension of the state
            obs2 : int, iterable of int, (required)
                The next state's index. Structure is the same as `obs1`
            action : int, iterable of int, (required)
                The action's (taken to transition from `obs1` to `obs2`) index.
                Structure is the same as `obs1`, but for actions
            reward : float
                The reward value for transitioning from `obs1` to `obs2`
        """
        obs1 = (obs1,) if isinstance(obs1, int) else obs1
        obs2 = (obs2,) if isinstance(obs2, int) else obs2
        action = (action,) if isinstance(action, int) else action

        self._qmat[obs1 + action] = self._learn_q(obs1, obs2, action, reward)

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
        possible_actions = self._qmat[obs] == np.max(self._qmat[obs])
        possible_actions = np.where(possible_actions)

        possible_actions = np.stack(possible_actions).T

        if len(possible_actions) > 1:
            action = np.random.choice(np.arange(len(possible_actions)))
            action = tuple(possible_actions[action].tolist())
        else:
            action = tuple(possible_actions[0])

        return action


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

        qmat_shape = (self._nstates, self._nactions)
        self._qmat = np.zeros(qmat_shape, dtype=np.float64)

    @property
    def lrate(self):
        """:obj:`float`: Learning rate. Affects exploration vs. exploitation

        Learning rate values must be [0, 1]
        """
        return self._lrate

    @property
    def drate(self):
        """:obj:`float`: Discount rate. Weight given to newly estimated Q-values

        Discount rate values must be [0, 1]
        """
        return self._drate

    @property
    def qmat(self):
        return self._qmat

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

    def _learn_q(self, obs1, obs2, action, reward):
        old_q = self._qmat[obs1, action]
        new_q = np.max(self._qmat[obs2, :])

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
        self._qmat[obs1, action] = self._learn_q(obs1, obs2, action, reward)

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
        possible_actions = self._qmat[obs, :] == np.max(self._qmat[obs, :])
        possible_actions = np.where(possible_actions)
        possible_actions = possible_actions[0]

        if len(possible_actions) > 1:
            action = np.random.choice(possible_actions)
        else:
            action = possible_actions[0]

        return action


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

def discretizer_yellow():
    """hello
    """