"""
Luby environment from
"Dynamic Algorithm Configuration:Foundation of a New Meta-Algorithmic Framework"
by A. Biedenkapp and H. F. Bozkurt and T. Eimer and F. Hutter and M. Lindauer.
Original environment authors: André Biedenkapp, H. Furkan Bozkurt
"""

from typing import List
import numpy as np

from dacbench import AbstractEnv


# Instance IDEA 1: shift luby seq -> feat is sum of skipped action values
# Instance IDEA 2: "Wiggle" luby i.e. luby(t + N(0, 0.1)) -> feat is sampled value


class LubyEnv(AbstractEnv):
    """
    Environment to learn Luby Sequence
    """

    def __init__(self, config) -> None:
        """
        Initialize Luby Env

        Parameters
        -------
        config : objdict
            Environment configuration
        """
        super().__init__(config)

        self._hist_len = config["hist_length"]
        self._ms = self.n_steps
        self._mi = config["min_steps"]
        self._state = np.array([-1 for _ in range(self._hist_len + 1)])
        self._r = 0
        self._genny = luby_gen(1)
        self._next_goal = next(self._genny)
        # Generate luby sequence up to 2*max_steps + 2 as mode 1 could potentially shift up to max_steps
        self._seq = np.log2(
            [next(luby_gen(i)) for i in range(1, 2 * config["cutoff"] + 2)]
        )
        self._jenny_i = 1
        self._start_dist = None
        self._sticky_dis = None
        self._sticky_shif = 0
        self._start_shift = 0
        self.__lower, self.__upper = 0, 0
        self.__error = 0
        self.done = None
        self.action = None

        if "reward_function" in config.keys():
            self.get_reward = config["reward_function"]
        else:
            self.get_reward = self.get_default_reward

        if "state_method" in config.keys():
            self.get_state = config["state_method"]
        else:
            self.get_state = self.get_default_state

    def step(self, action: int):
        """
        Execute environment step

        Parameters
        ----------
        action : int
            action to execute

        Returns
        -------
        np.array, float, bool, dict
            state, reward, done, info
        """
        self.done = super(LubyEnv, self).step_()
        self.prev_state = self._state.copy()
        self.action = action
        reward = self.get_reward(self)
        if (
            self.__error < self.__lower
        ):  # needed to avoid too long sequences of sticky actions
            self.__error += np.abs(self.__lower)
        elif self.__error > self.__upper:
            self.__error -= np.abs(self.__upper)
        self._jenny_i += 1
        self.__error += self._sticky_shif

        # next target in sequence at step luby_t is determined by the current time step (jenny_i), the start_shift
        # value and the sticky error. Additive sticky error leads to sometimes rounding to the next time_step and
        # thereby repeated actions. With check against lower/upper we reset the sequence to the correct timestep in
        # the t+1 timestep.
        luby_t = max(1, int(np.round(self._jenny_i + self._start_shift + self.__error)))
        self._next_goal = self._seq[luby_t - 1]
        return self.get_state(self), reward, self.done, {}

    def reset(self) -> List[int]:
        """
        Resets env

        Returns
        -------
        numpy.array
            Environment state
        """
        super(LubyEnv, self).reset_()
        self._start_shift = self.instance[0]
        self._sticky_shif = self.instance[1]
        self._r = 0
        self.n_steps = self._mi

        self.__error = 0 + self._sticky_shif
        self._jenny_i = 1
        luby_t = max(1, int(np.round(self._jenny_i + self._start_shift + self.__error)))
        self._next_goal = self._seq[luby_t - 1]
        self.done = False
        return self.get_state(self)

    def get_default_reward(self, _):
        if self.action == self._next_goal:
            self._r = 0  # we don't want to allow for exploiting large rewards by tending towards long sequences
        else:  # mean and var chosen s.t. ~1/4 of rewards are positive
            self._r = -1
        self._r = max(self.reward_range[0], min(self.reward_range[1], self._r))
        return self._r

    def get_default_state(self, _):
        if self.c_step == 0:
            self._state = [-1 for _ in range(self._hist_len + 1)]
        else:
            if self.c_step - 1 < self._hist_len:
                self._state[(self.c_step - 1)] = self.action
            else:
                self._state[:-2] = self._state[1:-1]
                self._state[-2] = self.action
            self._state[-1] = self.c_step - 1
        next_state = np.array(self._state if not self.done else self.prev_state)
        return next_state

    def close(self) -> bool:
        """
        Close Env

        Returns
        -------
        bool
            Closing confirmation
        """
        return True

    # TODO: this should render!

    def render(self, mode: str = "human") -> None:
        """
        Render env in human mode

        Parameters
        ----------
        mode : str
            Execution mode
        """
        if mode != "human":
            raise NotImplementedError

        pass


def luby_gen(i):
    """ Generator for the Luby Sequence """
    for k in range(1, 33):
        if i == ((1 << k) - 1):
            yield 1 << (k - 1)

    for k in range(1, 9999):
        if 1 << (k - 1) <= i < (1 << k) - 1:
            for x in luby_gen(i - (1 << (k - 1)) + 1):
                yield x
