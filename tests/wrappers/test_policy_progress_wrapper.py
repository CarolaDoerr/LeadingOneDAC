import unittest
from unittest import mock

import numpy as np
from dacbench.benchmarks import SigmoidBenchmark
from dacbench.wrappers import PolicyProgressWrapper


def _sig(x, scaling, inflection):
    return 1 / (1 + np.exp(-scaling * (x - inflection)))


def compute_optimal_sigmoid(instance):
    sig_values = [_sig(i, instance[1], instance[0]) for i in range(10)]
    optimal = [np.around(x) for x in sig_values]
    return optimal


class TestPolicyProgressWrapper(unittest.TestCase):
    def test_init(self):
        bench = SigmoidBenchmark()
        bench.set_action_values((3,))
        env = bench.get_environment()
        wrapped = PolicyProgressWrapper(env, compute_optimal_sigmoid)
        self.assertTrue(len(wrapped.policy_progress) == 0)
        self.assertTrue(len(wrapped.episode) == 0)
        self.assertFalse(wrapped.compute_optimal is None)

    def test_step(self):
        bench = SigmoidBenchmark()
        bench.set_action_values((3,))
        bench.config.instance_set = {0: [0, 0], 1: [1, 1], 2: [3, 4], 3: [5, 6]}
        env = bench.get_environment()
        wrapped = PolicyProgressWrapper(env, compute_optimal_sigmoid)

        wrapped.reset()
        _, _, done, _ = wrapped.step(1)
        self.assertTrue(len(wrapped.episode) == 1)
        while not done:
            _, _, done, _ = wrapped.step(1)
        self.assertTrue(len(wrapped.episode) == 0)
        self.assertTrue(len(wrapped.policy_progress) == 1)

    @mock.patch("dacbench.wrappers.policy_progress_wrapper.plt")
    def test_render(self, mock_plt):
        bench = SigmoidBenchmark()
        bench.set_action_values((3,))
        env = bench.get_environment()
        env = PolicyProgressWrapper(env, compute_optimal_sigmoid)
        for _ in range(2):
            done = False
            env.reset()
            while not done:
                _, _, done, _ = env.step(1)
        env.render_policy_progress()
        self.assertTrue(mock_plt.show.called)
