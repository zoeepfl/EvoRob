import os
import unittest
import gymnasium as gym
import numpy as np

from src.utils.Filesys import get_project_root
from src.world.envs.TestFunctions import f_reversed_ackley
from src.EA.EA import EA, EA_opts


ROOT_DIR = get_project_root()


class MyTestCase(unittest.TestCase):
    def test_gym(self):
        ENV_NAMES = [ "Ant-v5"] ### "HalfCheetah-v5", "Ant-v5"

        for ENV_NAME in ENV_NAMES:
            env = gym.make(
                ENV_NAME,
                render_mode='human')
            rewards = None
            env.reset()
            for step in range(1000):
                actions = np.random.uniform(low=-0.1, high=0.1, size=env.action_space.shape[0])
                observations, rewards, terminated, truncated, info = env.step(actions)
                if terminated:
                    break
            env.close()
            self.assertFalse(rewards is None, "Gym environment invalid")


    def test_functions(self):
        pop_size = 100
        n_params =2
        results_dir = os.path.join(ROOT_DIR, "results", "TEST")

        ea = EA(pop_size, n_params, EA_opts, results_dir)

        for _ in range(ea.n_gen):
            pop = ea.ask()
            fitnesses_gen = np.empty(ea.n_pop)
            for index, individual in enumerate(pop):
                fit_ind = f_reversed_ackley(*individual)
                fitnesses_gen[index] = fit_ind
            ea.tell(pop, fitnesses_gen)
        self.assertLess(-0.1, ea.x_best_so_far.max())


if __name__ == '__main__':
    unittest.main()
