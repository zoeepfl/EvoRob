import unittest
import numpy as np
import gymnasium as gym
from Exercise3_sol import AntWorld
from src.utils.Filesys import get_project_root
import os


class TestAntWorld(unittest.TestCase):
    def setUp(self):
        self.world = AntWorld()
        self.genotype_dim = self.world.n_params

    def test_antworld_single_individual(self):
        genotype = np.random.uniform(-1, 1, self.genotype_dim)

        fitness, multi_obj_fitness = self.world.evaluate_individual(genotype)

        # Vérifie que la fitness est bien définie et non NaN
        self.assertTrue(np.isfinite(fitness), "La fitness simple n'est pas valide")
        self.assertEqual(len(multi_obj_fitness), 2, "La fitness multi-objectif ne contient pas 2 objectifs")
        self.assertTrue(np.all(np.isfinite(multi_obj_fitness)), "La fitness multi-objectif contient des NaN")

        print(f"Fitness: {fitness}, Multi-objective: {multi_obj_fitness}")


if __name__ == '__main__':
    unittest.main()
