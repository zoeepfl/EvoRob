from src.EA.CMAES_sol import CMAES_sol, CMAES_opts
from src.EA.NSGA import NSGAII, NSGA_opts
from src.world.World import World
from src.world.robot.controllers import MLP
from src.world.robot.morphology.AntCustomRobot import AntRobot
from src.utils.Filesys import get_project_root
from gymnasium.vector import AsyncVectorEnv
from Exercise3_sol import AntWorld
from Exercise3_sol import AntRobot
from Exercise3_sol import visualise_individual

import xml.etree.ElementTree as xml
import gymnasium as gym
import numpy as np
import os

ROOT_DIR = get_project_root()
ENV_NAME = 'Ant_custom'


def visualize_genome(genome_path=None, generation=None, results_dir="results/Ant_custom/single"):
    """
    Visualise un génome donné sous forme de robot dans MuJoCo.
    
    Args:
        genome_path (str): chemin vers un fichier .npy contenant le vecteur génétique.
        generation (int): génération d'où charger le meilleur individu (si genome_path n'est pas fourni).
        results_dir (str): répertoire où sont stockés les résultats.
    """

    world = AntWorld()
    if genome_path is None and generation is None:
        raise ValueError("Tu dois spécifier soit `genome_path`, soit `generation`.")

    if genome_path is None:
        genome_path = os.path.join(results_dir, str(generation), "x_best.npy")


    # Charge le génome
    best_individual = np.load(genome_path)

    visualise_individual(best_individual)
    # Génère la vidéo
    #generate_best_individual_video(world)
    print(f"Visualisation terminée pour le génome : {genome_path}")

# Exemple d'utilisation :
if __name__ == "__main__":
    # Option 1 : Visualiser par génération
    for i in range(10):
        visualize_genome(generation=298)

    # Option 2 : Visualiser un génome précis
    # visualize_genome(genome_path="results/Ant_custom/single/35/x_best.npy")
