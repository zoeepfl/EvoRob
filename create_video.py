from src.EA.CMAES_sol import CMAES_sol, CMAES_opts
from src.EA.ES import ES, ES_opts
from src.EA.NSGA_sol import NSGAII_sol, NSGA_opts
from src.world.World import World
from src.world.robot.controllers import MLP
from src.world.robot.morphology.AntCustomRobot import AntRobot
from src.utils.Filesys import get_project_root
from gymnasium.vector import AsyncVectorEnv
from xml.etree.ElementTree import Element, SubElement, tostring, ElementTree
from Exercise3_sol import generate_best_individual_video, generate_random_ant_world
from Exercise3_sol import AntWorld,AntRobot

import xml.etree.ElementTree as xml
import xml.dom.minidom as minidom
import gymnasium as gym
import numpy as np
import os
import random
import time

import matplotlib.pyplot as plt

ROOT_DIR = get_project_root()
ENV_NAME = 'Ant_custom'

ROOT_EXT='/run/media/epuck/Tifaine Mez/EvoRobot_ results/'

def main():

    generate_random_ant_world(
        file_path=os.path.join(ROOT_DIR, "src", "world", "robot", "assets", "ant_world.xml"),
        #n_rocks=900,  # ou fixe : n_rocks=50
        #area_size=15,
        #min_size=0.005,
        #max_size=0.1
    )

    world = AntWorld()
    #results_dir = os.path.join(ROOT_DIR, 'results', ENV_NAME, 'single_es')
    results_dir = os.path.join(ROOT_EXT,'single_es')

    print("root dir",ROOT_DIR)
    print("result_dir",results_dir)

    # %% visualise
    # TODO: Make a video of the best individual, and plot the fitness curve.
    best_individual = np.load(os.path.join(results_dir, "128", "x_best.npy"))

    points, connectivity_mat = world.geno2pheno(best_individual)
    robot = AntRobot(points, connectivity_mat, world.joint_limits, world.joint_axis, verbose=False)
    robot.xml = robot.define_robot()
    robot.write_xml()

    # % Defining the Robot environment in MuJoCo
    world_xml = xml.parse(os.path.join(ROOT_DIR, 'src', 'world', 'robot', 'assets', "ant_world.xml"))
    robot_env = world_xml.getroot()

    robot_env.append(xml.Element("include", attrib={"file": "AntRobot.xml"}))
    world_xml = xml.tostring(robot_env, encoding='unicode')
    with open(world.world_file, "w") as f:
        f.write(world_xml)

    generate_best_individual_video(world,os.path.join(ROOT_EXT,'evo_100+25+25gen_v1.mp4'))


if __name__ == "__main__":
    main()