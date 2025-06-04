from src.EA.CMAES_sol import CMAES_sol, CMAES_opts
from src.EA.NSGA_sol import NSGAII_sol, NSGA_opts
from src.world.World import World
from src.world.robot.controllers import MLP
from src.world.robot.morphology.AntCustomRobot import AntRobot
from src.utils.Filesys import get_project_root
from gymnasium.vector import AsyncVectorEnv
from xml.etree.ElementTree import Element, SubElement, tostring, ElementTree

import xml.etree.ElementTree as xml
import xml.dom.minidom as minidom
import gymnasium as gym
import numpy as np
import os
import random
import time

import matplotlib.pyplot as plt


""" Large programming projects are often modularised in different components. 
    In the upcoming exercise(s) we will (re)build an evolutionary pipeline for robot evolution in MuJoCo.

    Exercise3 body-brain: This is your first full body+brain evolution by adapting a custom Ant-v5 gym environment. 
    We adjust both leg lengths and controller weights for a locomotion task.   
"""

ROOT_DIR = get_project_root()
ENV_NAME = 'Ant_custom'




def generate_random_ant_world(file_path, n_rocks=None, area_size=20, min_size=0.2, max_size=2, seed=None):
    if seed is not None:
        random.seed(seed)
    if n_rocks is None:
        n_rocks = random.randint(300, 500)

    mjcf = Element("mujoco", model="tensegrity default scene")

    # Header settings
    SubElement(mjcf, "compiler", angle="degree", coordinate="local", inertiafromgeom="true", autolimits="true")
    SubElement(mjcf, "option", integrator="RK4", timestep="0.01", gravity="0 0 -9.81")
    SubElement(mjcf, "statistic", center="0 0 .3", extent=str(area_size))

    # Visual section (as in walker_world)
    visual = SubElement(mjcf, "visual")
    SubElement(visual, "headlight", diffuse="0.6 0.6 0.6", ambient="0.3 0.3 0.3", specular="0 0 0")
    SubElement(visual, "rgba", haze="0.15 0.25 0.35 1")
    SubElement(visual, "global", azimuth="120", elevation="-20")

    # Assets
    asset = SubElement(mjcf, "asset")
    SubElement(asset, "texture", type="skybox", builtin="gradient", rgb1="0.3 0.5 0.7", rgb2="0 0 0", width="512", height="3072")
    SubElement(asset, "texture", type="2d", name="groundplane", builtin="checker", mark="edge",
               rgb1="0.2 0.3 0.4", rgb2="0.1 0.2 0.3", markrgb="0.8 0.8 0.8", width="300", height="300")
    SubElement(asset, "material", name="groundplane", texture="groundplane", texuniform="true", texrepeat="5 5", reflectance="0.2")
    SubElement(asset, "material", name="rock", rgba="0.4 0.3 0.2 1")

    # World body
    worldbody = SubElement(mjcf, "worldbody")
    SubElement(worldbody, "light", pos="2.5 0 3", dir="0 0 -1", directional="false")
    SubElement(worldbody, "geom", name="floor", size="0 0 0.05", type="plane",
               material="groundplane", friction="150 150 150")

    # Generate rocks
    i = 0
    attempts = 0
    max_attempts = n_rocks * 3  # pour éviter boucle infinie si area trop petit
    while i < n_rocks and attempts < max_attempts:
        sx, sy, sz = [round(random.uniform(min_size, max_size), 3) for _ in range(3)]
        x, y = [round(random.uniform(-area_size, area_size), 2) for _ in range(2)]
        z = sz
        distance = (x**2 + y**2)**0.5
        if distance < 1.0:  # zone sans cailloux autour du robot
            attempts += 1
            continue

        SubElement(worldbody, "geom", name=f"rock_{i}", type="ellipsoid",
                   size=f"{sx} {sy} {sz}",
                   pos=f"{x} {y} {z}",
                   material="rock",
                   contype="1", conaffinity="1", condim="3", density="500", friction="1 0.5 0.5")
        i += 1
        attempts += 1

    # Pretty print
    pretty_xml = minidom.parseString(tostring(mjcf, encoding="unicode")).toprettyxml(indent="  ")
    with open(file_path, "w") as f:
        f.write(pretty_xml)

def plot_ant_rewards(all_infos):
    """
    Agrège les rewards de tous les steps et les affiche sous forme de graphes.
    `all_infos` doit être une liste de dictionnaires retournés par l'env (clé 'info' de step()).
    """
    steps = len(all_infos)
    reward_keys = ["reward_forward", "healthy_reward", "yaw_reward", "Y penalty reward", "ctrl_cost", "cfrc_cost", "reward"]
    reward_data = {key: np.zeros(steps) for key in reward_keys}

    for i, info in enumerate(all_infos):
        for key in reward_keys:
            reward_data[key][i] = info.get(key, 0.0)

    # Tracer chaque courbe
    fig, ax = plt.subplots(figsize=(12, 6))
    for key, values in reward_data.items():
        ax.plot(values, label=key)

    ax.set_title("Reward components over time")
    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Reward value")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


class AntWorld(World):
    def __init__(self, ):
        action_space = 8  # https://gymnasium.farama.org/environments/mujoco/ant/#action-space
        state_space = 27  # https://gymnasium.farama.org/environments/mujoco/ant/#observation-space

        self.n_repeats = 1
        self.n_steps = 1000
        self.controller = MLP.NNController(state_space, action_space)
        self.n_weights = self.controller.n_params

        self.n_params = self.n_weights + 8
        self.world_file = os.path.join(ROOT_DIR, "AntEnv.xml")

        self.joint_limits = [[-30, 30], [30, 70],
                             [-30, 30], [-70, -30],
                             [-30, 30], [-70, -30],
                             [-30, 30], [30, 70], ]
        self.joint_axis = [[0, 0, 1], [-1, 1, 0],
                           [0, 0, 1], [1, 1, 0],
                           [0, 0, 1], [-1, 1, 0],
                           [0, 0, 1], [1, 1, 0],
                           ]
        self.rewards_forward = np.zeros((self.n_steps, self.n_repeats))


    def geno2pheno(self, genotype):
        control_weights = genotype[-self.n_weights:]

        # scale_weights = (genotype[:-self.n_weights] + 1.5) / 5 * 0.5 + 0.1
        scale_weights = (genotype[:-self.n_weights] + 1.5) / 3 * (0.2 - 0.1) + 0.1

        #body_params = (genotype[:-self.n_weights] + 1.5) / 5 * 0.5 + 0.1
        # body_params = ((genotype[:-self.n_weights]+ 1.5) / 3) * 1.4 + 0.1
        #body_params = ((genotype[:-self.n_weights] + 1) / 2) * (1.0 - 0.8) + 0.8
      # Un seul gène pour la longueur des jambes
        leg_length_gene = scale_weights[0]
        leg_length = ((leg_length_gene + 1.5) / 5) *0.5 +0.1  # map to [0.8, 1.0]

        ankle_gene = scale_weights[1]
        ankle_length = ((ankle_gene + 1) / 2) * (1.0 - 0.8) + 0.8


        # Construction de body_params avec jambes = leg_length, chevilles = ankle_length
        body_params = np.array([
            leg_length, ankle_length,   # front left
            leg_length, ankle_length,   # front right
            leg_length, ankle_length,   # back left
            leg_length, ankle_length    # back right
        ])

        # Vérifications
        assert len(body_params) == 8
        assert len(control_weights) == self.n_weights
        assert not np.any(body_params <= 0)

        # Conversion du contrôleur
        self.controller.geno2pheno(control_weights)

        # Décomposition
        front_left_leg, front_left_ankle, front_right_leg, front_right_ankle, back_left_leg, back_left_ankle, back_right_leg, back_right_ankle = body_params


        # Define the 3D coordinates of the relative tree structure
        front_left_hip_xyz = np.array([0.2, 0.2, 0])
        front_left_knee_xyz = np.array(
            [np.sqrt(0.5 * front_left_leg ** 2), np.sqrt(0.5 * front_left_leg ** 2), 0]) + front_left_hip_xyz
        front_left_toe_xyz = np.array(
            [np.sqrt(0.5 * front_left_ankle ** 2), np.sqrt(0.5 * front_left_ankle ** 2), 0]) + front_left_knee_xyz

        front_right_hip_xyz = np.array([-0.2, 0.2, 0])
        front_right_knee_xyz = np.array(
            [-np.sqrt(0.5 * front_right_leg ** 2), np.sqrt(0.5 * front_right_leg ** 2), 0]) + front_right_hip_xyz
        front_right_toe_xyz = np.array(
            [-np.sqrt(0.5 * front_right_ankle ** 2), np.sqrt(0.5 * front_right_ankle ** 2), 0]) + front_right_knee_xyz

        back_left_hip_xyz = np.array([-0.2, -0.2, 0])
        back_left_knee_xyz = np.array(
            [-np.sqrt(0.5 * back_left_leg ** 2), -np.sqrt(0.5 * back_left_leg ** 2), 0]) + back_left_hip_xyz
        back_left_toe_xyz = np.array(
            [-np.sqrt(0.5 * back_left_ankle ** 2), -np.sqrt(0.5 * back_left_ankle ** 2), 0]) + back_left_knee_xyz

        back_right_hip_xyz = np.array([0.2, -0.2, 0])
        back_right_knee_xyz = np.array(
            [np.sqrt(0.5 * back_right_leg ** 2), -np.sqrt(0.5 * back_right_leg ** 2), 0]) + back_right_hip_xyz
        back_right_toe_xyz = np.array(
            [np.sqrt(0.5 * back_right_ankle ** 2), -np.sqrt(0.5 * back_right_ankle ** 2), 0]) + back_right_knee_xyz

        points = np.vstack([front_left_hip_xyz,
                            front_left_knee_xyz,
                            front_left_toe_xyz,
                            front_right_hip_xyz,
                            front_right_knee_xyz,
                            front_right_toe_xyz,
                            back_left_hip_xyz,
                            back_left_knee_xyz,
                            back_left_toe_xyz,
                            back_right_hip_xyz,
                            back_right_knee_xyz,
                            back_right_toe_xyz,
                            ])

        # define the type of connections [FIXED ARCHITECTURE]
        connectivity_mat = np.array(
            [[150, np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 150, np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 150, np.inf, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 150, np.inf, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 150, np.inf, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 150, np.inf, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 150, np.inf, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 150, np.inf],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ]
        )
        return points, connectivity_mat

    def evaluate_individual(self, genotype):
        points, connectivity_mat = self.geno2pheno(genotype)

        robot = AntRobot(points, connectivity_mat, self.joint_limits, self.joint_axis, verbose=False)
        robot.xml = robot.define_robot()
        robot.write_xml()

        world = xml.parse(os.path.join(ROOT_DIR, 'src', 'world', 'robot', 'assets', "ant_world.xml"))

        robot_env = world.getroot()

        robot_env.append(xml.Element("include", attrib={"file": "AntRobot.xml"}))
        world_xml = xml.tostring(robot_env, encoding='unicode')

        with open(self.world_file, "w") as f:
            f.write(world_xml)
 
        envs = AsyncVectorEnv(
            [
                lambda i_env=i_env: gym.make(
                    ENV_NAME,
                    robot_path=self.world_file,
                    reset_noise_scale=0.1,
                    max_episode_steps=self.n_steps,
                )
                for i_env in range(self.n_repeats)
            ]
        )

        rewards_full = np.zeros((self.n_steps, self.n_repeats))
        # multi_obj_rewards_full = np.zeros((self.n_steps, self.n_repeats, 2))  # TODO

        observations, info = envs.reset()
        done_mask = np.zeros(self.n_repeats, dtype=bool)
        for step in range(self.n_steps):
            actions = np.where(done_mask[:, None], 0, self.controller.get_action(observations.T).T)
            observations, rewards, dones, truncated, infos = envs.step(actions)

            # Store rewards for active environments only
            rewards_full[step, done_mask == False] = rewards[done_mask == False]

            # multi_obj_reward = np.array([infos['reward_forward'], -infos['ctrl_cost']]).T  # TODO
            # multi_obj_rewards_full[step, done_mask == False] = multi_obj_reward[done_mask == False]

            # Update the done mask based on the "done" and "truncated" flags
            done_mask = done_mask | dones | truncated

            # Optionally, break if all environments have terminated
            if np.all(done_mask):
                break
        final_rewards = np.sum(rewards_full, axis=0)
        # final_multi_obj_rewards = np.sum(multi_obj_rewards_full, axis=0)
        # final_multi_obj_rewards = np.sum(multi_obj_rewards_full, axis=0)
        envs.close()
        return np.mean(final_rewards),infos
    # , np.mean(final_multi_obj_rewards, axis=0)

    # def plot_rewards(self):

    #     plt.figure(figsize=(10, 5))
    #     plt.plot(np.arange(self.n_steps), self.rewards_forward, label='Forward Rewards')
    #     plt.xlabel('Steps')
    #     plt.ylabel('Rewards')
    #     plt.title('Rewards Over Time')
    #     plt.legend()
    #     plt.grid()
    #     plt.show()

# def run_EA_single(ea_single, world):
#     for gen in range(ea_single.n_gen):
#         print(f"Generation {gen}")
#         pop = ea_single.ask()
#         fitnesses_gen = np.empty(len(pop))
#         for index, genotype in enumerate(pop):
#             fit_ind, _ = world.evaluate_individual(genotype)
#             fitnesses_gen[index] = fit_ind
#         ea_single.tell(pop, fitnesses_gen)

def run_EA_single(ea_single, world):
    start_gen = ea_single.current_gen + 1  # On continue après le checkpoint

    for gen in range(start_gen, ea_single.n_gen):
        print(f"Generation {gen}")
        # % Defining the Robot environment in MuJoCo
        generate_random_ant_world(
            file_path=os.path.join(ROOT_DIR, "src", "world", "robot", "assets", "ant_world.xml"),
            n_rocks=500,  # ou fixe : n_rocks=50
            area_size=20,
            min_size=0.005,
            max_size=0.4
        )
        pop = ea_single.ask()
        fitnesses_gen = np.empty(len(pop))
        for index, genotype in enumerate(pop):
            fit_ind,infos= world.evaluate_individual(genotype)
            fitnesses_gen[index] = fit_ind

        ea_single.tell(pop, fitnesses_gen,infos)  # Le checkpoint est automatiquement géré ici



def run_EA_multi(ea_multi, world):
    for gen in range(ea_multi.n_gen):
        pop = ea_multi.ask()
        fitnesses_gen = np.empty((len(pop), 2))
        for index, genotype in enumerate(pop):
            _, fit_ind = world.evaluate_individual(genotype)
            fitnesses_gen[index] = fit_ind
        ea_multi.tell(pop, fitnesses_gen)


def generate_best_individual_video(world, video_name: str = 'EvoRob5_video.mp4'):
    print("Creating video of the best individual...",video_name)
    env = gym.make(ENV_NAME,
                   robot_path=world.world_file,
                   render_mode="rgb_array")
    rewards_list = []

    observations, info = env.reset()
    frames = []
    for step in range(1000):
        frames.append(env.render())
        action = world.controller.get_action(observations)
        observations, rewards, terminated, truncated, info = env.step(action)
        rewards_list.append(rewards)
        # if terminated:
        #     break
    print(np.sum(rewards_list))

    import imageio
    imageio.mimsave(video_name, frames, fps=30)  # Set frames per second (fps)
    env.close()



import time

def visualise_individual(genotype):
    world = AntWorld()
    points, connectivity_mat = world.geno2pheno(genotype)
    robot = AntRobot(points, connectivity_mat, world.joint_limits, world.joint_axis, verbose=False)
    robot.xml = robot.define_robot()
    robot.write_xml()

    world_xml = xml.parse(os.path.join(ROOT_DIR, 'src', 'world', 'robot', 'assets', "ant_world.xml"))
    robot_env = world_xml.getroot()
    robot_env.append(xml.Element("include", attrib={"file": "AntRobot.xml"}))
    full_world_xml = xml.tostring(robot_env, encoding='unicode')

    with open(world.world_file, "w") as f:
        f.write(full_world_xml)

    env = gym.make(ENV_NAME, robot_path=world.world_file, render_mode="human")
    env = gym.make(ENV_NAME, robot_path=world.world_file, render_mode="human")
    rewards_list = []

    observations, info = env.reset()
    for step in range(1000):
        action = world.controller.get_action(observations)
        observations, rewards, terminated, truncated, info = env.step(action)
        rewards_list.append(rewards)
        
        env.render()  # <<< important pour que la fenêtre reste ouverte
        time.sleep(1 / 60)  # <<< ralenti la simulation
        if terminated or truncated:
            break


    env.close()
    print("Reward total:", np.sum(rewards_list))


def plot_rewards(value,title,ax=None, save_path='fitness_plot.png', data_path='full_fitness_max.csv'):
    """
    Plot the rewards over generations with lines and points, and save the plot and data in the 'reward_data' folder.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # ✅ Créer le dossier reward_data s'il n'existe pas
    output_dir = 'reward_data'
    os.makedirs(output_dir, exist_ok=True)

    # ✅ Mettre à jour les chemins avec le dossier
    save_path = os.path.join(output_dir, save_path)
    data_path = os.path.join(output_dir, data_path)

    if ax is None:
        fig, ax = plt.subplots()

    generations = range(len(value))
    ax.plot(generations, value, label='Fitness', marker='o', linestyle='-', color='blue')

    ax.set_xlabel('Generation')
    ax.set_ylabel('Reward')
    ax.set_title(title)
    ax.legend()

    # ✅ Sauvegarder le graphique
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

    # ✅ Sauvegarder les données
    np.savetxt(data_path, value, delimiter=',')
    print(f"Fitness data saved to {data_path}")

    plt.show()

def plot_all_rewards(reward_dict, save_path='combined_fitness_plot.png'):
    """
    Plot multiple reward curves on the same plot.

    Parameters:
        reward_dict (dict): Dictionary with title as key and list/array as value.
        save_path (str): Name of the file to save the plot.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # ✅ Créer le dossier reward_data s'il n'existe pas
    output_dir = 'reward_data'
    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(output_dir, save_path)

    # ✅ Création du plot
    fig, ax = plt.subplots()

    for label, values in reward_dict.items():
        generations = range(len(values))
        ax.plot(generations, values, marker='o', linestyle='-', label=label)

    ax.set_xlabel('Generation')
    ax.set_ylabel('Reward')
    ax.set_title('All Fitness Metrics Over Generations')
    ax.legend()
    ax.grid(True)

    plt.savefig(save_path)
    print(f"Combined plot saved to {save_path}")
    plt.show()



def main():
    # %% Understanding the world
    genotype = np.random.uniform(-1, 1, 953)  # 8 body parameters, 945 NN weights
    # for i in range(8):
    #     visualise_individual(genotype)
    world = AntWorld()

    n_parameters = world.n_params

    population_size = 250
    CMAES_opts["min"] = -1
    CMAES_opts["max"] = 1
    CMAES_opts["num_parents"] = 20
    CMAES_opts["num_generations"] = 5
    CMAES_opts["mutation_sigma"] = 0.33

    results_dir = os.path.join(ROOT_DIR, 'results', ENV_NAME, 'single')

    ea_single = CMAES_sol(population_size, n_parameters, CMAES_opts, results_dir)

    #Vérifie si un checkpoint existe
    # try:
    #     ea_single.load_checkpoint(1)
    #     print(f"Checkpoint trouvé ! Reprise depuis la génération {ea_single.current_gen}")
    # except AssertionError:
    #     print("Aucun checkpoint trouvé. Nouveau run.")

    run_EA_single(ea_single, world)
    #world.plot_rewards()
    print("full fitness",len(ea_single.full_fitness))
    print("max fitness:", len(ea_single.full_fitness_max))
    plot_rewards(ea_single.full_fitness_max,'full fitness',save_path='fitness_max_plot.png', data_path='full_fitness_max.csv')
    plot_rewards(ea_single.full_forward_fitness,'forward fitness',save_path='fitness_forward_plot.png', data_path='full_forward_fitness.csv')
    plot_rewards(ea_single.full_yaw_fitness,'yaw fitness',save_path='fitness_yaw_plot.png', data_path='full_yaw_fitness.csv')
    plot_rewards(ea_single.full_drift_fitness,'drift fitness (penalty)',save_path='fitness_drift_plot.png', data_path='full_drift_fitness.csv')

    plot_all_rewards({
    'Full fitness': ea_single.full_fitness_max,
    'Forward fitness': ea_single.full_forward_fitness,
    'Yaw fitness': ea_single.full_yaw_fitness,
    'Drift fitness (penalty)': ea_single.full_drift_fitness
    })


    # %% Optimise multi-objective
    # # TODO implement the NSGAII
    # # world = AntWorld()
    # # n_parameters = world.n_params
# 
    # # population_size = 250
    # # NSGA_opts["min"] = -1
    # # NSGA_opts["max"] = 1
    # # NSGA_opts["num_parents"] = population_size
    # # NSGA_opts["num_generations"] = 20
    # # NSGA_opts["mutation_prob"] = 0.3
    # # NSGA_opts["crossover_prob"] = 0.5
# 
    # # results_dir = os.path.join(ROOT_DIR, 'results', ENV_NAME, 'multi')
    # # ea_multi_obj = NSGAII_sol(population_size, n_parameters, NSGA_opts, results_dir)
# 
    # # run_EA_multi(ea_multi_obj, world)

    # %% visualise
    # TODO: Make a video of the best individual, and plot the fitness curve.
    best_individual = np.load(os.path.join(results_dir, "145", "x_best.npy"))

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

    generate_best_individual_video(world,'test_3.mp4')


if __name__ == "__main__":
    main()