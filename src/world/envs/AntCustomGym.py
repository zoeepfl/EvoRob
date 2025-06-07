from os import path
from typing import Dict, Union

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from src.utils.geometry import quat2rot

DEFAULT_CAMERA_CONFIG = {
    "distance": 5,
}


class AntCustomEnv(MujocoEnv, utils.EzPickle):
    r"""In this environment a Passive Dynamic Walker is tasked to locomote.
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        robot_path: str,
        frame_skip: int = 5,
        default_camera_config: Dict[str, float] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1,
        ctrl_cost_weight: float = 0.5,
        cfrc_cost_weight: float = 5e-4,
        main_body: Union[int, str] = 1,
        reset_noise_scale: float = 0.1,
        exclude_current_positions_from_observation: bool = True,
        include_cfrc_ext_in_observation: bool = False,
        pert_force=None,
        **kwargs,
    ):
        xml_file_path = path.join(
            path.dirname(path.realpath(__file__)),
            robot_path,
        )

        utils.EzPickle.__init__(
            self,
            xml_file_path,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            cfrc_cost_weight,
            main_body,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            pert_force,
            **kwargs,
        )
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._cfrc_cost_weight = cfrc_cost_weight
        self.prev_xpos = None
        self.no_progress_counter = 0
        self.min_movement_threshold = 0.01  # tolérance de déplacement
        self.patience_steps = 20           # nombre de pas avant abandon

        self._main_body = main_body

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        MujocoEnv.__init__(
            self,
            xml_file_path,
            frame_skip,
            observation_space=None,  # needs to be defined after
            default_camera_config=default_camera_config,
            width=832,
            height=496,
            camera_name="track",
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = self.data.qpos.size + self.data.qvel.size
        obs_size -= 2 * exclude_current_positions_from_observation
        obs_size += (
            self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation
        )

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 2 * exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size
            - 2 * exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
        }
        self.body_ids = None
        self.force = None
        self.previous_state = None
        self.stuck = 0
        if pert_force is not None:
            self.body_ids , self.force = pert_force



    def step(self, action):
        xy_position_before = self.data.body(self._main_body).xpos[:2].copy()
        if self.body_ids is not None:
            self.apply_force()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.body(self._main_body).xpos[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        # forward_reward = x_velocity * self._forward_reward_weight
        forward_reward = x_velocity * 2
        healthy_reward = 1
        ctrl_cost = np.linalg.norm(action)**2 * self._ctrl_cost_weight
        cfrc_cost = np.linalg.norm( self.data.cfrc_ext[1:])**2 * self._cfrc_cost_weight

        # Reward Yaw
        # quat = self.data.body(self._main_body).xquat
        # Convert quaternion to yaw (heading in radians)
        # yaw_rad = np.arctan2(
        #     2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
        #     1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2)
        # )
        import math

        def quaternion_to_yaw_deg(quat):
            """
            Convert quaternion (x, y, z, w) to yaw angle in degrees.
            """
            x, y, z, w = quat
            siny_cosp = 2.0 * (w * z + x * y)
            cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
            yaw_rad = math.atan2(siny_cosp, cosy_cosp)
            return math.degrees(yaw_rad)
        
        # Reward Yaw (orientation vers l’axe x)
        quat = self.data.body(self._main_body).xquat  # x, y, z, w
        yaw_deg = quaternion_to_yaw_deg(quat)
        reward_yaw = (math.exp(-(abs(yaw_deg) / 10.0)**2))
        # print("Yaw (deg):", yaw_deg, "| Reward:", reward_yaw)
        # print("x_velocity (m):", x_velocity, "| forward_reward:", forward_reward)

        # yaw = np.degrees(yaw_rad)
        # print("yaw : ", yaw)
        # reward_yaw = np.exp(-(yaw**2)/10)
        # print("yaw : ", yaw)

        # Y penalty reward (Y close to 0)
        drift = np.abs(self.data.qpos[1])
        drift_penalty = -1 * abs(drift)

        #TODO
        reward = healthy_reward + forward_reward + reward_yaw
        # reward = healthy_reward + forward_reward - ctrl_cost - cfrc_cost + reward_yaw + drift_penalty
        observation = self._get_obs()

        info = {
            "reward_forward": forward_reward,
            "healthy_reward": healthy_reward,
            "yaw_reward": reward_yaw,
            "drift penalty reward": drift_penalty,
            "ctrl_cost": - ctrl_cost,
            "cfrc_cost": - cfrc_cost,
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
        }
        terminated = False
        # Check for NaN, Inf, or huge values
        qacc = self.data.qacc
        if np.any(np.isnan(qacc)) or np.any(np.isinf(qacc)) or np.any(np.abs(qacc) > 1e6):
            print("too huge")
            DOF = np.argwhere((np.isnan(qacc)) + (np.isinf(qacc)) + (np.abs(qacc) > 1e6)).squeeze()[0]
            print(ValueError(f'MuJoCo Warning: Nan, Inf or huge value in QACC at DOF {DOF}'))
            terminated = True

        if self.data.qpos[2] < 0.1 or self.data.qpos[2] > 1.2:
            # print("hauteur",self.data.qpos[2])
            # print("hors bornes en z")
            terminated = True

        if np.isinf(observation).any():
            print("infini")
            terminated = True

        # Track forward movement
        xpos_now = self.data.qpos[0]  # Position en X actuelle
        
        if self.prev_xpos is None:
            self.prev_xpos = xpos_now
        
        movement = np.abs(xpos_now - self.prev_xpos)
        
        if movement < self.min_movement_threshold:
            self.no_progress_counter += 1
        else:
            self.no_progress_counter = 0
        
        self.prev_xpos = xpos_now
        
        if self.no_progress_counter >= self.patience_steps:
            # print(f"Terminated due to no forward progress for {self.patience_steps} steps")
            terminated = True


        from scipy.spatial.transform import Rotation as R
        quat_mujoco = self.data.qpos[3:7]  # [w, x, y, z] format MuJoCo
        quat_scipy = [quat_mujoco[1], quat_mujoco[2], quat_mujoco[3], quat_mujoco[0]]  # x, y, z, w
        z_axis = R.from_quat(quat_scipy).apply([0, 0, 1])
        if z_axis[2] < 0.5:
            #print("z_axis[2] : ", z_axis[2])
            #print("Ant retourné ou trop incliné")
            terminated = True

        self.previous_state = observation

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        return np.concatenate((position, velocity))


    def apply_force(self):
        body_id = self.body_ids
        force = self.force
        pert = self.np_random.uniform(
            low=-0.1, high=0.1, size=3)
        rot = quat2rot([1, *pert])
        force = np.dot(rot, force.reshape(2, 3).T).T.flatten()
        self.data.xfrc_applied[body_id] = force


    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale
            * self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)
        observation = self._get_obs()
        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }
