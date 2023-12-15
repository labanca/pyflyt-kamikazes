"""Multiagent QuadX Hover Environment."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
from gymnasium import spaces
from typing import Any

#from PyFlyt.pz_envs.quadx_envs.ma_quadx_base_env import MAQuadXBaseEnv
from envs.ma_quadx_base_env import MAQuadXBaseEnv

def _np_cross(x, y) -> np.ndarray:
    """__np_cross.

    Args:
        x:
        y:

    Returns:
        np.ndarray:
    """
    return np.cross(x, y)



class MAQuadXHoverEnv(MAQuadXBaseEnv):
    """Simple Multiagent Hover Environment.

    Actions are vp, vq, vr, T, ie: angular rates and thrust.
    The target is to not crash for the longest time possible.

    Args:
        start_pos (np.ndarray): start_pos
        start_orn (np.ndarray): start_orn
        sparse_reward (bool): sparse_reward
        flight_dome_size (float): flight_dome_size
        max_duration_seconds (float): max_duration_seconds
        angle_representation (str): angle_representation
        agent_hz (int): agent_hz
        render_mode (None | str): render_mode
    """

    metadata = {
        "render_modes": ["human"],
        "name": "ma_quadx_hover",
    }

    def __init__(
        self,
        spawn_settings: dict(),
        start_pos: np.ndarray = np.array(
            [[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [-1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        ),
        start_orn: np.ndarray = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        ),
        sparse_reward: bool = False,
        flight_dome_size: float = 10.0,
        max_duration_seconds: float = 30.0,
        angle_representation: str = "quaternion",
        agent_hz: int = 40,
        render_mode: None | str = None,
        uav_mapping: np.array = np.array(['lm', 'lm', 'lm', 'lm']),
        seed : int = None,
        num_lm: int = None
    ):
        """__init__.

        Args:
            start_pos (np.ndarray): start_pos
            start_orn (np.ndarray): start_orn
            sparse_reward (bool): sparse_reward
            flight_dome_size (float): flight_dome_size
            max_duration_seconds (float): max_duration_seconds
            angle_representation (str): angle_representation
            agent_hz (int): agent_hz
            render_mode (None | str): render_mode
        """
        super().__init__(
            start_pos=start_pos,
            start_orn=start_orn,
            flight_dome_size=flight_dome_size,
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,
            agent_hz=agent_hz,
            render_mode=render_mode,
            spawn_settings=spawn_settings,
            uav_mapping=uav_mapping,
            seed=seed,
            num_lm=num_lm,

        )

        self.lethal_distance = 0.5
        self.sparse_reward = sparse_reward
        self.spawn_setting = spawn_settings

        # observation space
        self._observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.combined_space.shape[0] + 12,),
            dtype=np.float64,
        )

    def observation_space(self, _):
        """observation_space.

        Args:
            _:
        """
        return self._observation_space

    def reset(self, seed=None, options=dict()) -> tuple[dict[str, Any], dict[str, Any]]:
        """reset.

        Args:
            seed: seed to pass to the base environment.
            options: None
        """

        super().begin_reset(seed, options)


        super().end_reset(seed, options)

        observations = {
            ag: self.compute_observation_by_id(self.agent_name_mapping[ag])
            for ag in self.agents
        }
        infos = {ag: dict() for ag in self.agents}
        return observations, infos

    def _compute_agent_states(self) -> None:
        """_compute_agent_states.

        Args:

        Returns:
            None:
        """

        self.past_magnitude = self.current_magnitude.copy()
        self.current_magnitude = [np.linalg.norm(action[np.r_[:2, 3]]) for action in self.current_actions]

        # get the states of all drones
        self.attitudes = np.stack(self.aviary.all_states, axis=0, dtype=np.float64)
        rotation, forward_vecs = self.compute_rotation_forward(self.attitudes[:, 1])

        # compute the vectors of each drone to each drone
        separation = self.attitudes[:, -1][:, np.newaxis, :] - self.attitudes[:, -1]
        self.previous_distance = self.current_distance.copy()

        # Compute the norm along the last axis for each pair of drones
        self.current_distance = np.linalg.norm(separation, axis=-1)

        # compute engagement angles
        self.previous_angles = self.current_angles.copy()
        self.current_angles = np.arccos(
            np.sum(separation * forward_vecs, axis=-1) / (self.current_distance+ 1e-10)
        )
        #np.nan_to_num(self.current_angles, copy=False)

        self.chasing = np.abs(self.current_angles) < (np.pi / 4.0) # I've tryed  /3.0 45º
        self.in_range = self.current_distance < self.lethal_distance


    def compute_observation_by_id(self, agent_id: int) -> np.ndarray:
        """compute_observation_by_id.

        Args:
            agent_id (int): agent_id

        Returns:
            np.ndarray:
        """

        # THIS IS THE COMPUTE_STATE
        # get all the relevant things

        self.last_obs_time = self.aviary.elapsed_time


        raw_state = self.compute_attitude_by_id(agent_id)
        aux_state = self.compute_auxiliary_by_id(agent_id)

        # # state breakdown
        # ang_vel = raw_state[0]
        # ang_pos = raw_state[1]
        # lin_vel = raw_state[2]
        # lin_pos = raw_state[3]
        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = raw_state

        target_attitude = self.aviary.state(self.find_nearest_drone(agent_id))

        # depending on angle representation, return the relevant thing
        if self.angle_representation == 0:
            return np.array(
                [
                    *ang_vel,
                    *ang_pos,
                    *lin_vel,
                    *lin_pos,
                    *aux_state,
                    *self.past_actions[agent_id],
                    *target_attitude.flatten(),
                ]
            )
        elif self.angle_representation == 1:
            return np.array(
                [
                    *ang_vel,
                    *quaternion,
                    *lin_vel,
                    *lin_pos,
                    *aux_state,
                    *self.past_actions[agent_id],
                    *target_attitude.flatten(),
                ]
            )
        else:
            raise AssertionError("Not supposed to end up here!")


    def compute_term_trunc_reward_info_by_id(
        self, agent_id: int
    ) -> tuple[bool, bool, float, dict[str, Any]]:
        """Computes the termination, truncation, and reward of the current timestep."""
        term, trunc, reward, info = super().compute_base_term_trunc_reward_info_by_id(agent_id)

        self._compute_agent_states()

        # don't recompute if we've already done it
        #if self.last_rew_time != self.aviary.elapsed_time and self.last_agent_id == agent_id:
        self.last_rew_time = self.aviary.elapsed_time
        self._compute_engagement_rewards(agent_id)

        reward += self.rewards[agent_id]


        return term, trunc, reward, info


    def _compute_engagement_rewards(self, agent_id) -> None:
        """_compute_engagement_rewards."""
        # reset rewards
        self.rewards[agent_id] *= 0.0

        target_id = self.find_nearest_drone(agent_id)
        # sparse reward computation
        if not self.sparse_reward:

            # reward for closing the distance
            rew_closing_distance = (np.clip(
                self.previous_distance[agent_id][target_id] - self.current_distance[agent_id][target_id],
                a_min=0.0,
                a_max=None,
                ) * (~self.in_range[agent_id][target_id] & self.chasing[agent_id][target_id]) * 1.0)

            # reward for progressing to engagement
            rew_progress_eng =(
                (self.previous_angles[agent_id][target_id] - self.current_angles[agent_id][target_id]) * self.in_range[agent_id][target_id] * 10.0
            )

            # reward for engaging the enemy
            rew_engaging_enemy = 3.0 / (self.current_angles[agent_id][target_id] + 0.1) * self.in_range[agent_id][target_id]

            ## reward for maintaning high speed.
            #self.rewards[agent_id] += 1.0 * self.current_magnitude[agent_id] * self.chasing[agent_id][target_id]
            #print(f'rew maintaning high speed {1.0 * self.current_magnitude[agent_id] * self.chasing[agent_id][target_id]}')

            self.rewards[agent_id] += rew_closing_distance + rew_progress_eng + rew_engaging_enemy

            # if (rew_engaging_enemy > 0) or (rew_progress_eng > 0) or (rew_closing_distance > 0):
            #     print(f'rew closing distance {rew_closing_distance}')
            #     print(f'rew progressing to engagement {rew_progress_eng}')
            #     print(f'rew engaging the enemy {rew_engaging_enemy}')
            #     print(f'------------------------------------------------------------')

    def lidar(self, allied_drone_position, enemy_positions):
        # Convert positions to numpy arrays for easier calculations
        allied_drone_position = np.array(allied_drone_position)
        enemy_positions = np.array(enemy_positions)

        # Calculate Euclidean distances between the allied drone and all enemy drones
        distances = np.linalg.norm(enemy_positions - allied_drone_position, axis=1)

        # Find the index of the nearest enemy drone
        nearest_enemy_index = np.argmin(distances)

        # Get the position of the nearest enemy drone
        nearest_enemy_position = enemy_positions[nearest_enemy_index]

        return nearest_enemy_position

    def find_nearest_drone(self, agent_id: int) -> int:
        # Assuming compute_observation_by_id has been called to update self.current_distance
        distances = self.current_distance[agent_id, :]


        # Find indices where uav_mapping is 'lw'
        lw_indices = np.where(self.uav_mapping == 'lw')[0]

        # Filter distances based on 'lw' indices
        lw_distances = distances[lw_indices]

        # Find the index of the minimum distance in lw_distances
        nearest_drone_index = lw_indices[np.argmin(lw_distances)]

        return nearest_drone_index

    def bodie_info(self, agent_id, substring):

        if 0 <= agent_id < len(self.agents):
            return substring in self.agents[agent_id]
        else:
            return False

    #def create_lw(self, start_pos):

    @staticmethod
    def compute_rotation_forward(orn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Computes the rotation matrix and forward vector of an aircraft given its orientation.

        Args:
            orn (np.ndarray): an [n, 3] array of each drone's orientation

        Returns:
            np.ndarray: an [n, 3, 3] rotation matrix of each aircraft
            np.ndarray: an [n, 3] forward vector of each aircraft
        """
        c, s = np.cos(orn), np.sin(orn)
        eye = np.stack([np.eye(3)] * orn.shape[0], axis=0)

        rx = eye.copy()
        rx[:, 1, 1] = c[..., 0]
        rx[:, 1, 2] = -s[..., 0]
        rx[:, 2, 1] = s[..., 0]
        rx[:, 2, 2] = c[..., 0]
        ry = eye.copy()
        ry[:, 0, 0] = c[..., 1]
        ry[:, 0, 2] = s[..., 1]
        ry[:, 2, 0] = -s[..., 1]
        ry[:, 2, 2] = c[..., 1]
        rz = eye.copy()
        rz[:, 0, 0] = c[..., 2]
        rz[:, 0, 1] = -s[..., 2]
        rz[:, 1, 0] = s[..., 2]
        rz[:, 1, 1] = c[..., 2]

        forward_vector = np.stack(
            [c[..., 2] * c[..., 1], s[..., 2] * c[..., 1], -s[..., 1]], axis=-1
        )

        # order of operations for multiplication matters here
        return rz @ ry @ rx, forward_vector
