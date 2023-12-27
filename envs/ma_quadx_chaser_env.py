
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
        agent_hz: int = 15,
        render_mode: None | str = None,
        uav_mapping: np.array = np.array(['lm', 'lm', 'lm', 'lm']),
        seed : int = None,
        num_lm: int = None,
        formation_center: np.ndarray = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        ),
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
            seed=seed,
            num_lm=num_lm,
            formation_center=formation_center

        )


        self.sparse_reward = sparse_reward
        self.spawn_setting = spawn_settings

        # observation space
        self._observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.combined_space.shape[0] + 19,),
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
        rotation, self.forward_vecs = self.compute_rotation_forward(self.attitudes[:, 1])
        self.drone_positions = self.attitudes[:, 3]

        # compute the vectors of each drone to each drone
        self.separation = self.attitudes[:, -1][:, np.newaxis, :] - self.attitudes[:, -1]
        self.previous_distance = self.current_distance.copy()

        # Compute the norm along the last axis for each pair of drones
        self.current_distance = np.linalg.norm(self.separation, axis=-1)

        # compute engagement angles (foward vectors angles?)
        self.previous_angles = self.current_angles.copy()

        x1 = np.sum(self.separation * self.forward_vecs, axis=-1)
        x2 = self.current_distance
        self.current_angles = np.arccos(np.divide(x1, x2, where=x2 != 0))

        # self.previous_traj_angles = self.current_traj_angles.copy()
        normalized_separation = self.separation / (self.current_distance[:, :, np.newaxis] + 1e-10)
        # x3 = np.sum(normalized_separation * self.forward_vecs, axis=-1)
        # self.current_traj_angles = np.arccos(np.divide(x3, x2, where=x2 != 0))

        self.previous_vel_angles = self.current_vel_angles.copy()
        lin_vel = self.attitudes[: ,2]
        normalized_lin_vel = lin_vel / (np.linalg.norm(lin_vel, axis=-1, keepdims=True) + 1e-10)
        x4 = np.sum(normalized_separation * normalized_lin_vel, axis=-1)
        self.current_vel_angles = np.arccos(np.clip(x4, -1.0, 1.0))  # angles between velocity and separation vectors


        self.in_cone = self.current_vel_angles < self.lethal_angle # lethal angle = 0.1
        self.in_range = self.current_distance < self.lethal_distance # lethal distance = 0.15
        self.chasing = np.abs(self.current_vel_angles) < (np.pi / 2.0)  # I've tryed  2.0



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
        target_id = self.find_nearest_lw(agent_id)
        near_ally_id = self.find_nearest_lm(agent_id, exclude_self=True)


        raw_state = self.compute_attitude_by_id(agent_id)
        aux_state = self.compute_auxiliary_by_id(agent_id)

        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = raw_state

        near_ally_attitude = self.compute_attitude_by_id(near_ally_id)
        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = raw_state

        ally_lin_vel = near_ally_attitude[2]
        ally_lin_pos = near_ally_attitude[3]

        # target_attitude = self.aviary.state(target_id)
        target_attitude = self.compute_attitude_by_id(target_id)
        ang_vel_target, ang_pos_target, lin_vel_target, lin_pos_target, quaternion_target = target_attitude

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
                    *np.array(quaternion),
                    *lin_vel,
                    *lin_pos,
                    *aux_state,
                    *self.past_actions[agent_id],

                    *ally_lin_vel,
                    *ally_lin_pos,

                    *ang_vel_target,
                    *np.array(quaternion_target),
                    *lin_vel_target,
                    *lin_pos_target,
                ]
            )
        else:
            raise AssertionError("Not supposed to end up here!")


    def compute_term_trunc_reward_info_by_id(
        self, agent_id: int
    ) -> tuple[bool, bool, float, dict[str, Any]]:
        """Computes the termination, truncation, and reward of the current timestep."""
        term, trunc, reward, info = super().compute_base_term_trunc_reward_info_by_id(agent_id)

        #self._compute_agent_states()

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

        target_id = self.find_nearest_lw(agent_id)
        # sparse reward computation
        if not self.sparse_reward:

            self.approaching = self.current_distance < self.previous_distance

            # reward for closing the distance
            rew_closing_distance = np.clip(
                self.previous_distance[agent_id][target_id] - self.current_distance[agent_id][target_id] * 1.0,
                a_min=-10.0,
                a_max=None,
            ) * (
                    self.chasing[agent_id][target_id]
                )

            # reward for engaging the enemy
            rew_engaging_enemy = np.divide(3.0, self.current_vel_angles[agent_id][target_id],
                                           where=self.current_vel_angles[agent_id][target_id] != 0) * (
                    self.chasing[agent_id][target_id]
                    * self.approaching[agent_id][target_id]
                    * 1.0
                )

            # # reward for progressing to engagement
            rew_near_engagement = (
                    (self.previous_vel_angles[agent_id][target_id] - self.current_vel_angles[agent_id][target_id])
                    * 10.0
                    * self.in_range[agent_id][target_id]
                    * self.approaching[agent_id][target_id]
            )

            # reward for maintaning linear velocities.
            rew_speed_magnitude =(
                    (self.current_magnitude[agent_id])
                    * self.chasing[agent_id][target_id]
                    * self.approaching[agent_id][target_id]
            )



            self.rewards[agent_id] += (
                    rew_closing_distance
                    + rew_engaging_enemy
                    + rew_speed_magnitude
            )

        step_data = {
            "agent_id": agent_id,
            "elapsed_time": self.aviary.elapsed_time,
            "rew_closing_distance": rew_closing_distance,
            "rew_engaging_enemy": rew_engaging_enemy,
            "rew_speed_magnitude": rew_speed_magnitude,
        }
        self.rewards_data.append(step_data)

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


    def save_runtim_data_by_id(self, agent_id, target_id, **rew_kwargs):

        ang_vel_a, ang_pos_a, lin_vel_a, lin_pos_a, quaternion_a = self.compute_attitude_by_id(agent_id)
        ang_vel_t, ang_pos_t, lin_vel_t, lin_pos_t, quaternion_t = self.compute_attitude_by_id(target_id)
        self.rew_log.append([
            self.aviary.elapsed_time,
             rew_kwargs['rew_closing_distance'],
             rew_kwargs['rew_progress_eng'],
             rew_kwargs['rew_engaging_enemy'],
             rew_kwargs['rew_last_distance'],
             ang_vel_a,
             ang_pos_a,
             lin_vel_a,
             lin_pos_a[0],
             lin_pos_a[1],
             lin_pos_a[2],
             quaternion_a,
             ang_vel_t,
             ang_pos_t,
             lin_vel_t,
             lin_pos_t[0],
             lin_pos_t[1],
             lin_pos_t[2],
             quaternion_t,
             self.current_distance[agent_id][target_id],
             self.current_angles[agent_id][target_id],
             ])

    def print_rewards(self, agent_id, **kargs):
        for k,v in kargs.items():
            print(f'{agent_id} {k} = {v}')

        print(f'{self.current_magnitude[agent_id]=}')
        print(f'{self.current_vel_angles[agent_id][1]=}')
        print(f'------------------------------------------------------------')

    def save_rewards_data(self, filename):
        # Save the rewards data to a file
        import csv

        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ["agent_id", "elapsed_time", "rew_closing_distance", "rew_engaging_enemy",
                          "rew_speed_magnitude"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write the header
            writer.writeheader()

            # Write the data
            for step_data in self.rewards_data:
                writer.writerow(step_data)



    def plot_rewards_data(self,filename):
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('TkAgg')

        # Read the CSV file into a DataFrame
        df = pd.read_csv(filename)

        # Plotting
        plt.figure(figsize=(10, 6))

        for agent_id in df['agent_id'].unique():
            agent_data = df[df['agent_id'] == agent_id]
            plt.plot(agent_data['elapsed_time'], agent_data['rew_closing_distance'], label=f'Agent {agent_id}')

        plt.xlabel('Elapsed Time')
        plt.ylabel('Reward - Closing Distance')
        plt.title('Rewards Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

