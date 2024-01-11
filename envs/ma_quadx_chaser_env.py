
from __future__ import annotations

import numpy as np

from gymnasium import spaces
from typing import Any
from pathlib import Path


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



class MAQuadXChaserEnv(MAQuadXBaseEnv):
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
        "name": "ma_quadx_chaser",

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
        angle_representation: str = "quaternion",
        render_mode: None | str = None,
        uav_mapping: np.array = np.array(['lm', 'lm', 'lm', 'lm']),
        seed : int = None,
        formation_center: np.ndarray = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        ),
        lw_stand_still: bool = True,
        lw_moves_random: bool = False,
        lw_chases: bool = False,
        lw_attacks: bool = False,
        lw_threat_radius: float = 4.0,
        lw_shoot_range: float = 1.0,
        agent_hz: int = 30,
        max_duration_seconds: float = 30.0,
        num_lm: int = 1,
        num_lw: int = 1,
        distance_factor: float = 1.0,
        proximity_factor: float = 1.0,
        speed_factor: float = 1.0,
        rew_exploding_target: float = 100,
        max_velocity_magnitude: float = 10,
        save_step_data: bool = False
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
            lw_stand_still=lw_stand_still,
            lw_moves_random=lw_moves_random,
            lw_chases=lw_chases,
            lw_attacks=lw_attacks,
            lw_threat_radius=lw_threat_radius,
            lw_shoot_range=lw_shoot_range,
            num_lm=num_lm,
            num_lw=num_lw,
            formation_center=formation_center,
            distance_factor=distance_factor,
            proximity_factor=proximity_factor,
            speed_factor=speed_factor,
            rew_exploding_target=rew_exploding_target,
            max_velocity_magnitude=max_velocity_magnitude,
            save_step_data=save_step_data,

        )

        self.sparse_reward = sparse_reward
        self.spawn_setting = spawn_settings

        # observation space
        self._observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.combined_space.shape[0] + 22,),
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

        self.previous_magnitude = self.current_magnitude.copy()
        self.previous_distance = self.current_distance.copy()
        self.previous_angles = self.current_angles.copy()
        self.previous_vel_angles = self.current_vel_angles.copy()
        self.previous_rel_vel_magnitude = self.current_rel_vel_magnitude.copy()

        # get the states of all drones
        self.attitudes = np.stack(self.aviary.all_states, axis=0, dtype=np.float64)
        self.linear_velocities = self.attitudes[:, 2]
        self.drone_positions = self.attitudes[:, 3]
        self.current_magnitude = np.linalg.norm(self.linear_velocities, axis=-1)


        # rotation matrix and forward vectors
        self.rotation, self.forward_vecs = self.compute_rotation_forward(self.attitudes[:, 1])

        # compute the separation vectors and distance of each drone to each drone
        self.separation = self.attitudes[:, -1][:, np.newaxis, :] - self.attitudes[:, -1]
        self.current_distance = np.linalg.norm(self.separation, axis=-1)

        # opponent velocity is relative to ours in our body frame
        ground_velocities: np.ndarray = (
            self.rotation @ np.expand_dims(self.attitudes[:, -2], axis=-1)
        ).reshape(self.attitudes.shape[0], 3)

        self.ground_velocities = ground_velocities

        # relative ground relative velocities and ground relative magnitude
        self.relative_lin_velocities = self.ground_velocities[:][:, np.newaxis, :] - self.ground_velocities[:]
        self.current_rel_vel_magnitude = np.linalg.norm(self.relative_lin_velocities, axis=-1)

        # angles between the ground velocity and separation
        for id in range(self.ground_velocities.shape[0]):
            separation = self.drone_positions[id] - self.drone_positions

            for i in range(self.separation.shape[0]):
                dot_products = np.dot(separation[i] ,ground_velocities[i],)
                norm_vectors = np.linalg.norm(separation[i]) * np.linalg.norm(ground_velocities[i])
                cosines = dot_products / (norm_vectors + 1e-10)
                self.current_vel_angles[id][i] = np.arccos(cosines)

        self.current_vel_angles = self.current_vel_angles.T
        self.in_cone = self.current_vel_angles < self.lethal_angle # lethal angle = 0.15
        self.in_range = self.current_distance <= self.lethal_distance # lethal distance = 1.0
        self.chasing = np.abs(self.current_vel_angles) < (np.pi / 2.0)  # if the drone is chasing another
        self.approaching = self.current_distance < self.previous_distance
        self.hit_probability = np.maximum(0.9 - self.current_magnitude/self.max_velocity_magnitude, 0.01)

    def compute_observation_by_id(self, agent_id: int) -> np.ndarray:
        """compute_observation_by_id.

        Args:
            agent_id (int): agent_id

        Returns:
            np.ndarray:

       attitude is (4, 3) array, where:
            - `state[0, :]` represents body frame angular velocity
            - `state[1, :]` represents ground frame angular position
            - `state[2, :]` represents body frame linear velocity
            - `state[3, :]` represents ground frame linear position
        """

        # ids
        target_id = self.find_nearest_lw(agent_id)
        near_ally_id = self.find_nearest_lm(agent_id, exclude_self=True)

        # states
        agent_state = self.compute_attitude_by_id(agent_id)
        agent_aux_state = self.compute_auxiliary_by_id(agent_id)
        target_state = self.compute_attitude_by_id(target_id)
        ally_state = self.compute_attitude_by_id(near_ally_id)

        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = agent_state
        target_ang_vel, target_ang_pos, target_lin_vel, target_lin_pos, target_quaternion = target_state
        ally_ang_vel, ally_ang_pos, ally_lin_vel, ally_lin_pos, ally_quaternion = ally_state

        # agent rotation matrix
        agent_rotation = np.array(self.aviary.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
        # delta_lin_vel = np.matmul((lin_vel), agent_rotation)
        # delta_ang_vel = np.matmul((ang_vel), agent_rotation)

        # target dynamics relative to the agent
        target_delta_lin_pos = np.matmul((target_lin_pos - lin_pos), agent_rotation)
        target_ang_vel = target_ang_vel # unchanged
        target_delta_ang_pos = ang_pos - target_ang_pos
        target_delta_lin_vel = np.matmul((target_lin_vel - lin_vel), agent_rotation)
        target_distance = float(np.linalg.norm(target_delta_lin_pos))

        ally_delta_lin_pos = np.matmul((ally_lin_pos - lin_pos), agent_rotation)
        ally_ang_vel = ally_ang_vel  # unchanged
        ally_delta_ang_pos = ang_pos - ally_ang_pos
        ally_delta_lin_vel = np.matmul((ally_lin_vel - lin_vel), agent_rotation)
        ally_distance = float(np.linalg.norm(ally_delta_lin_pos))


        if len(self.targets):
            if self.lw_manager.squad[target_id - self.num_lm].gun_loaded:
                hit_probability = self.hit_probability[agent_id]
            else:
                hit_probability = 0.0
        else:
            hit_probability = 0.0

        # depending on angle representation, return the relevant thing
        if self.angle_representation == 0:
            return np.array(
                [
                    *ang_vel,
                    *ang_pos,
                    *lin_vel,
                    *lin_pos,
                    *agent_aux_state,
                    *self.past_actions[agent_id],

                    *ally_ang_vel,
                    *ally_delta_ang_pos
                    *ally_delta_lin_vel,
                    *ally_delta_lin_pos,

                    *target_ang_vel,
                    *target_delta_ang_pos,
                    *target_delta_lin_vel,
                    *target_delta_lin_pos,
                    target_distance,
                    hit_probability,
                ]
            )
        elif self.angle_representation == 1:

            return np.array(
                [
                    *ang_vel,
                    *ang_pos,
                    *lin_vel,
                    *lin_pos,
                    *agent_aux_state,
                    *self.past_actions[agent_id],

                    *ally_ang_vel,
                    *ally_delta_ang_pos
                    *ally_delta_lin_vel,
                    *ally_delta_lin_pos,

                    *target_ang_vel,
                    *target_delta_ang_pos,
                    *target_delta_lin_vel,
                    *target_delta_lin_pos,
                    target_distance,
                    hit_probability,
                ]
            )
        else:
            raise AssertionError("Not supposed to end up here!")


    def compute_term_trunc_reward_info_by_id(
        self, agent_id: int
    ) -> tuple[bool, bool, float, dict[str, Any]]:
        """Computes the termination, truncation, and reward of the current timestep."""
        term, trunc, reward, info = super().compute_base_term_trunc_reward_info_by_id(agent_id)

        self._compute_engagement_rewards(agent_id)

        reward += self.rewards[agent_id]

        self.compute_collisions(agent_id)

        if term:
            self.disarm_drone(agent_id)

        return term, trunc, reward, info

    def _compute_engagement_rewards(self, agent_id) -> None:
        """_compute_engagement_rewards."""
        # reset rewards
        self.rewards[agent_id] *= 0.0
        target_id = self.find_nearest_lw(agent_id)
        self.current_target_id[agent_id] = target_id # stores targets

        # sparse reward computation
        if not self.sparse_reward:

            if target_id != agent_id:  #avoid the scenario where there are no targets, returns the last rewards in the last steps

                self.rew_closing_distance[agent_id] = np.clip(
                    (self.previous_distance[agent_id][target_id] - self.current_distance[agent_id][target_id]),
                    a_min=-10,
                    a_max=None,)

                exploding_distance = self.current_distance[agent_id][target_id] - 0.5
                self.rew_close_to_target[agent_id] = 1 / (exploding_distance
                                                          if exploding_distance > 0
                                                          else 0.09)  # if the 1 is to hight the kamikazes will circle the enemy. try a

                self.rew_speed_magnitude[agent_id] = - (self.hit_probability[agent_id] + 0.05)

                self.rew_closing_distance[agent_id] *= self.distance_factor
                self.rew_close_to_target[agent_id] *= self.proximity_factor
                self.rew_speed_magnitude[agent_id] *= self.speed_factor

                self.rewards[agent_id] += (
                                      self.rew_closing_distance[agent_id]
                                      + self.rew_close_to_target[agent_id]
                                      + self.rew_speed_magnitude[agent_id]
                )



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

    def print_rewards(self, agent_id, **kargs):
        for k,v in kargs.items():
            print(f'{agent_id} {k} = {v}')

        print(f'{self.current_magnitude[agent_id]=}')
        print(f'{self.current_vel_angles[agent_id][1]=}')
        print(f'------------------------------------------------------------')

    def write_step_data(self, filename):
        # write the rewards data to a file
        import csv

        filename.parent.mkdir(parents=True, exist_ok=True)

        with open(filename, 'w', newline='') as csvfile:
            fieldnames = [ "aviary_steps", "physics_steps", "step_count", "agent_id", "elapsed_time",
                          "rew_closing_distance", "rew_close_to_target", "rew_engaging_enemy", "rew_speed_magnitude", "rew_near_engagement", "acc_rewards",
                          "vel_angles", "rel_vel_magnitudade", "approaching", "chasing", "in_range", "in_cone", "current_term",
                          "info[downed]", "info[exploded_target]", "info[exploded_ally]",  "info[crashes]", "info[ally_collision]", "info[is_success]",
                            "info[mission_complete]", "info[out_of_bounds]", "info[timeover]"
                          ]
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

    def plot_agent_rewards(self, filename, agent_id):
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('TkAgg')

        # Read the CSV file into a DataFrame
        df = pd.read_csv(filename)

        # Filter data for the specified agent
        agent_data = df[df['agent_id'] == agent_id]

        # Plotting
        plt.figure(figsize=(12, 8))

        plt.plot(agent_data['elapsed_time'], agent_data['rew_closing_distance'], label='Closing Distance')
        plt.plot(agent_data['elapsed_time'], agent_data['rew_engaging_enemy'], label='Engaging Enemy')
        plt.plot(agent_data['elapsed_time'], agent_data['rew_speed_magnitude'], label='Speed Magnitude')
        plt.plot(agent_data['elapsed_time'], agent_data['rew_near_engagement'], label='Near Engagement')
        plt.plot(agent_data['elapsed_time'], agent_data['rel_vel_magnitudade'], label='rel_vel_magnitudade')
        #plt.plot(agent_data['elapsed_time'], agent_data['vel_angles'], label='vel_angles')
        #plt.plot(agent_data['elapsed_time'], agent_data['chasing'], label='chasing')


        plt.xlabel('Elapsed Time')
        plt.ylabel('Rewards')
        plt.title(f'Rewards Over Time for Agent {agent_id}')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_agent_infos(self, filename, agent_id):
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('TkAgg')

        # Read the CSV file into a DataFrame
        df = pd.read_csv(filename)

        # Filter data for the specified agent
        agent_data = df[df['agent_id'] == agent_id]

        # Plotting
        plt.figure(figsize=(12, 8))

        plt.plot(agent_data['elapsed_time'], agent_data['info[out_of_bounds]'], label='out_of_bounds')
        plt.plot(agent_data['elapsed_time'], agent_data['info[crashes]'], label='crashes')
        plt.plot(agent_data['elapsed_time'], agent_data['info[timeover]'], label='timeover')
        plt.plot(agent_data['elapsed_time'], agent_data['info[exploded_target]'], label='exploded_target')
        plt.plot(agent_data['elapsed_time'], agent_data['info[exploded_ally]'], label='exploded_ally')
        plt.plot(agent_data['elapsed_time'], agent_data['info[mission_complete]'], label='mission_complete')
        plt.plot(agent_data['elapsed_time'], agent_data['info[ally_collision]'], label='ally_collision')
        plt.plot(agent_data['elapsed_time'], agent_data['info[downed]'], label='downed')



        plt.xlabel('Elapsed Time')
        plt.ylabel('Rewards')
        plt.title(f'Rewards Over Time for Agent {agent_id}')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_agent_infos2(self, filename, agent_id):
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('TkAgg')

        # Read the CSV file into a DataFrame
        df = pd.read_csv(filename)

        # Filter data for the specified agent
        agent_data = df[df['agent_id'] == agent_id]

        # Plotting
        plt.figure(figsize=(12, 8))

        # Extract unique categories
        categories = [col.replace('info[', '').replace(']', '') for col in agent_data.columns if 'info[' in col]

        # Plot each category as a filled area
        for category in categories:
            plt.fill_between(agent_data['elapsed_time'], 0, agent_data[f'info[{category}]'], label=category, alpha=0.7)

        plt.xlabel('Elapsed Time')
        plt.ylabel('Rewards')
        plt.title(f'Rewards Over Time for Agent {agent_id}')
        plt.legend()
        plt.grid(True)
        plt.show()