"""Base Multiagent QuadX Environment."""
from __future__ import  annotations

import math
from copy import deepcopy
from typing import Any
from pprint import pprint

import numpy as np
import pandas as pd
import pybullet as p
from gymnasium import Space, spaces
from pettingzoo import ParallelEnv

from PyFlyt.core import Aviary
from modules.lwsfm import LWManager

class MAQuadXBaseEnv(ParallelEnv):
    """MAQuadXBaseEnv."""

    def __init__(
        self,
        spawn_settings: dict(),
        start_pos: np.ndarray = np.array(
            [[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [-1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        ),
        start_orn: np.ndarray = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        ),
        flight_dome_size: float = 10.0,
        max_duration_seconds: float = 20.0,
        angle_representation: str = "euler",
        agent_hz: int = 30,
        render_mode: None | str = None,
        seed: int = None,
        num_lm: int = 1,
        num_lw: int = 1,
        formation_center: np.ndarray = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        ),
        black_death: bool = False
    ):
        """__init__.

        Args:
            start_pos (np.ndarray): start_pos
            start_orn (np.ndarray): start_orn
            flight_dome_size (float): flight_dome_size
            max_duration_seconds (float): max_duration_seconds
            angle_representation (str): angle_representation
            agent_hz (int): agent_hz
            render_mode (None | str): render_mode
        """
        if 120 % agent_hz != 0:
            lowest = int(120 / (int(120 / agent_hz) + 1))
            highest = int(120 / int(120 / agent_hz))
            raise AssertionError(
                f"`agent_hz` must be round denominator of 120, try {lowest} or {highest}."
            )

        if render_mode is not None:
            assert (
                render_mode in self.metadata["render_modes"]
            ), f"Invalid render mode {render_mode}, only {self.metadata['render_modes']} allowed."
        self.render_mode = render_mode is not None

        """SPACES"""
        # attitude size increases by 1 for quaternion
        if angle_representation == "euler":
            attitude_shape = 12
        elif angle_representation == "quaternion":
            attitude_shape = 13
        else:
            raise AssertionError(
                f"angle_representation must be either `euler` or `quaternion`, not {angle_representation}"
            )

        # action space flight_mode 6: vx, vy, vr, vz
        angular_rate_limit = 10# np.pi
        thrust_limit = 10.0
        high = np.array(
            [
                thrust_limit,
                thrust_limit,
                angular_rate_limit,
                thrust_limit,
            ]
        )
        low = np.array(
            [
                -thrust_limit,
                -thrust_limit,
                -angular_rate_limit,
                -thrust_limit,
            ]
        )
        self._action_space = spaces.Box(low=low, high=high, dtype=np.float64)

        # observation space
        self.auxiliary_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64
        )
        self.combined_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                attitude_shape
                + self.auxiliary_space.shape[0]
                + self.action_space(None).shape[0],  # pyright: ignore
            ),
            dtype=np.float64,
        )



        """ENVIRONMENT CONSTANTS"""
        # check the start_pos shapes
        assert (
            len(start_pos.shape) == 2
        ), f"Expected `start_pos` to be of shape [num_agents, 3], got {start_pos.shape}."
        assert (
            start_pos.shape[-1] == 3
        ), f"Expected `start_pos` to be of shape [num_agents, 3], got {start_pos.shape}."
        assert (
            start_pos.shape == start_orn.shape
        ), f"Expected `start_pos` to be of shape [num_agents, 3], got {start_pos.shape}."

        #self.black_death = black_death
        self.start_pos = start_pos
        self.start_orn = start_orn
        self.spawn_settings = spawn_settings
        self.seed = seed
        self.num_drones = len(start_pos)
        self.num_lm = self.spawn_settings['num_lm'] if spawn_settings else num_lm
        self.num_lw = self.spawn_settings['num_lw'] if spawn_settings else num_lw
        self.formation_center = formation_center
        self.lethal_distance = 1
        self.lethal_angle = 0.1

        self.rewards_data = []
        self.rew_log = [
            [
                'self.aviary.elapsed_time'
               ,'rew_closing_distance'
               ,'rew_progress_eng'
               ,'rew_engaging_enemy'
               ,'rew_last_distance'
               ,'ang_vel_a'
               ,'ang_pos_a'
               ,'lin_vel_a'
               ,'lin_pos_a_x'
               ,'lin_pos_a_y'
               ,'lin_pos_a_z'
               ,'quaternion_a'
               ,'ang_vel_t'
               ,'ang_pos_t'
               ,'lin_vel_t'
               ,'lin_pos_t_x'
               ,'lin_pos_t_y'
               ,'lin_pos_t_z'
               ,'quaternion_t'
               ,'self.current_distance[target_id]'
               ,'self.current_angles[target_id]'
            ]
        ]


        self.flight_dome_size = flight_dome_size
        self.max_steps = int(agent_hz * max_duration_seconds)
        self.env_step_ratio = int(120 / agent_hz)
        if angle_representation == "euler":
            self.angle_representation = 0
        elif angle_representation == "quaternion":
            self.angle_representation = 1

        # select agents
        self.num_possible_agents = self.num_lm #len(start_pos)
        self.possible_agents = []
        self.agent_name_mapping = {}
        self.possible_targets = []
        self.target_name_mapping = {}

        self.possible_agents = [
            "agent_" + str(r) for r in range(self.num_possible_agents)
        ]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        self.possible_targets = [
            "target_" + str(r) for r in range(len(start_pos) - self.num_lm)
        ]

        self.drone_list = self.possible_agents + self.possible_targets

        self.uav_mapping = np.array(['lm'] * self.num_lm + ['lw'] * (len(start_pos) - self.num_lm))

        self.drone_id_mapping = dict(
            zip(list(range(len(self.drone_list))), self.drone_list)
        )

        self.drone_classes = dict(
            zip(range(self.num_drones), self.uav_mapping))


        """RUNTIME PARAMETERS"""
        self.current_actions = np.zeros(
            (
                self.num_drones,
                *self.action_space(None).shape,
            )
        )
        self.past_actions = np.zeros(
            (
                self.num_drones,
                *self.action_space(None).shape,
            )
        )



    def observation_space(self, _) -> Space:
        """observation_space.

        Args:
            _:

        Returns:
            Space:
        """
        raise NotImplementedError

    def action_space(self, _) -> spaces.Box:
        """action_space.

        Args:
            _:

        Returns:
            spaces.Box:
        """
        return self._action_space

    def close(self):
        """close."""
        if hasattr(self, "aviary"):
            self.aviary.disconnect()

    def reset(self, seed=None, options=dict()) -> tuple[dict[str, Any], dict[str, Any]]:
        """reset.

        Args:
            seed:
            options:

        Returns:
            tuple[dict[str, Any], dict[str, Any]]: observation and infos
        """
        raise NotImplementedError

    def begin_reset(self, seed=None, options=dict(), drone_options=dict()):
        """The first half of the reset function."""
        # if we already have an env, disconnect from it
        if hasattr(self, "aviary"):
            self.aviary.disconnect()
        self.step_count = 0

        np_random = np.random.RandomState(seed=seed)

        if self.spawn_settings is not None:
            self.start_pos, self.start_orn, self.formation_center = self.generate_start_pos_orn(**self.spawn_settings)

        self.agents = self.possible_agents[:]
        self.targets = self.possible_targets[:]

        self.current_term = {k: False for k in self.agents}
        self.current_trun = {k: False for k in self.agents}
        self.current_acc_rew = {k: 0.0 for k in self.agents}
        self.current_inf = {k: dict() for k in self.agents}
        self.current_obs = {k: [] for k in self.agents}
        self.rew_closing_distance = 0.0
        self.rew_engaging_enemy = 0.0
        self.rew_speed_magnitude = 0.0
        self.rew_near_engagement = 0.0
        self.current_target_id = np.zeros((self.num_possible_agents,), dtype=np.int32)

        self.drone_list = self.agents + self.targets
        self.num_drones = len(self.drone_list)
        self.uav_mapping = np.array(['lm'] * self.num_lm + ['lw'] * (len(self.start_pos) - self.num_lm))
        self.drone_id_mapping = dict(enumerate(self.drone_list))
        self.drone_classes = dict(enumerate(self.uav_mapping))

        self.attitudes= np.zeros((self.num_drones, 4,3), dtype=np.float64)
        self.drone_positions = np.zeros((self.num_drones, 3), dtype=np.float64)

        self.rewards = np.zeros((self.num_possible_agents,), dtype=np.float64)
        self.in_cone = np.zeros((self.num_drones,self.num_drones), dtype=bool)
        self.in_range = np.zeros((self.num_drones,self.num_drones), dtype=bool)
        self.chasing = np.zeros((self.num_drones,self.num_drones), dtype=bool)

        self.past_magnitude = np.zeros(self.num_drones, dtype=np.float64)
        self.current_magnitude = np.zeros(self.num_drones, dtype=np.float64)

        self.previous_rel_vel_magnitude = np.zeros((self.num_drones, self.num_drones), dtype=np.float64)
        self.current_rel_vel_magnitude = np.zeros((self.num_drones, self.num_drones), dtype=np.float64)

        self.forward_vecs = np.zeros((self.num_drones, self.num_drones), dtype=np.float64)
        self.separation = np.zeros((self.num_drones, self.num_drones, 3), dtype=np.float64)

        self.previous_angles = np.zeros((self.num_drones, self.num_drones), dtype=np.float64)
        self.current_angles = np.zeros((self.num_drones,self.num_drones), dtype=np.float64)

        self.previous_distance = np.zeros((self.num_drones, self.num_drones), dtype=np.float64)
        self.current_distance = np.zeros((self.num_drones,self.num_drones), dtype=np.float64)

        self.current_traj_angles = np.zeros((self.num_drones, self.num_drones), dtype=np.float64)
        self.previous_traj_angles = np.zeros((self.num_drones,self.num_drones), dtype=np.float64)

        self.current_vel_angles = np.zeros((self.num_drones, self.num_drones), dtype=np.float64)
        self.previous_vel_angles = np.zeros((self.num_drones, self.num_drones), dtype=np.float64)

        self.last_obs_time = -1.0
        self.last_rew_time = -1.0

        self.squad_id_mapping = {}

        # rebuild the environment
        self.aviary = Aviary(
            start_pos=self.start_pos,
            start_orn=self.start_orn,
            drone_type="quadx",
            render=bool(self.render_mode),
            drone_options=drone_options,
            seed=seed,
        )

        self.change_visuals()
        self.init_debug_vectors()

    def end_reset(self, seed=None, options=dict()):
        """The tailing half of the reset function."""
        # register all new collision bodies
        self.aviary.register_all_new_bodies()

        # set flight mode
        set_points = [6 if self.uav_mapping[i] == 'lm' else 7 for i in range(len(self.uav_mapping)) ]

        self.aviary.set_mode(set_points)

        self.update_control_lists()

        self._compute_agent_states()

        self.lw_manager = LWManager(env=self,
                                 formation_radius=1.0,
                                 threat_radius=4.0,
                                 shoot_range=2.0,
                                 )

        # wait for env to stabilize
        for _ in range(10):
            self.aviary.step()



    def compute_auxiliary_by_id(self, agent_id: int):
        """This returns the auxiliary state form the drone."""
        return self.aviary.aux_state(agent_id)

    def compute_attitude_by_id(
        self, agent_id: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """state.

        This returns the base attitude for the drone.
        - ang_vel (vector of 3 values)
        - ang_pos (vector of 3 values)
        - lin_vel (vector of 3 values)
        - lin_pos (vector of 3 values)
        - quaternion (vector of 4 values)
        """
        raw_state = self.aviary.state(agent_id)

        # state breakdown
        ang_vel = raw_state[0]
        ang_pos = raw_state[1]
        lin_vel = raw_state[2]
        lin_pos = raw_state[3]

        # quaternion angles
        quaternion = p.getQuaternionFromEuler(ang_pos)

        return ang_vel, ang_pos, lin_vel, lin_pos, quaternion

    def compute_observation_by_id(self, agent_id: int) -> Any:
        """compute_observation_by_id.

        Args:
            agent_id (int): agent_id

        Returns:
            Any:
        """
        raise NotImplementedError

    def _compute_agent_states(self) -> None:
        """compute_observation_by_id.

        Args:
            agent_id (int): agent_id

        Returns:
            Any:
        """
        raise NotImplementedError

    def compute_base_term_trunc_reward_info_by_id(
        self, agent_id: int
    ) -> tuple[bool, bool, float, dict[str, Any]]:
        """compute_base_term_trunc_reward_by_id."""
        # initialize
        term = False
        trunc = False
        reward = 0.0
        info = dict()

        # exceed step count
        if self.step_count > self.max_steps:
            trunc |= True
            info["timeover"] = True

        # collision with ground
        if np.any(self.aviary.contact_array[self.aviary.drones[agent_id].Id][0]):
            reward -= 100.0
            info["crashes"] = True
            term |= True

        # exceed flight dome
        if np.linalg.norm(self.aviary.state(agent_id)[-1]) > self.flight_dome_size:
            reward -= 100.0
            info["out_of_bounds"] = True
            term |= True

        # collide with other kamikaze
        colissions = self.get_friendlyfire_type_collision(agent_id)
        if 'lm' in colissions:
            reward -= 100.0
            info["ally_collision"] = True
            term |= True

        # Shoot down by a loyal wingman
        if self.lw_manager.downed_lm[agent_id]:
            reward -= 100
            term |= True
            info['downed'] = True

        # destroy any loyal wingman
        explosion_mapping = self.get_explosion_mapping(agent_id)
        if 'lw' in explosion_mapping.values():
            reward += 1000.0
            info["exploded_target"] = True
            term |= True

        if self.get_collateral_explosions(agent_id):
            reward += 0
            if not info.get("exploded_target", False): # avoid misscounting when exploded the same target
                info["exploded_by_ally"] = True
            term |= True

        # # being near of other lm without
        # if 'lm' in midrange_collisions:
        #     reward -= 1.0
        #     #info["exploded_ally"] = True
        #     #term |= True
        #     #trunc |= True

        return term, trunc, reward, info

    def compute_term_trunc_reward_info_by_id(
        self, agent_id: int
    ) -> tuple[bool, bool, float, dict[str, Any]]:
        """compute_term_trunc_reward_info_by_id.

        Args:
            agent_id (int): agent_id

        Returns:
            Tuple[bool, bool, float, dict[str, Any]]:
        """
        raise NotImplementedError

    def step(
        self, actions: dict[str, np.ndarray]
    ) -> tuple[
        dict[str, Any],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        """step.

        Args:
            actions (dict[str, np.ndarray]): actions

        Returns:
            tuple[dict[str, Any], dict[str, float], dict[str, bool], dict[str, bool], dict[str, dict[str, Any]]]:
        """

        # copy over the past values

        self.past_actions = deepcopy(self.current_actions)

        # set the new actions and send to aviary
        self.current_actions *= 0.0

        # tune up the linear velocities
        #actions = {k: v * np.array([1, 1, 5, 5]) for k, v in actions.items()}

        for id, uav in self.armed_uavs.items():
            if self.get_drone_type_by_id(id) == 'lm':
                self.current_actions[id] = actions[uav]
                self.aviary.set_setpoint(id, self.current_actions[id])

        # observation and rewards dictionary
        observations = dict()
        terminations = {k: False for k in self.agents}
        truncations = {k: False for k in self.agents}
        rewards = {k: 0.0 for k in self.agents}
        infos = {k: dict() for k in self.agents}

        # step enough times for one RL step
        for _ in range(self.env_step_ratio):

            self.aviary.step()
            self._compute_agent_states()
            self.collision_matrix = self.create_collision_matrix(distance_threshold=0.5)
            self.lw_manager.update(stand_still=False)

            # update reward, term, trunc, for each agent
            for ag in self.agents:
                ag_id = self.agent_name_mapping[ag]

                # compute term trunc reward
                term, trunc, rew, info = self.compute_term_trunc_reward_info_by_id(ag_id)

                terminations[ag] |= term
                truncations[ag] |= trunc
                rewards[ag] += rew
                infos[ag] = {**infos[ag], **info}

                # compute observations
                observations[ag] = self.compute_observation_by_id(ag_id)

                self.current_term[ag] = term
                self.current_trun[ag] = trunc
                self.current_acc_rew[ag] += rew
                self.current_inf[ag] = {**infos[ag], **info}
                self.current_obs[ag] = observations[ag]
                #self.save_step_data(ag)

        # increment step count and cull dead agents for the next round
        self.step_count += 1
        self.agents = [
            agent
            for agent in self.agents
            if not (terminations[agent] or truncations[agent])
        ]
        self.update_control_lists()

        # all targets destroyed, End.
        if self.targets == []:
            terminations = {k: True for k in self.agents}
            truncations = {k: True for k in self.agents}
            infos = {key: {'survived': True, **infos[key]} if infos[key] == {} else infos[key] for key in infos.keys() }

        return observations, rewards, terminations, truncations, infos

    #------------------------------ Env End ----------------------------------------------------

    def compute_collisions(self,ag_id):

        explosion_mapping = self.get_explosion_mapping(ag_id)

        for id, type in explosion_mapping.items():
            target = self.drone_id_mapping[id]
            if type == 'lw':
                self.disarm_drone(id)
                if target in self.targets:  # avoid double removing in the same iteration
                    self.targets.remove(target)



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

    @staticmethod
    def generate_start_pos_orn(seed=None, lw_center_bounds=5.0, lw_spawn_radius=1.0, num_lw=3, min_z=1.0,
                               lm_center_bounds=5, lm_spawn_radius=10, num_lm=3,):

        np_random = np.random.RandomState(seed=seed)
        lw_formation_center = [np.random.uniform(-lw_center_bounds, lw_center_bounds),
                               np.random.uniform(-lw_center_bounds, lw_center_bounds),
                               np.random.uniform(min_z, lw_center_bounds + min_z)]

        start_pos_lw = LWManager.generate_formation_pos(lw_formation_center, num_lw, lw_spawn_radius)
        start_orn_lw = np.zeros_like(start_pos_lw)

        lm_spawn_center = [np.random.uniform(-lm_center_bounds, lm_center_bounds),
                           np.random.uniform(-lm_center_bounds, lm_center_bounds),
                           np.random.uniform(min_z, lm_center_bounds + min_z)]

        start_pos_lm = MAQuadXBaseEnv.generate_random_coordinates(lw_formation_center, lw_spawn_radius,
                                                                   lm_spawn_center, lm_spawn_radius, num_lm, min_z)

        start_orn_lm = (np_random.rand(num_lm, 3) - 0.5) * 2.0 * np.array([1.0, 1.0, 2 * np.pi])

        return np.concatenate([start_pos_lm, start_pos_lw]), np.concatenate([start_orn_lm, start_orn_lw]), lw_formation_center


    @staticmethod
    def generate_random_coordinates(lw_formation_center, lw_spawn_radius, lm_spawn_center, lm_spawn_radius, num_lm,
                                    min_z):
        # Ensure the formation center and spawn center are NumPy arrays
        lw_formation_center = np.array(lw_formation_center)
        lm_spawn_center = np.array(lm_spawn_center)

        # Generate random coordinates within the specified spawn radius and above the minimum z
        lm_coordinates = []
        while len(lm_coordinates) < num_lm:
            x = np.random.uniform(low=lm_spawn_center[0] - lm_spawn_radius, high=lm_spawn_center[0] + lm_spawn_radius)
            y = np.random.uniform(low=lm_spawn_center[1] - lm_spawn_radius, high=lm_spawn_center[1] + lm_spawn_radius)
            z = np.random.uniform(low=min_z, high=lm_spawn_center[2] + lm_spawn_radius)

            # Check if the generated coordinates are outside the exclusion area of the lw formation
            lm_distance = np.linalg.norm(lw_formation_center[:2] - np.array([x, y]))
            if lm_distance > lw_spawn_radius * 3:
                lm_coordinates.append([x, y, z])

        return np.array(lm_coordinates)

    def get_drone_type(self, agent):

        if self.drone_classes[self.agents[agent]] == 'lw':
            return 'lw'
        elif self.drone_classes[self.agents[agent]] == 'lm':
            return 'lm'
        else:
            return None

    def get_drone_type_by_id(self, agent_id):

        if self.drone_classes[agent_id] == 'lw':
            return 'lw'
        elif self.drone_classes[agent_id] == 'lm':
            return 'lm'
        else:
            return None

    def get_lw_action(self, target_id):

        return np.array([*self.start_pos[target_id][0:2], 0, self.start_pos[target_id][2]])


    def update_control_lists(self):

        self.drone_list = self.agents + self.targets

        self.armed_uavs = {key:value for key, value in self.drone_id_mapping.items() if value in self.agents or value in self.targets}

        self.armed_uav_types = {key:self.get_drone_type_by_id(key) for key, value in self.armed_uavs.items()}

        self.uav_mapping = np.array(['lm'] * len(self.agents) + ['lw'] * len(self.targets))

        self.num_drones = len(self.agents) + len(self.targets)


    def draw_forward_vector(self, drone_index, line_id = None, length=1.0, lineColorRGB=[1, 0, 0] ):
        #Calculate the forward vector based on the drone's orientation

        drone_pos, drone_orientation = p.getBasePositionAndOrientation(drone_index)
        forward_vector = [math.cos(drone_orientation[2]), math.sin(drone_orientation[2]), 0]

        # Calculate the end point of the vector
        end_point = [drone_pos[0] + length * forward_vector[0],
                     drone_pos[1] + length * forward_vector[1],
                     drone_pos[2] + length * forward_vector[2]]

        drone_pos, drone_orientation = p.getBasePositionAndOrientation(drone_index)

        # Convert quaternion to rotation matrix
        rotation_matrix = np.array(p.getMatrixFromQuaternion(drone_orientation)).reshape((3, 3))

        # Extract the forward vector (assuming the convention that forward is along the x-axis)
        forward_vector = rotation_matrix[:, 0]

        # Normalize the vector
        forward_vector /= np.linalg.norm(forward_vector)

        # Calculate the end point of the vector
        end_point = [drone_pos[0] + length * forward_vector[0],
                     drone_pos[1] + length * forward_vector[1],
                     drone_pos[2] + length * forward_vector[2]]

        # Draw the line in PyBullet
        if line_id is not None:

            self.forward_debug_line = self.aviary.addUserDebugLine(
                drone_pos, end_point, lineColorRGB=lineColorRGB,
                lineWidth=2,  replaceItemUniqueId=line_id
            )
        else:
            self.forward_debug_line = self.aviary.addUserDebugLine(
                drone_pos, end_point, lineColorRGB=lineColorRGB,
                lineWidth=2, parentObjectUniqueId=drone_index
            )

        return self.forward_debug_line

    def draw_vel_vector(self, drone_index, line_id = None, length=1.0, lineColorRGB=[1, 1, 0] ):
        # Calculate the forward vector based on the drone's orientation

        drone_pos, drone_orientation = p.getBasePositionAndOrientation(drone_index)
        vel_vector = self.aviary.state(drone_index-1)[2]


        # Calculate the end point of the vector
        end_point = [drone_pos[0] + length * vel_vector[0],
                     drone_pos[1] + length * vel_vector[1],
                     drone_pos[2] + length * vel_vector[2]]

        # Draw the line in PyBullet
        if line_id is not None:

            self.forward_debug_line = self.aviary.addUserDebugLine(
                drone_pos, end_point, lineColorRGB=lineColorRGB,
                lineWidth=2,  replaceItemUniqueId=line_id
            )
        else:
            self.forward_debug_line = self.aviary.addUserDebugLine(
                drone_pos, end_point, lineColorRGB=lineColorRGB,
                lineWidth=2, parentObjectUniqueId=drone_index
            )

        return self.forward_debug_line

    def draw_v1v2_vector(self, v1, v2, line_id = None, length=1.0, lineColorRGB=[1, 1, 0]):
        # Calculate the forward vector based on the drone's orientation

        # Calculate the end point of the vector
        end_point = [v1[0] - length * v2[0],
                     v2[1] - length * v2[1],
                     v2[2] - length * v2[2]]

        # Draw the line in PyBullet
        if line_id is not None:

            self.forward_debug_line = self.aviary.addUserDebugLine(
                v1, end_point, lineColorRGB=lineColorRGB,
                lineWidth=2,  replaceItemUniqueId=line_id
            )


        return self.forward_debug_line



    def draw_separation_vector(self, drone_index, separation_vector, line_id = None, length=1.0, lineColorRGB=[1, 0, 0] ):
        # Calculate the forward vector based on the drone's orientation

        drone_pos, drone_orientation = p.getBasePositionAndOrientation(drone_index)
        forward_vector = [math.cos(drone_orientation[2]), math.sin(drone_orientation[2]), 0]

        # Calculate the end point of the vector
        end_point = [drone_pos[0] + length * separation_vector[0],
                     drone_pos[1] + length * separation_vector[1],
                     drone_pos[2] + length * separation_vector[2]]

        # Draw the line in PyBullet
        if line_id is not None:

            self.forward_debug_line = self.aviary.addUserDebugLine(
                drone_pos, end_point, lineColorRGB=lineColorRGB,
                lineWidth=2, replaceItemUniqueId=line_id
            )
        else:
            self.forward_debug_line = self.aviary.addUserDebugLine(
                drone_pos, end_point, lineColorRGB=lineColorRGB,
                lineWidth=2, parentObjectUniqueId=drone_index)

        return self.forward_debug_line

    def find_nearest_lw(self, agent_id: int) -> int:

        distances = self.current_distance[agent_id, :]

        self.update_control_lists()

        lw_indices = np.array([key for key, value in self.armed_uav_types.items() if value == 'lw'])

        if not lw_indices.any(): # TODO seems wrong
            return self.find_nearest_lm(agent_id)

        # Filter distances based on 'lw' indices
        lw_distances = distances[lw_indices]

        # Find the index of the minimum distance in lw_distances
        nearest_lw_index = lw_indices[np.argmin(lw_distances)]

        return nearest_lw_index

    def find_nearest_lm(self, agent_id: int, exclude_self = False):

        self.update_control_lists()

        distances = self.current_distance[agent_id, :]

        if exclude_self:
            lm_indices = np.array([key for key, value in self.armed_uav_types.items() if value == 'lm' and key != agent_id])
        else:
            lm_indices = np.array([key for key, value in self.armed_uav_types.items() if value == 'lm'])

        if not len(lm_indices) > 0:
            if self.drone_classes == 'lw':
                return None # for LWManager works
            else:
                return -1

        # Filter distances based on 'lw' indices
        lm_distances = distances[lm_indices]

        # Find the index of the minimum distance in lw_distances
        nearest_lm_index = lm_indices[np.argmin(lm_distances)]

        return nearest_lm_index

    def disarm_drone(self, agent_id):

        armed_drones_ids = {drone.Id for drone in self.aviary.armed_drones}
        armed_status_list = [drone.Id in armed_drones_ids for drone in self.aviary.drones]
        armed_status_list[agent_id] = False

        self.aviary.set_armed(armed_status_list)

    def get_type_collision(self, agent_id):

        collisions = np.where(self.collision_matrix[self.aviary.drones[agent_id].Id][1:])[0]
        collision_types = [value for key, value in self.drone_classes.items() if key in collisions]
        return collision_types

    def get_friendlyfire_type_collision(self, agent_id):

        collisions = np.where(self.aviary.contact_array[self.aviary.drones[agent_id].Id][1:])[0]
        collision_types = [value for key, value in self.drone_classes.items() if key in collisions]
        return collision_types

    def get_collision_ids(self, agent_id):

        collisions = np.where(self.collision_matrix[self.aviary.drones[agent_id].Id][1:])[0]
        return collisions

    def get_explosion_mapping(self, agent_id):
        collisions = np.where(self.collision_matrix[self.aviary.drones[agent_id].Id][1:])[0]
        explosion_mapping = {key: value for key, value in self.drone_classes.items() if key in collisions}

        return explosion_mapping

    def get_collateral_explosions(self, agent_id):
        explosions_in_range = self.get_collision_ids(agent_id)

        collateral_explosions = [id for id in explosions_in_range if 'lw' in self.get_type_collision(id)]

        if len(collateral_explosions) > 0:
            return True
        else:
            return False

    def create_collision_matrix(self, distance_threshold):
        # Initialize a num_bodies x num_bodies matrix with False values
        num_bodies = np.max([self.aviary.getBodyUniqueId(i) for i in range(self.aviary.getNumBodies())]) + 1

        collision_matrix = np.array([[False] * num_bodies for _ in range(num_bodies)])

        # Iterate through all pairs of bodies
        for i in range(num_bodies):
            for j in range(i + 1, num_bodies):
                # Check for collisions between body i and body j
                points = p.getClosestPoints(bodyA=i, bodyB=j, distance=distance_threshold)

                # If there are points, a collision occurred
                if points:
                    collision_matrix[i][j] = True
                    collision_matrix[j][i] = True  # The matrix is symmetric

        return collision_matrix


    def draw_debug_vectors(self):

        self.agent_forward_line = self.draw_forward_vector(
            1, line_id=self.agent_forward_line, length=0.35, lineColorRGB=[1, 0, 0]
        )
        self.target_forward_line = self.draw_forward_vector(
            1, line_id=self.target_forward_line, length=0.35, lineColorRGB=[0, 0, 1]
        )

        self.agent_vel_line = self.draw_vel_vector(
            1, line_id=self.agent_vel_line, length=0.35, lineColorRGB=[1, 1, 0]
        )
        self.target_vel_line = self.draw_vel_vector(
            1, line_id=self.target_vel_line, length=0.35, lineColorRGB=[1, 1, 0]
        )

        self.target_traj_line = self.draw_separation_vector(
            1,
            line_id=self.agent_traj_line,
            separation_vector=self.separation[2][0],
            lineColorRGB=[0, 1, 0]
        )


    def change_visuals(self):
        LightBlue = [0.5, 0.5, 1, 1]
        Red = [1, 0, 0, 1]
        LightRed = [1, 0.5, 0.5, 1]
        DarkBlue = [0, 0, 0.8, 1]
        [self.aviary.changeVisualShape(drone.Id, -1, rgbaColor=DarkBlue)
         for i, drone in enumerate(self.aviary.drones)
         if self.get_drone_type_by_id(i) == 'lw']

    def init_debug_vectors(self):
        # self.time_elapsed = self.aviary.addUserDebugText(
        #     text=str(self.aviary.elapsed_time), textPosition=[2, 2, 2], textColorRGB=[1, 0, 0], )
        self.agent_forward_line = self.aviary.addUserDebugLine([0,0,0], [0,0,1], lineColorRGB=[1,0,0], lineWidth=2)
        self.target_forward_line = self.aviary.addUserDebugLine([0,0,0], [0,0,1], lineColorRGB=[1, 0, 0], lineWidth=2)
        self.agent_vel_line = self.aviary.addUserDebugLine([0,0,0], [0,0,1], lineColorRGB=[1,1,0], lineWidth=2)
        self.target_vel_line = self.aviary.addUserDebugLine([0,0,0], [0,0,1], lineColorRGB=[1,1,0], lineWidth=2)
        self.agent_traj_line = self.aviary.addUserDebugLine([0,0,0], [0,0,1], lineColorRGB=[0,1,0], lineWidth=2)


    def decode_observation(self, obs) -> dict:

        ang_vel = obs[:3]
        quaternion = obs[3:7]
        lin_vel = obs[7:10]
        lin_pos = obs[10:13]
        aux_state = obs[13:17]
        past_actions = obs[17:21]
        vel_magnitude = obs[21]
        ally_lin_vel = obs[22:25]
        ally_lin_pos = obs[25:28]
        ang_vel_target = obs[28:31]
        quaternion_target = obs[31:35]
        lin_vel_target = obs[35:38]
        lin_pos_target = obs[38:41]
        target_last_shot_time = obs[41]

        return {
            "ang_vel": ang_vel,
            "quaternion": quaternion,
            "lin_vel": lin_vel,
            "lin_pos": lin_pos,
            "aux_state": aux_state,
            "past_actions": past_actions,
            'vel_magnitude': vel_magnitude,
            "ally_lin_vel": ally_lin_vel,
            "ally_lin_pos": ally_lin_pos,
            "ang_vel_target": ang_vel_target,
            "quaternion_target": quaternion_target,
            "lin_vel_target": lin_vel_target,
            "lin_pos_target": lin_pos_target,
            "target_last_shot_time": target_last_shot_time,
        }

    def print_obs_variables(self, obs):
        ang_vel = obs[:3]
        quaternion = obs[3:7]
        lin_vel = obs[7:10]
        lin_pos = obs[10:13]
        aux_state = obs[13:17]
        past_actions = obs[17:21]
        ally_lin_vel = obs[21:24]
        ally_lin_pos = obs[24:27]
        ang_vel_target = obs[27:30]
        quaternion_target = obs[30:34]
        lin_vel_target = obs[34:37]
        lin_pos_target = obs[37:40]

        print(f"ang_vel: {ang_vel}")
        print(f"quaternion: {quaternion}")
        print(f"lin_vel: {lin_vel}")
        print(f"lin_pos: {lin_pos}")
        print(f"aux_state: {aux_state}")
        print(f"past_actions: {past_actions}")
        print(f"ally_lin_vel: {ally_lin_vel}")
        print(f"ally_lin_pos: {ally_lin_pos}")
        print(f"ang_vel_target: {ang_vel_target}")
        print(f"quaternion_target: {quaternion_target}")
        print(f"lin_vel_target: {lin_vel_target}")
        print(f"lin_pos_target: {lin_pos_target}")


    def save_step_data(self, agent):
        agent_id = self.agent_name_mapping[agent]
        target_id = self.current_target_id[agent_id]
        step_data = {
            "agent_id": agent_id,
            "elapsed_time": self.aviary.elapsed_time,
            "rew_closing_distance": self.rew_closing_distance,
            "rew_engaging_enemy": self.rew_engaging_enemy,
            "rew_speed_magnitude": self.rew_speed_magnitude,
            "rew_near_engagement": self.rew_near_engagement,
            "acc_rewards": self.current_acc_rew[agent],
            "vel_angles": self.current_vel_angles[agent_id][target_id],
            "rel_vel_magnitudade": self.current_rel_vel_magnitude[agent_id][target_id],
            "approaching": int(self.approaching[agent_id][target_id]),
            "chasing": int(self.chasing[agent_id][target_id]),
            "in_range": int(self.in_range[agent_id][target_id]),
            "current_term": int(self.current_term[agent]),
            "info[downed]": int(self.current_inf[agent].get('downed', False)),
            "info[exploded_target]": int(self.current_inf[agent].get('exploded_target', False)),
            "info[exploded_ally]": int(self.current_inf[agent].get('exploded_ally', False)),
            "info[crashes]": int(self.current_inf[agent].get('crashes', False)),
            "info[ally_collision]": int(self.current_inf[agent].get('ally_collision', False)),
            "info[mission_complete]": int(self.current_inf[agent].get('mission_complete', False)),
            "info[out_of_bounds]": int(self.current_inf[agent].get('out_of_bounds', False)),
            "info[timeover]": int(self.current_inf[agent].get('timeover', False)),

        }
        self.rewards_data.append(step_data)

    def uav_alive(self, drone_id):
        if drone_id in self.armed_uavs.keys():
            return True
        else:
            return False
