"""Base Multiagent QuadX Environment."""
from __future__ import  annotations
from copy import deepcopy
from typing import Any

import numpy as np
import pybullet as p
from gymnasium import Space, spaces
from pettingzoo import ParallelEnv

from PyFlyt.core import Aviary


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
        max_duration_seconds: float = 10.0,
        angle_representation: str = "euler",
        agent_hz: int = 15,
        render_mode: None | str = None,
        uav_mapping: np.array = np.array(['lm', 'lm', 'lm', 'lm']),
        seed: int = None,
        num_lm: int = 1

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
        angular_rate_limit = np.pi
        thrust_limit = 4.0
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

        self.start_pos = start_pos
        self.start_orn = start_orn
        self.spawn_settings = spawn_settings
        self.seed = seed
        self.num_lm = num_lm
        self.num_drones = len(start_pos)

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

        #update lists to reset death agents
        self.agents = self.possible_agents[:]
        self.targets = self.possible_targets
        self.update_control_lists()

        self.rewards = np.zeros((self.num_possible_agents,), dtype=np.float64)
        self.in_cone = np.zeros((self.num_drones,self.num_drones), dtype=bool)
        self.in_range = np.zeros((self.num_drones,self.num_drones), dtype=bool)
        self.chasing = np.zeros((self.num_drones,self.num_drones), dtype=bool)

        self.current_angles = np.zeros((self.num_drones,self.num_drones), dtype=np.float64)
        self.current_distance = np.zeros((self.num_drones,self.num_drones), dtype=np.float64)
        self.previous_angles = np.zeros((self.num_drones,self.num_drones), dtype=np.float64)
        self.previous_distance = np.zeros((self.num_drones,self.num_drones), dtype=np.float64)
        self.last_obs_time = -1.0
        self.last_rew_time = -1.0
        self.past_magnitude = np.zeros(self.num_drones, dtype=np.float64)
        self.current_magnitude = np.zeros(self.num_drones, dtype=np.float64)

        if self.spawn_settings is not None:
            self.start_pos, self.start_orn = self.get_start_pos_orn(**self.spawn_settings, num_lm=self.num_lm)



        # rebuild the environment
        self.aviary = Aviary(
            start_pos=self.start_pos,
            start_orn=self.start_orn,
            drone_type="quadx",
            render=bool(self.render_mode),
            drone_options=drone_options,
            seed=seed,
        )

        LightBlue = [0.5, 0.5, 1, 1]
        Red = [1, 0, 0, 1]
        LightRed = [1, 0.5, 0.5, 1]
        DarkBlue = [0, 0, 0.8, 1]
        [self.aviary.changeVisualShape(drone.Id, -1, rgbaColor=DarkBlue)
         for i, drone in enumerate(self.aviary.drones)
         if self.drone_classes[i] == 'lw']


    def end_reset(self, seed=None, options=dict()):
        """The tailing half of the reset function."""
        # register all new collision bodies
        self.aviary.register_all_new_bodies()

        # set flight mode
        self.aviary.set_mode(6)

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
        trunc |= self.step_count > self.max_steps

        # collision with ground
        if np.any(self.aviary.contact_array[self.aviary.drones[agent_id].Id][0]):
            reward -= 100.0
            info["collision"] = True
            term |= True

        # exceed flight dome
        if np.linalg.norm(self.aviary.state(agent_id)[-1]) > self.flight_dome_size:
            reward -= 100.0
            info["out_of_bounds"] = True
            term |= True

        # collide with any loyal wingmen
        if 'lw' in self.uav_mapping[np.where(self.aviary.contact_array[self.aviary.drones[agent_id].Id][1:])]:
            reward += 10000.0
            info["sucess"] = True
            term |= True


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

        for ag_id, ag in self.drone_id_mapping.items():
            if self.drone_classes[ag_id] == 'lm':
                self.current_actions[ag_id] = actions[ag]
            elif self.drone_classes[ag_id] == 'lw':
                self.current_actions[ag_id] = self.get_lw_action(ag)

        self.aviary.set_all_setpoints(self.current_actions)

        # observation and rewards dictionary
        observations = dict()
        terminations = {k: False for k in self.agents}
        truncations = {k: False for k in self.agents}
        rewards = {k: 0.0 for k in self.agents}
        infos = {k: dict() for k in self.agents}

        # step enough times for one RL step
        for _ in range(self.env_step_ratio):
            self.aviary.step()

        # update reward, term, trunc, for each agent
        for ag in self.agents:
            ag_id = self.agent_name_mapping[ag]

            # compute term trunc reward
            term, trunc, rew, info = self.compute_term_trunc_reward_info_by_id(
                ag_id
            )
            terminations[ag] |= term
            truncations[ag] |= trunc
            rewards[ag] += rew
            infos[ag] = {**infos[ag], **info}

            # compute observations
            observations[ag] = self.compute_observation_by_id(ag_id)

            # TODO: To solve: File "C:\projects\pyflyt_parallel\venv\lib\site-packages\pettingzoo\utils\conversions.py", line 357, in step
            #     assert action is None
            # AssertionError
            if terminations[ag]:
                 truncations[ag] = True


        # increment step count and cull dead agents for the next round
        self.step_count += 1
        self.agents = [
            agent
            for agent in self.agents
            if not (terminations[agent] or truncations[agent])
        ]

        self.update_control_lists()

        # Trunc if all lm are terminated (dead)
        #if 'lm' not in self.uav_mapping[[self.agent_name_mapping[agent] for agent in self.agents]]:
        # if all(terminations):
        #      #truncations = {key: True for key in truncations}
        #     truncations[ag] = True

        return observations, rewards, terminations, truncations, infos

    def get_start_pos_orn(self, num_drones, min_distance, spawn_radius, center, num_lm, seed=None):
        start_pos = np.empty((num_drones, 3))
        min_z = 1
        np_random = np.random.RandomState(seed=seed)

        for i in range(num_drones):
            while True:
                # Generate random coordinates within the spawn area centered at 'center'
                x = np_random.uniform(center[0] - spawn_radius, center[0] + spawn_radius)
                y = np_random.uniform(center[1] - spawn_radius, center[1] + spawn_radius)
                z = np_random.uniform(max(center[2], min_z), center[2] + spawn_radius)  # Ensure z-axis is within range

                # Check if the minimum distance condition is met with existing drones
                if i == 0 or np.min(np.linalg.norm(start_pos[:i] - np.array([x, y, z]), axis=1)) >= min_distance:
                    start_pos[i] = [x, y, z]
                    break

        start_orn = (np_random.rand(num_drones, 3) - 0.5) * 2.0 * np.array([1.0, 1.0, 2 * np.pi])
        start_orn[num_lm:] = np.zeros((len(start_pos) - num_lm, 3), dtype=np.float64)

        return start_pos, start_orn

    def get_drone_type(self, agent):

        if self.uav_classes[agent] == 'lw':
            return 'lw'
        elif self.uav_classes[agent] == 'lm':
            return 'lm'
        else:
            return None

    def get_drone_type_by_id(self, agent_id):

        if self.uav_classes[self.agents[agent_id]] == 'lw':
            return 'lw'
        elif self.uav_classes[self.agents[agent_id]] == 'lm':
            return 'lm'
        else:
            return None

    def get_lw_action(self, agent_id):

        return np.array([0, 0, 0, 0])


    def speed_ratio(self, agent_id):

        speed_ratio = self.current_magnitude[agent_id] /(
            self.past_magnitude[agent_id] if self.past_magnitude[agent_id] > 0 else 1)

        return speed_ratio


    def update_control_lists(self):

        # self.agent_name_mapping = dict(
        #     zip(self.agents, list(range(len(self.agents))))
        # )

        self.drone_list = self.agents + self.targets

        self.uav_mapping = np.array(['lm'] * len(self.agents) + ['lw'] * len(self.targets))

        self.drone_id_mapping = dict(
            zip(list(range(len(self.drone_list))), self.drone_list)
        )

        self.num_drones = len(self.agents) + len(self.targets)

        self.drone_classes = dict(
            zip(range(self.num_drones), self.uav_mapping))
