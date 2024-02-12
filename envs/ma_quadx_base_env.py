"""Base Multiagent QuadX Environment."""
from __future__ import annotations

from copy import deepcopy
from typing import Any
from PIL import Image

import numpy as np
import pybullet as p
from PyFlyt.core import Aviary
from gymnasium import Space, spaces
from pettingzoo import ParallelEnv
from PyFlyt.core.utils.compile_helpers import check_numpy

from modules.lwsfm import LWManager
from modules.utils import *


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
            angle_representation: str = "euler",
            render_mode: None | str = None,
            seed: int = None,
            formation_center: np.ndarray = np.array(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            ),
            lw_stand_still: bool = True,
            lw_moves_random: bool = False,
            lw_chases: bool = False,
            lw_attacks: bool = False,
            lw_threat_radius: float = 4.0,
            lw_shoot_range: float = 1.0,
            lethal_angle: float = 0.15,
            lethal_distance: float = 2.0,
            agent_hz: int = 30,
            max_duration_seconds: float = 30.0,
            num_lm: int = 1,
            num_lw: int = 1,
            distance_factor: float = 1.0,
            proximity_factor: float = 1.0,
            speed_factor: float = 1.0,
            rew_exploding_target: float = 100,
            max_velocity_magnitude: float = 10,
            save_step_data: bool = False,
            reward_type: int = 0,
            observation_type: int = 0,
            explosion_radius: float = 0.5,
            thrust_limit: float = 10.0,
            angular_rate_limit: float = np.pi,
            direct_control: bool = False,
            lw_weapon_cooldown: float = 2.0,
            custom_spawn: bool = False,
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
        self.thrust_limit = thrust_limit
        self.angular_rate_limit = angular_rate_limit
        self.direct_control = direct_control
        self.custom_spawn = custom_spawn

        self.action_bounds = 1

        high = np.array(
            [
                angular_rate_limit,
                angular_rate_limit,
                angular_rate_limit,
                self.thrust_limit,
            ]
        )
        low = np.array(
            [
                -angular_rate_limit,
                -angular_rate_limit,
                -angular_rate_limit,
                0.0,
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

        self.flight_dome_size = flight_dome_size
        self.max_steps = int(agent_hz * max_duration_seconds)
        self.max_duration_seconds = max_duration_seconds
        self.env_step_ratio = int(120 / agent_hz)
        if angle_representation == "euler":
            self.angle_representation = 0
        elif angle_representation == "quaternion":
            self.angle_representation = 1
        self.seed = seed
        self.rewards_data = []
        self.obs_data = []
        self.ep_data = {}

        """TRAINING PARAMETERS"""
        self.start_pos = start_pos
        self.start_orn = start_orn
        self.spawn_settings = spawn_settings
        self.num_lm = self.spawn_settings['num_lm'] if spawn_settings else num_lm
        self.num_lw = self.spawn_settings['num_lw'] if spawn_settings else num_lw
        self.lethal_distance = lethal_distance
        self.lethal_angle = lethal_angle
        self.distance_factor = distance_factor
        self.proximity_factor = proximity_factor
        self.speed_factor = speed_factor
        self.lw_stand_still = lw_stand_still
        self.lw_chases = lw_chases
        self.lw_moves_random = lw_moves_random
        self.lw_attacks = lw_attacks
        self.lw_threat_radius = lw_threat_radius
        self.lw_shoot_range = lw_shoot_range
        self.lw_weapon_cooldown = lw_weapon_cooldown
        self.rew_exploding_target = rew_exploding_target
        self.max_velocity_magnitude = max_velocity_magnitude
        self.save_step_data = save_step_data,
        self.reward_type = reward_type
        self.observation_type = observation_type
        self.explosion_radius = explosion_radius
        self.max_achieved_speed = 0
        self.max_achieved_rel_speed = 0

        """ PETTINGZOO """
        self.num_drones = len(start_pos)
        self.formation_center = formation_center

        self.num_possible_agents = self.num_lm
        # self.possible_agents = []
        # self.agent_name_mapping = {}
        # self.possible_targets = []
        # self.target_name_mapping = {}

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

    def get_rgb_image(self) -> np.ndarray:
        """render."""
        check_numpy()
        assert (
            self.render_mode is not None
        ), "Please set `render_mode='human'` or `render_mode='rgb_array'` to use this function."
        self.aviary.resetDebugVisualizerCamera(cameraDistance=self.cameraDistance, cameraYaw=self.cameraYaw,
                                               cameraPitch=self.cameraPitch,
                                               cameraTargetPosition=self.cameraTargetPosition)
        # self.camera_parameters = self.aviary.getDebugVisualizerCamera(cameraDistance=4.24, cameraYaw=-198.40,
        #                                                                 cameraPitch=-25.60,
        #                                                                 cameraTargetPosition=[-1.39, -2.66, 2.38])
        _, _, rgbaImg, _, _ = self.aviary.getCameraImage(
            width=self.render_resolution[1],
            height=self.render_resolution[0],
            viewMatrix=self.camera_parameters[2],
            projectionMatrix=self.camera_parameters[3],
        )

        rgbaImg = np.asarray(rgbaImg).reshape(
            self.render_resolution[0], self.render_resolution[1], -1
        )

        return rgbaImg.astype(np.uint8)

    def begin_reset(self, seed=None, options=dict(), drone_options=dict()):
        """The first half of the reset function."""
        # if we already have an env, disconnect from it
        if hasattr(self, "aviary"):
            self.aviary.disconnect()
        self.lw_manager = None

        self.step_count = 0

        self.overlay = None
        self.render_resolution = np.array([1080, 1920])

        if (self.spawn_settings is not None) and ( not self.custom_spawn):
            self.start_pos, self.start_orn, self.formation_center = generate_start_pos_orn(**self.spawn_settings)

        self.agents = self.possible_agents[:]
        self.targets = self.possible_targets[:]


        self.current_term = {k: False for k in self.agents}
        self.current_trun = {k: False for k in self.agents}
        self.current_acc_rew = {k: 0.0 for k in self.agents}
        self.current_inf = {k: dict() for k in self.agents}
        self.current_obs = {k: [] for k in self.agents}
        self.info_counters = {'out_of_bounds': 0, 'crashes': 0, 'timeover': 0, 'exploded_target': 0,
                              'exploded_by_ally': 0, 'survived': 0, 'ally_collision': 0, 'downed': 0,
                              'is_success': 0, 'mission_complete': 0}
        self.max_achieved_speed = 0
        self.max_achieved_rel_speed = 0

        self.rew_closing_distance = np.zeros((self.num_possible_agents), dtype=np.float64)
        self.rew_close_to_target = np.zeros((self.num_possible_agents), dtype=np.float64)
        self.rew_engaging_enemy = np.zeros((self.num_possible_agents), dtype=np.float64)
        self.rew_speed_magnitude = np.zeros((self.num_possible_agents), dtype=np.float64)
        self.rew_near_engagement = np.zeros((self.num_possible_agents), dtype=np.float64)
        self.rewards = np.zeros((self.num_possible_agents,), dtype=np.float64)

        self.acc_rew_closing_distance = np.zeros((self.num_possible_agents), dtype=np.float64)
        self.acc_rew_close_to_target = np.zeros((self.num_possible_agents), dtype=np.float64)
        self.acc_rew_speed_magnitude = np.zeros((self.num_possible_agents), dtype=np.float64)


        self.current_target_ids = np.array([-1] * self.num_possible_agents, dtype=np.int32)
        self.drone_list = self.agents + self.targets
        self.num_drones = len(self.drone_list)
        self.uav_mapping = np.array(['lm'] * self.num_lm + ['lw'] * (len(self.start_pos) - self.num_lm))
        self.drone_id_mapping = dict(enumerate(self.drone_list))
        self.drone_classes = dict(enumerate(self.uav_mapping))

        self.current_actions = np.zeros((self.num_drones, *self.action_space(None).shape,))
        self.past_actions = np.zeros((self.num_drones, *self.action_space(None).shape,))

        self.attitudes = np.zeros((self.num_drones, 4, 3), dtype=np.float64)
        self.drone_positions = np.zeros((self.num_drones, 3), dtype=np.float64)

        self.in_cone = np.zeros((self.num_drones, self.num_drones), dtype=bool)
        self.in_range = np.zeros((self.num_drones, self.num_drones), dtype=bool)
        self.chasing = np.zeros((self.num_drones, self.num_drones), dtype=bool)
        self.approaching = np.zeros((self.num_drones, self.num_drones), dtype=bool)
        self.hit_probability = np.zeros((self.num_drones, self.num_drones), dtype=bool)

        self.previous_vel_magnitude = np.zeros(self.num_drones, dtype=np.float64)
        self.current_vel_magnitude = np.zeros(self.num_drones, dtype=np.float64)

        self.previous_rel_vel_magnitude = np.zeros((self.num_drones, self.num_drones), dtype=np.float64)
        self.current_rel_vel_magnitude = np.zeros((self.num_drones, self.num_drones), dtype=np.float64)

        self.forward_vecs = np.zeros((self.num_drones, self.num_drones), dtype=np.float64)
        self.separation = np.zeros((self.num_drones, self.num_drones, 3), dtype=np.float64)
        self.ground_velocities = np.zeros((self.num_drones, self.num_drones, 3), dtype=np.float64)

        self.previous_angles = np.zeros((self.num_drones, self.num_drones), dtype=np.float64)
        self.current_angles = np.zeros((self.num_drones, self.num_drones), dtype=np.float64)

        self.previous_distance = np.zeros((self.num_drones, self.num_drones), dtype=np.float64)
        self.current_distance = np.zeros((self.num_drones, self.num_drones), dtype=np.float64)

        self.current_traj_angles = np.zeros((self.num_drones, self.num_drones), dtype=np.float64)
        self.previous_traj_angles = np.zeros((self.num_drones, self.num_drones), dtype=np.float64)

        self.previous_vel_angles = np.zeros((self.num_drones, self.num_drones), dtype=np.float64)
        self.current_vel_angles = np.zeros((self.num_drones, self.num_drones), dtype=np.float64)

        self.ground_velocities = np.zeros((self.num_drones, 3), dtype=np.float64)

        self.last_obs_time = -1.0
        self.last_rew_time = -1.0

        self.squad_id_mapping = {}
        self.observation_dict = {}
        self.rewards_data = []

        self.desired_vel = np.array([0, 0, 0, 0])

        if not self.direct_control:
            drone_options = [{'drone_model': 'cf2x'} if self.uav_mapping[i] == 'lm'
                             else {'drone_model': 'custom_cf2x'}
                             for i in range(len(self.uav_mapping))]
        else:
            drone_options = [{'drone_model': 'custom_cf2x'} if self.uav_mapping[i] == 'lm'
                             else {'drone_model': 'custom_cf2x'}
                             for i in range(len(self.uav_mapping))]

        # rebuild the environment
        self.aviary = Aviary(
            start_pos=self.start_pos,
            start_orn=self.start_orn,
            drone_type="quadx",
            render= bool(self.render_mode),
            drone_options=drone_options,
            seed=seed,
        )

        if self.lw_stand_still:
            [
                self.aviary.changeDynamics(self.aviary.drones[k].Id, -1, mass=0.0, localInertiaDiagonal=0.0, )
                for k, v in self.drone_classes.items()
                if v == 'lw'
            ]

        if self.render_mode:
            self.debuglines = [self.aviary.addUserDebugLine([0, 0, 0], [0, 0, 1], lineColorRGB=[1, 0, 0], lineWidth=2)
                               for id in range(3)]
            self.change_visuals()
            self.init_debug_vectors()


    def end_reset(self, seed=None, options=dict()):
        """The tailing half of the reset function."""
        # register all new collision bodies
        self.aviary.register_all_new_bodies()

        # set flight mode
        if not self.direct_control:
            flight_modes = [0 if self.uav_mapping[i] == 'lm' else 7 for i in range(len(self.uav_mapping))]
        else:
            flight_modes = [6 if self.uav_mapping[i] == 'lm' else 7 for i in range(len(self.uav_mapping))]

        self.aviary.set_mode(flight_modes)

        self.update_control_lists()

        self._compute_agent_states()

        self.lw_manager = LWManager(env=self,
                                    formation_radius=self.spawn_settings[
                                        'lw_spawn_radius'] if self.spawn_settings is not None else 1.0,
                                    threat_radius=self.lw_threat_radius,
                                    weapon_cooldown=self.lw_weapon_cooldown,
                                    shoot_range=self.lw_shoot_range,
                                    )

        self.cameraDistance=5.12
        self.cameraYaw=-185.20
        self.cameraPitch=-41.20
        self.cameraTargetPosition=[-1.39, -0.66, 1.38]


        if self.render_mode and self.custom_spawn:
            self.aviary.resetDebugVisualizerCamera(cameraDistance=self.cameraDistance, cameraYaw=self.cameraYaw,
                                                   cameraPitch=self.cameraPitch,cameraTargetPosition=self.cameraTargetPosition)
            self.aviary.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)

            self.camera_parameters = self.aviary.getDebugVisualizerCamera(ccameraDistance=self.cameraDistance, cameraYaw=self.cameraYaw,
                                                   cameraPitch=self.cameraPitch,cameraTargetPosition=self.cameraTargetPosition)

        if self.lw_moves_random:
            for id, uav in self.armed_uavs.items():
                if self.get_drone_type_by_id(id) == 'lw' :
                    random_setpoint = np.zeros(4, np.float64)
                    random_setpoint[0] = np.random.uniform(-self.flight_dome_size/np.sqrt(3), self.flight_dome_size/np.sqrt(3))/2
                    random_setpoint[1] = np.random.uniform(-self.flight_dome_size/np.sqrt(3), self.flight_dome_size/np.sqrt(3))/2
                    random_setpoint[2] = 0
                    random_setpoint[3] = np.random.uniform(2 ,self.flight_dome_size/np.sqrt(3))/2
                    self.aviary.set_setpoint(id, random_setpoint )

        #self.update_targets()

        # wait for env to stabilize
        for _ in range(10):
            self.aviary.step()

    def compute_auxiliary_by_id(self, agent_id: int):
        """This returns the auxiliary state form the drone."""
        return self.aviary.aux_state(agent_id)

    def update_targets(self):


        for agent in self.agents:
            agent_id = self.agent_name_mapping[agent]

            if self.current_target_ids[agent_id] == -1:
                self.current_target_ids[agent_id] = np.argmin(np.where(self.current_distance[agent_id] != 0,
                                                                       self.current_distance[agent_id], -1))


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
            self.info_counters["timeover"] += 1

        # collision with ground
        if np.any(self.aviary.contact_array[self.aviary.drones[agent_id].Id][0]):
            reward -= 100.0
            info["crashes"] = True
            term |= True
            self.info_counters["crashes"] += 1

        # exceed flight dome
        if np.linalg.norm(self.aviary.state(agent_id)[-1]) > self.flight_dome_size:
            reward -= 100.0
            info["out_of_bounds"] = True
            term |= True
            self.info_counters["out_of_bounds"] += 1

        # collide with other kamikaze
        colissions = self.get_friendlyfire_type_collision(agent_id)
        if 'lm' in colissions:
            reward -= 100.0
            info["ally_collision"] = True
            term |= True
            self.info_counters["ally_collision"] += 1

        # Shoot down by a loyal wingman
        if self.lw_manager.downed_lm[agent_id]:
            reward -= 100
            term |= True
            info['downed'] = True
            self.info_counters["downed"] += 1

        # destroy any loyal wingman
        explosion_mapping = self.get_explosion_mapping(agent_id)
        if 'lw' in explosion_mapping.values():
            num_lw_exploded = sum(np.array(list(explosion_mapping.values())) == 'lw')
            reward += self.rew_exploding_target
            info["exploded_target"] = num_lw_exploded
            info["is_success"] = True
            term |= True
            self.info_counters["exploded_target"] += 1
            self.info_counters["is_success"] += 1

        elif self.get_collateral_explosions(agent_id):
            reward -= 100
            if not info.get("exploded_target", False):  # avoid misscounting when exploded the same target
                info["exploded_by_ally"] = True
                self.info_counters["exploded_by_ally"] += 1
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




        for id, uav in self.armed_uavs.items():
            if self.get_drone_type_by_id(id) == 'lm':
                if not self.direct_control:
                    self.current_actions[id] = actions[uav]  # rescale actions
                    self.aviary.set_setpoint(id, self.current_actions[id])  # denormalize actions
                else:
                    target_id = self.find_nearest_lw(id) # retão
                    self.current_actions[id] = self.direct_control_action(id) # np.insert(self.drone_positions[target_id], 2, 0)
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
            # if self.aviary.isConnected():  # avoid pybullet.error: Not connected to physics server.
            self.collision_matrix = self.create_collision_matrix(distance_threshold=self.explosion_radius)
            self.lw_manager.update(stand_still=self.lw_stand_still)



            # update reward, term, trunc, for each agent
            for ag in self.agents:
                ag_id = self.agent_name_mapping[ag]
                # self.draw_vel_vector(ag_id, line_id=self.debuglines[ag_id])
                # self.draw_vel_vector(ag_id, line_id=self.debuglines[ag_id])

                # compute term trunc reward
                term, trunc, rew, info = self.compute_term_trunc_reward_info_by_id(ag_id)

                terminations[ag] |= term
                truncations[ag] |= trunc
                rewards[ag] += rew
                infos[ag] = {**infos[ag], **info}

                # compute observations
                observations[ag] = self.compute_observation_by_id(ag_id)

                if self.targets == [] and (term == True):
                    infos[ag] = {**infos[ag], 'is_success': True}

                self.current_term[ag] = term
                self.current_trun[ag] = trunc
                self.current_acc_rew[ag] += rew
                self.current_inf[ag] = {**infos[ag], **info}
                self.current_obs[ag] = observations[ag]
                if self.save_step_data:
                    self.append_step_data(ag)
                    self.append_obs_data(self.observation_dict)

            for agent in self.agents:
                self.compute_collisions(agent)
                if terminations[agent]:
                    self.disarm_drone(self.agent_name_mapping[agent])
                    #self.summarize_infos()


            # increment step count and cull dead agents for the next round

            self.agents = [
                agent
                for agent in self.agents
                if not (terminations[agent] or truncations[agent])
            ]
            self.update_control_lists()

        self.step_count += 1

        if self.step_count % 5 == 0 and self.custom_spawn:

            if self.overlay is None:
                self.overlay = self.get_rgb_image()[..., :3]
            else:
                self.overlay = np.min(np.stack([self.overlay, self.get_rgb_image()[..., :3]], axis=0), axis=0)

        # all targets destroyed, End.
        if self.targets == []:
            terminations = {k: True if terminations[k] == False else v for k, v in terminations.items()}

            infos = {key: {'survived': True, **infos[key]} if infos[key] == {} else infos[key] for key in infos.keys()}
            self.current_inf = {key: {'survived': True, **infos[key]} if infos[key] == {} else infos[key] for key in
                                infos.keys()}

            infos = {key: {'is_success': True, **infos[key]} if infos[key].keys() == {'survived'} else infos[key] for
                     key in infos.keys()}
            self.current_inf = {
                key: {'is_success': True, **infos[key]} if infos[key].keys() == {'survived'} else infos[key] for key in
                infos.keys()}

            for agent in self.agents:
                self.info_counters['survived'] +=1
                self.info_counters['is_success'] += 1

            #self.summarize_infos()
            self.info_counters['mission_complete'] = 1
            self.create_dict_ep_data()
            if self.custom_spawn:
                self.overlay = np.min(np.stack([self.overlay, self.get_rgb_image()[..., :3]], axis=0), axis=0)
                im = Image.fromarray(self.overlay)
                im.save("images/quadx_trajectory.png")

            return observations, rewards, terminations, truncations, infos

        elif self.agents == []:
            #self.summarize_infos()
            self.create_dict_ep_data()
            if self.custom_spawn:
                self.overlay = np.min(np.stack([self.overlay, self.get_rgb_image()[..., :3]], axis=0), axis=0)
                im = Image.fromarray(self.overlay)
                im.save("images/quadx_trajectory.png")

            return observations, rewards, terminations, truncations, infos

        elif all(truncations.values()):
            #self.summarize_infos()
            self.create_dict_ep_data()
            if self.custom_spawn:
                self.overlay = np.min(np.stack([self.overlay, self.get_rgb_image()[..., :3]], axis=0), axis=0)
                im = Image.fromarray(self.overlay)
                im.save("images/quadx_trajectory.png")

        return observations, rewards, terminations, truncations, infos

    # ------------------------------ Env End ----------------------------------------------------

    def summarize_infos(self):

        for agent_key, agent_data in self.current_inf.items():
            for key, value in agent_data.items():
                if key in self.info_counters:
                    self.info_counters[key] += value

    def normalize_action(self, action: np.ndarray):
        """
        Convert the given command to a setpoint that can be used by the quadcopter's propulsion system.
        Parameters:
        - command: The command to be converted. It is composed by:
            - direction (3d) and magnitude (1d) of the desired velocity in the x, y, and z axes.
        Returns:
        - The converted setpoint.
        """

        return action / self.thrust_limit

    def compute_collisions(self, agent):

        ag_id = self.agent_name_mapping[agent]

        explosion_mapping = self.get_explosion_mapping(ag_id)

        for id, type in explosion_mapping.items():
            target = self.drone_id_mapping[id]
            if type == 'lw':
                self.disarm_drone(id)
                if target in self.targets:  # avoid double removing in the same iteration
                    self.targets.remove(target)

        armed_lw = {k: v for k, v in self.armed_uavs.items() if self.drone_classes[k] == 'lw'}
        for target_id, target in armed_lw.items():
            if (
                    np.any(self.aviary.contact_array[self.aviary.drones[target_id].Id][0])  # lw collide with the ground
                    or np.any(self.aviary.contact_array[self.aviary.drones[target_id].Id][1:])
            # lw collide with any uav
            ):
                self.disarm_drone(target_id)
                if target in self.targets:  # avoid double removing in the same iteration
                    self.targets.remove(target)

        # get_in_explosion = [k for k,v in self.armed_uavs.items()
        #                     if self.current_distance[ag_id][k] < (self.explosion_radius + 0.5)
        #                     and k != ag_id ]

        # destroy lw in the range of explosion
        # if any(explosion_mapping.keys()):
        #     for id in get_in_explosion:
        #         if self.drone_classes[id] == 'lw':
        #             self.disarm_drone(id)
        #             target = self.drone_id_mapping[id]
        #             if target in self.targets:  # avoid double removing in the same iteration
        #                 self.targets.remove(target)

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

        self.armed_uavs = {key: value for key, value in self.drone_id_mapping.items() if
                           value in self.agents or value in self.targets}

        self.armed_uav_types = {key: self.get_drone_type_by_id(key) for key, value in self.armed_uavs.items()}

        self.uav_mapping = np.array(['lm'] * len(self.agents) + ['lw'] * len(self.targets))

        self.num_drones = len(self.agents) + len(self.targets)

    def draw_vel_vector(self, agent_id, line_id=None, length=0.3, lineColorRGB=[1, 1, 0]):
        # Calculate the forward vector based on the drone's orientation

        debug_line = self.aviary.addUserDebugLine(self.aviary.state(agent_id)[-1],
                                                  self.aviary.state(agent_id)[-1] + self.ground_velocities[agent_id],
                                                  lineWidth=3, replaceItemUniqueId=line_id,
                                                  lineColorRGB=[1, 1, 1])
        return debug_line

    def find_nearest_lw(self, agent_id: int) -> int:

        distances = self.current_distance[agent_id, :]

        self.update_control_lists()

        lw_indices = np.array([key for key, value in self.armed_uav_types.items() if value == 'lw'])

        if not lw_indices.any():  # TODO seems wrong
            return self.find_nearest_lm(agent_id)

        # Filter distances based on 'lw' indices
        lw_distances = distances[lw_indices]

        # Find the index of the minimum distance in lw_distances
        nearest_lw_index = lw_indices[np.argmin(lw_distances)]

        return nearest_lw_index

    def find_nearest_lm(self, agent_id: int, exclude_self=False):

        self.update_control_lists()

        distances = self.current_distance[agent_id, :]

        if exclude_self:
            lm_indices = np.array(
                [key for key, value in self.armed_uav_types.items() if value == 'lm' and key != agent_id])
        else:
            lm_indices = np.array([key for key, value in self.armed_uav_types.items() if value == 'lm'])

        if not (len(lm_indices) > 0):
            if self.drone_classes == 'lw':
                return None  # for LWManager works
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
        collisions = [i for i in collisions if i in self.armed_uavs.keys()]
        return collisions

    def get_explosion_mapping(self, agent_id):
        collisions = np.where(self.collision_matrix[self.aviary.drones[agent_id].Id][1:])[0]
        explosion_mapping = {key: value for key, value in self.drone_classes.items() if
                             (key in collisions) and (key in self.armed_uavs.keys())}

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
                points = self.aviary.getClosestPoints(bodyA=i, bodyB=j, distance=distance_threshold)

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

        self.agent_forward_line = self.aviary.addUserDebugLine([0, 0, 0], [0, 0, 1], lineColorRGB=[1, 0, 0],
                                                               lineWidth=2)
        self.target_forward_line = self.aviary.addUserDebugLine([0, 0, 0], [0, 0, 1], lineColorRGB=[1, 0, 0],
                                                                lineWidth=2)
        self.agent_vel_line = self.aviary.addUserDebugLine([0, 0, 0], [0, 0, 1], lineColorRGB=[1, 1, 0], lineWidth=2)
        self.target_vel_line = self.aviary.addUserDebugLine([0, 0, 0], [0, 0, 1], lineColorRGB=[1, 1, 0], lineWidth=2)
        self.agent_traj_line = self.aviary.addUserDebugLine([0, 0, 0], [0, 0, 1], lineColorRGB=[0, 1, 0], lineWidth=2)


    def append_step_data(self, agent):
        agent_id = self.agent_name_mapping[agent]
        target_id = self.current_target_ids[agent_id]
        step_data = {
            "aviary_steps": self.aviary.aviary_steps,
            "physics_steps": self.aviary.physics_steps,
            "step_count": self.step_count,
            "agent_id": agent_id,
            "elapsed_time": self.aviary.elapsed_time,
            "rew_closing_distance": self.rew_closing_distance[agent_id],
            "rew_close_to_target": self.rew_close_to_target[agent_id],
            "rew_engaging_enemy": self.rew_engaging_enemy[agent_id],
            "rew_speed_magnitude": self.rew_speed_magnitude[agent_id],
            "rew_near_engagement": self.rew_near_engagement[agent_id],
            "acc_rewards": self.current_acc_rew[agent],
            "vel_angles": self.current_vel_angles[agent_id][target_id],
            "rel_vel_magnitudade": self.current_rel_vel_magnitude[agent_id][target_id],
            "approaching": int(self.approaching[agent_id][target_id]),
            "chasing": int(self.chasing[agent_id][target_id]),
            "in_range": int(self.in_range[agent_id][target_id]),
            "in_cone": int(self.in_cone[agent_id][target_id]),
            "current_term": int(self.current_term[agent]),
            "info[downed]": int(self.current_inf[agent].get('downed', False)),
            "info[exploded_target]": int(self.current_inf[agent].get('exploded_target', False)),
            "info[exploded_ally]": int(self.current_inf[agent].get('exploded_ally', False)),
            "info[crashes]": int(self.current_inf[agent].get('crashes', False)),
            "info[ally_collision]": int(self.current_inf[agent].get('ally_collision', False)),
            "info[is_success]": int(self.current_inf[agent].get('is_success', False)),
            "info[mission_complete]": int(self.current_inf[agent].get('mission_complete', False)),
            "info[out_of_bounds]": int(self.current_inf[agent].get('out_of_bounds', False)),
            "info[timeover]": int(self.current_inf[agent].get('timeover', False)),

        }
        self.rewards_data.append(step_data)

    def create_dict_ep_data(self):

        self.ep_data = {
            'episode': 0,
            "aviary_steps": self.aviary.aviary_steps,
            "physics_steps": self.aviary.physics_steps,
            'step_count': self.step_count,
            "elapsed_time": self.aviary.elapsed_time,
            "agents_acc_distance_rewards": self.acc_rew_closing_distance.sum(),
            "agents_acc_proximity_rewards": self.acc_rew_close_to_target.sum(),
            "agents_acc_speed_rewards": self.acc_rew_speed_magnitude.sum(),

            "agents_mean_distance_rewards": self.acc_rew_closing_distance.mean(),
            "agents_mean_proximity_rewards": self.acc_rew_close_to_target.mean(),
            "agents_mean_speed_rewards": self.acc_rew_speed_magnitude.mean(),
            "agents_mean_acc_rewards": sum(self.current_acc_rew.values())/len(self.current_acc_rew),
            "agents_total_acc_rewards": sum(self.current_acc_rew.values()),

            "max_achieved_speed": self.max_achieved_speed,
            "max_achieved_rel_speed": self.max_achieved_rel_speed,
            'num_lm': self.num_lm,
            'num_lw': self.num_lw,
            'out_of_bounds': self.info_counters['out_of_bounds'],
            'crashes': self.info_counters['crashes'],
            'timeover': self.info_counters['timeover'],
            'exploded_target': self.info_counters['exploded_target'],
            'mission_complete': self.info_counters['mission_complete'],
            'ally_collision': self.info_counters['ally_collision'],
            'exploded_by_ally': self.info_counters['exploded_by_ally'],
            'downed': self.info_counters['downed'],
            'is_success': self.info_counters['is_success'],
            'survived': self.info_counters['survived'],
            'flight_dome_size': self.flight_dome_size,
            'explosion_radius': self.explosion_radius,
            'max_duration_seconds': self.max_duration_seconds,
            'distance_factor': self.distance_factor,
            'speed_factor': self.speed_factor,
            'proximity_factor': self.proximity_factor,
            'rew_exploding_target': self.rew_exploding_target,
            'reward_type': self.reward_type,
            'observation_type': self.observation_type,
            'max_velocity_magnitude': self.max_velocity_magnitude,
            'lethal_angle': self.lethal_angle,
            'lethal_distance': self.lethal_distance,
            'lw_stand_still': self.lw_stand_still,
            'lw_attacks': self.lw_attacks,
            'lw_chases': self.lw_chases,
            'lw_moves_random': self.lw_moves_random,
            'lw_threat_radius': self.lw_threat_radius,
            'lw_shoot_range': self.lw_shoot_range,
#            'lm_center_bounds': self.spawn_settings['lm_center_bounds'],
#            'lm_spawn_radius': self.spawn_settings['lm_spawn_radius'],
#            'lw_center_bounds': self.spawn_settings['lw_center_bounds'],
#            'lw_spawn_radius': self.spawn_settings['lw_spawn_radius'],
#            'min_z': self.spawn_settings['min_z'],

        }



    def append_obs_data(self, agent):

        self.obs_data.append(self.observation_dict)

    def uav_alive(self, drone_id):
        if drone_id in self.armed_uavs.keys():
            return True
        else:
            return False

    def normalize_linear(self, array):
        linear_min = -self.flight_dome_size
        linear_max = self.flight_dome_size
        return (array - linear_min) / (linear_max - linear_min)

    def normalize_angular(self, array):
        linear_min = -np.pi
        linear_max = np.pi
        return (array - linear_min) / (linear_max - linear_min)

    def sizeof_fmt(self, num, suffix='B'):
        ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
        for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
            if abs(num) < 1024.0:
                return "%3.1f %s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f %s%s" % (num, 'Yi', suffix)


    def direct_control_action(self, agent_id):

        target_id = self.find_nearest_lw(agent_id)
        agent_state = self.compute_attitude_by_id(agent_id)
        target_state = self.compute_attitude_by_id(target_id)

        _, _, _, lin_pos, agent_quaternion = agent_state
        _, _, _, target, target_quaternion = target_state

        # rotation matrix
        rotation = np.array(self.aviary.getMatrixFromQuaternion(target_quaternion)).reshape(3, 3)

        # drone to target
        target_deltas = target - lin_pos

        # velocity v_x, v_y, v_z in the drone body frame axis
        velocity = ( target_deltas / np.linalg.norm(target_deltas) ) * 7.0 #  max(self.current_distance[agent_id][target_id], 5.0)

        # rotate the velocity to the target
        velocity = np.matmul( velocity, rotation.T)

        # return the drone setpoint in the format [v_x, v_y,
        return np.insert(velocity, 2, 0)
