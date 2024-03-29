from PyFlyt.pz_envs.quadx_envs.ma_quadx_hover_env import MAQuadXBaseEnv

from modules.utils import *


class State:
    def __init__(self, lwfsm):
        self.drone_fsm = lwfsm

    def enter(self):
        pass

    def execute(self):
        pass

    def exit(self):
        pass


class IdleState(State):
    def enter(self):
        # print(f"Drone {self.drone_fsm.id} is entering idle state.")
        pass

    def execute(self):
        if self.drone_fsm.idle:
            # print(f"Drone {self.drone_fsm.id} is idling.")
            self.drone_fsm.idle = True
        if self.drone_fsm.detect_threat():
            self.drone_fsm.change_state('ChaseThreatState')

    def exit(self):
        # print(f"Drone {self.drone_fsm.id} is exiting idle state.")
        self.drone_fsm.idle = False


class ThreatChaseState(State):

    def enter(self):
        # print(f"Drone {self.drone_fsm.id} is entering threat chase state.")
        self.drone_fsm.chase_threat()

    def execute(self):
        if self.drone_fsm.current_threat_id is None:
            # print(f"Drone {self.drone_fsm.id} is chasing a threat.")
            self.drone_fsm.change_state('GoToFormationState')

        elif not self.drone_fsm.manager.env.uav_alive(self.drone_fsm.current_threat_id):
            self.drone_fsm.change_state('GoToFormationState')

        elif self.drone_fsm.at_shoot_distance():
            self.drone_fsm.change_state('ShootThreatState')

        elif self.drone_fsm.distance_to_threat() < 6.0:
            self.drone_fsm.chase_threat()

        else:
            self.drone_fsm.change_state('GoToFormationState')

    def exit(self):
        # print(f"Drone {self.drone_fsm.id} is exiting threat chase state.")
        self.drone_fsm.chasing = False


class ShootThreatState(State):
    def enter(self):
        # print(f"Drone {self.drone_fsm.id} is entering shoot threat state.")
        pass

    def execute(self):
        if self.drone_fsm.current_threat_id is None:
            self.drone_fsm.change_state('GoToFormationState')

        elif self.drone_fsm.at_shoot_distance():
            if self.drone_fsm.shoot_target():
                self.drone_fsm.change_state('GoToFormationState')

        elif self.drone_fsm.distance_to_threat() > 6.0:
            self.drone_fsm.change_state('GoToFormationState')
        else:
            self.drone_fsm.change_state('ChaseThreatState')

    def exit(self):
        # print(f"Drone {self.drone_fsm.id} is exiting shoot threat state.")
        self.drone_fsm.shooting = False


class GoToFormationState(State):
    def enter(self):
        # print(f"Drone {self.drone_fsm.id} is entering go to formation state.")
        self.drone_fsm.return_drone_to_formation()

    def execute(self):
        if not self.drone_fsm.returning:
            # print(f"Drone {self.drone_fsm.id} is going to formation.")
            self.drone_fsm.returning = True

        if self.drone_fsm.detect_threat() and self.drone_fsm.gun_loaded:
            self.drone_fsm.change_state('ChaseThreatState')

        elif self.drone_fsm.at_formation():
            self.drone_fsm.change_state('IdleState')

    def exit(self):
        # print(f"Drone {self.drone_fsm.id} is exiting go to formation state.")
        self.drone_fsm.returning = False


class LWManager:

    def __init__(self, env: MAQuadXBaseEnv, formation_radius, threat_radius, shoot_range):
        self.env = env
        self.aviary = self.env.aviary
        self.shoot_range = shoot_range
        self.current_distance = self.env.current_distance
        self.downed_lm = {k: 0 for k, v in self.env.armed_uav_types.items() if v == 'lm'}

        # create the finite state machine for each lw drone inside LWManager
        self.squad = [LWFSM(lw_id=k,
                            threat_radius=threat_radius,
                            shoot_range=shoot_range,
                            manager=self) for k, v in self.env.armed_uav_types.items() if v == 'lw']

        #self.env.squad_id_mapping = {self.squad[i].id: i for i, v in list(enumerate(self.squad))}

    def update(self, stand_still=False):

        for lwfsw in self.squad:
            if not stand_still:
                lwfsw.update()

    def get_squad_setpoints(self):
        """
        Generates setpoints for a squad of drones to move to a location while maintaining formation.

        Returns:
        A list of numpy arrays, each representing the setpoint for an individual drone.
        """
        positions = generate_formation_pos(self.env.formation_center, self.env.num_lw)
        setpoints = []

        for position in positions:
            setpoint = np.insert(position, 2, 0)  # Insert 0 as angular position (r)
            setpoints.append(setpoint)

        return np.array(setpoints)

    def move_squad(self, target_pos):

        squad_setpoints = self.get_squad_setpoints(target_pos)

        for lw in self.squad:
            self.aviary.set_setpoint(lw.id, squad_setpoints[lw.id])


# Finite State machine to handle the lw drones
class LWFSM:

    def __init__(self,
                 lw_id: int,
                 manager: LWManager,
                 threat_radius: float,
                 shoot_range: float,
                 ):
        self.last_shot_time = 0
        self.current_threat_id = None
        # self.current_threat_pos = None
        self.id = lw_id
        self.thread_radius = threat_radius
        self.shoot_range = shoot_range
        self.manager = manager
        self.states = {
            'IdleState': IdleState(self),
            'ChaseThreatState': ThreatChaseState(self),
            'ShootThreatState': ShootThreatState(self),
            'GoToFormationState': GoToFormationState(self),
        }
        self.current_state = self.states['IdleState']
        self.gun_loaded = True
        self.recharge_time = 2.0
        self.chasing = False
        self.shooting = False
        self.reloading = False
        self.returning = False
        self.idle = False

    def change_state(self, new_state):
        self.current_state.exit()
        self.current_state = self.states[new_state]
        self.current_state.enter()

    def update(self):
        self.upkeep()
        self.current_state.execute()

    def upkeep(self):
        if abs(self.manager.aviary.elapsed_time - self.last_shot_time) >= self.recharge_time:
            self.gun_loaded = True

        if self.current_threat_id is not None:
            if self.manager.downed_lm[self.current_threat_id]:
                self.current_threat_id = None
                # self.current_threat_pos = None

        if self.current_threat_id not in self.manager.env.armed_uavs:
            self.current_threat_id = None

    def distance_to_threat(self):

        return np.linalg.norm(
            self.manager.env.drone_positions[self.current_threat_id, :] - self.manager.env.drone_positions[self.id, :])

    def detect_threat(self):
        # Replace with actual logic to detect a threat

        nearest_threat = self.find_nearest_lm()  # self.manager.env.find_nearest_lm(self.id)

        if nearest_threat is None:
            return False

        if self.manager.env.current_distance[self.id][nearest_threat] < self.thread_radius:

            self.current_threat_id = nearest_threat
            # self.current_threat_pos = self.manager.env.drone_positions[nearest_threat]
            return True
        else:
            self.current_threat_id = None
            # self.current_threat_pos = None
            return False

    def at_shoot_distance(self):
        # Return True if the LW is at shoot range from the threat
        if self.manager.env.current_distance[self.id][self.current_threat_id] <= self.shoot_range:
            return True
        else:
            return False

    def at_formation(self):
        if abs(self.manager.env.drone_positions[self.id].sum() - self.current_setpoint.sum()) < 0.1:
            return True
        else:
            return False

    def return_drone_to_formation(self):
        """
        Returns a drone to its original position within the formation.
        """

        # Get the original position of the drone based on its ID
        squad_positions = self.manager.get_squad_setpoints()
        setpoint = squad_positions[self.id - self.manager.env.num_lm]  # TODO this is very bad
        self.current_setpoint = setpoint

        # Send the setpoint to the drone to move it back to formation
        self.manager.aviary.set_setpoint(self.id, setpoint)

    def chase_threat(self, ):
        """
        Moves the LW to the halfway point between its current position and the threat's position.
        """

        lw_position = self.manager.env.drone_positions[self.id]
        current_threat_pos = self.manager.env.drone_positions[self.current_threat_id]

        # Calculate the direction vector from the LW to the threat
        direction_vector = current_threat_pos - lw_position

        # Calculate the halfway point
        halfway_point = lw_position + direction_vector

        # Create a setpoint with 0 for angular position (r) in the third position
        setpoint = np.insert(halfway_point, 2, 0)  # Insert r using np.insert

        # store the current setpoint of the lw
        self.current_setpoint = setpoint

        # Send the setpoint to the LW
        self.manager.aviary.set_setpoint(self.id, setpoint)

    def shoot_target(self, ):
        """
        Simulates shooting at a target drone within a specified radius,
        considering weapon cooldown.

        Returns:
          True if a target was hit and disarmed, False otherwise.
        """

        # Check if weapon is ready (cooldown elapsed)
        if self.gun_loaded:
            if self.manager.env.current_distance[self.id][self.current_threat_id] < self.shoot_range:

                # Calculate hit probability based on velocity
                # target_drone_velocity = self.manager.env.attitudes[self.current_threat_id][2, :]
                hit_chance = self.manager.env.hit_probability[self.id][self.current_threat_id]
                max_hit_probability = 0.9
                hit_probability = max(hit_chance, 0.01)

                # Determine if the shot hits
                shot_outcome = np.random.random()
                hit = shot_outcome < hit_probability
                self.gun_loaded = False
                self.last_shot_time = self.manager.aviary.elapsed_time

                # print(f'LW {self.id} {"hit" if hit else "misses"} lm {self.current_threat_id} {self.manager.aviary.elapsed_time=}')
                if hit:
                    # print(f'Drone {self.id} hit threat {self.current_threat_id}!')
                    # Disarm the target drone if hit
                    self.manager.downed_lm[self.current_threat_id] += 1

                    # if self.current_threat_id in self.manager.env.armed_uav_types.keys():
                    #     self.manager.env.armed_uav_types.pop(self.current_threat_id)

                    self.current_threat_id = None
                    # self.current_threat_pos = None

                    return True
                else:
                    # print(f'Drone {self.id} miss {self.current_threat_id}!')
                    return False  # misses shot
            else:
                return False  # Weapon o"n cooldown
        else:
            # print(f'Drone {self.id} gun is not loaded!')
            return False

    def disarm_drone(self, env, agent_id):

        armed_drones_ids = {drone.Id for drone in env.armed_drones}
        armed_status_list = [drone.Id in armed_drones_ids for drone in env.drones]
        armed_status_list[agent_id] = False

        env.set_armed(armed_status_list)

    def find_nearest_lm(self, ):
        # Assuming compute_observation_by_id has been called to update
        distances = self.manager.env.current_distance[self.id, :]

        lm_indices = np.array([key for key, value in self.manager.env.armed_uav_types.items() if value == 'lm'])

        if lm_indices.size == 0:
            return None

        # Filter distances based on 'lw' indices
        lm_distances = distances[lm_indices]

        # Find the index of the minimum distance in lw_distances
        nearest_lm_index = lm_indices[np.argmin(lm_distances)]

        return nearest_lm_index
