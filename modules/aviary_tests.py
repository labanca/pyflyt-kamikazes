import copy
import time
import random

import numpy as np
import math
from PyFlyt.core import Aviary


class State:
    def __init__(self, lwfsm ):
        self.drone_fsm = lwfsm

    def enter(self):
        pass

    def execute(self):
        pass

    def exit(self):
        pass


class IdleState(State):
    def enter(self):
        print(f"Drone {self.drone_fsm.id} is entering idle state.")


    def execute(self):
        print(f"Drone {self.drone_fsm.id} is idling.")
        if self.drone_fsm.detect_threat():
            self.drone_fsm.change_state('ChaseThreatState')


    def exit(self):
        print(f"Drone {self.drone_fsm.id} is exiting idle state.")

class ThreatChaseState(State):

    def enter(self):
        print(f"Drone {self.drone_fsm.id} is entering threat chase state.")
        self.drone_fsm.chase_treat()

    def execute(self):
            print(f"Drone {self.drone_fsm.id} is chasing a threat.")
            if self.drone_fsm.shoot_distance_to_threat():
                self.drone_fsm.change_state('ShootThreatState')

    def exit(self):
        print(f"Drone {self.drone_fsm.id} is exiting threat chase state.")


class ShootThreatState(State):
    def enter(self):
        print(f"Drone {self.drone_fsm.id} is entering shoot threat state.")

    def execute(self):
        if self.drone_fsm.gun_loaded:
            if self.drone_fsm.shoot_target():
                print(f"Drone {self.drone_fsm.id} neutralized the threat.")
                self.drone_fsm.change_state('GoToFormationState')
        else:
            print(f"Drone {self.drone_fsm.id} gun is not loaded.")


    def exit(self):
        print(f"Drone {self.drone_fsm.id} is exiting shoot threat state.")


class GoToFormationState(State):
    def enter(self):
        print(f"Drone {self.drone_fsm.id} is entering go to formation state.")
        self.drone_fsm.return_drone_to_formation()

    def execute(self):
        print(f"Drone {self.drone_fsm.id} is going to formation.")
        if self.drone_fsm.detect_threat():
            self.drone_fsm.change_state('ChaseThreatState')

        elif self.drone_fsm.at_formation():
            self.drone_fsm.change_state('IdleState')

    def exit(self):
        print(f"Drone {self.drone_fsm.id} is exiting go to formation state.")

class PatrolState(State):

    def enter(self):
        print(f"Drone {self.drone_fsm.id} is entering patrol state.")
        target_pos = [element + random.uniform(-5, 5) for element in self.drone_fsm.manager.formation_center]
        self.drone_fsm.manager.move_squad(self, target_pos)

    def execute(self):
        print(f"Drone {self.drone_fsm.id} is patrolling.")
        if self.drone_fsm.detect_threat():
            self.drone_fsm.half_distance_to_threat('threat_chase')

    def exit(self):
        print(f"Drone {self.drone_fsm.id} is exiting patrol state.")



class LWManager:

    def __init__(self, start_pos, armed_uav_types, uav_id_types, aviary, formation_radius, detect_threat_radius, shoot_range, formation_center):
        self.armed_uav_types = copy.deepcopy(armed_uav_types)
        self.uav_id_types = copy.deepcopy(uav_id_types)
        self.env = aviary
        self.threat_radius = detect_threat_radius
        self.formation_radius = formation_radius
        self.max_velocity = np.linalg.norm(np.linalg.norm([6, 6, 6]))
        self.num_drones = len(armed_uav_types)
        self.num_lw = len([k for k,v in self.armed_uav_types.items() if v == 'lw'] )
        self.num_lm = num_drones - self.num_lw
        self.formation_center = formation_center

        self.attitudes = np.stack(self.env.all_states, axis=0, dtype=np.float64)
        self.current_distance = np.linalg.norm(self.attitudes[:, -1][:, np.newaxis, :] - self.attitudes[:, -1], axis=-1)
        self.drone_positions = start_pos
        self.squad = [LWFSM(lw_id=k,
                            detect_thread_radius=detect_threat_radius,
                            shoot_range=shoot_range,
                            manager=self) for k, v in armed_uav_types.items() if v == 'lw']

    def compute_state(self):

        self.attitudes = np.stack(self.env.all_states, axis=0, dtype=np.float64)
        self.current_distance = np.linalg.norm(self.attitudes[:, -1][:, np.newaxis, :] - self.attitudes[:, -1], axis=-1)
        self.drone_positions = np.vstack([self.get_all_lm_postions(), self.get_all_lw_positions()])

    def update(self):

        for lw in self.squad:
            lw.update()


    def get_all_lm_postions(self):

        return [self.attitudes[id][3] for id, type in self.uav_id_types.items() if type == 'lm']

    def get_all_lw_positions(self,):

        return [self.attitudes[id][3] for id, type in self.uav_id_types.items() if type == 'lw']

    @staticmethod
    def generate_formation_pos( formation_center, num_drones, radius=0.5):
        # Ensure the formation center is a NumPy array
        formation_center = np.array(formation_center)

        # Generate angles evenly distributed around a circle
        angles = np.linspace(0, 2 * np.pi, num_drones, endpoint=False)

        # Calculate drone positions in a radial formation
        x_positions = formation_center[0] + radius * np.cos(angles)
        y_positions = formation_center[1] + radius * np.sin(angles)

        # Set z coordinates to zero (you can modify this based on your specific requirements)
        z_positions = formation_center[2] + np.zeros_like(x_positions)

        # Combine x, y, and z coordinates into a 3D array
        drone_positions = np.column_stack((x_positions, y_positions, z_positions))

        return np.array(drone_positions)


    def get_squad_setpoints(self):
        """
        Generates setpoints for a squad of drones to move to a location while maintaining formation.

        Args:
        setpoint: A numpy array representing the target location for the central drone,
                   in the format [x, y, r, z].
        num_drones: The number of drones in the squad.
        radius: The desired radius of the squad formation.

        Returns:
        A list of numpy arrays, each representing the setpoint for an individual drone.
        """
        positions = self.generate_formation_pos(self.formation_center, self.num_lw)
        setpoints = []

        for position in positions:
            setpoint = np.insert(position, 2, 0)  # Insert 0 as angular position (r)
            setpoints.append(setpoint)

        return np.array(setpoints)

    def move_squad(self, target_pos):

        squad_setpoints = self.get_squad_setpoints(target_pos)

        for lw in self.squad:
            self.env.set_setpoint(lw.id, squad_setpoints[lw.id])


class LWFSM:

    def __init__(self,
                 lw_id : int,
                 manager: LWManager,
                 detect_thread_radius: float,
                 shoot_range: float,
                 ):
        self.last_shot_time = 0
        self.current_threat_id = None
        self.current_threat_pos = None
        self.id = lw_id
        self.thread_radius = detect_thread_radius
        self.shoot_range = shoot_range
        self.manager = manager
        self.states = {
            'IdleState': IdleState(self),
            'PatrolSate': PatrolState(self),
            'ChaseThreatState': ThreatChaseState(self),
            'ShootThreatState': ShootThreatState(self),
            'GoToFormationState': GoToFormationState(self),
        }
        self.current_state = self.states['IdleState']
        self.change_state('IdleState')
        self.gun_loaded = True
        self.recharge_time = 2.0
        self.chasing = False
        self.shooting = False
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
        if abs(self.manager.env.elapsed_time - self.last_shot_time) >= self.recharge_time:
            self.gun_loaded = True

    def detect_threat(self):
        # Replace with actual logic to detect a threat

        nearest_threat = self.find_nearest_lm()

        if nearest_threat is None:
            return False

        if self.manager.current_distance[self.id][nearest_threat] < self.thread_radius:

            self.current_threat_id = nearest_threat
            self.current_threat_pos = self.manager.attitudes[nearest_threat, 3]
            return True
        else:
            self.current_threat_id = None
            self.current_threat_pos = None
            return False

    def shoot_distance_to_threat(self):
        # Return True if the LW is at shoot range from the threat
        if self.manager.current_distance[self.id][self.current_threat_id] <= self.shoot_range:
            return True
        else:
            return False

    def at_formation(self):
        if abs(self.manager.drone_positions[self.id].sum() - self.current_setpoint.sum()) < 0.1:
            return True
        else:
            return False

    def find_nearest_lm(self,):
        # Assuming compute_observation_by_id has been called to update self.current_distance
        distances = self.manager.current_distance[self.id, :]

        lm_indices = np.array([key for key, value in self.manager.armed_uav_types.items() if value == 'lm'])

        if lm_indices.size == 0:
            return None

        # Filter distances based on 'lw' indices
        lm_distances = distances[lm_indices]

        # Find the index of the minimum distance in lw_distances
        nearest_lm_index = lm_indices[np.argmin(lm_distances)]

        return nearest_lm_index

    def return_drone_to_formation(self):
        """
        Returns a drone to its original position within the formation.

        Args:
        env: The environment object with the `set_setpoint` function.
        drone_id: The ID of the drone that has left the formation.
        leader_center: The current center of the formation (leader's position).
        num_drones: The total number of drones in the squad.
        radius: The desired radius of the formation.
        """

        # Get the original position of the drone based on its ID
        squad_positions = self.manager.get_squad_setpoints()
        setpoint = squad_positions[self.id - self.manager.num_lm] # TODO this is very bad
        self.current_setpoint = setpoint

        # Send the setpoint to the drone to move it back to formation
        self.manager.env.set_setpoint(self.id, setpoint)

    def chase_treat(self,):
        """
        Moves the LW to the halfway point between its current position and the threat's position.

        Args:
        env: The environment object with the `set_setpoint` function.
        lw_id: The ID of the LW to move.
        threat_position: A numpy array representing the threat's position (x, y, z).
        lw_position: A numpy array representing the LW's current position (x, y, z).
        """

        lw_position = self.manager.drone_positions[self.id]

        # Calculate the direction vector from the LW to the threat
        direction_vector = self.current_threat_pos - lw_position

        # Calculate the halfway point
        halfway_point = lw_position + direction_vector

        # Create a setpoint with 0 for angular position (r) in the third position
        setpoint = np.insert(halfway_point, 2, 0)  # Insert r using np.insert

        #store the current setpoint of the lw
        self.current_setpoint = setpoint

        # Send the setpoint to the LW
        self.manager.env.set_setpoint(self.id, setpoint)

    def shoot_target(self,):
        """
        Simulates shooting at a target drone within a specified radius,
        considering weapon cooldown.

        Args:
          env: The environment object with functions to get drone states and disarm drones.
          drone_id: The ID of the drone attempting to shoot.
          radius: The radius within which to search for targets.

        Returns:
          True if a target was hit and disarmed, False otherwise.
        """


        # Check if weapon is ready (cooldown elapsed)
        if self.gun_loaded:
            # Get states of drones within radius
            target_drone_state = self.manager.attitudes[self.current_threat_id]
            target_drone_velocity = target_drone_state[2, :]

            if self.manager.current_distance[self.id][self.current_threat_id] < self.shoot_range:

                # Calculate hit probability based on velocity
                velocity_magnitude = np.linalg.norm(target_drone_velocity)
                max_hit_probability = 0.9
                hit_probability = max_hit_probability - velocity_magnitude / self.manager.max_velocity

                # Determine if the shot hits
                hit = np.random.random() < hit_probability

                if hit:
                    print(f'Drone {self.id} hit threat {self.current_threat_id}!')
                    # Disarm the target drone if hit
                    self.disarm_drone(env, self.current_threat_id)
                    self.last_shot_time = self.manager.env.elapsed_time  # Update last shot time
                    self.gun_loaded = False

                    if self.current_threat_id in self.manager.armed_uav_types.keys():
                        self.manager.armed_uav_types.pop(self.current_threat_id)

                    self.current_threat_id = None
                    self.current_threat_pos = None

                    return True
                else:
                    print(f'Drone {self.id} miss {self.current_threat_id}!')
                    return False  # No targets in range
            else:
                return False  # Weapon o"n cooldown
        else:
            print(f'Drone {self.id} gun is not loaded!')
            return False
    def disarm_drone(self, env, agent_id):

        armed_drones_ids = {drone.Id for drone in env.armed_drones}
        armed_status_list = [drone.Id in armed_drones_ids for drone in env.drones]
        armed_status_list[agent_id] = False

        env.set_armed(armed_status_list)

# -------- TEST ---------------------------


formation_center = np.array([0, 0, 2])
num_lm = 3
num_lw = 3
num_drones = num_lm + num_lw
formation_radius = 1

start_pos = LWManager.generate_formation_pos(formation_center, num_lw, formation_radius)
threat_pos = np.array([4, 0, 5])

threat_pos2 = np.array([-5, 0, 5])
threat_pos3 = np.array([4.5, 1, 5])


start_pos = np.vstack([threat_pos, threat_pos2, threat_pos3, start_pos])
start_orn = np.zeros_like(start_pos)
render_mode = 'human'
drone_options = dict()
seed=None

env = Aviary(
            start_pos=start_pos,
            start_orn=start_orn,
            drone_type="quadx",
            render=bool(render_mode),
            drone_options=drone_options,
            seed=seed,
        )

detect_threat_radius = 5
shoot_range = 2
armed_uav_types = {i: 'lm' if i < num_lm else 'lw' for i in range(num_drones)}
uav_id_types = {i: 'lm' if i < num_lm else 'lw' for i in range(num_drones)}


DarkBlue = [0, 0, 0.8, 1]
[env.changeVisualShape(drone.Id, -1, rgbaColor=DarkBlue)
 for i, drone in enumerate(env.drones)
 if uav_id_types[i] == 'lw']

manager = LWManager(start_pos=start_pos,
                    armed_uav_types=armed_uav_types,
                    uav_id_types=uav_id_types,
                    aviary=env,
                    formation_radius=formation_radius,
                    detect_threat_radius=detect_threat_radius,
                    shoot_range=shoot_range,
                    formation_center=formation_center)

env.set_mode(7)


# Step 6: step the physics
for i in range(10000):
    env.step()
    manager.compute_state()
    if i % 7 == 0:
        manager.update()

    if i == 300:
        env.set_setpoint(1, np.array([0, 0, 0, 2]))

# Gracefully close
env.close()