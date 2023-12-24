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
        #self.idle_start_time = time.time()

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
        #self.drone_fsm.shoot_target()


    def execute(self):
        print(f"Drone {self.drone_fsm.id} is shooting at a threat.")
        self.drone_fsm.shoot_target()
        if self.drone_fsm.current_threat_id is None:
            self.drone_fsm.change_state('GoToFormationState')

    def exit(self):
        print(f"Drone {self.drone_fsm.id} is exiting shoot threat state.")


class GoToFormationState(State):
    def enter(self):
        print(f"Drone {self.drone_fsm.id} is entering go to formation state.")
        self.drone_fsm.return_drone_to_formation()


    def execute(self):
        print(f"Drone {self.drone_fsm.id} is going to formation.")
        if self.drone_fsm.at_formation():
            self.drone_fsm.change_state('IdleState')

    def exit(self):
        print(f"Drone {self.drone_fsm.id} is exiting go to formation state.")

class PatrolState(State):
    def enter(self):
        print(f"Drone {self.drone_fsm.id} is entering patrol state.")
        target_pos = [element + random.uniform(-5, 5) for element in self.drone_fsm.manager.leader_pos]
        self.drone_fsm.manager.move_squad(self, target_pos)

    def execute(self):
        print(f"Drone {self.drone_fsm.id} is patrolling.")
        if self.drone_fsm.detect_threat():
            self.drone_fsm.half_distance_to_threat('threat_chase')

    def exit(self):
        print(f"Drone {self.drone_fsm.id} is exiting patrol state.")



class LWManager:

    def __init__(self, start_pos, armed_uav_types, aviary, formation_radius, detect_threat_radius, shoot_range, leader_pos):
        self.armed_uav_types = armed_uav_types
        self.env = aviary
        self.threat_radius = detect_threat_radius
        self.formation_radius = formation_radius
        self.max_velocity = np.linalg.norm(np.linalg.norm([6, 6, 6]))
        self.num_drones = len(armed_uav_types)
        self.num_lw = len([k for k,v in self.armed_uav_types.items() if v == 'lw'] )
        self.num_lm = num_drones - self.num_lw
        self.leader_pos = leader_pos

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
        self.compute_state()

        for lw in self.squad:
            lw.update()


    def get_all_lm_postions(self):

        return [self.attitudes[id][3] for id, type in self.armed_uav_types.items() if type == 'lm']

    def get_all_lw_positions(self,):

        return [self.attitudes[id][3] for id, type in self.armed_uav_types.items() if type == 'lw']


    def get_formation_pos(self, central_pos):
        """
        This function generates the initial positions for a given number of drones
        around a central point, within a specified radius.

        Args:
            center: A tuple representing the central point (x, y, z).
            num_drones: The number of drones to position.
            radius: The maximum distance from the center where drones can spawn.

        Returns:
            A list of tuples representing the initial positions (x, y, z) for each drone.
        """

        positions = []
        angle_step = 2 * math.pi / self.num_lw

        for i in range(self.num_lw):
            angle = i * angle_step
            distance = self.formation_radius * math.sqrt(1 - (i / (self.num_lw - 1)) ** 2)  # Adjust distance for even spacing
            x_offset = distance * math.cos(angle)
            y_offset = distance * math.sin(angle)
            position = (central_pos[0] + x_offset, central_pos[1] + y_offset, central_pos[2])
            positions.append(position)

        # leader coordinates in the halfway of the pos array
        positions = np.array(positions)
        c = len(np.array(positions)) // 2
        positions[[0, c], :] = positions[[c, 0], :]
        self.leader_id = c
        self.leader_pos = positions[c]


        return np.array(positions)

    def get_squad_setpoints(self, target_pos):
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

        center = np.array(target_pos)  # Extract linear coordinates (x, y, z)
        positions = self.get_formation_pos(center)
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



    def change_state(self, new_state):
        self.current_state.exit()
        self.current_state = self.states[new_state]
        self.current_state.enter()

    def update(self):
        self.current_state.execute()

    def detect_threat(self):
        # Replace with actual logic to detect a threat

        nearest_threat = self.find_nearest_lm()

        if self.manager.current_distance[self.id][nearest_threat] < self.thread_radius:

            self.current_threat_id = nearest_threat
            self.current_threat_pos = self.manager.attitudes[nearest_threat,3]
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
        if self.manager.drone_positions[self.id] self.current_setpoint
        # Replace with actual logic to check if drone is at the formation position
        return True

    def find_nearest_lm(self,) -> int:
        # Assuming compute_observation_by_id has been called to update self.current_distance
        distances = self.manager.current_distance[self.id, :]

        lm_indices = np.array([key for key, value in self.manager.armed_uav_types.items() if value == 'lm'])

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
      original_positions = self.manager.get_squad_setpoints(self.manager.leader_pos)
      original_position = original_positions[self.id - self.manager.num_lw + 1] # TODO this is very bad

      # Create a setpoint to move the drone back to its original position
      # Preserve the original angular position (r)
      setpoint = np.insert(original_position, 2, 0)

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

        current_time = self.manager.env.elapsed_time

        lw_pos = self.manager.current_distance[self.id]

        # Check if weapon is ready (cooldown elapsed)
        if abs(current_time - self.last_shot_time) >= 2:
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
                    # Disarm the target drone if hit
                    self.disarm_drone(env, self.current_threat_id)
                    self.last_shot_time = current_time  # Update last shot time
                    self.current_threat_id = None
                    self.current_threat_pos = None
                    return True
                else:
                    return False  # No targets in range
            else:
                return False  # Weapon o"n cooldown

    def disarm_drone(self, env, agent_id):

        armed_drones_ids = {drone.Id for drone in env.armed_drones}
        armed_status_list = [drone.Id in armed_drones_ids for drone in env.drones]
        armed_status_list[agent_id] = False

        env.set_armed(armed_status_list)

# -------- TEST ---------------------------

def get_all_lw_positions(center, num_lw, radius):
    """
    This function generates the initial positions for a given number of drones
    around a central point, within a specified radius.

    Args:
        center: A tuple representing the central point (x, y, z).
        num_drones: The number of drones to position.
        radius: The maximum distance from the center where drones can spawn.

    Returns:
        A list of tuples representing the initial positions (x, y, z) for each drone.
    """

    positions = []
    angle_step = 2 * math.pi / num_lw

    for i in range(num_lw):
        angle = i * angle_step
        distance = radius * math.sqrt(1 - (i / (num_lw - 1)) ** 2)  # Adjust distance for even spacing
        x_offset = distance * math.cos(angle)
        y_offset = distance * math.sin(angle)
        position = (center[0] + x_offset, center[1] + y_offset, center[2])
        positions.append(position)

    positions = np.array(positions)
    c = len(np.array(positions)) // 2

    positions[[0, c], :] = positions[[c, 0], :]


    return positions

leader_pos = np.array([0, 0, 2])

num_lm = 2
num_lw = 3
num_drones = num_lm + num_lw
formation_radius = 2

start_pos = get_all_lw_positions(center=leader_pos, num_lw=num_lw, radius=formation_radius)
threat_pos = np.array([6,0,5])
threat_pos2 = np.array([-6,0,5])
start_pos = np.vstack([threat_pos, threat_pos2, start_pos])
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


manager = LWManager(start_pos=start_pos,
                    armed_uav_types=armed_uav_types,
                    aviary=env,
                    formation_radius=formation_radius,
                    detect_threat_radius=detect_threat_radius,
                    shoot_range=shoot_range,
                    leader_pos=leader_pos)

env.set_mode(7)

# Step 6: step the physics
for i in range(10000):
    env.step()
    manager.compute_state()
    manager.update()

# Gracefully close
env.close()