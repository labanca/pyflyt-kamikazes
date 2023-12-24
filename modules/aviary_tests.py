import time
import random

import numpy as np
import math
from PyFlyt.core import Aviary


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
        print(f"Drone {self.drone_fsm.id} is entering idle state.")
        #self.idle_start_time = time.time()

    def execute(self):
        print(f"Drone {self.drone_fsm.id} is idling.")
        if self.drone_fsm.detect_threat():
            self.drone_fsm.change_state('ChaseThreatState')

    def exit(self):
        print(f"Drone {self.drone_fsm.id} is exiting idle state.")



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


class ThreatChaseState(State):
    def enter(self):
        print(f"Drone {self.drone_fsm.id} is entering threat chase state.")
        self.drone_fsm._move_half_distance_to_threat(self.current_threat)


    def execute(self):
        print(f"Drone {self.drone_fsm.id} is chasing a threat.")
        if self.drone_fsm.half_distance_to_threat():
            self.drone_fsm.change_state('shoot_threat')

    def exit(self):
        print(f"Drone {self.drone_fsm.id} is exiting threat chase state.")


class ShootThreatState(State):
    def enter(self):
        print(f"Drone {self.drone_fsm.id} is entering shoot threat state.")

    def execute(self):
        print(f"Drone {self.drone_fsm.id} is shooting at a threat.")
        if not self.drone_fsm.detect_threat():
            self.drone_fsm.change_state('go_to_formation')

    def exit(self):
        print(f"Drone {self.drone_fsm.id} is exiting shoot threat state.")


class GoToFormationState(State):
    def enter(self):
        print(f"Drone {self.drone_fsm.id} is entering go to formation state.")

    def execute(self):
        print(f"Drone {self.drone_fsm.id} is going to formation.")
        if self.drone_fsm.at_formation():
            self.drone_fsm.change_state('idle')

    def exit(self):
        print(f"Drone {self.drone_fsm.id} is exiting go to formation state.")


class LWManager:

    def __init__(self, drone_id_types, aviary, formation_radius, threat_radius, leader_pos):
        self.drone_id_types = drone_id_types
        self.env = aviary
        self.threat_radius = threat_radius
        self.formation_radius = formation_radius
        self.max_velocity = np.linalg.norm(np.linalg.norm([6, 6, 6]))
        self.squad = [LWSFM(k, threat_radius, self) for k, v in drone_id_types.items() if v == 'lw']
        self.num_lw = len(self.squad)
        self.leader_id = len(np.array(self.env.start_pos[0:self.num_lw])) // 2
        self.leader_pos = leader_pos

    def compute_state(self):

        self.states = self.env.all_states
        self.leader_pos = self.states[self.leader_id][3]


    def control(self):
        self.compute_state()

        for lw in self.squad:
            lw.update()

    def get_all_lw_positions(self, center, num_drones, radius):
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
        angle_step = 2 * math.pi / num_drones

        for i in range(num_drones):
            angle = i * angle_step
            distance = radius * math.sqrt(1 - (i / (num_drones - 1)) ** 2)  # Adjust distance for even spacing
            x_offset = distance * math.cos(angle)
            y_offset = distance * math.sin(angle)
            position = (center[0] + x_offset, center[1] + y_offset, center[2])
            positions.append(position)

        positions = np.array(positions)
        c = len(np.array(positions)) // 2
        positions[[0, c], :] = positions[[c, 0], :]

        self.leader_id = c
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
        positions = self.get_all_lw_positions(center, self.num_lw, self.formation_radius)
        setpoints = []

        for position in positions:
            setpoint = np.insert(position, 2, 0)  # Insert 0 as angular position (r)
            setpoints.append(setpoint)

        return np.array(setpoints)

    def move_squad(self, target_pos):

        squad_setpoints = self.get_squad_setpoints(target_pos)

        for lw in self.squad:
            self.env.set_setpoint(lw.id, squad_setpoints[lw.id])


class LWSFM:

    def __init__(self, lw_id, thread_radius, manager ):
        self.last_shot_time = 0
        self.current_threat = None
        self.current_state = 'Formation'
        self.id = lw_id
        self.thread_radius = thread_radius
        self.manager = manager
        self.states = {
            'idle': IdleState(self),
            'patrol': PatrolState(self),
            'threat_chase': ThreatChaseState(self),
            'shoot_threat': ShootThreatState(self),
            'go_to_formation': GoToFormationState(self),
        }
        self.current_state = self.states['idle']



    def change_state(self, new_state):
        self.current_state.exit()
        self.current_state = self.states[new_state]
        self.current_state.enter()

    def update(self):
        self.current_state.execute()

    def detect_threat(self):
        # Replace with actual logic to detect a threat
        return True

    def half_distance_to_threat(self):
        # Replace with actual logic to check if drone is half the distance to the threat
        return True

    def at_formation(self):
        # Replace with actual logic to check if drone is at the formation position
        return True

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
      original_positions = self.manager.get_all_lw_positions(self.manager.leader_pos,
                                                            self.manager.num_lw,
                                                            self.manager.formation_radius)
      original_position = original_positions[self.id]

      # Create a setpoint to move the drone back to its original position
      # Preserve the original angular position (r)
      setpoint = np.insert(original_position, 2, 0)

      # Send the setpoint to the drone to move it back to formation
      self.manager.env.set_setpoint(self.id, setpoint)

    def move_half_distance_threat(self, threat_position):
        """
        Moves the LW to the halfway point between its current position and the threat's position.

        Args:
        env: The environment object with the `set_setpoint` function.
        lw_id: The ID of the LW to move.
        threat_position: A numpy array representing the threat's position (x, y, z).
        lw_position: A numpy array representing the LW's current position (x, y, z).
        """

        lw_position = self.manager.env.state(self.id)[3]

        # Calculate the direction vector from the LW to the threat
        direction_vector = threat_position - lw_position

        # Calculate the halfway point
        halfway_point = lw_position + 0.5 * direction_vector

        # Create a setpoint with 0 for angular position (r) in the third position
        setpoint = np.insert(halfway_point, 2, 0)  # Insert r using np.insert

        # Send the setpoint to the LW
        self.manager.env.set_setpoint(self.id, setpoint)

    def shoot_target(self, env, lw_id, lm_id, radius):
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

        current_time = env.elapsed_time

        lw_pos = env.state(lw_id)[3]

        # Check if weapon is ready (cooldown elapsed)
        if abs(current_time - self.last_shot_time) >= 2:
            # Get states of drones within radius
            target_drone_state = env.state(lm_id)
            target_drone_position = target_drone_state[3, :]
            target_drone_velocity = target_drone_state[2, :]

            enemy_drone_near = np.linalg.norm(target_drone_position- lw_pos) < radius

            if enemy_drone_near:

                # Calculate hit probability based on velocity
                velocity_magnitude = np.linalg.norm(target_drone_velocity)
                max_hit_probability = 0.9
                hit_probability = max_hit_probability - velocity_magnitude / np.linalg.norm(self.manager.max_velocity)

                # Determine if the shot hits
                hit = np.random.random() < hit_probability

                if hit:
                    # Disarm the target drone if hit
                    self.disarm_drone(env, lm_id)
                    self.last_shot_time = current_time  # Update last shot time
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

def get_all_lw_positions(center, num_drones, radius):
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
    angle_step = 2 * math.pi / num_drones

    for i in range(num_drones):
        angle = i * angle_step
        distance = radius * math.sqrt(1 - (i / (num_drones - 1)) ** 2)  # Adjust distance for even spacing
        x_offset = distance * math.cos(angle)
        y_offset = distance * math.sin(angle)
        position = (center[0] + x_offset, center[1] + y_offset, center[2])
        positions.append(position)

    positions = np.array(positions)
    c = len(np.array(positions)) // 2
    positions[[0, c], :] = positions[[c, 0], :]


    return positions

leader_pos = np.array([0, 0, 2])
num_drones = 7
num_lm = 7
formation_radius = 2

start_pos = get_all_lw_positions(center=leader_pos, num_drones=num_drones, radius=formation_radius)
threat_pos = np.array([6,0,5])
threat_pos2 = np.array([7,0,5])
threat_pos3 = np.array([5,0,5])
start_pos = np.vstack([start_pos, threat_pos, threat_pos2, threat_pos3])
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

threat_radius = 5
drone_id_types = {i: 'lw' if i < num_lm else 'lm' for i in range(num_drones)}


manager = LWManager(drone_id_types, env, formation_radius, threat_radius, leader_pos)

destination = [5,5,5]
destination_setpoints = manager.get_squad_setpoints(destination)

env.set_mode(7)

# Step 5: (Optional) define a setpoint for the first drone (at index 0) in the aviary
setpoint1 = np.array([3.0, 3.0, 0.0, 3.0])
setpoint2 = np.array([3.0, 3.0, 0.0, 3.0])



# Step 6: step the physics
for i in range(10000):
    env.step()
    manager.compute_state()
    if i == 100:

        #env.set_all_setpoints(destination_setpoints)
        #return_drone_to_formation(env, 0, [0,0,2] , num_drones, formation_radius)
        #return_drone_to_formation(env, 1, [0, 0, 2], num_drones, formation_radius)
        manager.squad[0].move_lw_to_halfway_threat(env, 0, threat_pos2  , env.state(0))
        manager.move_squad([7,7,7])


    if i == 300:

        print(manager.squad[0].shoot_target(env, 0, 6, 7))

    if i == 600:
        print(manager.squad[0].shoot_target(env, 0, 5, 7))

    if i == 900:
        manager.squad[0].return_drone_to_formation()

# Gracefully close
env.close()