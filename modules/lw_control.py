import math

import PyFlyt
import time
import numpy
import numpy as np


class LWFSM:
    states = ['IdleState', 'LeaveFormationState' 'GoToFormationState', 'ApproachFormationState',
              'ShootThreatState' ]

    def __init__(self, drone_id, formation_leader_id, threat_detection_range):
        self.current_state = self.states[0]  # Start in ChaseThreatState
        self.drone_id = drone_id
        self.formation_leader_id = formation_leader_id
        self.threat_detection_range = threat_detection_range
        # ... other initializations (ammo count, etc.)

        self.current_setpoint = np.zeros(4)
        self.type = 'lw'


    def update(self):

        if self.detect_Threat():
            self.transition_to('ChaseThreatState')
        if self.current_state == 'ChaseThreatState':
            self.chase_threat_behavior()
        elif self.current_state == 'GoToFormationState':
            self.return_to_formation_behavior()
        elif self.current_state == 'PatrolState':
            self.patrol_behavior()
        elif self.current_state == 'ShootThreatState':
            self.shoot_threat_behavior()
        elif self.current_state == 'FreezeThreatState':
            self.freeze_threat_behavior()
        # Add other states as needed

    def chase_threat_behavior(self, ):
        # Implement chase threat behavior using PyFlyt API
        print(f'Drone {self.drone_id} is chasing the threat.')

    def return_to_formation_behavior(self, ):
        # Implement return to formation behavior using PyFlyt API
        print(f'Drone {self.drone_id} is returning to formation.')

    def patrol_behavior(self, ):
        # Implement patrol behavior using PyFlyt API
        print(f'Drone {self.drone_id} is patrolling.')

    def shoot_threat_behavior(self, ):
        # Implement shoot threat behavior using PyFlyt API
        print(f'Drone {self.drone_id} is shooting at the threat.')

    def freeze_threat_behavior(self, ):
        # Implement freeze threat behavior using PyFlyt API
        print(f'Drone {self.drone_id} is freezing in response to the threat.')

    def transition_to(self, new_state):
        print(f'Drone {self.drone_id} transitioning from {self.current_state} to {new_state}.')
        self.current_state = new_state

    def detect_Threat(self):
        return False

    def get_type(self):
        return self.type



class LWManager:

    def __init__(self, start_pos, drone_id_classes, leader_id, thread_detection_range):
        self.start_pos = start_pos
        self.drone_id_classes = drone_id_classes
        self.formation = [LWFSM(drone_id=k,
                                formation_leader_id=leader_id,
                                threat_detection_range=thread_detection_range)
                          for k,v in self.drone_id_classes.items() if v == 'lw' ]

    def update(self):
        # Coordinate LWFSM actions, ensuring safety and effectiveness
        for lw in self.formation:
            lw.update()

            # Implement additional safety checks and coordination logic here
            # - Prevent collisions between LW drones
            # - Avoid targeting protected areas or non-threats
            # - Prioritize actions that minimize risk to humans or infrastructure

    def get_drone_positions(self, center, num_drones, radius):
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

        return np.array(positions)


    def get_squad_setpoints(self, setpoint, num_drones, radius):
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

        center = np.array(setpoint[:3])  # Extract linear coordinates (x, y, z)
        positions = self.get_drone_positions(center, num_drones, radius)
        setpoints = []

        for position in positions:
            setpoint = np.append(position, setpoint[3])  # Append original angular position (r)
            setpoints.append(setpoint)

        return setpoints







