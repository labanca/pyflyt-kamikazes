import numpy as np
from PyFlyt.pz_envs.quadx_envs.ma_quadx_hover_env import MAQuadXBaseEnv


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
        #print(f"Drone {self.drone_fsm.id} is entering idle state.")
        pass

    def execute(self):
        if self.drone_fsm.idle:
            #print(f"Drone {self.drone_fsm.id} is idling.")
            self.drone_fsm.idle = True
        if self.drone_fsm.detect_threat():
            self.drone_fsm.change_state('ChaseThreatState')


    def exit(self):
        #print(f"Drone {self.drone_fsm.id} is exiting idle state.")
        self.drone_fsm.idle = False

class ThreatChaseState(State):

    def enter(self):
        #print(f"Drone {self.drone_fsm.id} is entering threat chase state.")
        self.drone_fsm.chase_threat()

    def execute(self):
            if not self.drone_fsm.chasing:
                #print(f"Drone {self.drone_fsm.id} is chasing a threat.")
                self.drone_fsm.chasing = True
            if self.drone_fsm.shoot_distance_to_threat():
                self.drone_fsm.change_state('ShootThreatState')
            elif self.drone_fsm.drone_distance_pos(self.drone_fsm.current_threat_id,
                                self.drone_fsm.current_threat_pos) > 0.3:
                self.drone_fsm.chase_threat()

    def exit(self):
        #print(f"Drone {self.drone_fsm.id} is exiting threat chase state.")
        self.drone_fsm.chasing = False

class ShootThreatState(State):
    def enter(self):
        #print(f"Drone {self.drone_fsm.id} is entering shoot threat state.")
        pass

    def execute(self):
        if self.drone_fsm.gun_loaded:
            if self.drone_fsm.shoot_target():
                #print(f'Drone {self.drone_fsm.id} hit threat {self.drone_fsm.current_threat_id}!')
                self.drone_fsm.change_state('GoToFormationState')

        elif not self.drone_fsm.reloading:
            #print(f"Drone {self.drone_fsm.id} gun is not loaded.")
            self.drone_fsm.reloading = True


    def exit(self):
        #print(f"Drone {self.drone_fsm.id} is exiting shoot threat state.")
        self.drone_fsm.shooting = False

class GoToFormationState(State):
    def enter(self):
        #print(f"Drone {self.drone_fsm.id} is entering go to formation state.")
        self.drone_fsm.return_drone_to_formation()


    def execute(self):
        if not self.drone_fsm.returning:
            #print(f"Drone {self.drone_fsm.id} is going to formation.")
            self.drone_fsm.returning = True

        if self.drone_fsm.detect_threat() and self.drone_fsm.gun_loaded:
            self.drone_fsm.change_state('ChaseThreatState')

        elif self.drone_fsm.at_formation():
            self.drone_fsm.change_state('IdleState')

    def exit(self):
        #print(f"Drone {self.drone_fsm.id} is exiting go to formation state.")
        self.drone_fsm.returning = False

class LWManager:

    def __init__(self, env: MAQuadXBaseEnv , formation_radius, threat_radius, shoot_range):
        self.env = env
        self.armed_uav_types = self.env.armed_uav_types
        self.uav_id_types = self.env.drone_classes
        self.aviary = self.env.aviary
        self.threat_radius = threat_radius
        self.formation_radius = formation_radius
        self.max_velocity = np.linalg.norm(np.linalg.norm([6, 6, 6]))
        self.num_drones = self.env.num_drones
        self.num_lw = self.env.num_lw
        self.num_lm = self.env.num_lm
        self.formation_center = self.env.formation_center
        self.shoot_range = shoot_range

        self.downed_lm = {k: 0 for k, v in self.env.armed_uav_types.items() if v == 'lm'}
        self.attitudes = self.env.attitudes
        self.current_distance = self.env.current_distance
        self.drone_positions = self.env.drone_positions

        # create the finite state machine for each lw drone inside LWManager
        self.squad = [LWFSM(lw_id=k,
                            threat_radius=threat_radius,
                            shoot_range=shoot_range,
                            manager=self) for k, v in self.env.armed_uav_types.items() if v == 'lw']

    def update(self, stand_still = False):

        for lwfsw in self.squad:
            if not stand_still:
                lwfsw.update()



    @staticmethod
    def generate_formation_pos( formation_center, num_drones, radius=0.5, min_z = 1.0):
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

        Returns:
        A list of numpy arrays, each representing the setpoint for an individual drone.
        """
        positions = self.generate_formation_pos(self.env.formation_center, self.env.num_lw)
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
                 lw_id : int,
                 manager: LWManager,
                 threat_radius: float,
                 shoot_range: float,
                 ):
        self.last_shot_time = 0
        self.current_threat_id = None
        self.current_threat_pos = None
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

    def drone_distance_pos(self, drone_id, vector):

        return np.linalg.norm(self.manager.env.drone_positions[drone_id, :] - vector)

    def detect_threat(self):
        # Replace with actual logic to detect a threat

        nearest_threat = self.find_nearest_lm() #self.manager.env.find_nearest_lm(self.id)

        if nearest_threat is None:
            return False

        if self.manager.env.current_distance[self.id][nearest_threat] < self.thread_radius:

            self.current_threat_id = nearest_threat
            self.current_threat_pos = self.manager.env.drone_positions[nearest_threat]
            return True
        else:
            self.current_threat_id = None
            self.current_threat_pos = None
            return False

    def shoot_distance_to_threat(self):
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
        setpoint = squad_positions[self.id - self.manager.num_lm] # TODO this is very bad
        self.current_setpoint = setpoint

        # Send the setpoint to the drone to move it back to formation
        self.manager.aviary.set_setpoint(self.id, setpoint)

    def chase_threat(self,):
        """
        Moves the LW to the halfway point between its current position and the threat's position.
        """

        lw_position = self.manager.env.drone_positions[self.id]

        # Calculate the direction vector from the LW to the threat
        direction_vector = self.current_threat_pos - lw_position

        # Calculate the halfway point
        halfway_point = lw_position + direction_vector

        # Create a setpoint with 0 for angular position (r) in the third position
        setpoint = np.insert(halfway_point, 2, 0)  # Insert r using np.insert

        #store the current setpoint of the lw
        self.current_setpoint = setpoint

        # Send the setpoint to the LW
        self.manager.aviary.set_setpoint(self.id, setpoint)

    def shoot_target(self,):
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
                #target_drone_velocity = self.manager.env.attitudes[self.current_threat_id][2, :]
                velocity_magnitude = self.manager.env.current_magnitude[self.current_threat_id] #np.linalg.norm(target_drone_velocity)
                max_hit_probability = 0.9
                hit_probability = max_hit_probability - velocity_magnitude / self.manager.max_velocity

                # Determine if the shot hits
                hit = np.random.random() < hit_probability

                if hit:

                    # Disarm the target drone if hit
                    #self.manager.env.disarm_drone(self.current_threat_id)
                    self.manager.downed_lm[self.current_threat_id] +=1

                    self.last_shot_time = self.manager.aviary.elapsed_time  # Update last shot time
                    self.gun_loaded = False

                    if self.current_threat_id in self.manager.env.armed_uav_types.keys():
                        self.manager.env.armed_uav_types.pop(self.current_threat_id)

                    self.current_threat_id = None
                    self.current_threat_pos = None

                    return True
                else:
                    #print(f'Drone {self.id} miss {self.current_threat_id}!')
                    return False  # No targets in range
            else:
                return False  # Weapon o"n cooldown
        else:
            #print(f'Drone {self.id} gun is not loaded!')
            return False



    def disarm_drone(self, env, agent_id):

        armed_drones_ids = {drone.Id for drone in env.armed_drones}
        armed_status_list = [drone.Id in armed_drones_ids for drone in env.drones]
        armed_status_list[agent_id] = False

        env.set_armed(armed_status_list)

    def find_nearest_lm(self, ):
        # Assuming compute_observation_by_id has been called to update self.current_distance
        distances = self.manager.current_distance[self.id, :]

        lm_indices = np.array([key for key, value in self.manager.env.armed_uav_types.items() if value == 'lm'])

        if lm_indices.size == 0:
            return None

        # Filter distances based on 'lw' indices
        lm_distances = distances[lm_indices]

        # Find the index of the minimum distance in lw_distances
        nearest_lm_index = lm_indices[np.argmin(lm_distances)]

        return nearest_lm_index