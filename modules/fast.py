import numpy as np
from PyFlyt.core import Aviary



def generate_radial_formation(formation_center, num_drones, radius=0.25):
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


formation_center = np.array([0 ,0 , 3])

start_pos = generate_radial_formation(formation_center, 5, 0.5 )
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

env.set_mode(7)

for i in range(10000):
    env.step()

env.close()
