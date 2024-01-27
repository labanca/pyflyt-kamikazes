def plane_floor_is_body_zero(env):
    if env.getBodyInfo(0)[-1].decode("UTF-8") == 'plane':
        return True
    else:
        return False


def test_aviary():
    """Spawn a single drone on x=0, y=0, z=1, with 0 rpy."""
    # Step 1: import things
    import numpy as np
    from PyFlyt.core import Aviary

    # Step 2: define starting positions and orientations
    start_pos = np.array([[0.0, 0.0, 3.0]])
    start_orn = np.array([[0.0, 0.0, 0.0]])

    # Step 3: instantiate aviary
    env = Aviary(start_pos=start_pos, start_orn=start_orn, render=True, drone_type="quadx")

    # Step 4: (Optional) define control mode to use for drone
    env.set_mode(6)

    # Step 5: (Optional) define a setpoint for the first drone (at index 0) in the aviary
    setpoint = np.array([20.0, 0.0, 0.0, 1.0])
    env.set_setpoint(0, setpoint)

    # Step 6: step the physics
    for i in range(1000):
        env.step()

        # if i == 500:
        #     Aviary(start_pos=np.array([[2, 2, 2]]), start_orn=start_orn, render=False, drone_type="quadx")
        #     env.register_all_new_bodies()

    # Gracefully close
    env.close()


def test_plot_vectors():
    import matplotlib.pyplot as plt

    # Drone positions and forward vectors
    drone_a_position = np.array([1, 0, 0])
    drone_b_position = np.array([-1, 0, 0])

    drone_a_forward = np.array([1, 0, 0])
    drone_b_forward = np.array([1, 0, 0])  # Reversed for opposite direction

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot drone positions as spheres
    ax.scatter(drone_a_position[0], drone_a_position[1], drone_a_position[2], color='blue', marker='o', label='Drone A')
    ax.scatter(drone_b_position[0], drone_b_position[1], drone_b_position[2], color='red', marker='o', label='Drone B')

    # Plot drone forward vectors as arrows
    arrow_len = 0.5  # Adjust arrow length as needed
    ax.quiver(drone_a_position[0], drone_a_position[1], drone_a_position[2], drone_a_forward[0], drone_a_forward[1],
              drone_a_forward[2], arrow_length=arrow_len, color='blue', label='Drone A Forward')
    ax.quiver(drone_b_position[0], drone_b_position[1], drone_b_position[2], drone_b_forward[0], drone_b_forward[1],
              drone_b_forward[2], arrow_length=arrow_len, color='red', label='Drone B Forward')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Drone Positions and Forward Vectors')

    # Add legend and show the plot
    ax.legend()
    plt.show()


test_aviary()
