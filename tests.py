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
    start_pos = np.array([[0.0, 0.0, 1.0]])
    start_orn = np.array([[0.0, 0.0, 0.0]])

    # Step 3: instantiate aviary
    env = Aviary(start_pos=start_pos, start_orn=start_orn, render=True, drone_type="quadx")

    # Step 4: (Optional) define control mode to use for drone
    env.set_mode(7)

    # Step 5: (Optional) define a setpoint for the first drone (at index 0) in the aviary
    setpoint = np.array([1.0, 0.0, 0.0, 1.0])
    env.set_setpoint(0, setpoint)

    # Step 6: step the physics
    for i in range(1000):
        env.step()

        if i == 500:
            Aviary(start_pos=np.array([[2, 2, 2]]), start_orn=start_orn, render=False, drone_type="quadx")
            env.register_all_new_bodies()

    # Gracefully close
    env.close()

test_aviary()