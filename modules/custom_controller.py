import matplotlib.pyplot as plt
import numpy as np
from PyFlyt.core import Aviary
import matplotlib

matplotlib.use('TkAgg')

def try_velocity(target_velocity: float, log) -> bool:
    """Returns True if the target_velocity is achievable"""
    num_steps = 5000

    # spawn the UAV
    start_pos = np.array([[0.0, 0.0, 2.0]])
    start_orn = np.array([[0.0, 0.0, 0.0]])
    env = Aviary(start_pos=start_pos, start_orn=start_orn, drone_type="quadx")

    # set to [u, v, vr, z]
    env.set_mode(4)

    # set the UAV to fly at constant x m/s, maintain 2 m altitude
    env.set_setpoint(0, np.array([target_velocity, 0.0, 0.0, 2.0]))

    # run the simulation and log the results
    for i in range(num_steps):
        env.step()

        # record the linear position state
        log[i] = env.state(0)[-1]

        # if we go below the ground, we failed to maintain the target velocity
        if env.state(0)[3, -1] <= 0.0:
            plt.plot(np.arange(i), log[:i+1])
            plt.show()

            return False

    # if we reach this point, the drone successfully reached the target velocity without hitting the floor
    return True

if __name__ == "__main__":
    log = np.zeros((5000, 3), dtype=np.float32)

    for i in range(7, 20):
        if try_velocity(float(i), log):
            print(f"Successfully reached {i} m/s.")
        else:
            print(f"Failed to reach {i} m/s.")
            break

    plt.plot(np.arange(1000), log)
    plt.show()