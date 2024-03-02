import matplotlib.pyplot as plt
import numpy as np
from PyFlyt.core import Aviary
import matplotlib
matplotlib.use('TkAgg')

if __name__ == "__main__":
    num_steps = 250
    start_height = 2000.0

    # spawn the UAV sideways in the sky
    start_pos = np.array([[0.0, 0.0, start_height]])
    start_orn = np.array([[0.0, np.pi / 2.0, 0.0]])
    env = Aviary(start_pos=start_pos, start_orn=start_orn, drone_type="quadx")

    # manually set motor noise to 0
    env.drones[0].motors.noise_ratio = 0.02

    # set to raw motor control
    env.set_mode(0)

    # all motors full power
    env.set_setpoint(0, np.array([0.0, 0.0, 0.0, 1.0]))

    # log for velocity
    time = np.arange(num_steps).astype(np.float32) * env.drones[0].control_period
    altitude_gain = np.zeros_like(time)
    velocity_gain = np.zeros_like(time)

    # run the simulation and log the results
    for i in range(num_steps):
        env.step()
        altitude_gain[i] = env.state(0)[3, -1] - start_height
        velocity_gain[i] = max([env.state(0)[2, -1], np.max(velocity_gain)])

    plt.plot(time, altitude_gain)
    plt.plot(time, velocity_gain)
    plt.xlabel('altitude_gain')
    plt.ylabel('velocity_gain')
    plt.show()