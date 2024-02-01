"""Spawn a single drone, then command it to go to two setpoints consecutively, and plots the xyz output."""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from PyFlyt.core import Aviary
matplotlib.use('TkAgg')



# the starting position and orientations
start_pos = np.array([[0.0, 0.0, 1.0]])
start_orn = np.array([[0.0, 0.0, 0.0]])

drone_options = {}
drone_options['drone_model'] = 'custom_cf2x'
# environment setup
env = Aviary(start_pos=start_pos, start_orn=start_orn, render=True, drone_type="quadx", drone_options=drone_options)

# set to position control
env.set_mode(7)
max_speed = 0



steps = 1000
# initialize the log
log = np.zeros((steps, 3), dtype=np.float32)

setpoint = np.array([15.0, 15.0, 0.0, 2.0])
env.set_setpoint(0, setpoint)

for i in range(steps):
    env.step()

    # record the linear position state
    log[i] = env.state(0)[-1]
    current_speed = np.linalg.norm(env.state(0)[2])
    if current_speed > max_speed:
        max_speed = current_speed
        max_lin_vel = env.state(0)[2]

# for the next 500 steps, go to x=0, y=0, z=2, rotate 45 degrees
# plot stuff out
print(f'{max_speed=}')
print(f'{max_lin_vel=}')
plt.plot(np.arange(steps), log[:, 0], label='x')  # log[:, 0] assumes that x is in the first column
plt.plot(np.arange(steps), log[:, 1], label='y')  # log[:, 1] assumes that y is in the second column
plt.plot(np.arange(steps), log[:, 2], label='z')
plt.legend()
plt.show()