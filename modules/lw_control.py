import numpy as np


def reward_function(t0, t1, A, VA, D, VD):
    """Calculates the reward for a UAV trajectory based on minimum time to satisfy constraints.

    Args:
        t0: Initial time (float).
        t1: Final time (float).
        A: Initial position (numpy array).
        VA: Initial velocity (numpy array).
        D: Final position (numpy array).
        VD: Final velocity (numpy array).

    Returns:
        float: The reward value.
    """

    dt = t1 - t0  # Time difference

    # Calculate control points for Bezier curve
    B = A + VA * dt / 3
    C = D - VD * dt / 3

    def find_minimum_time_satisfying_constraints(A, VA, D, VD):
        v_max = np.array([10, 10, 5])  # Maximum allowable velocities
        a_max = np.array([2, 2, 1])  # Maximum allowable accelerations

        dt_min = 0.01  # Initial time step
        dt = dt_min

        while True:
            _, v, a = calculate_state(1)  # Calculate velocity and acceleration at the end of the trajectory
            if np.all(np.abs(v) <= v_max) and np.all(np.abs(a) <= a_max):
                return dt
            else:
                dt += dt_min

    # Define a function to calculate position, velocity, and acceleration at any time tau
    def calculate_state(tau):
        r = (1 - tau)**3 * A + 3 * tau * (1 - tau)**2 * B + 3 * tau**2 * (1 - tau) * C + tau**3 * D
        v = 3 * (1 - tau)**2 * (B - A) + 6 * tau * (1 - tau) * (C - B) + 3 * tau**2 * (D - C) / dt
        a = 6 * (tau - 1) * (B - A) + 6 * (1 - 2 * tau) * (C - B) + 6 * tau * (D - C) / dt**2
        return r, v, a

    # Find the minimum time Atmin that satisfies constraints on velocities and accelerations
    # (implementation for finding Atmin would depend on specific constraints)
    Atmin = find_minimum_time_satisfying_constraints(A, VA, D, VD)

    # Calculate the reward
    Amin = dt
    r = Amin - Atmin


    return r


t0 = 0
t1 = 10
A = np.array([1,1])
VA = 1
D = np.array([10,11])
VD = 5

print(reward_function(t0, t1, A, VA, D, VD))