from modules.envs_test.ma_fixedwing_dogfight_env import MAFixedwingDogfightEnv
import numpy as np

def decode_obs(obs):

    ang_vel = obs[:3]
    ang_pos = obs[3:6]
    lin_vel = obs[6:9]
    lin_pos = obs[9:12]
    health = obs[12]
    op_ang_vel = obs[13:16]
    op_ang_pos = obs[16:19]
    op_lin_vel = obs[19:22]
    op_lin_pos = obs[22:25]
    op_health = obs[25]
    past_actions = obs[26:29]

    return {
        "ang_vel": ang_vel,
        "ang_pos": ang_pos,
        "lin_vel": lin_vel,
        "lin_pos": lin_pos,
        "health": health,
        "op_ang_vel": op_ang_vel,
        "op_ang_pos": op_ang_pos,
        "op_lin_vel": op_lin_vel,
        "op_lin_pos": op_lin_pos,
        "op_health": op_health,
        "past_actions": past_actions,
    }

def draw_chasing_velocity(lin_pos_uav1, lin_vel_uav1, line_id1, lin_pos_uav0, lin_vel_uav0, line_id0, line_id2, line_id3 ):
    # Calculate relative velocity
    rel_vel = np.array([lin_vel_uav1[i] - lin_vel_uav0[i] for i in range(3)])
    rel_vel_magnitude = np.linalg.norm(rel_vel)

    # Draw position and velocity vectors for each UAV
    draw_vector(lin_pos_uav0, lin_vel_uav0, line_id=line_id0, lineColorRGB=[0, 1, 1])  # UAV 0
    draw_vector(lin_pos_uav1, lin_vel_uav1, line_id=line_id1, lineColorRGB=[0, 1, 1])  # UAV 1

    # Draw relative velocity vector
    draw_vector(lin_pos_uav0, rel_vel, line_id=line_id2, lineColorRGB=[1, 1, 1])  # Relative velocity

    # Draw chasing velocity vector for UAV_0
    if rel_vel_magnitude > 0:
        chasing_vel_dir = rel_vel / rel_vel_magnitude
        chasing_vel_magnitude = min(1.0, rel_vel_magnitude)  # Limit the length for better visualization
        chasing_vel = chasing_vel_magnitude * chasing_vel_dir
        draw_vector(lin_pos_uav0, chasing_vel, line_id=line_id3, lineColorRGB=[0, 1, 1])

def draw_vector( lin_pos, vec, line_id=None, length=1.0, lineColorRGB=[1, 1, 0]):
    # Calculate the forward vector based on the drone's orientation

    end_point = [lin_pos[i] + length * vec[i] for i in range(3)]
    env.aviary.addUserDebugLine(lin_pos, end_point, lineColorRGB=lineColorRGB, replaceItemUniqueId=line_id, lineWidth=2)




env = MAFixedwingDogfightEnv(render_mode="human", )
observations, infos = env.reset()

agent_vel_line = env.aviary.addUserDebugLine([0,0,0], [0,0,1.1], lineColorRGB=[1,1,0], lineWidth=2)
target_vel_line = env.aviary.addUserDebugLine([0,0,0], [0,0,1.1], lineColorRGB=[1,1,0], lineWidth=2)
rel_vel_lin = env.aviary.addUserDebugLine([0,0,0], [0,0,1.2], lineColorRGB=[0,1,0], lineWidth=2)
chasing_vel_line = env.aviary.addUserDebugLine([0,0,0], [0,0,1.2], lineColorRGB=[0,1,0], lineWidth=2)

while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    actions['uav_0'] = np.array([0, 0, -1, 0.5])
    actions['uav_1'] = np.array([0, 0,  1, 0.5])

    observations, rewards, terminations, truncations, infos = env.step(actions)

    lin_pos_0 = decode_obs(observations['uav_0'])['lin_pos']
    lin_pos_1 = decode_obs(observations['uav_1'])['lin_pos']
    lin_vel_0 = decode_obs(observations['uav_0'])['lin_vel']
    lin_vel_1 = decode_obs(observations['uav_1'])['lin_vel']

    draw_chasing_velocity(lin_pos_1, lin_vel_1, agent_vel_line,
        lin_pos_0, lin_vel_0, target_vel_line,

                          rel_vel_lin, chasing_vel_line)






env.close()


