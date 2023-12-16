from envs.ma_quadx_chaser_env import MAQuadXHoverEnv
from stable_baselines3 import PPO
import numpy as np
import time
import os

def get_start_pos_orn(num_drones, min_distance, spawn_radius, center, num_lm, seed=None):
    start_pos = np.empty((num_drones, 3))
    min_z = 1
    np_random = np.random.RandomState(seed=seed)

    for i in range(num_drones):
        while True:
            # Generate random coordinates within the spawn area centered at 'center'
            x = np_random.uniform(center[0] - spawn_radius, center[0] + spawn_radius)
            y = np_random.uniform(center[1] - spawn_radius, center[1] + spawn_radius)
            z = np_random.uniform(max(center[2], min_z), center[2] + spawn_radius)  # Ensure z-axis is within range

            # Check if the minimum distance condition is met with existing drones
            if i == 0 or np.min(np.linalg.norm(start_pos[:i] - np.array([x, y, z]), axis=1)) >= min_distance:
                start_pos[i] = [x, y, z]
                break

    start_orn = (np_random.rand(num_drones, 3) - 0.5) * 2.0 * np.array([1.0, 1.0, 2 * np.pi])
    start_orn[num_lm:] = np.zeros((len(start_pos) - num_lm, 3), dtype=np.float64)

    return start_pos, start_orn

model = PPO.load('models/ma_quadx_hover_20231215-145039.zip')
seed=None

print((os.cpu_count() or 1))
spawn_settings = dict(
    num_drones=4,
    min_distance=3.0,
    spawn_radius=5.0,
    center=(0, 0, 0),
    seed=seed,
)

env_kwargs = {}
env_kwargs['num_lm'] = 1
env_kwargs['start_pos'], env_kwargs['start_orn'] = get_start_pos_orn(**spawn_settings, num_lm=env_kwargs['num_lm']) # np.array([[0, 0, 1], [4,0,1]])
#env_kwargs['start_orn'] = np.zeros_like(env_kwargs['start_pos'])
env_kwargs['flight_dome_size'] = (6.75 * (spawn_settings['spawn_radius'] + 1) ** 2) ** 0.5  # dome size 50% bigger than the spawn radius
#env_kwargs['uav_mapping'] = np.array(['lm', 'lm', 'lm', 'lm', 'lm','lm', 'lw', 'lw', 'lw', 'lw'])
env_kwargs['seed'] = seed


env = MAQuadXHoverEnv(render_mode="human", spawn_settings=spawn_settings, **env_kwargs)
observations, infos = env.reset(seed=seed)

last_term = {}
first_time = True
last_start_pos = env_kwargs['start_pos']
while env.agents:
    # this is where you would insert your policy
    #actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    actions = {agent: model.predict(observations[agent], deterministic=True)[0] for agent in env.agents}

    #print(f'{actions=}')

    # if len(env.agents) >= 2 :
    #     actions['uav_0'][0] = 0.8
    #     actions['uav_0'][1] = 0
    #     actions['uav_0'][2] = 0
    #     actions['uav_0'][3] = 0.01


    observations, rewards, terminations, truncations, infos = env.step(actions)

    #print(f'{terminations=}')
    #print(f'{truncations=}\n')
    #print(f'{rewards=}{terminations=}{truncations=}\n')
    #print(infos)
    #print(observations)

    if first_time == True:
        first_time = False

    elif set(terminations.values()) != set(last_term.values() ):
        print(f"| An agent terminated |")
        print(f'{terminations=}')
        print(f'{truncations=}')
        print(f'{infos=}\n')

    last_term = terminations

    if all(terminations.values()) or all(truncations.values()):
        print(f'********* EPISODE END **********\n')
        print(f'{rewards=}\n')
        print(f'{terminations=} {truncations=}\n')
        print(f'{infos=}\n\n\n')
        time.sleep(3)

        last_start_pos = env.start_pos
        observations, infos = env.reset(seed=seed)
        new_start_pos = env.start_pos
        print(f'{last_start_pos=}\n {new_start_pos=}')

env.close()

