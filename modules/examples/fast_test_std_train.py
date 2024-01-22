from pathlib import Path

from envs.ma_quadx_chaser_env import MAQuadXChaserEnv
from stable_baselines3 import PPO
import numpy as np
import time

from modules.utils import generate_start_pos_orn, read_yaml_file

seed=None

model_path = Path('apps/models/ma_quadx_chaser_20240121-132008/ma_quadx_chaser-4000000.zip')
model_name = model_path.stem
model_folder = model_path.parent
model = PPO.load(model_path)

params_path = f'modules/examples/train_params_test.yaml'
spawn_settings, env_kwargs, train_kwargs = read_yaml_file(params_path)

start_pos = np.array([ [-4, -4, 1], [0, 0, 1] ])
start_orn = np.zeros_like(start_pos)
env_kwargs['start_pos'] = start_pos
env_kwargs['num_lm'] = 1
env_kwargs['spawn_settings'] = None
env_kwargs['formation_center'] = np.array([0,0,1])

env = MAQuadXChaserEnv(render_mode='human', **env_kwargs)
observations, infos = env.reset(seed=seed)

last_term = {}
counters = {'out_of_bounds': 0, 'crashes': 0, 'timeover': 0, 'exploded_target': 0, 'exploded_by_ally': 0,
            'survived': 0, 'ally_collision': 0, 'downed': 0, 'is_success': 0}
first_time = True
num_games = 1
i = 1
last_start_pos = env_kwargs['start_pos']
while env.agents:


    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    #actions = {agent: model.predict(observations[agent], deterministic=True)[0] for agent in env.agents}

    #always chase
    actions['agent_0'] = np.array([4, 4, 0, 0.5]) # np.array([i, i, 0, 0.123*i])env.desired_vel
    #actions['agent_1'] = np.array([-1, 0, 0, 0.8])
    #actions['agent_2'] = np.array([5, 2, 0, 0.8])
    #actions['agent_3'] = np.array([0, 0, 0, 0])
    i +=1

    observations, rewards, terminations, truncations, infos = env.step(actions)


    if first_time == True:
        first_time = False


    if any(terminations.values()) or any(truncations.values()):

        for agent_key, agent_data in infos.items():
            for key, value in agent_data.items():
                if key in counters:
                    counters[key] += 1

    if all(terminations.values()) or all(truncations.values()):
        print(f'********* EPISODE END **********\n')
        print(f'{rewards=}\n')
        print(f'{terminations=} {truncations=}\n')
        print(f'{infos=}\n\n\n')
        time.sleep(0)
        env.write_step_data(Path('modules/examples/step_data.csv'))
        env.write_obs_data(Path('modules/examples/obs_data.csv'))
        #env.plot_rewards_data('reward_data.csv')
        #env.plot_agent_rewards('reward_data.csv', 0)
        #env.plot_agent_infos2('reward_data.csv', 0)
        observations, infos = env.reset(seed=seed)
        num_games -= 1
        print(f'Remaining games: {num_games}')


    if num_games == 0:
        print(counters)
        print(f'{i=}')
        break

env.close()


