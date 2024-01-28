from pathlib import Path

from stable_baselines3 import PPO

from envs.ma_quadx_chaser_env import MAQuadXChaserEnv
from modules.utils import *

# model_path = Path('apps/models/ma_quadx_chaser_20240111-002615/ma_quadx_chaser-3145728.zip') always chase
# model_path = Path('apps/models/ma_quadx_chaser_20240117-054612/ma_quadx_chaser-10013504.zip')
model_path = Path('apps/models/ma_quadx_chaser_20240127-193805/saved_models/model_16000000.zip')
model_name = model_path.stem

model_folder = model_path.parent
if 'saved_models' in model_folder.parts:
    model_folder = model_folder.parent
model = PPO.load(model_path)

try:
    params_path = f'{model_folder}/{model_name}.yaml'
    spawn_settings, env_kwargs, train_kwargs = read_yaml_file(params_path)
except:
    params_path = f'{model_folder}\\saved_models\\train_params.yaml' # {model_folder.name}
    spawn_settings, env_kwargs, train_kwargs = read_yaml_file(params_path)
#spawn_settings, env_kwargs, train_kwargs = read_yaml_file(params_path)

env = MAQuadXChaserEnv(render_mode='human', **env_kwargs)
observations, infos = env.reset(seed=spawn_settings['seed'])

last_term = {}
counters = {'out_of_bounds': 0, 'crashes': 0, 'timeover': 0, 'exploded_target': 0, 'exploded_by_ally': 0,
            'survived': 0, 'ally_collision': 0, 'downed': 0, 'is_success': 0}
first_time = True
num_games = 1

while env.agents:
    print(env.aviary.elapsed_time)
    # actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    actions = {agent: model.predict(observations[agent], deterministic=True)[0] for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)
    if first_time == True:
        first_time = False

    elif set(terminations.values()) != set(last_term.values()):
        print(f"| An agent terminated |")
        print(f'{terminations=}')
        print(f'{truncations=}')
        print(f'{infos=}\n')

    last_term = terminations

    # if any(terminations.values()) or any(truncations.values()):
    #
    #     for agent_key, agent_data in infos.items():
    #         for key, value in agent_data.items():
    #             if key in counters:
    #                 counters[key] += 1

    if all(terminations.values()) or all(truncations.values()):
        print(f'********* EPISODE END **********\n')
        print(f'{rewards=}\n')
        print(f'{terminations=}')
        print(f'{truncations=}\n')
        print(f'{infos=}\n\n\n')
        # time.sleep(5)
        if env.save_step_data:
            env.write_step_data(Path(model_folder, 'run-data', f'{model_name}.csv'))
            env.write_obs_data(Path(model_folder, 'run-data', f'obs-{model_name}.csv'))
        # env.plot_agent_rewards('reward_data.csv', 0)
        # env.plot_agent_infos2('reward_data.csv', 0)

        num_games -= 1

        print(f'Remaining games: {num_games}')

    if num_games == 0:
        print(env.info_counters)
        break

env.close()
