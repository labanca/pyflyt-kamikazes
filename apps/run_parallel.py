from pathlib import Path
from stable_baselines3 import PPO
from envs.ma_quadx_chaser_env import MAQuadXChaserEnv
from modules.utils import *

model_path = Path('apps/models/ma_quadx_chaser_20240204-120343/ma_quadx_chaser-10000000.zip')
model_name = model_path.stem
model_folder = model_path.parent

if 'saved_models' in model_folder.parts:
    model_folder = model_folder.parent
elif 'logs' in model_folder.parts:
    model_folder = model_folder.parent.parent

model = PPO.load(model_path)

params_path = f'{model_folder}/{model_name}.yaml'
spawn_settings, env_kwargs, train_kwargs = read_yaml_file(params_path)

env = MAQuadXChaserEnv(render_mode='human', **env_kwargs)
observations, infos = env.reset(seed=spawn_settings['seed'])

num_games = 1

print(env_kwargs)

while env.agents:
    print(env.aviary.elapsed_time)
    actions = {agent: model.predict(observations[agent], deterministic=True)[0] for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)

    if all(terminations.values()) or all(truncations.values()):
        if env.save_step_data:
            env.write_step_data(Path(model_folder, 'run-data', f'{model_name}.csv'))
            env.write_obs_data(Path(model_folder, 'run-data', f'obs-{model_name}.csv'))

        num_games -= 1
        print(f'Remaining games: {num_games}')

    if num_games == 0:
        print(env.info_counters)
        break

env.close()
