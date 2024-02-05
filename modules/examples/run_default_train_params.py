import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from envs.ma_quadx_chaser_env import MAQuadXChaserEnv
from modules.utils import read_yaml_file

seed = None

model_path = Path('apps/models/ma_quadx_chaser_20240204-120343/saved_models/model_15500000.zip')
model_name = model_path.stem
model_folder = model_path.parent
model = PPO.load(model_path)

params_path = f'apps/train_params.yaml'
spawn_settings, env_kwargs, train_kwargs = read_yaml_file(params_path)

env = MAQuadXChaserEnv(render_mode='human', **env_kwargs)
observations, infos = env.reset(seed=seed)

num_games = 1

while env.agents:

    # actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    actions = {agent: model.predict(observations[agent], deterministic=True)[0] for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)

    if all(terminations.values()) or all(truncations.values()):
        env.write_step_data(Path('modules/examples/step_data.csv'))
        env.write_obs_data(Path('modules/examples/obs_data.csv'))
        print(env.info_counters)
        time.sleep(2)
        observations, infos = env.reset(seed=seed)
        num_games -= 1
        print(f'Remaining games: {num_games}')

    if num_games == 0:
        break

env.close()
