import csv
import time
from pathlib import Path
from timeit import timeit

import numpy as np
from stable_baselines3 import PPO

from envs.ma_quadx_chaser_env import MAQuadXChaserEnv
from modules.utils import read_yaml_file, save_dict_to_csv


def write_inferation_data(data, filename):
    filename.parent.mkdir(parents=True, exist_ok=True)

    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['i', 'start_time', 'end_time', 'inference_time']

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header
        writer.writeheader()

        # Write the data
        for step_data in data:
            writer.writerow(step_data)

seed = None

model_path = Path('apps/models/ma_quadx_chaser_20240204-120343/model_39500000.zip')
model_name = model_path.stem
model_folder = model_path.parent
model = PPO.load(model_path)

params_path = f'modules/examples/train_params_test.yaml'
spawn_settings, env_kwargs, train_kwargs = read_yaml_file(params_path)

env = MAQuadXChaserEnv(render_mode=None, **env_kwargs)
observations, infos = env.reset(seed=seed)

inferences = []

num_iterations = 100_000
for i in range(num_iterations):

    start_time = time.perf_counter()
    model.predict(observations['agent_0'], deterministic=True)
    end_time = time.perf_counter()

    inferences.append({'i':i, 'start_time':start_time, 'end_time':end_time, 'inference_time':round(end_time - start_time,10)})

    #observations, rewards, terminations, truncations, infos = env.step(actions)
filename = Path(f'{model_folder}/nn_inference/inference_times.csv')
filename.parent.mkdir(parents=True, exist_ok=True)
write_inferation_data(inferences, filename)


