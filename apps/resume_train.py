"""Uses Stable-Baselines3 to train agents to play the Waterworld environment using SuperSuit vector envs.

For more information, see https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

Author: Elliot (https://github.com/elliottower)
"""
from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from pprint import pprint

import supersuit as ss
from gymnasium.utils import EzPickle
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.ppo import MlpPolicy

from envs.ma_quadx_chaser_env import MAQuadXChaserEnv
from modules.callbacks import TensorboardCallback
from modules.utils import *


def train_butterfly_supersuit(
        env_fn, steps: int = 10_000, seed: int | None = 0, train_desc='', model_name='', model_dir='',
        env_kwargs={}, train_kwargs={}):
    """ Train a single model to play as each agent in a cooperative Parallel environment """

    # unpack train kwargs
    num_vec_envs = train_kwargs['num_vec_envs']
    num_cpus = num_vec_envs
    device = train_kwargs['device']
    batch_size = train_kwargs['batch_size']
    lr = train_kwargs['lr']
    discount_factor = train_kwargs['discount_factor']
    nn_t = train_kwargs['nn_t']
    policy_kwargs = dict(
        net_arch=dict(pi=nn_t, vf=nn_t)
    )
    env = env_fn(**env_kwargs)
    env.reset(seed=seed)
    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env, )
    env = ss.concat_vec_envs_v1(env, num_vec_envs, num_cpus=num_cpus, base_class="stable_baselines3", )

    model_path = os.path.join('apps\\models', model_dir, model_name)

    if not os.path.exists(model_path):
        print(f"-> Model {model_path} do not exist. Creating a new one.")
        model_dir = f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}"
        model_name = f"{env.unwrapped.metadata.get('name')}-{steps}"
        folder_name = os.path.join("apps\\models", model_dir)
        filename = os.path.join(folder_name, model_name)
        log_dir = os.path.join(folder_name, 'logs', model_name)

        model = PPO(
            MlpPolicy,
            env,
            verbose=1,
            learning_rate=lr,
            batch_size=batch_size,
            policy_kwargs=policy_kwargs,
            device=device,
            gamma=discount_factor,
        )

        new_logger = configure(log_dir, ["csv", "tensorboard"])
        model.set_logger(new_logger)

        callback = TensorboardCallback(verbose=1, save_path=folder_name,
                                       save_freq=250_000, check_freq=500, log_dir=log_dir)

        model.learn(total_timesteps=steps, callback=callback, progress_bar=False)

        model.save(filename)

        print(f"Model {model_name} has been saved.")
        new_model_name = model_name

    else:
        print(f"\nModel {model_path} found. Resuming training.\n")

        model = PPO.load(model_path, env=env, device=device)

        new_total_timesteps = (steps + model.num_timesteps)
        new_model_name = f"{env.unwrapped.metadata.get('name')}-{new_total_timesteps}"
        folder_name = os.path.join("apps\\models", model_dir)
        filename = os.path.join(folder_name, new_model_name)
        log_dir = os.path.join(folder_name, 'logs', new_model_name)

        new_logger = configure(log_dir, ["csv", "tensorboard"])
        model.set_logger(new_logger)
        callback = TensorboardCallback(verbose=1, save_path=folder_name,
                                       save_freq=500_000, check_freq=1000, log_dir=log_dir)

        print(f"Starting resume training on {model_name}")
        model.learn(total_timesteps=steps, reset_num_timesteps=False, callback=callback)
        model.save(filename)

        print(f"Model {new_model_name} has been saved.")

    with open(f'{filename}.txt', 'w') as file:
        # Write train params to file
        start_datetime = datetime.fromtimestamp(model.start_time / 1e9)
        current_time = datetime.now()
        elapsed_time = current_time - start_datetime
        file.write(f'{train_desc=}\n')
        file.write(f'{__file__=}\n')
        file.write(f'model.start_datetime={start_datetime}\n')
        file.write(f'elapsed_time={elapsed_time}\n')
        file.write(f'completion_datetime={current_time}\n')
        file.write(f'{model.num_timesteps=:n}\n')
        file.write(f'{model.device=}\n')
        file.write(f'{model.n_envs=}\n')
        file.write(f'{model.n_steps=}\n')
        file.write(f'{model.n_epochs=}\n')
        file.write(f'{model.batch_size=}\n')
        file.write(f'{model.action_space=}\n')
        file.write(f'{model.observation_space=}\n')
        file.write(f'{model.policy_kwargs=}\n')
        file.write(f'{model.policy=}\n')
        file.write(f'{model.policy_aliases=}\n')
        file.write(f'{model.policy_class=}\n')

    env.close()

    return f'{new_model_name}.zip', model_dir


class EZPEnv(EzPickle, MAQuadXChaserEnv):
    def __init__(self, *args, **kwargs):
        EzPickle.__init__(self, *args, **kwargs)
        MAQuadXChaserEnv.__init__(self, *args, **kwargs)


if __name__ == "__main__":
    env_fn = EZPEnv

    train_desc = """  """

    params_path = 'apps/train_params.yaml'
    spawn_settings, env_kwargs, train_kwargs = read_yaml_file(params_path)

    root_dir = 'apps/models'
    model_dir = 'ma_quadx_chaser_20240204-120343'
    model_name = 'ma_quadx_chaser-20027008.zip'

    steps = 40_000_000
    num_resumes = 4
    reset_model = False

    for i in range(num_resumes):

        if i >= 1: # lw fsm on
            model_name = 'a'
            #steps = 10_000_000
            #env_kwargs['reward_type'] = 5 # hit prob in penalty
            #env_kwargs['observation_type'] = 3 # hit prob in obs

            #train_kwargs['num_vec_envs'] = 2
            #env_kwargs["lw_stand_still"] = False
            #env_kwargs["lw_moves_random"] = False
            #env_kwargs['max_velocity_magnitude'] = 20
            #env_kwargs['speed_factor'] = 0.2
            #env_kwargs['rew_exploding_target'] = 200
            #env_kwargs["spawn_settings"]["num_lm"] = 10
            #env_kwargs["spawn_settings"]["num_lw"] = 2
            #env_kwargs["num_lm"] = 10
            #env_kwargs["num_lw"] = 2

        pprint(f'{env_kwargs["num_lm"]=}')
        pprint(f'{env_kwargs["num_lw"]=}')
        pprint(f'{env_kwargs["spawn_settings"]["num_lm"]=}')
        pprint(f'{env_kwargs["spawn_settings"]["num_lw"]=}')
        pprint(f'{env_kwargs["lw_stand_still"]=}')
        pprint(f'{env_kwargs["lw_moves_random"]=}')
        pprint(f'{env_kwargs["lw_shoot_range"]=}')
        pprint(f'{env_kwargs["lw_threat_radius"]=}')
        pprint(f'{env_kwargs["lw_weapon_cooldown"]=}')
        pprint(f'{env_kwargs["rew_exploding_target"]=}')
        pprint(f'{env_kwargs["distance_factor"]=}')
        pprint(f'{env_kwargs["proximity_factor"]=}')
        pprint(f'{env_kwargs["speed_factor"]=}')
        pprint(f'{env_kwargs["max_velocity_magnitude"]=}')
        pprint(f'{env_kwargs["observation_type"]=}')
        pprint(f'{env_kwargs["reward_type"]=}')

        model_name, model_dir = train_butterfly_supersuit(
            env_fn=env_fn, steps=steps, train_desc=train_desc,
            model_name=model_name, model_dir=model_dir,
            env_kwargs=env_kwargs, train_kwargs=train_kwargs)

        save_dicts_to_yaml(env_kwargs['spawn_settings'], env_kwargs, train_kwargs,
                           Path(root_dir, model_dir, f'{model_name.split(".")[0]}.yaml'))



    # tensorboard --logdir C:/projects/pyflyt-kamikazes/apps/models/ma_quadx_chaser_20240104-161545/tensorboard/
