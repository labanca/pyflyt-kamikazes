"""Uses Stable-Baselines3 to train agents to play the Waterworld environment using SuperSuit vector envs.

For more information, see https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

Author: Elliot (https://github.com/elliottower)
"""
from __future__ import annotations

import os
import time
from datetime import datetime

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import get_device
from stable_baselines3.ppo import MlpPolicy
from gymnasium.utils import EzPickle

from envs.ma_quadx_chaser_env import MAQuadXChaserEnv
from modules.callbacks import TensorboardCallback


def train_butterfly_supersuit(
    env_fn, steps: int = 10_000, seed: int | None = 0, train_desc = '', model_name='', model_dir='', **env_kwargs):

    # Train a single model to play as each agent in a cooperative Parallel environment
    env = env_fn(**env_kwargs)

    env.reset(seed=seed)

    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env,)

    num_vec_envs = 16
    num_cpus = num_vec_envs
    env = ss.concat_vec_envs_v1(env, num_vec_envs, num_cpus=num_cpus, base_class="stable_baselines3", )

    device = get_device('cuda')
    batch_size = 512  # 512 davi
    lr = 1e-4
    discount_factor = 0.99
    nn_t = [128, 128, 128]
    policy_kwargs = dict(
        net_arch=dict(pi=nn_t, vf=nn_t)
    )

    model_path = os.path.join('apps\\models', model_dir, model_name)

    if not os.path.exists(model_path):
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
            tensorboard_log=log_dir
        )

        new_logger = configure(log_dir, ["csv", "tensorboard"])
        model.set_logger(new_logger)

        callback = TensorboardCallback(verbose=1)

        model.learn(total_timesteps=steps, callback=callback)

        model.save(filename)

        print("Model has been saved.")

        print(f"Finished training on {model_name}.")

    else:
        model = PPO.load(model_path, env=env)

        new_total_timesteps = model.num_timesteps + steps
        new_model_name = f"{env.unwrapped.metadata.get('name')}-{new_total_timesteps}"
        folder_name = os.path.join("apps\\models", model_dir )
        filename = os.path.join(folder_name, new_model_name)

        print(f"Starting resume training on {model_name} to {new_total_timesteps} steps.")

        logs_dir = os.path.join(folder_name, 'logs', new_model_name)
        new_logger = configure(logs_dir, [ "csv", "tensorboard"])
        model.set_logger(new_logger)

        callback = TensorboardCallback(verbose=1)

        model.learn(total_timesteps=steps, reset_num_timesteps=False, callback=callback )
        model.save(filename)

        print("Model has been saved.")

        print(f"Finished training on {new_model_name}.")

    with open(f'{filename}.txt', 'w') as file:
        # Write train params to file
        start_datetime = datetime.fromtimestamp(model.start_time / 1e9)
        current_time = datetime.now()
        elapsed_time = current_time - start_datetime
        file.write(f'{train_desc=}\n')
        file.write(f'{__file__=}\n')
        file.write(f'model.start_datetime={start_datetime}\n')
        file.write(f'elapsed_time={elapsed_time}\n')
        file.write(f'{model.num_timesteps=:n}\n')
        file.write(f'lw_stand_still={env_kwargs["lw_stand_still"]}\n')
        file.write(f'{device=}\n')
        file.write(f'{seed=}\n')
        file.write(f'{batch_size=}\n')
        file.write(f'{model.learning_rate=}\n')
        file.write(f'{discount_factor=}\n')
        file.write(f'{nn_t=}\n')
        file.write(f'{num_cpus=}\n')
        file.write(f'{num_vec_envs=}\n')
        file.write(f'{model.n_envs=}\n')
        file.write(f'{model.n_steps=}\n')
        file.write(f'{model.n_epochs=}\n')
        file.write(f'{model.batch_size=}\n')
        file.write(f'random_respawn={True if spawn_settings != None else False}\n')
        file.write(f'{spawn_settings=}\n')
        file.write(f'completion_datetime={current_time}\n')
        file.write(f'{model.action_space=}\n')
        file.write(f'{model.observation_space=}\n')
        file.write(f'{env_kwargs=}\n')
        file.write(f'{model.policy_kwargs=}\n')
        file.write(f'{model.policy=}\n')
        file.write(f'{model.policy_aliases=}\n')
        file.write(f'{model.policy_class=}\n')

    env.close()

    return new_model_name

class EZPEnv(EzPickle, MAQuadXChaserEnv):
    def __init__(self, *args, **kwargs):
        EzPickle.__init__(self, *args, **kwargs)
        MAQuadXChaserEnv.__init__(self, *args, **kwargs)




if __name__ == "__main__":
    env_fn = EZPEnv

    train_desc = """ more explode reward, more ep len .

            # reward for closing the distance
            self.rew_closing_distance = np.clip(
                self.previous_distance[agent_id][target_id] - self.current_distance[agent_id][target_id],
                a_min=-10.0,
                a_max=None,
            ) * self.chasing[agent_id][target_id]

            self.rew_close_to_target = 1 / (
                self.current_distance[agent_id][target_id]
                if self.current_distance[agent_id][target_id] > 0 else 0.1 )   #if the 1 is to hight the kamikazes will circle the enemy. try a


            self.rewards[agent_id] += (
                    self.rew_closing_distance
                    + self.rew_close_to_target * self.reward_coef # regularizations
            )
"""

    spawn_settings = dict(
        lw_center_bounds=10.0,
        lw_spawn_radius=1.0,
        lm_center_bounds=5.0,
        lm_spawn_radius=10.0,
        min_z=1.0,
        seed=None,
        num_lw=1,
        num_lm=1,
    )

    env_kwargs = {}
    env_kwargs['start_pos'], env_kwargs['start_orn'], env_kwargs['formation_center'] = MAQuadXChaserEnv.generate_start_pos_orn(**spawn_settings)
    env_kwargs['flight_dome_size'] = (spawn_settings['lw_spawn_radius'] + spawn_settings['lm_spawn_radius']
                                      + spawn_settings['lw_center_bounds'] + spawn_settings['lm_center_bounds']) * 1.5
    env_kwargs['seed'] = spawn_settings['seed']
    env_kwargs['spawn_settings'] = spawn_settings
    env_kwargs['num_lm'] = spawn_settings['num_lm']
    env_kwargs['num_lw'] = spawn_settings['num_lw']
    env_kwargs['max_duration_seconds'] = 30
    env_kwargs['reward_coef'] = 1.0
    env_kwargs['lw_stand_still'] = False

    model_name = 'ma_quadx_chaser-5038656.zip'
    model_dir = 'ma_quadx_chaser_20240105-210345'

    num_resumes = 15
    for i in range(num_resumes):
        model_name = train_butterfly_supersuit(env_fn, steps=1_000_000, train_desc=train_desc,
                                  model_name=model_name, model_dir=model_dir,
                                  **env_kwargs)



    # tensorboard --logdir C:/projects/pyflyt-kamikazes/apps/models/ma_quadx_chaser_20240104-161545/tensorboard/
