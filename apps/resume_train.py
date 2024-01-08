"""Uses Stable-Baselines3 to train agents to play the Waterworld environment using SuperSuit vector envs.

For more information, see https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

Author: Elliot (https://github.com/elliottower)
"""
from __future__ import annotations

import os
import time
import inspect
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
    env_fn, steps: int = 10_000, seed: int | None = 0, train_desc = '', model_name='', model_dir='',
        env_kwargs={}, train_kwargs={} ):

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
    env = ss.pettingzoo_env_to_vec_env_v1(env,)
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

        callback = TensorboardCallback(verbose=1)

        model.learn(total_timesteps=steps, callback=callback, progress_bar=True)
        model.save(filename)

        print(f"Model {model_name} has been saved.")
        new_model_name = model_name

    else:
        print(f"\nModel {model_path} found. Resuming training.\n")

        model = PPO.load(model_path, env=env, device=device)

        new_total_timesteps = model.num_timesteps + steps
        new_model_name = f"{env.unwrapped.metadata.get('name')}-{new_total_timesteps}"
        folder_name = os.path.join("apps\\models", model_dir )
        filename = os.path.join(folder_name, new_model_name)
        log_dir = os.path.join(folder_name, 'logs', new_model_name)

        new_logger = configure(log_dir, ["csv", "tensorboard"])
        model.set_logger(new_logger)
        callback = TensorboardCallback(verbose=1)

        print(f"Starting resume training on {model_name}")
        model.learn(total_timesteps=steps, reset_num_timesteps=False, callback=callback )
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
        file.write(f'model_name={model_name}\n')
        file.write(f'model_name={new_model_name}\n')
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
        file.write(f'{train_kwargs=}\n')
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

    train_desc = """ Include the explosion radius (0.5) in the reward computation to incentivize collission and negative reward for current distance. 
                            
                if target_id != agent_id:  #avoid the scenario where there are no targets, returns the last rewards in the last steps

                # reward for closing the distance
                self.rew_closing_distance[agent_id] = np.clip(
                    self.previous_distance[agent_id][target_id] - self.current_distance[agent_id][target_id],
                    a_min=-10.0,
                    a_max=None,
                ) * self.chasing[agent_id][target_id]

                

                exploding_distance = self.current_distance[agent_id][target_id] - 0.5

                self.rew_close_to_target[agent_id] = - exploding_distance
                 
                # self.rew_close_to_target[agent_id] = 1 / (exploding_distance
                #                                 if exploding_distance > 0
                #                                 else 0.09)   #if the 1 is to hight the kamikazes will circle the enemy. try a


            self.rewards[agent_id] += (
                    self.rew_closing_distance[agent_id]
                    + self.rew_close_to_target[agent_id] * self.reward_coef #* (1 - self.step_count/self.max_steps) # regularizations

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
    env_kwargs['lw_stand_still'] = True
    env_kwargs['rew_exploding_target'] = 1000

    nn_t = [256, 256, 512]
    train_kwargs = dict(
        device=get_device('cuda'),
        batch_size=1024,
        lr=1e-4,
        discount_factor=0.98,
        nn_t=nn_t,
        num_vec_envs=16,
    )


    model_dir = 'ma_quadx_chaser_20240107-202245'
    model_name = 'ma_quadx_chaser-5063232.zip'

    num_resumes = 5
    for i in range(num_resumes):
        model_name, model_dir = train_butterfly_supersuit(
                                    env_fn=env_fn, steps=1_000_000, train_desc=train_desc,
                                    model_name=model_name, model_dir=model_dir,
                                    env_kwargs=env_kwargs, train_kwargs=train_kwargs)

    # tensorboard --logdir C:/projects/pyflyt-kamikazes/apps/models/ma_quadx_chaser_20240104-161545/tensorboard/
