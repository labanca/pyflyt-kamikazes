"""Uses Stable-Baselines3 to train agents to play the Waterworld environment using SuperSuit vector envs.

For more information, see https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

Author: Elliot (https://github.com/elliottower)
"""
from __future__ import annotations

import stable_baselines3.common.monitor
import yaml

import glob
import os
import time
from datetime import datetime

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.utils import get_device
from gymnasium.utils import EzPickle
from pettingzoo.utils import parallel_to_aec
from modules.callbacks import TensorboardCallback

from envs.ma_quadx_chaser_env import MAQuadXChaserEnv


def train_butterfly_supersuit(
    env_fn, steps: int = 10_000, seed: int | None = 0, train_desc = '', **env_kwargs
):
    # Train a single model to play as each agent in a cooperative Parallel environment
    env = env_fn(**env_kwargs)

    env.reset(seed=seed)

    print(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.black_death_v3(env,)
    env = ss.pettingzoo_env_to_vec_env_v1(env )


    num_vec_envs = 16
    num_cpus = num_vec_envs
    env = ss.concat_vec_envs_v1(env, num_vec_envs, num_cpus=num_cpus, base_class="stable_baselines3")


    device = get_device('cpu')
    batch_size = 512 # 512 davi
    lr = 1e-4
    discount_factor = 0.98
    nn_t = [128, 128, 128]
    policy_kwargs = dict(
        net_arch=dict(pi=nn_t, vf=nn_t)
    )



    folder_name = f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}"
    model_name = f"{env.unwrapped.metadata.get('name')}-{steps}"
    folder_path = os.path.join("apps\\models", folder_name,)
    log_path = os.path.join(folder_path, 'tensorboard')
    filename = os.path.join(folder_path, model_name )

    log_dir  = os.path.join(folder_path, 'logs', model_name)


    model = PPO(
        MlpPolicy,
        env,
        verbose=1,
        learning_rate=lr,
        batch_size=batch_size,
        policy_kwargs=policy_kwargs,
        device=device,
        gamma=discount_factor,
        tensorboard_log=log_path
    )


    new_logger = configure(log_dir, ["csv", "tensorboard"])

    model.set_logger(new_logger)

    callback = TensorboardCallback()
    model.learn(total_timesteps=steps, callback=callback)

    model.save(filename)

    print("Model has been saved.")

    print(f"Finished training on {model_name}.")


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


def eval(env_fn, num_games: int = 100, render_mode: str | None = None, **env_kwargs):
    # Evaluate a trained agent vs a random agent
    env = env_fn(render_mode=render_mode, **env_kwargs )
    env = parallel_to_aec(env)

    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    try:
        latest_policy = max(
            glob.glob(f"models/{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)
    print(f"Using {latest_policy} as model.")
    model = PPO.load(latest_policy)

    rewards = {agent: 0.0 for agent in env.possible_agents}

    # Note: We train using the Parallel API but evaluate using the AEC API
    # SB3 backup are designed for single-agent settings, we get around this by using he same model for every agent

    for i in range(num_games):
        last_term = False
        env.reset(seed=i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            for a in env.agents:
                rewards[a] += env.rewards[a]
            if truncation : #and info.get('mission_complete') == True
                print(f'terminate with {agent=} {termination=} {truncation=} {info=}')
                break
            else:
                act = model.predict(obs, deterministic=True)[0]
                #act = np.array([1,1,0,0])
            if termination != last_term:
                print(f'| A agent terminated |')
                print(f'{obs=}')
                print(f'{agent=}')
                print(f'{termination=}')
                print(f'{truncation=}\n')
                print(f'{reward=}\n')
                print(f'{info}')
            env.step(act)
            #print(f'{reward=}')



    env.close()

    avg_reward_per_agent = sum(rewards.values()) / len(rewards.values()  )
    avg_reward_per_game = sum(rewards.values()) / num_games
    print("\nRewards: ", rewards)
    print(f"Avg reward per agent: {avg_reward_per_agent}")
    print(f"Avg reward per game: {avg_reward_per_game}")
    return avg_reward_per_game


class EZPEnv(EzPickle, MAQuadXChaserEnv):
    def __init__(self, *args, **kwargs):
        EzPickle.__init__(self, *args, **kwargs)
        MAQuadXChaserEnv.__init__(self, *args, **kwargs)


if __name__ == "__main__":
    env_fn = EZPEnv

    spawn_settings = dict(
        lw_center_bounds=10.0,
        lm_center_bounds=5.0,
        lw_spawn_radius=1.0,
        lm_spawn_radius=10.0,
        min_z=1.0,
        seed=None,
        num_lw=1,
        num_lm=1,
    )

    env_kwargs = {}
    env_kwargs['start_pos'], env_kwargs['start_orn'], env_kwargs['formation_center'] = MAQuadXChaserEnv.generate_start_pos_orn(**spawn_settings)
    env_kwargs['flight_dome_size'] = (spawn_settings['lw_spawn_radius'] + spawn_settings['lm_spawn_radius']
                                      + spawn_settings['lw_center_bounds'] + spawn_settings[
                                          'lm_spawn_radius'])
    env_kwargs['seed'] = spawn_settings['seed']
    env_kwargs['spawn_settings'] = spawn_settings
    env_kwargs['num_lm'] = spawn_settings['num_lm']
    env_kwargs['num_lw'] = spawn_settings['num_lw']
    env_kwargs['max_duration_seconds'] = 10
    env_kwargs['reward_coef'] = 1.0
    env_kwargs['lw_stand_still'] = True

    #seed = 42
    train_desc = """new rewards. Current_vel_angles still broken.

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



    #Train a model
    train_butterfly_supersuit(env_fn, steps=1_000_000, train_desc=train_desc, **env_kwargs)

    # Evaluate 10 games (average reward should be positive but can vary significantly)
    #eval(env_fn, num_games=10, render_mode=None, **env_kwargs)

    #eval(env_fn, num_games=1, render_mode="human", **env_kwargs)
