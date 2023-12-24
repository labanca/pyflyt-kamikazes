"""Uses Stable-Baselines3 to train agents to play the Waterworld environment using SuperSuit vector envs.

For more information, see https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

Author: Elliot (https://github.com/elliottower)
"""
from __future__ import annotations

import glob
import os
import time
import numpy as np
import torch
from datetime import datetime

import supersuit as ss
import torch
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.utils import get_device
from gymnasium.utils import EzPickle
from pettingzoo.utils import parallel_to_aec

from envs.ma_quadx_chaser_env import MAQuadXHoverEnv


def train_butterfly_supersuit(
    env_fn, steps: int = 10_000, seed: int | None = 0, train_desc = '', **env_kwargs
):
    # Train a single model to play as each agent in a cooperative Parallel environment
    env = env_fn(**env_kwargs)

    env.reset(seed=seed)

    print(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)

    num_vec_envs = 12 #8
    num_cpus = 12 #(os.cpu_count() or 1)
    env = ss.concat_vec_envs_v1(env, num_vec_envs, num_cpus=num_cpus, base_class="stable_baselines3")


    device = get_device('cuda')
    batch_size = 512 # 512 davi
    lr = 1e-4
    nn_t = [256, 256, 256]
    policy_kwargs = dict(
        net_arch=dict(pi=nn_t, vf=nn_t)
    )

    model = PPO(
        MlpPolicy,
        env,
        verbose=1,
        learning_rate=lr,
        batch_size=batch_size,
        policy_kwargs=policy_kwargs,
        device=device,
    )

    model.learn(total_timesteps=steps)
    model_name = f"models/{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}"
    model.save(model_name)

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    with open(f'{model_name}.txt', 'w') as file:
        # Write train params to file
        start_datetime = datetime.fromtimestamp(model.start_time / 1e9)
        current_time = datetime.now()
        elapsed_time = current_time - start_datetime
        file.write(f'{train_desc=}\n')
        file.write(f'{__file__=}\n')
        file.write(f'model.start_datetime={start_datetime}\n')
        file.write(f'elapsed_time={elapsed_time}\n')
        file.write(f'{model.num_timesteps=:n}\n')
        file.write(f'{device=}\n')
        file.write(f'{seed=}\n')
        file.write(f'{batch_size=}\n')
        file.write(f'{model.learning_rate=}\n')
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


class EZPEnv(EzPickle, MAQuadXHoverEnv):
    def __init__(self, *args, **kwargs):
        EzPickle.__init__(self, *args, **kwargs)
        MAQuadXHoverEnv.__init__(self, *args, **kwargs)


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

import numpy as np

def get_start_pos_orn_scenario1(num_drones, min_distance, spawn_radius, center, num_lm, lw_spawn_radius_ratio=0.25, seed=None):
    start_pos = np.empty((num_drones, 3))
    min_z = 1
    np_random = np.random.RandomState(seed=seed)

    # Calculate the radius within which lw drones can spawn
    lw_spawn_radius = lw_spawn_radius_ratio * spawn_radius

    for i in range(num_drones):
        while True:
            # Generate random coordinates
            x = np_random.uniform(center[0] - spawn_radius, center[0] + spawn_radius)
            y = np_random.uniform(center[1] - spawn_radius, center[1] + spawn_radius)
            z = np_random.uniform(max(center[2], min_z), center[2] + spawn_radius)

            if i < num_lm:
                # For lm drones, check if the generated coordinates are outside the lw_spawn_radius in 3D
                while np.linalg.norm([x - center[0], y - center[1], z - center[2]]) < lw_spawn_radius:
                    x = np_random.uniform(center[0] - spawn_radius, center[0] + spawn_radius)
                    y = np_random.uniform(center[1] - spawn_radius, center[1] + spawn_radius)
                    z = np_random.uniform(max(center[2], min_z), center[2] + spawn_radius)

            # Check if the minimum distance condition is met with existing drones
            if i == 0 or np.min(np.linalg.norm(start_pos[:i] - np.array([x, y, z]), axis=1)) >= min_distance:
                start_pos[i] = [x, y, z]
                break

    start_orn = (np_random.rand(num_drones, 3) - 0.5) * 2.0 * np.array([1.0, 1.0, 2 * np.pi])
    start_orn[num_lm:] = np.zeros((len(start_pos) - num_lm, 3), dtype=np.float64)

    return start_pos, start_orn

seed=None

#find a better way to store it
spawn_settings = dict(
    num_drones=2,
    min_distance=2,
    spawn_radius=5,
    center=(0, 0, 0),
    seed=seed,
)

if __name__ == "__main__":
    env_fn = EZPEnv

    env_kwargs = {}
    env_kwargs['num_lm'] = 1
    env_kwargs['start_pos'], env_kwargs['start_orn']  = get_start_pos_orn(**spawn_settings, num_lm=env_kwargs['num_lm']) #np.array([[1, 1, 1], [-1, -1, 2]])
    #env_kwargs['start_orn'] =  #np.zeros_like(env_kwargs['start_pos']) np.array([[-1, -1, 2], [0, 0, 0]])
    env_kwargs['flight_dome_size'] = (6.75 * (spawn_settings['spawn_radius'] + 1) ** 2) ** 0.5  # dome size 50% bigger than the spawn radius
    env_kwargs['seed'] = seed
    env_kwargs['spawn_settings'] = spawn_settings

    #seed = 42
    train_desc = """10 times more rew_closing_distance and 3 times more rew_speed_magitude with less requirement of magnitude.
            # reward for closing the distance
            rew_closing_distance = np.clip(
                self.previous_distance[agent_id][target_id] - self.current_distance[agent_id][target_id],
                a_min=-10.0,
                a_max=None,
            ) * (
                    self.chasing[agent_id][target_id] * 10.0
                )

            # reward for engaging the enemy
            rew_engaging_enemy = np.divide(3.0, self.current_vel_angles[agent_id][target_id],
                                           where=self.current_vel_angles[agent_id][target_id] != 0) * (
                    self.chasing[agent_id][target_id]
                    * self.approaching[agent_id][target_id]
                    * 1.0
                )

            # reward for maintaning linear velocities.
            rew_speed_magitude =(
                    (3.0 if self.current_magnitude[agent_id] > 3.0 else 0.0)
                    * self.chasing[agent_id][target_id]
                    * self.approaching[agent_id][target_id]
            )



            self.rewards[agent_id] += (
                    rew_closing_distance
                    + rew_engaging_enemy
                    + rew_speed_magitude
            )
"""

    #Train a model (takes ~3 minutes on GPU)
    train_butterfly_supersuit(env_fn, steps=15_000_000,train_desc=train_desc, **env_kwargs)

    # Evaluate 10 games (average reward should be positive but can vary significantly)
    #eval(env_fn, num_games=10, render_mode=None, **env_kwargs)

    #eval(env_fn, num_games=1, render_mode="human", **env_kwargs)
