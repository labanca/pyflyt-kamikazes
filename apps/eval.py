import glob
import os.path
import yaml


from gymnasium.utils import EzPickle
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from pathlib import Path

import supersuit as ss
from envs.ma_quadx_chaser_env import MAQuadXChaserEnv
from stable_baselines3 import PPO
import numpy as np
import time


def save_dict_to_yaml(data, file_path):
    with open(file_path, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False, sort_keys=False)

class EZPEnv(EzPickle, MAQuadXChaserEnv):
    def __init__(self, *args, **kwargs):
        EzPickle.__init__(self, *args, **kwargs)
        MAQuadXChaserEnv.__init__(self, *args, **kwargs)

def eval(env_fn, n_eval_episodes: int = 100, num_vec_envs: int =1, model_name: str = '',
         render_mode: str | None = None, **kwargs):
    # Train a single model to play as each agent in a cooperative Parallel environment
    env = env_fn(**kwargs)

    env.reset(seed=kwargs['seed'])

    print(f"Starting eval on {model_name}.")

    env = ss.black_death_v3(env,)
    env = ss.pettingzoo_env_to_vec_env_v1(env )

    num_cpus = num_vec_envs
    env = ss.concat_vec_envs_v1(env, num_vec_envs, num_cpus=num_cpus, base_class="stable_baselines3")

    model = PPO.load(model_name)

    deterministic = True
    rewards, lengths = evaluate_policy(
                    model,
                    env,
                    render=False,
                    n_eval_episodes=n_eval_episodes,
                    return_episode_rewards=True,
                    deterministic=deterministic,
                )

    output = dict(
        ep_mean_rew=float(np.array(rewards).mean()),
        ep_mean_len=float(np.array(lengths).mean()),
        reward=[float(r) for r in rewards],
        lengths=[int(i) for i in lengths],

    )

    return output

if __name__ == "__main__":

    env_fn = EZPEnv

    spawn_settings = dict(
        lw_center_bounds=5.0,
        lm_center_bounds=5.0,
        lw_spawn_radius=1.0,
        lm_spawn_radius=5.0,
        min_z=1.0,
        seed=None,
        num_lw=1,
        num_lm=1,
    )

    env_kwargs = {}
    env_kwargs['start_pos'], env_kwargs['start_orn'], env_kwargs[
        'formation_center'] = MAQuadXChaserEnv.generate_start_pos_orn(**spawn_settings)
    env_kwargs['flight_dome_size'] = (spawn_settings['lw_spawn_radius'] + spawn_settings['lm_spawn_radius']
                                      + spawn_settings['lw_center_bounds'] + spawn_settings[
                                          'lm_center_bounds'])  # dome size 50% bigger than the spawn radius
    env_kwargs['seed'] = None
    env_kwargs['spawn_settings'] = spawn_settings
    env_kwargs['num_lm'] = spawn_settings['num_lm']
    env_kwargs['num_lw'] = spawn_settings['num_lw']
    env_kwargs['max_duration_seconds'] = 30.0
    env_kwargs['reward_coef'] = 1.0
    env_kwargs['lw_stand_still'] = True


    counters = {'out_of_bounds': 0, 'crashes': 0, 'timeover': 0, 'exploded_target': 0, 'mission_complete': 0, 'ally_collision': 0,
                'exploded_by_ally': 0, 'downed': 0}


    n_eval_episodes = 100
    model_dir = 'ma_quadx_chaser_20240107-173953'
    root_path = Path('apps/models')
    model_folder = Path.joinpath(root_path, model_dir)
    files_paths = glob.glob(f"{model_folder}/*.zip")
    model_names = [Path(file).name for file in files_paths]

    data = dict()
    result_dict = dict()

    for file, model_name in zip(files_paths, model_names):

        result_dict = eval(env_fn, n_eval_episodes=n_eval_episodes, num_vec_envs=1, model_name=file,  **env_kwargs)
        data[model_name] = {**result_dict}

    yaml_file = Path(model_folder, model_dir +".yaml")
    save_dict_to_yaml(data, yaml_file)





