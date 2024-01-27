import glob
from pathlib import Path

import numpy as np
import supersuit as ss
import yaml
from gymnasium.utils import EzPickle
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from envs.ma_quadx_chaser_env import MAQuadXChaserEnv
from modules.utils import read_yaml_file
from modules.utils import save_agg_dict_to_csv


def save_dict_to_yaml(data, file_path):
    Path(file_path.parent).mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False, sort_keys=False)


class EZPEnv(EzPickle, MAQuadXChaserEnv):
    def __init__(self, *args, **kwargs):
        EzPickle.__init__(self, *args, **kwargs)
        MAQuadXChaserEnv.__init__(self, *args, **kwargs)


def eval(env_fn, n_eval_episodes: int = 100, num_vec_envs: int = 1, model_name: str = '',
         render_mode: str | None = None, **kwargs):
    # Train a single model to play as each agent in a cooperative Parallel environment
    env = env_fn(**kwargs)

    env.reset(seed=kwargs['seed'])

    print(f"Starting eval on {model_name}.")

    env = ss.black_death_v3(env, )
    env = ss.pettingzoo_env_to_vec_env_v1(env)

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
        ep_max_rew=float(max(rewards)),
        ep_min_rew=float(min(rewards)),
        ep_std_rew=float(np.std(np.array(rewards))),

        ep_mean_len=float(np.array(lengths).mean()),
        ep_max_len=float(max(lengths)),
        ep_min_len=float(min(lengths)),
        ep_std_len=float(np.std(np.array(lengths))),

        reward=[float(r) for r in rewards],
        lengths=[int(i) for i in lengths],

    )

    return output


if __name__ == "__main__":

    env_fn = EZPEnv

    counters = {'out_of_bounds': 0, 'crashes': 0, 'timeover': 0, 'exploded_target': 0, 'mission_complete': 0,
                'ally_collision': 0,
                'exploded_by_ally': 0, 'downed': 0}

    n_eval_episodes = 100
    model_dir = 'ma_quadx_chaser_20240112-013125'
    root_path = Path('apps/models')
    model_folder = Path.joinpath(root_path, model_dir)
    files_paths = glob.glob(f"{model_folder}/*.zip")
    model_names = [Path(file).name for file in files_paths]

    data = dict()
    result_dict = dict()

    for file, model_name in zip(files_paths, model_names):
        params_path = f'{model_folder}/{model_name.split(".")[0]}.yaml'
        spawn_settings, env_kwargs, train_kwargs = read_yaml_file(params_path)
        result_dict = eval(env_fn, n_eval_episodes=n_eval_episodes, num_vec_envs=1, model_name=file, **env_kwargs)

        agg_results = {key: value for key, value in result_dict.items() if key not in ['reward', 'lengths']}
        yaml_model_filename = Path(model_folder, 'eval', model_name.split('.')[0] + ".yaml")
        save_dict_to_yaml(agg_results, yaml_model_filename)
        # save_dict_to_csv(agg_results, str(yaml_model_filename).split('.')[0] + '.csv')
        data[model_name] = {**result_dict}

    final_yaml_filename = Path(model_folder, 'eval', "agg-" + model_dir + ".yaml")
    final_csv_filename = Path(model_folder, 'eval', "agg-" + model_dir + ".csv")
    save_dict_to_yaml(data, final_yaml_filename)
    save_agg_dict_to_csv(data, final_csv_filename)
