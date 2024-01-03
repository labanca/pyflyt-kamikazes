from gymnasium.utils import EzPickle
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
import supersuit as ss

from envs.ma_quadx_chaser_env import MAQuadXChaserEnv
from stable_baselines3 import PPO
import numpy as np
import time


class EZPEnv(EzPickle, MAQuadXChaserEnv):
    def __init__(self, *args, **kwargs):
        EzPickle.__init__(self, *args, **kwargs)
        MAQuadXHoverEnv.__init__(self, *args, **kwargs)

def eval(env_fn, n_eval_episodes: int = 100, num_vec_envs: int =1, model_name: str = '',
         render_mode: str | None = None, **kwargs):
    # Train a single model to play as each agent in a cooperative Parallel environment
    env = env_fn(**kwargs)

    env.reset(seed=kwargs['seed'])

    print(f"Starting eval on {str(env.metadata['name'])}.")

    env = ss.black_death_v3(env,)
    env = ss.pettingzoo_env_to_vec_env_v1(env )

    num_cpus = num_vec_envs
    env = ss.concat_vec_envs_v1(env, num_vec_envs, num_cpus=num_cpus, base_class="stable_baselines3")

    model = PPO.load(model_name)
    # new_logger = configure('apps/logs', ["csv", "tensorboard"])
    # model.set_logger(new_logger)

    deterministic = True
    episode_rewards, dontknow = evaluate_policy(
                    model,
                    env,
                    render=False,
                    n_eval_episodes=n_eval_episodes,
                    return_episode_rewards=True,
                    deterministic=deterministic,
                )

    print(episode_rewards)
    print(dontknow)

if __name__ == "__main__":

    env_fn = EZPEnv
    spawn_settings = dict(
        lw_center_bounds=5.0,
        lm_center_bounds=5.0,
        lw_spawn_radius=1.0,
        lm_spawn_radius=5.0,
        min_z=1.0,
        seed=None,
        num_lw=3,
        num_lm=6,
    )

    env_kwargs = {}
    env_kwargs['start_pos'], env_kwargs['start_orn'], env_kwargs[
        'formation_center'] = MAQuadXChaserEnv.generate_start_pos_orn(**spawn_settings)
    env_kwargs['flight_dome_size'] = (spawn_settings['lw_spawn_radius'] + spawn_settings['lm_spawn_radius'] + spawn_settings['lw_center_bounds']) * 2.5  # dome size 50% bigger than the spawn radius
    env_kwargs['seed'] = spawn_settings['seed']
    env_kwargs['spawn_settings'] = spawn_settings
    env_kwargs['num_lm'] = spawn_settings['num_lm']
    env_kwargs['num_lw'] = spawn_settings['num_lw']
    counters = {'out_of_bounds': 0, 'crashes': 0, 'timeover': 0, 'exploded_target': 0, 'mission_complete': 0, 'ally_collision': 0,
                'exploded_by_ally': 0, 'downed': 0}


    n_eval_episodes = 10
    model_name = 'C:/projects/pyflyt-kamikazes/apps/models/ma_quadx_hover_20240103-103330.zip'

    eval(env_fn, n_eval_episodes=n_eval_episodes, num_vec_envs=1, model_name=model_name,  **env_kwargs)





