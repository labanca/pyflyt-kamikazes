import glob
import time
from pathlib import Path
from stable_baselines3 import PPO
from envs.ma_quadx_chaser_env import MAQuadXChaserEnv
from modules.utils import *

def evaluate_agent(model_path, custom_params, params_path, eval_mode):

    model_name = model_path.stem
    model_folder = model_path.parent
    scenario_name = params_path.stem

    if not custom_params:
        params_path = f'{model_folder}/{model_name}.yaml'

    if eval_mode == 'rl':
        model = PPO.load(model_path)
    spawn_settings, env_kwargs, train_kwargs = read_yaml_file(params_path)

    env = MAQuadXChaserEnv(render_mode=None, **env_kwargs)
    observations, infos = env.reset(seed=spawn_settings['seed'])

    #print(f'{env_kwargs=}')

    elapsed_games = 1
    num_victories = 0
    episode_summary = []

    csv_path = Path(model_folder, 'ep_data', model_name, eval_mode, f'{scenario_name}.csv')

    if csv_path.is_file():
        csv_path.unlink()

    print(f'{scenario_name=}')
    print(f'{env_kwargs["spawn_settings"]["num_lm"]=}')
    print(f'{env_kwargs["spawn_settings"]["num_lw"]=}')
    print(f'{env_kwargs["lw_stand_still"]=}')
    print(f'{env_kwargs["lw_moves_random"]=}')
    print(f'{env_kwargs["direct_control"]=}')


    start_time = time.time()
    while env.agents:
        if elapsed_games > num_games:
            break

        if eval_mode == 'rl':
            actions = {agent: model.predict(observations[agent], deterministic=True)[0] for agent in env.agents}
        else:
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}

        observations, rewards, terminations, truncations, infos = env.step(actions)

        if all(terminations.values()) or all(truncations.values()):

            env.ep_data['episode'] = elapsed_games
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            episode_summary.append(env.ep_data)
            num_victories += env.ep_data['mission_complete']
            elapsed_games += 1
            observations, infos = env.reset(seed=spawn_settings['seed'])
            print(f'Episodes elapsed: {elapsed_games}')

    for dictionary in episode_summary:
        dictionary['scenario'] = scenario_name
    env.write_eval_data(episode_summary, csv_path)
    end_time = time.time()
    print(f"Win rate: {num_victories/num_games:.2%}")
    print(f"Execution time: {end_time - start_time} seconds")

    env.close()

# -------------------------------------------------

model_path = Path('apps/models/ma_quadx_chaser_20240202-014543/ma_quadx_chaser-30000000.zip')
mode_dir = model_path.parent
eval_mode = 'rl'

custom_params = True
num_games = 100

params_paths = glob.glob(f"{mode_dir}\\eval_scenarios\\{eval_mode}\\*.yaml")

print(params_paths)

params_paths = [params_paths[4]]




for i in range(1):
    for scenario_param in params_paths:

        evaluate_agent(model_path=model_path, custom_params=custom_params, params_path=Path(scenario_param), eval_mode=eval_mode)

    eval_mode = 'dc'

