from pathlib import Path
from stable_baselines3 import PPO
from envs.ma_quadx_chaser_env import MAQuadXChaserEnv
from modules.utils import *

model_path = Path('apps/models/ma_quadx_chaser_20240204-120343/model_39500000.zip')
model_name = model_path.stem
model_folder = model_path.parent

if 'saved_models' in model_folder.parts:
    model_folder = model_folder.parent
elif 'logs' in model_folder.parts:
    model_folder = model_folder.parent.parent

model = PPO.load(model_path)

params_path = f'modules/examples/train_params_test.yaml'
spawn_settings, env_kwargs, train_kwargs = read_yaml_file(params_path)



num_games = 1

print(env_kwargs)

start_pos = np.array([[-7, -5.5, 5],  [0, 0, 4]]) #[5, 4, 2.5],
start_orn = np.zeros_like(start_pos)
spawn_settings['start_pos'] = start_pos
spawn_settings['start_orn'] = start_orn
env_kwargs['start_pos'] = start_pos
env_kwargs['start_orn'] = start_orn

env = MAQuadXChaserEnv(render_mode='human', **env_kwargs)
observations, infos = env.reset(seed=spawn_settings['seed'])

env.aviary.resetDebugVisualizerCamera(cameraDistance=4.24, cameraYaw=-198.40, cameraPitch=-25.60, cameraTargetPosition=[-1.39,-2.66,2.38])
env.aviary.configureDebugVisualizer(env.aviary.COV_ENABLE_WIREFRAME, 0)


while env.agents:
    print(env.aviary.elapsed_time)
    actions = {agent: model.predict(observations[agent], deterministic=True)[0] for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)

    if env.step_count % 5 == 0:

        if env.overlay is None:
            env.overlay = env.render()[..., :3]
        else:
            env.overlay = np.min(np.stack([env.overlay, env.render()[..., :3]], axis=0), axis=0)

    if all(terminations.values()) or all(truncations.values()):
        if env.save_step_data:
            env.write_step_data(Path(model_folder, 'run-data', f'{model_name}.csv'))
            env.write_obs_data(Path(model_folder, 'run-data', f'obs-{model_name}.csv'))

        num_games -= 1
        print(f'Remaining games: {num_games}')

    if num_games == 0:
        print(env.info_counters)
        break

env.close()
