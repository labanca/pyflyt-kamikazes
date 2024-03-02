from pathlib import Path
from stable_baselines3 import PPO
from envs.ma_quadx_chaser_env import MAQuadXChaserEnv
from modules.utils import *

model_path = Path('apps/models/ma_quadx_chaser_20240208-112945/ma_quadx_chaser-40056448.zip')
model_name = model_path.stem
model_folder = model_path.parent

model = PPO.load(model_path)

params_path = f'modules/video_params.yaml'
spawn_settings, env_kwargs, train_kwargs = read_yaml_file(params_path)

lw_formation_center = [0, 0, 2]


lm_1x5 = np.array([[12, 0, 2]])
lm_5x5 = np.array([[8, 0, 6], [7,-6,4], [7.5, 10, 3], [10, -2, 1.5], [12, 9, 3], [-10, 5, 5] ])


lm_swarm = lm_5x5
lw_squad = generate_formation_pos(formation_center=lw_formation_center, num_drones=5, radius=1.0, min_z=1)

start_pos = np.concatenate([lm_swarm, lw_squad], axis=0)
start_orn = np.zeros_like(start_pos)
spawn_settings['start_pos'] = start_pos
spawn_settings['start_orn'] = start_orn
spawn_settings['num_lm'] = lm_swarm.shape[0]
spawn_settings['num_lw'] = lw_squad.shape[0]
env_kwargs['num_lm'] = lm_swarm.shape[0]
env_kwargs['num_lw'] = lw_squad.shape[0]
env_kwargs['formation_center'] = lw_formation_center
env_kwargs['start_pos'] = start_pos
env_kwargs['start_orn'] = start_orn

env = MAQuadXChaserEnv(render_mode='human', **env_kwargs)
observations, infos = env.reset(seed=spawn_settings['seed'])

#env.aviary.resetDebugVisualizerCamera(cameraDistance=4.24, cameraYaw=-198.40, cameraPitch=-25.60, cameraTargetPosition=[-1.39,-2.66,2.38])
#env.aviary.configureDebugVisualizer(env.aviary.COV_ENABLE_WIREFRAME, 0)


while env.agents:

    actions = {agent: model.predict(observations[agent], deterministic=True)[0] for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)

    # if all(terminations.values() or all(truncations.values())):
    #     break
print(env.info_counters)
env.close()
