from envs.ma_quadx_chaser_env import MAQuadXChaserEnv
from stable_baselines3 import PPO
import numpy as np
import time

model = PPO.load('apps/models/ma_quadx_chaser_20240105-210345/ma_quadx_chaser-20267584.zip')
#model = PPO.load('apps/resumes/20231230-223741_res_20231230-232053')
seed=None

#print((os.cpu_count() or 1))
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
env_kwargs['start_pos'], env_kwargs['start_orn'], env_kwargs['formation_center'] = MAQuadXChaserEnv.generate_start_pos_orn(**spawn_settings)
env_kwargs['flight_dome_size'] = (spawn_settings['lw_spawn_radius'] + spawn_settings['lm_spawn_radius']
                                  + spawn_settings['lw_center_bounds'] + spawn_settings['lm_center_bounds'])  # dome size 50% bigger than the spawn radius
env_kwargs['seed'] = seed
env_kwargs['spawn_settings'] = spawn_settings
env_kwargs['num_lm'] = spawn_settings['num_lm']
env_kwargs['num_lw'] = spawn_settings['num_lw']
env_kwargs['max_duration_seconds'] = 30.0
env_kwargs['reward_coef'] = 1.0
env_kwargs['lw_stand_still'] = False

env = MAQuadXChaserEnv(render_mode='human', **env_kwargs)
observations, infos = env.reset(seed=seed)

last_term = {}
counters = {'out_of_bounds': 0, 'crashes': 0, 'timeover': 0, 'exploded_target': 0, 'exploded_by_ally': 0,
            'survived': 0, 'ally_collision': 0, 'downed': 0, 'is_success': 0}
first_time = True
num_games = 1

while env.agents:


    #actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    actions = {agent: model.predict(observations[agent], deterministic=True)[0] for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)
    if first_time == True:
        first_time = False

    elif set(terminations.values()) != set(last_term.values()):
        print(f"| An agent terminated |")
        print(f'{terminations=}')
        print(f'{truncations=}')
        print(f'{infos=}\n')

    last_term = terminations

    if any(terminations.values()) or any(truncations.values()):

        for agent_key, agent_data in infos.items():
            for key, value in agent_data.items():
                if key in counters:
                    counters[key] += 1

    if all(terminations.values()) or all(truncations.values()):
        print(f'********* EPISODE END **********\n')
        print(f'{rewards=}\n')
        print(f'{terminations=}')
        print(f'{truncations=}\n')
        print(f'{infos=}\n\n\n')
        #time.sleep(5)
        #env.write_step_data('reward_data.csv')
        #env.plot_agent_rewards('reward_data.csv', 0)
        #env.plot_agent_infos2('reward_data.csv', 0)


        #observations, infos = env.reset(seed=seed)
        num_games -= 1
        print(f'Remaining games: {num_games}')

    if num_games == 0:
        print(counters)
        break


env.close()


