from envs.ma_quadx_chaser_env import MAQuadXHoverEnv
from stable_baselines3 import PPO
import numpy as np
import time

model = PPO.load('C:/projects/pyflyt-kamikazes/apps/models/ma_quadx_hover_20231227-205852.zip')
seed=None

#print((os.cpu_count() or 1))
spawn_settings = dict(
    lw_center_bounds=5.0,
    lm_center_bounds=10.0,
    lw_spawn_radius=1.0,
    lm_spawn_radius=10,
    min_z=1.0,
    seed=None,
    num_lw=3,
    num_lm=2,
)


env_kwargs = {}
env_kwargs['start_pos'] = np.array([ [3.5, 0 ,1], [5, 5, 5],  [0.5, 0, 1], [0, 0.5, 1], [0, 0, 1] ])
env_kwargs['start_orn'] = np.zeros_like(env_kwargs['start_pos'])
env_kwargs['formation_center'] = np.array([0, 0, 1])
env_kwargs['flight_dome_size'] = (spawn_settings['lw_spawn_radius'] + spawn_settings['lm_spawn_radius'] + spawn_settings['lw_center_bounds']) * 2.5  # dome size 50% bigger than the spawn radius
env_kwargs['seed'] = seed
env_kwargs['spawn_settings'] = None
env_kwargs['num_lm'] = spawn_settings['num_lm']
env_kwargs['num_lw'] = spawn_settings['num_lw']

env = MAQuadXHoverEnv(render_mode='human', **env_kwargs)
observations, infos = env.reset(seed=seed)

last_term = {}
counters = {'out_of_bounds': 0, 'crashes': 0, 'timeover': 0, 'exploded_target': 0, 'mission_complete': 0, 'ally_collision': 0,
            'exploded_by_ally': 0, 'downed': 0}
first_time = True
num_games = 1
i = 1
last_start_pos = env_kwargs['start_pos']
while env.agents:


    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    #actions = {agent: model.predict(observations[agent], deterministic=True)[0] for agent in env.agents}


    actions['agent_0'] = np.array([2, 0, 0, 0.15]) # np.array([i, i, 0, 0.123*i])
    actions['agent_1'] = np.array([0, 0, 0, 0])
    #actions['agent_2'] = np.array([0, 0, 0, 0])
    i +=1
    observations, rewards, terminations, truncations, infos = env.step(actions)

    if first_time == True:
        first_time = False

    # elif set(terminations.values()) != set(last_term.values()) and (len(terminations.values()) == len(last_term.values())):
    #     print(f"| An agent terminated |")
    #     print(f'{terminations=}')
    #     print(f'{truncations=}')
    #     print(f'{infos=}\n')
    #
    # last_term = terminations



    if any(terminations.values()) or any(truncations.values()):

        for agent_key, agent_data in infos.items():
            for key, value in agent_data.items():
                if key in counters:
                    counters[key] += 1

    if all(terminations.values()) or all(truncations.values()):
        print(f'********* EPISODE END **********\n')
        print(f'{rewards=}\n')
        print(f'{terminations=} {truncations=}\n')
        print(f'{infos=}\n\n\n')
        time.sleep(0)
        env.write_step_data('reward_data.csv')
        #env.plot_rewards_data('reward_data.csv')
        env.plot_agent_rewards('reward_data.csv', 0)
        env.plot_agent_infos2('reward_data.csv', 0)
        observations, infos = env.reset(seed=seed)
        num_games -= 1
        print(f'Remaining games: {num_games}')


    if num_games == 0:
        print(counters)
        print(f'{i=}')
        break

env.close()


