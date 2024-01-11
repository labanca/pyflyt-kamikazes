from envs.ma_quadx_chaser_env import MAQuadXChaserEnv
from stable_baselines3 import PPO
import numpy as np
import time





#model = PPO.load('apps/models/ma_quadx_chaser_20240104-195408/ma_quadx_chaser-11682368.zip')
seed=None

#print((os.cpu_count() or 1))
spawn_settings = dict(
    lw_center_bounds=5.0,
    lm_center_bounds=10.0,
    lw_spawn_radius=1.0,
    lm_spawn_radius=10,
    min_z=1.0,
    seed=None,
    num_lw=2,
    num_lm=3,
)

flight_dome_size = spawn_settings['lw_spawn_radius'] + spawn_settings['lm_spawn_radius'] + spawn_settings['lw_center_bounds'] + spawn_settings['lm_center_bounds'],
start_pos = np.array([ [4, 4, 1], [7,7, 1], [12, 7, 1], [0, 0, 1], [10, 10, 1]])
start_orn = np.zeros_like(start_pos)
formation_center = np.array([0, 0, 1])

env_kwargs = dict(
    start_pos=start_pos,
    start_orn=start_orn,
    formation_center=formation_center,
    flight_dome_size=flight_dome_size,
    seed=None,
    spawn_settings=None,
    num_lm=spawn_settings['num_lm'],
    num_lw=spawn_settings['num_lm'],
    max_duration_seconds=30,
    lw_moves_random=False,
    lw_stand_still=True,
    lw_chases=False,
    lw_attacks=False,
    lw_threat_radius=4.0,
    lw_shoot_range=1.0,
    agent_hz=30,
    max_velocity_magnitude=10,
    rew_exploding_target=100,
    distance_factor=0.5,
    speed_factor=0.1,

)

env = MAQuadXChaserEnv(render_mode='human', **env_kwargs)
observations, infos = env.reset(seed=seed)

last_term = {}
counters = {'out_of_bounds': 0, 'crashes': 0, 'timeover': 0, 'exploded_target': 0, 'exploded_by_ally': 0,
            'survived': 0, 'ally_collision': 0, 'downed': 0, 'is_success': 0}
first_time = True
num_games = 1
i = 1
last_start_pos = env_kwargs['start_pos']
while env.agents:


    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    #actions = {agent: model.predict(observations[agent], deterministic=True)[0] for agent in env.agents}


    actions['agent_0'] = np.array([-3, -3, 0, 0.8]) # np.array([i, i, 0, 0.123*i])
    actions['agent_1'] = np.array([4, 4, 0, 0.8])
    actions['agent_2'] = np.array([-5, -2, 0, 0.8])
    #actions['agent_3'] = np.array([0, 0, 0, 0])
    i +=1

    observations, rewards, terminations, truncations, infos = env.step(actions)


    if first_time == True:
        first_time = False


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
        #env.plot_agent_rewards('reward_data.csv', 0)
        #env.plot_agent_infos2('reward_data.csv', 0)
        observations, infos = env.reset(seed=seed)
        num_games -= 1
        print(f'Remaining games: {num_games}')


    if num_games == 0:
        print(counters)
        print(f'{i=}')
        break

env.close()


