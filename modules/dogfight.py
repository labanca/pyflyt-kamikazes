from modules.envs_test.ma_fixedwing_dogfight_env import MAFixedwingDogfightEnv

env = MAFixedwingDogfightEnv(render_mode="human", )
observations, infos = env.reset()

while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()


