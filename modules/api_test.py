from pettingzoo.test import api_test

from envs.ma_quadx_chaser_env import MAQuadXChaserEnv

env = MAQuadXChaserEnv(spawn_settings=None)
api_test(env, num_cycles=1000, verbose_progress=False)
