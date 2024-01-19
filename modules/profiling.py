import cProfile
from pathlib import Path

from gymnasium.utils import EzPickle

from apps.resume_train import train_butterfly_supersuit
from envs.ma_quadx_chaser_env import MAQuadXChaserEnv
from modules.utils import save_dicts_to_yaml, read_yaml_file


class EZPEnv(EzPickle, MAQuadXChaserEnv):
    def __init__(self, *args, **kwargs):
        EzPickle.__init__(self, *args, **kwargs)
        MAQuadXChaserEnv.__init__(self, *args, **kwargs)


def main():

    env_fn = EZPEnv

    train_desc = """ take ma_quadx_chaser_20240116-142312 and train with the speed vector reward with diffent speed_factor values, trying to gain overall speed"""

    params_path = 'apps/train_params.yaml'
    spawn_settings, env_kwargs, train_kwargs = read_yaml_file(params_path)

    root_dir = 'apps/models'
    model_dir = 'ma_quadx_chaser_20240117-174627'
    model_name = 'ma_quadx_chaser-6063232.zip'

    steps = 100_000
    num_resumes = 2
    reset_model = False

    for i in range(num_resumes):
        model_name, model_dir = train_butterfly_supersuit(
            env_fn=env_fn, steps=steps, train_desc=train_desc,
            model_name=model_name, model_dir=model_dir,
            env_kwargs=env_kwargs, train_kwargs=train_kwargs)

        env_kwargs['reward_type'] = 2
        env_kwargs['thrust_limit'] = 10
        env_kwargs['explosion_radius'] = 0.0

        save_dicts_to_yaml(spawn_settings, env_kwargs, train_kwargs,
                           Path(root_dir, model_dir, f'{model_name.split(".")[0]}.yaml'))

        model_name = 'a'


if __name__ == "__main__":
    # Create a cProfile object
    profiler = cProfile.Profile()

    # Start profiling
    profiler.enable()

    # Run your main code
    main()

    # Stop profiling
    profiler.disable()

    # Print the profiling results
    profiler.print_stats()
    profiler.dump_stats('cprofiler.csv')
