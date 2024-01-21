import os
import sys

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam


class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
            "device": self.model.device,
            "etc": self.model.get_parameters()
        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0.0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    #def __init__(self, env, verbose=0):
        #super().__init__(verbose)

    def __init__(self, verbose=0):
        super().__init__(verbose)
        #super().init(verbose)
        self.mean_rew_vec_envs = 0

    def _on_step(self) -> bool:

        self.mean_rew_vec_envs += self.locals['rewards']
        #self.logger.record("rew_vec_envs", self.mean_rew_vec_envs)

        if self.num_timesteps % (self.model.n_steps) == 0:
            self.logger.record("rew_vec_envs", sum(self.mean_rew_vec_envs)/self.training_env.num_envs)
            #self.logger.dump(self.num_timesteps)
            self.mean_rew_vec_envs = 0

        #try this later _locals['self'].num_timesteps https://github.com/hill-a/stable-baselines/issues/62#issuecomment-565665707

        return True

