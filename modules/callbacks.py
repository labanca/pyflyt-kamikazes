import os
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



# Define your custom callback class
class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0,):
        super(TensorboardCallback, self).__init__(verbose)




    def _on_step(self) -> bool:
        # self.logger.record('reward', self.CustomEnvironment1.get_attr('total_reward')[0])
        # ep_rewards1 = self.locals['infos']['Seeker1'][0]
        # ep_rewards2 = self.locals['infos']['Seeker2'][0]
        ep_rewards1 = self.locals['rewards'][0]
        ep_rewards2 = self.locals['rewards'][1]
        self.logger.record("rewards1", ep_rewards1)
        self.logger.record("rewards2", ep_rewards2)
        if (self.num_timesteps % 100 == 0):
            self.logger.dump(self.num_timesteps)

        return True

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass


    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    #def __init__(self, env, verbose=0):
        #super().__init__(verbose)

    def init(self, verbose=0):
        super().init(verbose)

    def _on_step(self) -> bool:

        mean_rew_vec_envs = np.array([rew for rew in self.locals['rewards']]).mean()
        self.logger.record("rew_vec_envs", mean_rew_vec_envs)

        if (self.num_timesteps % 1024 == 0):
            self.logger.dump(self.num_timesteps)

        return True