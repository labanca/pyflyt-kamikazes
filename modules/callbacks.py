import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy

def last_non_zero(arr):
    non_zero_indices = np.nonzero(arr)[0]
    return arr[non_zero_indices[-1]] if non_zero_indices.size > 0 else 0


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    # def __init__(self, env, verbose=0):
    # super().__init__(verbose)

    def __init__(self, save_path, save_freq, check_freq, log_dir, verbose=1):
        super().__init__(verbose)
        self.last_100_episode_rewards = []
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.models_path = save_path
        self.save_freq = save_freq

        self.rew_vec_envs = 0
        self.mean_reward_last_100 = 0
        self.best_mean_reward = -np.inf
        self.last_ep_mean_rew = -np.inf
        self.last_100_episode_rewards = []


    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.env_episode_rewards= [[] for _ in range(len(self.training_env.venv.idx_starts)-1)]

        return True


    def _on_step(self) -> bool:
        self.rew_vec_envs += self.locals['rewards']
        # self.logger.record("rew_vec_envs", self.mean_rew_vec_envs)

        idx = self.training_env.venv.idx_starts

        env_dones = np.array([self.locals['dones'][idx[i]:idx[i + 1]] for i in range(len(idx) - 1)])

        if self.num_timesteps % (self.model.n_steps) == 0:
            current_mean_reward_vec_env = sum(self.rew_vec_envs) / self.training_env.num_envs
            self.logger.record("rew_vec_envs", current_mean_reward_vec_env)
            self.rew_vec_envs = 0

            ep_total_rew = np.concatenate([array for sublist in self.env_episode_rewards for array in sublist])
            ep_mean_rew = np.mean(ep_total_rew)
            self.logger.record("ep_mean_rew", ep_mean_rew)

            self.last_ep_mean_rew = ep_mean_rew

        for id, dones in enumerate(env_dones):
            if dones.all():

                # take the last reward int the rollout buffer of the episode for that env
                env_rollout_rewards = self.locals['rollout_buffer'].rewards[:, idx[id]:idx[id+1]]
                ep_last_rewards = np.array([last_non_zero(arr) for arr in env_rollout_rewards.T], dtype=np.float64)

                self.env_episode_rewards[id].append(ep_last_rewards)
                if len(self.env_episode_rewards[id]) > 100:
                    self.env_episode_rewards[id] = self.env_episode_rewards[id][-100:]



        # Save the model every `save_freq` steps
        if self.num_timesteps % self.save_freq == 0:
            save_name = os.path.join(self.models_path, 'saved_models', f'model_{self.num_timesteps}.zip')
            self.model.save(save_name)
            print(f"Model saved at timestep: {self.num_timesteps}")

        # try this later _locals['self'].num_timesteps https://github.com/hill-a/stable-baselines/issues/62#issuecomment-565665707
        if self.n_calls % self.check_freq == 0:

            # New best model, you could save the agent here
            if self.last_ep_mean_rew > self.best_mean_reward:
                self.best_mean_reward = self.last_ep_mean_rew
                # Example for saving best model
                if self.verbose > 0:
                    print(f"Saving new best model at {self.num_timesteps} with mean reward {self.best_mean_reward}")
                    print(f"Saving new best model to {self.save_path}-{self.num_timesteps}.zip")
                self.model.save(self.save_path)

        return True
