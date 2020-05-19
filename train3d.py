import os
import gym
import gym_auv
import stable_baselines.results_plotter as results_plotter
import numpy as np
import tensorflow as tf

from stable_baselines.bench import Monitor
from stable_baselines.common.policies import MlpPolicy, LstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.schedules import LinearSchedule
from utils import parse_experiment_info


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
scenarios = ["beginner", "intermediate", "proficient", "advanced", "expert"]
hyperparams = {
    'n_steps': 1024,
    'nminibatches': 256,
    'learning_rate': 1e-5,
    'nminibatches': 32,
    'lam': 0.95,
    'gamma': 0.99,
    'noptepochs': 4,
    'cliprange': 0.2,
    'ent_coef': 0.01,
    'verbose': 2
    }


def callback(_locals, _globals):
    global n_steps, best_mean_reward
    if (n_steps + 1) % 5 == 0:
        _locals['self'].save(os.path.join(agents_dir, "model_" + str(n_steps+1) + ".pkl"))
    n_steps += 1
    return True


if __name__ == '__main__':
    experiment_dir, _, _ = parse_experiment_info()
    
    for i, scen in enumerate(scenarios):
        agents_dir = os.path.join(experiment_dir, scen, "agents")
        tensorboard_dir = os.path.join(experiment_dir, scen, "tensorboard")
        os.makedirs(agents_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)
        hyperparams["tensorboard_log"] = tensorboard_dir

        num_envs = 4
        if num_envs > 1:
            env = SubprocVecEnv([lambda: Monitor(gym.make("PathColav3d-v0", scenario=scen), agents_dir, allow_early_resets=True) for i in range(num_envs)])
        else:
            env = DummyVecEnv([lambda: Monitor(gym.make("PathColav3d-v0", scenario=scen), agents_dir, allow_early_resets=True)])

        if scen == "beginner":
            agent = PPO2(MlpPolicy, env, **hyperparams)
        else:
            continual_model = os.path.join(experiment_dir, scenarios2[i+1], "agents", "last_model.pkl")
            agent = PPO2.load(continual_model, env=env, **hyperparams)
            agent.setup_model()
        best_mean_reward, n_steps, timesteps = -np.inf, 0, int(300e3 + i*150e3)
        agent.learn(total_timesteps=timesteps, tb_log_name="PPO2", callback=callback2)
        save_path = os.path.join(agents_dir, "last_model.pkl")
        agent.save(save_path)