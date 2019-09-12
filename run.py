import os
import sys
import subprocess
import numpy as np
from time import time, sleep
import argparse
import tensorflow as tf
import gym
import gym_auv

from stable_baselines.common import set_global_seeds
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import VecVideoRecorder, DummyVecEnv, SubprocVecEnv
from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines import PPO2, DDPG
from pyglet.window import key

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def _preprocess_custom_envconfig(rawconfig):
    custom_envconfig = dict(zip(args.envconfig[::2], args.envconfig[1::2]))
    for key in custom_envconfig:
        try:
            custom_envconfig[key] = float(custom_envconfig[key])
            if (custom_envconfig[key] == int(custom_envconfig[key])):
                custom_envconfig[key] = int(custom_envconfig[key])
        except ValueError:
            pass
    return custom_envconfig

def create_env(env_id, custom_envconfig):
    envconfig = gym_auv.SCENARIOS[env_id.split(':')[-1]]['config']
    envconfig.update(custom_envconfig)
    env = gym.make(env_id, env_config=envconfig)
    return env

def make_mp_env(env_id, rank, custom_envconfig, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = create_env(env_id, custom_envconfig)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init


def play_scenario(env, args):


    key_input = np.array([0.0, 0.0])

    def key_press(k, mod):
        if k == key.DOWN:  key_input[0] = -1
        if k == key.UP:    key_input[0] = 1
        if k == key.LEFT:  key_input[1] = -1
        if k == key.RIGHT: key_input[1] = 1

    def key_release(k, mod):
        nonlocal restart, quit
        if k == key.R:
            restart = True
            print('Restart')
        if k == key.Q:
            quit = True
            print('quit')
        if k == key.UP:    key_input[0] = 0
        if k == key.DOWN:  key_input[0] = 0
        if k == key.LEFT and key_input[1] != 0: key_input[1] = 0
        if k == key.RIGHT and key_input[1] != 0: key_input[1] = 0

    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

    try:
        while True:
            a = np.array([0.0, 0.0])
            t = time()
            restart = False
            steps = 0
            quit = False
            while True:
                t, dt = time(), time()-t
                a[0] = key_input[0]
                a[1] = key_input[1]

                obs, r, done, info = env.step(a)
                if (steps % 1 == 0 or done) and args.verbose > 0:
                    print(', '.join('{:.2f}'.format(x) for x in obs))
                env.render()
                steps += 1

                if quit: raise KeyboardInterrupt
                if done or restart: break
            
            env.reset()

    except KeyboardInterrupt:
        pass

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'mode',
        help='Which program mode to run.',
        choices=['play', 'train', 'enjoy', 'test'],
    )
    parser.add_argument(
        'env',
        help='Name of the gym environment to run.',
        choices=gym_auv.SCENARIOS.keys(),
    )
    parser.add_argument(
        '--agent',
        help='Path to the RL agent to simulate.',
    )
    parser.add_argument(
        '--video-dir',
        help='Dir for output video.',
        default='./logs/videos/'
    )
    parser.add_argument(
        '--video-name',
        help='Name of output video.',
        default='latest'
    )
    parser.add_argument(
        '--length',
        help='Timesteps to simulate.',
        type=int,
        default=8000
    )
    parser.add_argument(
        '--mp',
        help='Whether to use multiple CPU cores for training.',
        action='store_true'
    )
    parser.add_argument(
        '--deterministic',
        help='Whether to use use deterministic policy for testing.',
        action='store_true'
    )
    parser.add_argument(
        '--episodes',
        help='Number of episodes to simulate in test mode.',
        type=int,
        default=5
    )
    parser.add_argument(
        '--verbose',
        help='How much to print.',
        type=int,
        default=1
    )
    parser.add_argument(
        '--envconfig',
        help='Override environment config parameters.',
        nargs='*'
    )
    args = parser.parse_args()
    custom_envconfig = _preprocess_custom_envconfig(args.envconfig) if args.envconfig is not None else {}
    NUM_CPU = 8
    TIMESTAMP = str(int(time()))
    env_id = 'gym_auv:' + args.env

    if (args.mode == 'play'):
        env = create_env(env_id, custom_envconfig)
        play_scenario(env, args)


    elif (args.mode == 'enjoy'):
        agent = PPO2.load(args.agent)
        env = create_env(env_id, custom_envconfig)
        vec_env = DummyVecEnv([lambda: env])
        recorded_env = VecVideoRecorder(vec_env, args.video_dir, record_video_trigger=lambda x: x == 0, 
            video_length=args.length, name_prefix=args.video_name
        )
        obs = recorded_env.reset()
        for i in range(args.length):
            action, _states = agent.predict(obs)
            obs, reward, done, info = recorded_env.step(action)
            recorded_env.render()
        recorded_env.env.close()


    elif (args.mode == 'train'):
        video_folder = os.path.join(DIR_PATH, 'logs', 'videos', args.env, TIMESTAMP)
        video_length = 8000
        os.makedirs(video_folder, exist_ok=True)
        agent_folder = os.path.join(DIR_PATH, 'logs', 'agents', args.env, TIMESTAMP)
        os.makedirs(agent_folder, exist_ok=True)
        tensorboard_log = os.path.join('tensorboard', args.env)
        tensorboard_port = 6006

        if (args.mp):
            num_cpu = NUM_CPU
            vec_env = SubprocVecEnv([make_mp_env(env_id, i, custom_envconfig) for i in range(num_cpu)])
        else:
            num_cpu = 1
            vec_env = DummyVecEnv([lambda: create_env(env_id, custom_envconfig)])
            vec_env = VecVideoRecorder(vec_env, args.video_dir, record_video_trigger=lambda x: x % 10000 == 0, 
                video_length=args.length, name_prefix=args.env
            )

        if (args.agent is not None):
            agent = PPO2.load(args.agent)
            agent.set_env(vec_env)
        else:
            # hyperparams = {
            #     'n_steps': 1024,
            #     'nminibatches': 4,
            #     #'learning_rate': 5e-5,
            #     'lam': 0.98,
            #     'gamma': 0.999,
            #     'noptepochs': 4,
            #     'ent_coef': 0.01,
            #     # 'policy_kwargs': dict(
            #     #     net_arch=[64, 64, 64],
            #     #     act_fun=tf.nn.relu
            #     # )
            # }
            hyperparams = {
                'n_steps': 1024,
                'nminibatches': 32,
                'lam': 0.98,
                'gamma': 0.999,
                'noptepochs': 4,
                'ent_coef': 0.01
            }
            
            agent = PPO2(MlpPolicy, vec_env, verbose=1, tensorboard_log=tensorboard_log, **hyperparams)
        
        print('Training {} agent on "{}"'.format('PPO', env_id))

        n_updates = 0
        def callback(_locals, _globals):
            global n_updates

            total_t_steps = _locals['self'].get_env().get_attr('total_t_steps')[0]*num_cpu
            agent_filepath = os.path.join(agent_folder, str(total_t_steps) + '.pkl')
            _locals['self'].save(agent_filepath)

            if (n_updates % 25 == 0 and args.mp):
                cmd = 'python run.py enjoy {} --agent "{}" --video-dir "{}" --video-name "{}" --length {}' .format(
                    args.env, agent_filepath, video_folder, args.env + '-' + str(total_t_steps), video_length
                )
                subprocess.Popen(cmd)
        
            n_updates += 1
            
        agent.learn(
            total_timesteps=10000000, 
            tb_log_name='log',
            callback=callback
        )


    elif (args.mode == 'test'):
        figure_folder = os.path.join(DIR_PATH, 'logs', 'figures', args.env, TIMESTAMP)
        os.makedirs(figure_folder, exist_ok=True)
        agent = PPO2.load(args.agent)
        env = create_env(env_id, custom_envconfig)
        vec_env = DummyVecEnv([lambda: env])

        print('Testing scenario "{}" for {} episodes.\n '.format(args.env, args.episodes))
        report_msg_header = '{:<20}{:<20}{:<20}{:<20}'.format('Episode', 'Time Steps', 'Tot. Reward', 'Progress')
        print(report_msg_header)
        print('-'*len(report_msg_header))

        for episode in range(args.episodes):
            obs = vec_env.reset()
            cumulative_reward = 0
            t_steps = 0
            while 1:
                action, _states = agent.predict(obs)
                obs, reward, done, info = vec_env.step(action)
                t_steps += 1
                cumulative_reward += reward[0]
                report_msg = '{:<20}{:<20}{:<20.2f}{:<20.2%}\r'.format(episode, t_steps, cumulative_reward, info[0]['progress'])
                sys.stdout.write(report_msg)
                sys.stdout.flush()
                if (done):
                    env.plot(fig_dir=figure_folder, fig_name=args.env + '_ep_{}'.format(episode))
                    print()
                    break