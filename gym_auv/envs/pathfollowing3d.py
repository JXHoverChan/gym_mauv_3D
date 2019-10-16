import gym
import numpy as np
import matplotlib.pyplot as plt
import gym_auv.utils.geomutils as geom

from math import inf
from mpl_toolkits.mplot3d import Axes3D
from gym_auv.objects.path3d import Path3D, generate_random_waypoints
from gym_auv.objects.auv3d import AUV3D


env_config = {
    "reward_ds": 1,
    "reward_speed_error": -0.08,
    "reward_cross_track_error": -100,
    "reward_vertical_track_error": -100,
    "reward_heading_error": -10,
    "reward_pitch_error": -10,
    "reward_rudderchange": -0.01,
    "t_step": 0.1,
    "cruise_speed": 1.5,
    "la_dist": 50,
    "min_reward": -500,
    "max_timestemps": 10000}


class PathFollowing3d(gym.Env):
    def __init__(self, env_config):
        self.config = env_config
        self.nstates = 7
        self.n_observations = self.nstates
        self.vessel = None
        self.path = None

        self.np_random = None

        self.reward = 0
        self.path_prog = []
        self.past_actions = []
        self.past_obs = []
        self.t_step = self.config["t_step"]
        self.total_t_steps = 0

        self.action_space = gym.spaces.Box(low=np.array([0, -1, -1]),
                                           high=np.array([1, 1, 1]),
                                           dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.array([-inf]*self.n_observations),
            high=np.array([inf]*self.n_observations),
            dtype=np.float32)

        self.generate()


    def generate(self):
        nwaypoints = np.random.randint(10,15)
        waypoints = generate_random_waypoints(nwaypoints)
        self.path = Path3D(waypoints)
        init_pos = self.path(0) + 0*np.random.rand(3)
        init_angle = np.array([0, self.path.elevation_angles[0], self.path.azimuth_angles[0]])
        initial_state = np.hstack([init_pos, init_angle])

        self.vessel = AUV3D(self.t_step, initial_state)
        self.path_prog.append(self.path.get_closest_s(self.vessel.position))


    def step(self, action):
        action = np.clip(action, np.array([0, -1, -1]), np.array([1, 1, 1]))
        self.past_actions.append(action)
        self.vessel.step(action)

        prog = self.path.get_closest_s(self.vessel.position)
        self.path_prog.append(prog)

        obs = self.observe()
        self.past_obs.append(obs)

        done, step_reward = self.step_reward()
        info = {}

        self.total_t_steps += 1

        return obs, step_reward, done, info


    def observe(self):
        la_distance = self.config["la_dist"]
        la_point = self.path_prog[-1] + la_distance
        la_azimuth_angle, la_elevation_angle = self.path.get_direction(la_point)

        vessel_position = self.vessel.position
        path_position = self.path.get_closest_point(vessel_position)
        tracking_errors = geom.Rzyx(0, la_elevation_angle, la_azimuth_angle).dot(vessel_position-path_position)

        along_track_distance = tracking_errors[0]
        cross_track_error = tracking_errors[1]
        vertical_track_error = tracking_errors[2]

        azimuth_error = np.arctan2(-cross_track_error, la_distance)
        elevation_error = np.arctan2(vertical_track_error, np.sqrt(cross_track_error**2 + la_distance**2))
        
        obs = self.vessel.velocity
        obs = np.vstack([*obs, azimuth_error, elevation_error, cross_track_error, vertical_track_error])
        return np.reshape(obs, (7,))


    def step_reward(self):
        obs = self.past_obs[-1]
        done = False
        step_reward = 0

        delta_path_prog = self.path_prog[-1] - self.path_prog[-2]
        max_prog = self.config["cruise_speed"]*self.t_step


        speed_error = self.vessel.speed - self.config["cruise_speed"]
        relative_progress = delta_path_prog/max_prog
        endpoint_distance = np.linalg.norm(self.vessel.position - self.path.get_endpoint())

        step_reward += relative_progress * self.config["reward_ds"]
        step_reward += speed_error * self.config["reward_speed_error"]
        step_reward += obs[-3] * self.config["reward_heading_error"]
        step_reward += obs[-4] * self.config["reward_pitch_error"]
        step_reward += obs[-5] * self.config["reward_cross_track_error"]
        step_reward += obs[-6] * self.config["reward_vertical_track_error"]

        self.reward = step_reward

        if step_reward < self.config["min_reward"] or endpoint_distance < 10:
            done = True

        return done, step_reward


    def reset(self):
        self.vessel = None
        self.path = None
        self.reward = 0
        self.path_prog = []
        self.past_actions = []
        self.past_obs = []

        if self.np_random is None:
            self.seed()

        self.generate()
        obs = self.observe()
        self.past_obs.append(obs)
        return obs


    def seed(self, seed=5):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]


if __name__ == "__main__":
    env = PathFollowing3d(env_config)
    ax1 = env.path.plot_path(label="Path")
    ax1.scatter3D(*env.vessel.position)
    ax1.legend()
    plt.show()
    env.reset()
    ax1 = env.path.plot_path(label="Path")
    ax1.legend()
    plt.show()