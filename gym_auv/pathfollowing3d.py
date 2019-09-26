import gym
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from objects.path3d import ParamCurve3d

class PathFollowing3d(gym.Env):
    def __init__(self, env_config):
        self.config = env_config
        self.nstates = 12
        self.n_observations = self.nstates
        self.vessel = None
        self.path = None

        self.np_random = None

        self.reward = 0
        self.path_prog = None
        self.past_actions = None
        self.past_obs = None
        self.t_step = None

        self.action_space = gym.spaces.Box(low=np.array([0, -1, -1]),
                                           high=np.array([1, 1, 1]),
                                           dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.array([-1]*self.n_observations),
            high=np.array([1]*self.n_observations),
            dtype=np.float32)

        self.generate()

    def generate(self):
        waypoints = [(0,0,0), (1,1,0), (3,3,2)]
        self.path = ParamCurve3d(waypoints, 10000)
        init_pos = self.path.path_coords[0]
        init_angle = self.path.psi, self.path.theta
        """
        init_pos[0] += 50*(self.np_random.rand()-0.5)
        init_pos[1] += 50*(self.np_random.rand()-0.5)
        init_angle = geom.princip(init_angle
                                  + 2*np.pi*(self.np_random.rand()-0.5))
        self.t_step = self.config["t_step"]
        self.vessel = AUV2D(self.t_step,
                            np.hstack([init_pos, init_angle]))
        self.path_prog = np.array([
            self.path.get_closest_arclength(self.vessel.position)])"""

    def step(self, action):
        action = np.clip(action, np.array([0, -1, -1]), np.array([1, 1, 1]))
        self.past_actions = np.vstack([self.past_actions, action])
        self.vessel.step(action)

        prog = self.path.get_closest_arclength(self.vessel.position)
        self.path_prog = np.append(self.path_prog, prog)

        obs = self.observe()
        self.past_obs = np.vstack([self.past_obs, obs])
        done, step_reward = self.step_reward()
        info = {}

        return obs, step_reward, done, info

    def reset(self):
        self.vessel = None
        self.path = None
        self.reward = 0
        self.path_prog = None
        self.past_actions = np.array([[0, 0]])
        self.t_step = None

        if self.np_random is None:
            self.seed()

        self.generate()
        obs = self.observe()
        self.past_obs = np.array([obs])
        return obs


if __name__ == "__main__":
    env = PathFollowing3d({})
    ax = env.path.plot_path()
    plt.show()