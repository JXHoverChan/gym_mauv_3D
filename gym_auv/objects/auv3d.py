import numpy as np
import gym_auv.utils.state_space_3d as ss
import gym_auv.utils.geomutils as geom
import matplotlib.pyplot as plt

from gym_auv.objects.current3d import Current
from mpl_toolkits.mplot3d import Axes3D


def odesolver45(f, y, h, nu_c):
    """Calculate the next step of an IVP of a time-invariant ODE with a RHS
    described by f, with an order 4 approx. and an order 5 approx.
    Parameters:
        f: function. RHS of ODE.
        y: float. Current position.
        h: float. Step length.
    Returns:
        q: float. Order 4 approx.
        w: float. Order 5 approx.
    """

    """
    翻译：
    计算由f描述的时间不变ODE的IVP的下一步，其中f的RHS，具有4阶近似和5阶近似。
    参数：
        f：函数。ODE的RHS。
        y：浮点数。当前位置。
        h：浮点数。步长。
    返回：
        q：浮点数。4阶近似。
        w：浮点数。5阶近似。
    """
    s1 = f(y, nu_c)
    s2 = f(y + h*s1/4.0, nu_c)
    s3 = f(y + 3.0*h*s1/32.0 + 9.0*h*s2/32.0, nu_c)
    s4 = f(y + 1932.0*h*s1/2197.0 - 7200.0*h*s2/2197.0 + 7296.0*h*s3/2197.0, nu_c)
    s5 = f(y + 439.0*h*s1/216.0 - 8.0*h*s2 + 3680.0*h*s3/513.0 - 845.0*h*s4/4104.0, nu_c)
    s6 = f(y - 8.0*h*s1/27.0 + 2*h*s2 - 3544.0*h*s3/2565 + 1859.0*h*s4/4104.0 - 11.0*h*s5/40.0, nu_c)
    w = y  +  h*(25.0*s1/216.0 + 1408.0*s3/2565.0 + 2197.0*s4/4104.0 - s5/5.0)
    q = y  +  h*(16.0*s1/135.0 + 6656.0*s3/12825.0 + 28561.0*s4/56430.0 - 9.0*s5/50.0 + 2.0*s6/55.0)
    return w, q


"""
def odesolver45(f, y, h, nu_c):
    """"""Calculate the next step of an IVP of a time-invariant ODE with a RHS
    described by f, with an order 4 approx. and an order 5 approx.
    Parameters:
        f: function. RHS of ODE.
        y: float. Current position.
        h: float. Step length.
    Returns:
        q: float. Order 4 approx.
        w: float. Order 5 approx.
    """"""
    s1 = f(y, nu_c)
    s2 = f(y + h*s1/5.0, nu_c)
    s3 = f(y + 3.0*h*s1/40.0 + 9.0*h*s2/40.0, nu_c)
    s4 = f(y + 44.0*h*s1/45.0 - 56.0*h*s2/15.0 + 32.0*h*s3/9.0, nu_c)
    s5 = f(y + 19372.0*h*s1/6561.0 - 25360.0*h*s2/2187.0 + 64448.0*h*s3/6561.0 - 212.0*h*s4/729.0, nu_c)
    s6 = f(y + 9017.0*h*s1/3168.0 - 355.0*h*s2/33.0 + 46732.0*h*s3/5247.0 + 49.0*h*s4/176.0 - 5103.0*h*s5/18656.0, nu_c)
    w = y  +  h*(35.0*s1/384.0 + 500.0*s3/1113.0 + 125.0*s4/192.0 - -2187.0*s5/6784.0 + 11.0*s6/84.0)
    q = y  +  h*(16.0*s1/135.0 + 6656.0*s3/12825.0 + 28561.0*s4/56430.0 - 9.0*s5/50.0 + 2.0*s6/55.0)
    return w, q
"""

class AUV3D():
    """
    Implementation of AUV dynamics.     翻译：AUV动力学的实现。
    """
    def __init__(self, step_size, init_eta, safety_radius=1):
        self.state = np.hstack([init_eta, np.zeros((6,))])  # AUV状态，[N, E, D, phi, theta, psi, u, v, w, p, q, r]，前六个是位置，后六个是速度
        self.step_size = step_size  # 步长
        #self.alpha = self.step_size/(self.step_size + 1)
        self.alpha = self.step_size/(self.step_size + 0.2)  # 低通滤波器的系数，用于平滑舵角和升降舵角
        self.input = np.zeros(3)    # 输入，[推进力，舵角，升降舵角]
        self.position_dot = np.zeros(3) # 位置导数
        self.safety_radius = safety_radius  # 安全半径
        self.safety_radius = 1


    def step(self, action, nu_c):
        prev_rudder_angle = self.input[1] # 舵角
        prev_elevator_angle = self.input[2] # 升降舵角

        # Un-normalize actions from neural network
        thrust = _surge(action[0])  # 推进力
        commanded_rudder = _steer(action[1])    # 舵角
        commanded_elevator = _steer(action[2])  # 升降舵角
        # Lowpass filter the rudder and elevator
        rudder_angle = self.alpha*commanded_rudder + (1-self.alpha)*prev_rudder_angle   # 低通滤波器，平滑舵角
        elevator_angle = self.alpha*commanded_elevator + (1-self.alpha)*prev_elevator_angle # 低通滤波器，平滑升降舵角

        #self.input = np.array([thrust, commanded_rudder, commanded_elevator])
        self.input = np.array([thrust, rudder_angle, elevator_angle])   # 输入，[推进力，舵角，升降舵角]
        self._sim(nu_c) # 模拟


    def _sim(self, nu_c):
        #self.state += self.state_dot(nu_c)*self.step_size
        w, q = odesolver45(self.state_dot, self.state, self.step_size, nu_c)    # 用4阶和5阶近似计算下一步
        #self.state = q
        self.state = w
        self.state[3] = geom.ssa(self.state[3])
        self.state[4] = geom.ssa(self.state[4])
        self.state[5] = geom.ssa(self.state[5]) # 归一化
        self.position_dot = self.state_dot(self.state, nu_c)[0:3]


    def state_dot(self, state, nu_c):
        """
        The right hand side of the 12 ODEs governing the AUV dyanmics.   翻译：控制AUV动力学的12个ODE的右手边。
        """
        eta = self.state[:6]    # 位置
        nu_r = self.state[6:]   # 速度

        eta_dot = geom.J(eta).dot(nu_r+nu_c)    # 位置导数
        nu_r_dot = ss.M_inv().dot(  # 速度导数
            ss.B(nu_r).dot(self.input)
            - ss.D(nu_r).dot(nu_r)
            - ss.C(nu_r).dot(nu_r)
            - ss.G(eta))
        state_dot = np.hstack([eta_dot, nu_r_dot])
        return state_dot

    @property
    def position(self):
        """
        Returns an array holding the position of the AUV in NED     翻译：返回一个数组，其中包含AUV在NED中的位置
        coordinates.
        """
        return self.state[0:3]


    @property
    def attitude(self):
        """
        Returns an array holding the attitude of the AUV wrt. to NED coordinates.   翻译：返回一个数组，其中包含AUV相对于NED坐标的姿态。
        """
        return self.state[3:6]

    @property
    def heading(self):
        """
        Returns the heading of the AUV wrt true north.  翻译：返回AUV相对于真北的航向。
        """
        return geom.ssa(self.state[5])

    @property
    def pitch(self):
        """
        Returns the pitch of the AUV wrt NED.   翻译：返回AUV相对于NED的俯仰。
        """
        return geom.ssa(self.state[4])

    @property
    def roll(self):
        """
        Returns the roll of the AUV wrt NED.    翻译：返回AUV相对于NED的滚动。
        """
        return geom.ssa(self.state[3])

    @property
    def relative_velocity(self):
        """
        Returns the surge, sway and heave velocity of the AUV.  翻译：返回AUV的前进、摇摆和起伏速度。
        """
        return self.state[6:9]

    @property
    def relative_speed(self):
        """
        Returns the length of the velocity vector of the AUV.   翻译：返回AUV速度矢量的长度。
        """
        return np.linalg.norm(self.relative_velocity)

    @property
    def angular_velocity(self):
        """
        Returns the rate of rotation about the NED frame.   翻译：返回围绕NED框架的旋转速率。
        """
        return self.state[9:12]
    
    @property
    def chi(self):
        """
        Returns the rate of rotation about the NED frame.   翻译：返回围绕NED框架的旋转速率。
        """
        [N_dot, E_dot, D_dot] = self.position_dot
        return np.arctan2(E_dot, N_dot)
        
    @property
    def upsilon(self):
        """
        Returns the rate of rotation about the NED frame.   翻译：返回围绕NED框架的旋转速率。
        """
        [N_dot, E_dot, D_dot] = self.position_dot
        return np.arctan2(-D_dot, np.sqrt(N_dot**2 + E_dot**2))


def _surge(surge):
    surge = np.clip(surge, 0, 1)
    return surge*ss.thrust_max


def _steer(steer):
    steer = np.clip(steer, -1, 1)
    return steer*ss.rudder_max
