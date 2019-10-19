import numpy as np
import gym_auv.utils.geomutils as geom

class Current():
    def __init__(self, mu, Vmin, Vmax, Vc_init, alpha_init, beta_init, t_step_size):
        self.mu = mu
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.Vc = Vc_init
        self.alpha = alpha_init
        self.beta = beta_init
        self.t_step_size = t_step_size
       

    def __call__(self, eta):
        """Returns the current velcotiy in {b} to use in in AUV kinematics"""
        phi = eta[3]
        theta = eta[4]
        psi = eta[5]

        vc_n = np.array([self.Vc*np.cos(self.alpha)*np.cos(self.beta), self.Vc*np.sin(self.beta), self.Vc*np.sin(self.alpha)*np.cos(self.beta)])
        vc_b = np.transpose(geom.Rzyx(phi, theta, psi)).dot(vc_n)
        vc_dot = -geom.S_skew(vc_b)

        nu_c = np.array([*vc_b, 0, 0, 0])

        return nu_c 


    def sim(self):
        w = np.random.normal(0, 1)
        if self.Vc <= self.Vmax and self.Vc >= self.Vmin:
            Vc_dot = -self.mu*self.Vc + w
        else:
            Vc_dot = 0
        self.Vc += Vc_dot*self.t_step_size