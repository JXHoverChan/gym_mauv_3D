import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt

from scipy.optimize import fminbound
from mpl_toolkits.mplot3d import Axes3D


class Path3D():
    def __init__(self, waypoints):
        self.waypoints = np.array(waypoints)
        self.nwaypoints = len(waypoints)
        self.segment_lengths = self._get_path_lengths()
        self.length = np.sum(self.segment_lengths) 
        self.azimuth_angles = np.array([])
        self.elevation_angles = np.array([])

        self._get_parametric_params()


    def __call__(self, s):
        seg_start, seg_index = self._get_segment_start(s)
        alpha = self.azimuth_angles[seg_index]
        beta = self.elevation_angles[seg_index]
        seg_distance = s - seg_start
        x_start, y_start, z_start = self.waypoints[seg_index]

        x = x_start + seg_distance*np.cos(alpha)*np.cos(beta)
        y = y_start + seg_distance*np.sin(alpha)*np.cos(beta)
        z = z_start - seg_distance*np.sin(beta)

        return np.array([x,y,z])


    def _get_segment_start(self, s):
        seg_start = 0
        for i, sl in enumerate(self.segment_lengths):
            if s <= seg_start+sl:
                return seg_start, i
            else:
                seg_start += sl


    def _get_parametric_params(self):
        diff = np.diff(self.waypoints, axis=0)
        for i in range(self.nwaypoints-1):
            derivative = diff[i] / self.segment_lengths[i]
            alpha = np.arctan2(derivative[1], derivative[0])
            beta = np.arctan2(-derivative[2], np.sqrt(derivative[0]**2 + derivative[1]**2))
            self.azimuth_angles = np.append(self.azimuth_angles, alpha)
            self.elevation_angles = np.append(self.elevation_angles, beta)


    def plot_path(self, label,*opts):
        x = []
        y = []
        z = []
        s = np.linspace(0, self.length, 10000)
        
        for ds in s:
            x.append(self(ds)[0])
            y.append(self(ds)[1])
            z.append(self(ds)[2])

        ax = plt.axes(projection='3d')
        ax.plot(x, y, z, label=label, *opts)
        return ax
    
    
    def get_closest_s(self, position):
        s = fminbound(lambda s: np.linalg.norm(self(s) - position),
                    x1=0, x2=self.length, xtol=1e-6,
                    maxfun=10000)
        return s


    def get_closest_point(self, position):
        s = self.get_closest_s(position)
        return self(s)


    def _get_path_lengths(self):
        waypoints = self.waypoints
        diff = np.diff(waypoints, axis=0)
        seg_lengths = np.sqrt(np.sum(diff**2, axis=1))
        return seg_lengths

    
    def get_endpoint(self):
        return self(self.length)

    
    def get_direction(self, s):
        _, seg_index = self._get_segment_start(s)
        return self.azimuth_angles[seg_index], self.elevation_angles[seg_index]


def generate_random_waypoints(nwaypoints):
    waypoints = [np.array([0,0,0])]
    for i in range(nwaypoints-1):
        azimuth = np.random.rand() * np.pi/2
        elevation = np.random.rand() * np.pi/2
        dist = np.random.randint(50, 150)

        x = waypoints[i][0] + dist*np.cos(azimuth)*np.cos(elevation)
        y = waypoints[i][1] + dist*np.sin(azimuth)*np.cos(elevation)
        z = waypoints[i][2] - dist*np.sin(elevation)
        wp = np.array([x, y, z])
        waypoints.append(wp)
    return waypoints

if __name__ == "__main__":
    x = np.linspace(0,6,100)
    waypoints = generate_random_waypoints(8)
    position = waypoints[4] + [30, 20, 10]
    path = Path3D(waypoints)
    p = path.get_closest_point(position)
    ax = path.plot_path()
    ax.scatter3D(*position)
    ax.scatter3D(*p)
    plt.show()
