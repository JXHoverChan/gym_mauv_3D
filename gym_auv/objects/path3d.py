import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


class ParamCurve3d():
    def __init__(self, waypoints, ncoords):
        self.ncoords = ncoords
        self.segment_lengths = get_path_lengths(waypoints)
        self.length = np.sum(self.segment_lengths)
        self.path_coords = self._get_path_coords_from_waypoints(waypoints)


    def _get_path_coords_from_waypoints(self, waypoints):
        path_coords = []

        ds = self.length / self.ncoords
        s_seg = np.round(self.segment_lengths / ds)
        
        for i, s in enumerate(s_seg):            
            start = waypoints[i]
            stop = waypoints[i+1]
            interpolation = np.linspace(start, stop, int(s)+1)
            interpolation = interpolation[:-1]
            interpolation = interpolation.tolist()
            path_coords.extend(interpolation)
        path_coords.append(np.array((waypoints[-1])))
        self.ncoords = len(path_coords) #update ncoords due to quantization error

        return path_coords


    def plot_path(self, *opts):
        x = []
        y = []
        z = []
        
        for coord in self.path_coords:
            x.append(coord[0])
            y.append(coord[1])
            z.append(coord[2])
        
        ax = plt.axes(projection='3d')
        ax.plot(x, y, z, *opts)
        return ax
    
    
    def get_closest_point_distance(self, position):
        best_distance = 1000000
        for i in range(self.ncoords):
            coords = np.array(self.path_coords[i])
            distance = linalg.norm(coords-position)
            if distance < best_distance:
                best_distance = distance
                best_i = i
        closest_point = self.path_coords[best_i]
        return closest_point, best_distance


def get_path_lengths(waypoints):
    diff = np.diff(waypoints, axis=0)
    seg_lengths = np.sqrt(np.sum(diff**2, axis=1))
    return seg_lengths


if __name__ == "__main__":
    x = np.linspace(0,6,100)
    waypoints = [(0,0,0), (1,1,1), (4,3,3)]
    waypoints = []
    for i in x:
        waypoint = (5*np.sin(i), 5*np.cos(i), 0.05*i)
        waypoints.append(waypoint)
    curve = ParamCurve3d(waypoints, 1000)
    position = np.array([2,5,0])
    c, bd = curve.get_closest_point_distance(position)
    print("Closest_distance: {}".format(bd))
    ax = curve.plot_path()
    ax.scatter3D(*position)
    ax.scatter3D(*c)
    plt.show()
