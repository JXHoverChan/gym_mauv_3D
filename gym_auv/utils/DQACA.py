import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

class DQACA:
    def __init__(self, start_points, task_points, resources, params):
        """
        初始化DQACA算法
        :param start_points: AUV起始点坐标列表(三维坐标)
        :param task_points: 任务点坐标列表(三维坐标)
        :param resources: 每个AUV的任务资源限制列表
        :param params: 算法参数
        """
        # 合并起点和任务点（添加虚拟节点）
        self.all_points = np.vstack([start_points, task_points])
        self.num_auv = len(start_points)
        self.resources = resources
        self.num_tasks = len(task_points)
        
        # 参数设置
        self.alpha = params.get('alpha', 1)
        self.beta = params.get('beta', 10)
        self.rho = params.get('rho', 0.1)
        self.Q = params.get('Q', 100)
        self.max_iter = params.get('max_iter', 2000)
        self.ant_count = params.get('ant_count', 51)
        self.epsilon_0 = params.get('epsilon_0', 0.1)
        
        # 计算距离矩阵（三维欧氏距离）
        self.dist_matrix = cdist(self.all_points, self.all_points, 'euclidean')
        np.fill_diagonal(self.dist_matrix, np.inf)  # 禁止原地停留
        
        # 初始化信息素矩阵
        self.pheromone = np.ones_like(self.dist_matrix) * 1e-6
        
        # 虚拟节点处理：起始点之间不可达
        for i in range(self.num_auv):
            for j in range(self.num_auv):
                if i != j:
                    self.dist_matrix[i,j] = np.inf

    def dynamic_Q(self, iter):
        """动态Q信息素调整函数"""
        if iter < self.max_iter/3:
            return 0.5 * self.Q
        elif iter < 2*self.max_iter/3:
            return 0.8 * self.Q
        else:
            return self.Q

    def epsilon_greedy(self, iter):
        """变ε-greedy策略"""
        if iter < self.max_iter/3:
            return self.epsilon_0
        else:
            return self.epsilon_0 * (self.max_iter - iter) / self.max_iter

    def run(self):
        self.best_paths = None
        best_distance = np.inf
        self.totaldistances = []
        self.bestdistances = []
        for iteration in range(self.max_iter):
            all_paths = []
            total_distances = []
            
            # 每只蚂蚁独立搜索路径
            for _ in range(self.ant_count):
                paths = [[] for _ in range(self.num_auv)]
                visited = set()
                current_pos = list(range(self.num_auv))  # 初始位置为虚拟节点
                remaining_res = self.resources.copy()
                
                # 构建路径（修改部分）
                for auv_id in range(self.num_auv):
                    path = [current_pos[auv_id]]  # 起始虚拟节点
                    while remaining_res[auv_id] > 0:
                        current_node = path[-1]
                        
                        # 获取可行节点（排除所有起点）
                        feasible = [n for n in range(len(self.all_points)) 
                                if n not in visited 
                                and n >= self.num_auv  # 只选择任务节点
                                and n != current_node]
                        
                        if not feasible:
                            break
                            
                        # 变ε-greedy策略
                        epsilon = self.epsilon_greedy(iteration)
                        if np.random.rand() < epsilon:
                            # 探索：随机选择
                            next_node = np.random.choice(feasible)
                        else:
                            # 利用：根据概率选择
                            pheromone = self.pheromone[current_node, feasible] ** self.alpha
                            heuristic = (1.0 / (self.dist_matrix[current_node, feasible] + 1e-6)) ** self.beta
                            probs = pheromone * heuristic
                            probs /= probs.sum()
                            next_node = np.random.choice(feasible, p=probs)
                        
                        # 更新路径和资源
                        path.append(next_node)
                        visited.add(next_node)
                        remaining_res[auv_id] -= 1
                        
                        # 返回起点
                    if len(path) > 1:  # 确保有实际路径
                        path.append(path[0])  # 最终返回起始点
                        
                    paths[auv_id] = path
                    
                # 计算总距离
                total_dist = sum(self.dist_matrix[path[i], path[i+1]] 
                                for path in paths 
                                for i in range(len(path)-1))
                total_distances.append(total_dist)
                all_paths.append(paths)
                
                # 更新最优解
                if total_dist < best_distance:
                    best_distance = total_dist
                    self.best_paths = paths
                    
            # 动态Q信息素更新
            delta_pheromone = np.zeros_like(self.pheromone)
            Q = self.dynamic_Q(iteration)
            
            for paths in all_paths:
                for auv_path in paths:
                    path = auv_path
                    path_dist = sum(self.dist_matrix[path[i], path[i+1]] 
                                  for i in range(len(path)-1))
                    if path_dist == 0:
                        continue
                        
                    for i in range(len(path)-1):
                        delta = Q / path_dist
                        delta_pheromone[path[i], path[i+1]] += delta
                        
            # 信息素挥发和更新
            self.pheromone = (1 - self.rho) * self.pheromone + delta_pheromone
            
            # 早停条件：连续20代无改进
            if iteration > 20 and len(set(total_distances[-20:])) == 1:
                break
            self.totaldistances.append(total_distances)
            self.bestdistances.append(best_distance)
        return self.best_paths, best_distance, self.totaldistances, self.bestdistances
    
    def data_plot(self):
        """绘制距离变化图"""
        plt.figure()
        # 每20代的平均距离
        average_distances = [np.mean(d) for d in self.totaldistances]
        plt.plot(range(len(average_distances)), average_distances, label='Average Distance')
        plt.plot(range(len(self.bestdistances)), self.bestdistances, label='Best Distance')
        # plt.plot(range(len(self.totaldistances)), [min(d) for d in self.totaldistances], label='Average Distance')
        plt.xlabel('Iteration')
        plt.ylabel('Distance')
        plt.title('Best Distance Over Iterations')
        plt.legend()
        plt.show()

    def route_plot(self):
        """绘制路径图"""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # 绘制所有点
        all_points = np.vstack([self.all_points])
        ax.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2], c='b', marker='o')
        # 绘制路径
        for i, path in enumerate(self.best_paths):
            point_path = [self.all_points[n] for n in path]
            xs = [p[0] for p in point_path]
            ys = [p[1] for p in point_path]
            zs = [p[2] for p in point_path]
            ax.plot(xs, ys, zs, label=f'AUV {i+1}')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Z Coordinate')
        ax.legend()
        plt.show()

# 示例使用
if __name__ == "__main__":
    # 参数配置
    params = {
        'alpha': 1,
        'beta': 10,
        'rho': 0.1,
        'Q': 100,
        'max_iter': 20,
        'ant_count': 51,
        'epsilon_0': 0.1
    }

    # 示例数据（三维坐标）
    start_points = np.array([[35,50,35], [35,50,35], [35,50,35], [35,50,35]])  # 4个AUV的起始点
    task_points = np.array([
        [30, 194, 140], [35, 156, 94], [121, 94, 99], [13, 32, 62], [185, 141, 104],
        [32, 148, 85], [113, 40, 117], [171, 11, 88], [24, 128, 150], [89, 34, 104],
        [78, 5, 25], [84, 22, 145], [2, 120, 57], [90, 96, 53], [75, 152, 146],
        [130, 14, 133], [114, 8, 125], [82, 144, 61], [198, 128, 120], [200, 183, 127]
    ])  # 20个任务点
    resources = [6, 6, 4, 4]  # 每个AUV的任务资源限制

    # 验证任务资源是否足够
    total_tasks = len(task_points)
    if sum(resources) < total_tasks:
        print("任务资源不足，无法分配所有任务点。")
        exit()
    else:
        print("任务资源足够，开始运行算法。")

    # 运行算法
    solver = DQACA(start_points, task_points, resources, params)
    best_paths, best_distance, total, best = solver.run()

    # 结果解析
    print(f"最优总距离: {best_distance}")
    waypoints = []
    for i, path in enumerate(best_paths):
        point_path = [solver.all_points[n] for n in path]
        waypoints.append(point_path)
        print(f"AUV {i+1} 的路径:")
        for p in point_path:
            print(f"({p[0]}, {p[1]}, {p[2]})")
        print()
    print("所有AUV的路径:")
    for wp in waypoints:
        print(wp)