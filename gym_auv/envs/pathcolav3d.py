import numpy as np
import gym
import gym_auv.utils.geomutils as geom
import matplotlib.pyplot as plt
import skimage.measure

from gym_auv.objects.auv3d import AUV3D
from gym_auv.objects.current3d import Current
from gym_auv.objects.QPMI import QPMI, generate_random_waypoints
from gym_auv.objects.path3d import Path3D
from gym_auv.objects.obstacle3d import Obstacle
from gym_auv.utils.controllers import PI, PID


test_waypoints = np.array([np.array([0,0,0]), np.array([20,10,15]), np.array([50,20,20]), np.array([80,20,40]), np.array([90,50,50]),
                           np.array([80,80,60]), np.array([50,80,20]), np.array([20,60,15]), np.array([20,40,10]), np.array([0,0,0])])

test_waypoints = np.array([np.array([0,0,0]), np.array([50,15,5]), np.array([80,5,-5]), np.array([120,10,0]), np.array([150,0,0])])


class PathColav3d(gym.Env):
    """
    Creates an environment with a vessel, a path and obstacles. 翻译：创建一个带有船舶、路径和障碍物的环境。
    """
    def __init__(self, env_config, scenario="beginner"):
        for key in env_config:
            setattr(self, key, env_config[key])
        self.n_observations = (self.n_obs_states * self.num_vessels+ 
                               self.n_obs_errors * self.num_vessels+ 
                               self.n_obs_inputs * self.num_vessels+ 
                               self.sensor_input_size[0]* self.sensor_input_size[1]* self.num_vessels)
        self.action_space = gym.spaces.Box(low=np.array([-1]*self.n_actuators*self.num_vessels, dtype=np.float32),
                                        high=np.array([1]*self.n_actuators*self.num_vessels, dtype=np.float32),
                                        dtype=np.float32)    # 创建动作空间
        self.observation_space = gym.spaces.Box(low=np.array([-1]*self.n_observations, dtype=np.float32),
                                                high=np.array([1]*self.n_observations, dtype=np.float32),
                                                dtype=np.float32)   # 创建观察空间
        
        self.scenario = scenario    # 训练场景
        
        self.n_sensor_readings = self.sensor_suite[0]*self.sensor_suite[1]
        max_horizontal_angle = self.sensor_span[0]/2
        max_vertical_angle = self.sensor_span[1]/2
        self.sectors_horizontal = np.linspace(-max_horizontal_angle*np.pi/180, max_horizontal_angle*np.pi/180, self.sensor_suite[0])
        self.sectors_vertical =  np.linspace(-max_vertical_angle*np.pi/180, max_vertical_angle*np.pi/180, self.sensor_suite[1])
        self.update_sensor_step= 1/(self.step_size*self.sensor_frequency)
        
        self.scenario_switch = {
            # Training scenarios
            "beginner": self.scenario_beginner,
            "intermediate": self.scenario_intermediate,
            "proficient": self.scenario_proficient,
            "advanced": self.scenario_advanced,
            "expert": self.scenario_expert,
            "m_beginner": self.scenario_beginner_multi,
            "m_intermediate": self.scenario_intermediate_multi,
            "m_proficient": self.scenario_proficient_multi,
            "m_advanced": self.scenario_advanced_multi,
            # Testing scenarios
            "test_path": self.scenario_test_path,
            "m_test_path": self.scenario_test_path_multi,  # Multi-vessel test path
            "test_path_current": self.scenario_test_path_current,
            "test": self.scenario_test,
            "m_test": self.scenario_test_multi,  # Multi-vessel test
            "test_current": self.scenario_test_current,
            "horizontal": self.scenario_horizontal_test,
            "vertical": self.scenario_vertical_test,
            "deadend": self.scenario_deadend_test
        }

        self.reset()


    def reset(self):
        """
        Resets environment to initial state. 
        """
        #print("ENVIRONMENT RESET INITIATED")
        self.vessel = None  # AUV object
        self.path = None    # Path object
        self.u_error = None # Cruise speed error
        self.e = None    # Cross-track error
        self.h = None
        self.chi_error = None
        self.upsilon_error = None
        self.waypoint_index = 0
        self.prog = 0
        self.path_prog = []
        self.success = False
        self.obstacles = []
        self.nearby_obstacles = []
        self.sensor_readings = np.zeros(shape=self.sensor_suite, dtype=float)
        self.collided = False
        self.penalize_control = 0.0
        self.observation = None
        self.action_derivative = np.zeros(self.n_actuators)
        self.past_states = []
        self.past_actions = []
        self.past_errors = []
        self.past_obs = []
        self.current_history = []

        # For multiple vessels
        self.vessels = []  # Clear vessels list, in case of multiple vessels
        self.multivessels_sensor_readings = np.zeros((self.num_vessels, self.sensor_suite[0], self.sensor_suite[1]), dtype=float) # Sensor readings for multiple vessels
        self.multivessels_collided = np.zeros(self.num_vessels, dtype=bool) # Track collisions for multiple vessels
        self.multivessels_current_history = [[] for _ in range(self.num_vessels)]
        self.u_error_multi = [0.0 for _ in range(self.num_vessels)]  # Cruise speed error for multiple vessels
        self.chi_error_multi = [0.0 for _ in range(self.num_vessels)]  # Course error for multiple vessels
        self.e_multi = [0.0 for _ in range(self.num_vessels)]  # Cross-track error for multiple vessels
        self.upsilon_error_multi = [0.0 for _ in range(self.num_vessels)]  # Elevation error for multiple vessels
        self.h_multi = [0.0 for _ in range(self.num_vessels)]  # Additional error term for multiple vessels
        self.prog_multi = [0.0 for _ in range(self.num_vessels)]  # Progress along the path for each vessel, initialized to 0
        self.path_prog_multi = [[] for _ in range(self.num_vessels)]  # Track progress along the path for each vessel
        self.success_multi = [False for _ in range(self.num_vessels)]  # Success flags for multiple vessels
        self.path_multi = [None for _ in range(self.num_vessels)]  # Path objects for multiple vessels
        self.past_actions_multi = [[] for _ in range(self.num_vessels)]  # Past actions for multiple vessels
        self.action_derivative_multi = np.zeros((self.num_vessels, self.n_actuators))  # Action derivatives for multiple vessels
        self.past_states_multi = [[] for _ in range(self.num_vessels)]  # Past states for multiple vessels
        self.past_errors_multi = [[] for _ in range(self.num_vessels)]  # Past errors for multiple vessels
        # self.path_multi = [None for _ in range(self.num_vessels)]  # Path objects for multiple vessels, initialized to None
        self.waypoint_index_multi = [0 for _ in range(self.num_vessels)]  # Waypoint index for multiple vessels
        self.observation_multi = [None for _ in range(self.num_vessels)]  # Observations for multiple vessels
        self.past_obs_multi = [[] for _ in range(self.num_vessels)]  # Past observations for multiple vessels
        self.nearby_obstacles_multi = [[] for _ in range(self.num_vessels)]  # Nearby obstacles for multiple vessels, initialized to empty lists

        self.time = []
        self.total_t_steps = 0
        self.reward = 0

        self.generate_environment()
        #print("\tENVIRONMENT GENERATED")
        if self.num_vessels == 1:
            self.update_control_errors()
        elif self.num_vessels > 1:
            # For multiple vessels, update control errors for each vessel
            self.update_control_errors_multi()
        #print("\tCONTROL ERRORS UPDATED")
        self.observation = self.observe(np.zeros(6, dtype=float))
        self.observation_multi = [self.observe(np.zeros(6, dtype=float)) for _ in range(self.num_vessels)]  # Initialize observations for multiple vessels
        #print("COMPLETE")
        return self.observation_multi


    def generate_environment(self):
        """
        Generates environment with a vessel, potentially ocean current and a 3D path.
        """     
        # Generate training/test scenario
        scenario = self.scenario_switch.get(self.scenario, lambda: print("Invalid scenario"))
        #print("\tGENERATING", self.scenario.upper())
        init_state = scenario()
        # Generate AUV
        #print("\tGENERATING AUV")
        if self.num_vessels == 1:
            self.vessel = AUV3D(self.step_size, init_state)
            print("\tGENERATING PI-CONTROLLER")
            self.thrust_controller = PI()   # PI-controller for thrust

        elif self.num_vessels > 1:
            # For multiple vessels, create a list of AUVs
            for i in range(self.num_vessels):
                # Create a new AUV for each vessel with a unique initial state
                self.vessels.append(AUV3D(self.step_size, init_state[i]))  # Pass the i-th initial state for multiple vessels
            print("\tGENERATING PI-CONTROLLER for multiple vessels")
            self.thrust_controllers = [PI() for _ in range(self.num_vessels)]  # One controller per vessel


    

    def plot_section3(self):
        plt.rc('lines', linewidth=3)
        ax = self.plot3D(wps_on=False)
        ax.set_xlabel(xlabel="North [m]", fontsize=14)
        ax.set_ylabel(ylabel="East [m]", fontsize=14)
        ax.set_zlabel(zlabel="Down [m]", fontsize=14)
        ax.xaxis.set_tick_params(labelsize=12)
        ax.yaxis.set_tick_params(labelsize=12)
        ax.zaxis.set_tick_params(labelsize=12)
        ax.set_xticks([0, 50, 100])
        ax.set_yticks([-50, 0, 50])
        ax.set_zticks([-50, 0, 50])
        ax.view_init(elev=-165, azim=-35)
        if self.num_vessels == 1:
            ax.scatter3D(*self.vessel.position, label="Initial Position", color="y")

        # For multiple vessels, plot their initial positions
        elif self.num_vessels > 1:
            for i, vessel in enumerate(self.vessels):
                ax.scatter3D(*vessel.position, label="Initial Position Vessel {}".format(i+1), color="y")

        self.axis_equal3d(ax)
        ax.legend(fontsize=14)
        plt.show()

    def step(self, action):
        """
        Simulates the environment one time-step. 
        """
        # Simulate Current
        self.current.sim()
        nu_c = self.current(self.vessel.state)
        self.current_history.append(nu_c[0:3])

        # Simulate AUV dynamics one time-step and save action and state
        self.update_control_errors()
        thrust = self.thrust_controller.u(self.u_error)
        action = np.hstack((thrust, action))
        action = np.clip(action, np.array([0, -1, -1]), np.array([1, 1, 1]))
        if len(self.past_actions) > 0:
            self.action_derivative = (action[1:]-self.past_actions[-1][1:])/(self.step_size)
        
        self.vessel.step(action, nu_c)
        self.past_states.append(np.copy(self.vessel.state))
        self.past_errors.append(np.array([self.u_error, self.chi_error, self.e, self.upsilon_error, self.h]))
        self.past_actions.append(self.vessel.input)

        if self.path:
            self.prog = self.path.get_closest_u(self.vessel.position, self.waypoint_index)
            self.path_prog.append(self.prog)
            
            # Check if a waypoint is passed
            k = self.path.get_u_index(self.prog)
            if k > self.waypoint_index:
                print("Passed waypoint {:d}".format(k+1))
                self.waypoint_index = k
        
        # Calculate reward based on observation and actions 翻译：根据观察和行动计算奖励
        done, step_reward = self.step_reward(self.observation, action)
        info = {}

        # Make next observation
        self.observation = self.observe(nu_c)
        self.past_obs.append(self.observation)

        # Save sim time info
        self.total_t_steps += 1
        self.time.append(self.total_t_steps*self.step_size)
        
        return self.observation, step_reward, done, info

    def step_multi(self, actions):
        """
        Simulates the environment for multiple vessels one time-step.
        """
        # Simulate Current
        self.current.sim()
        nu_c_multi = [self.current(vessel.state) for vessel in self.vessels]
        for i in range(self.num_vessels):
            self.multivessels_current_history[i].append(nu_c_multi[i][0: 3])
        
        # Simulate AUV dynamics for each vessel one time-step and save action and state
        self.update_control_errors_multi()
        thrust_multi = [self.thrust_controllers[i].u(self.u_error_multi[i]) for i in range(self.num_vessels)]
        actions_multi = [np.hstack((thrust_multi[i], actions[i])) for i in range(self.num_vessels)]  # Combine thrust with other actions for each vessel
        # print("\tThrust Multi:", thrust_multi)  # Debugging: Print thrust for each vessel
        # print("\tActions", actions)  # Debugging: Print combined actions for each vessel
        # print("\tActions Multi:", actions_multi)  # Debugging: Print actions for each vessel
        for i in range(self.num_vessels):
            # Clip actions for each vessel
            actions_multi[i] = np.clip(actions_multi[i], np.array([0, -1, -1]), np.array([1, 1, 1]))
            if len(self.past_actions_multi[i]) > 0:
                self.action_derivative_multi[i] = (actions_multi[i][1:]-self.past_actions_multi[i][-1][1:])/(self.step_size)  # Calculate action derivative for each vessel
            # print("\tTime :{},Vessel {}: Action Derivative: {}, Actions Multi{}: {}, Past Actions: {}".format(
                self.total_t_steps*self.step_size,
                i+1, 
                self.action_derivative_multi[i], 
                i+1, 
                actions_multi[i], 
                self.past_actions_multi[i][-1][1:] if len(self.past_actions_multi[i]) > 0 else "N/A"
            # print("actions_multi: {}".format(actions_multi))  # Debugging: Print actions for each vessel
            # print("nu_c_multi[{}]: {}".format(i, nu_c_multi[i]))  # Debugging: Print current for each vessel

            self.vessels[i].step(actions_multi[i], nu_c_multi[i])  # Step the i-th vessel
            self.past_states_multi[i].append(np.copy(self.vessels[i].state))  # Save the state of the i-th vessel
            self.past_errors_multi[i].append(np.array([self.u_error_multi[i], self.chi_error_multi[i], self.e_multi[i], self.upsilon_error_multi[i], self.h_multi[i]]))
            self.past_actions_multi[i].append(self.vessels[i].input)  # Save the input of the i-th vessel
            if self.path_multi[i]:
                # Update progress along the path for each vessel
                self.prog_multi[i] = self.path_multi[i].get_closest_u(self.vessels[i].position, self.waypoint_index_multi[i])
                self.path_prog_multi[i].append(self.prog_multi[i])
                
                # Check if a waypoint is passed for each vessel
                k = self.path_multi[i].get_u_index(self.prog_multi[i])
                if k > self.waypoint_index_multi[i]:
                    print("Vessel {:d} passed waypoint {:d}".format(i+1, k+1))
                    self.waypoint_index_multi[i] = k
            # Calculate reward based on observation and actions for each vessel
            # print("actions_multi[{}]: {}".format(i, actions_multi[i]))  # Debugging: Print actions for each vessel
            done, step_reward = self.step_reward_multi(self.observe(nu_c_multi[i]), actions_multi[i])
            self.observation_multi[i] = self.observe(nu_c_multi[i])  # Update observation for the i-th vessel
            self.past_obs_multi[i].append(self.observation_multi[i])  # Save the observation for the i-th vessel
            # if self.multivessels_collided[i]:
                # print("Vessel {:d} collided!".format(i+1))
                # print(np.round(self.multivessels_sensor_readings[i], 2))  # Print the sensor readings for the i-th vessel at collision
        self.total_t_steps += 1  # Increment total time steps
        self.time.append(self.total_t_steps*self.step_size)  # Save the current simulation time

        # Return the observations, rewards, done flags, and info for multiple vessels
        return self.observation_multi, step_reward, done, {}  # Return observations for multiple vessels, total reward, done flag, and info


    def observe(self, nu_c):
        """
        Returns observations of the environment. 
        """
        if self.num_vessels == 1:
            obs = np.zeros((self.n_observations,))
            obs[0] = np.clip(self.vessel.relative_velocity[0] / 2, -1, 1)
            obs[1] = np.clip(self.vessel.relative_velocity[1] / 0.3, -1, 1)
            obs[2] = np.clip(self.vessel.relative_velocity[2] / 0.3, -1, 1)
            obs[3] = np.clip(self.vessel.roll / np.pi, -1, 1)
            obs[4] = np.clip(self.vessel.pitch / np.pi, -1, 1)
            obs[5] = np.clip(self.vessel.heading / np.pi, -1, 1)
            obs[6] = np.clip(self.vessel.angular_velocity[0] / 1.2, -1, 1)
            obs[7] = np.clip(self.vessel.angular_velocity[1] / 0.4, -1, 1)
            obs[8] = np.clip(self.vessel.angular_velocity[2] / 0.4, -1, 1)
            obs[9] = np.clip(nu_c[0] / 1, -1, 1)
            obs[10] = np.clip(nu_c[1] / 1, -1, 1)
            obs[11] = np.clip(nu_c[2] / 1, -1, 1)
            obs[12] = self.chi_error
            obs[13] = self.upsilon_error
        elif self.num_vessels > 1:
            """
            For multiple vessels, construct the observation for each vessel and concatenate them.
            """
            obs = np.zeros((self.num_vessels, self.n_observations // self.num_vessels))
            for i in range(self.num_vessels):
                obs[i][0] = np.clip(self.vessels[i].relative_velocity[0] / 2, -1, 1)
                obs[i][1] = np.clip(self.vessels[i].relative_velocity[1] / 0.3, -1, 1)
                obs[i][2] = np.clip(self.vessels[i].relative_velocity[2] / 0.3, -1, 1)
                obs[i][3] = np.clip(self.vessels[i].roll / np.pi, -1, 1)
                obs[i][4] = np.clip(self.vessels[i].pitch / np.pi, -1, 1)
                obs[i][5] = np.clip(self.vessels[i].heading / np.pi, -1, 1)
                obs[i][6] = np.clip(self.vessels[i].angular_velocity[0] / 1.2, -1, 1)
                obs[i][7] = np.clip(self.vessels[i].angular_velocity[1] / 0.4, -1, 1)
                obs[i][8] = np.clip(self.vessels[i].angular_velocity[2] / 0.4, -1, 1)
                obs[i][9] = np.clip(nu_c[0]/1, -1, 1)
                obs[i][10] = np.clip(nu_c[1]/1, -1, 1)
                obs[i][11] = np.clip(nu_c[2]/1, -1, 1)
                obs[i][12] = self.chi_error_multi[i]  # Course error for the i-th vessel
                obs[i][13] = self.upsilon_error_multi[i]  # Elevation error for the i-th vessel

        # Update nearby obstacles and calculate distances
        if self.total_t_steps % self.update_sensor_step == 0:
            if self.num_vessels == 1:
                self.update_nearby_obstacles()
                self.sonar_observations = skimage.measure.block_reduce(self.sensor_readings, (2,2), np.max)
                self.update_sensor_readings()
                obs[14:] = self.sonar_observations.flatten()
            else:
                self.update_nearby_obstacles_multi()
                self.sonar_observations_multi = np.zeros((self.num_vessels, self.sensor_input_size[0], self.sensor_input_size[1]), dtype=float)
                self.update_sensor_readings_multi()
                for i in range(self.num_vessels):
                    self.sonar_observations_multi[i] = skimage.measure.block_reduce(self.multivessels_sensor_readings[i], (2,2), np.max)
                    obs[i][14:] = self.sonar_observations_multi[i].flatten()
            # self.update_sensor_readings_with_plots_multi()  # For debugging purposes, plot sensor readings for multiple vessels
        return obs


    def step_reward(self, obs, action):
        """
        Calculates the reward function for one time step. Also checks if the episode should end. 
        翻译： 计算一个时间步的奖励函数。还检查是否应该结束该集。
        """
        done = False
        step_reward = 0 

        reward_roll = self.vessel.roll**2*self.reward_roll + self.vessel.angular_velocity[0]**2*self.reward_rollrate
        reward_control = action[1]**2*self.reward_use_rudder + action[2]**2*self.reward_use_elevator
        reward_path_following = self.chi_error**2*self.reward_heading_error + self.upsilon_error**2*self.reward_pitch_error
        reward_collision_avoidance = self.penalize_obstacle_closeness()

        step_reward = self.lambda_reward*reward_path_following + (1-self.lambda_reward)*reward_collision_avoidance \
                    + reward_roll + reward_control
        self.reward += step_reward

        # Check collision
        for obstacle in self.nearby_obstacles:
            if np.linalg.norm(obstacle.position - self.vessel.position) <= obstacle.radius + self.vessel.safety_radius:
                self.collided = True
        
        end_cond_1 = self.reward < self.min_reward
        end_cond_2 = self.total_t_steps >= self.max_t_steps
        end_cond_3 = np.linalg.norm(self.path.get_endpoint()-self.vessel.position) < self.accept_rad and self.waypoint_index == self.n_waypoints-2
        end_cond_4 = abs(self.prog - self.path.length) <= self.accept_rad/2.0

        if end_cond_1 or end_cond_2 or end_cond_3 or end_cond_4:
            if end_cond_3:
                print("AUV reached target!")
                self.success = True
            elif self.collided:
                print("AUV collided!")
                print(np.round(self.sensor_readings,2))
                self.success = False
            print("Episode finished after {} timesteps with reward: {}".format(self.total_t_steps, self.reward.round(1)))
            done = True
        return done, step_reward
    
    def step_reward_multi(self, obs_multi, actions_multi):
        """
        Calculates the reward function for multiple vessels for one time step. Also checks if the episode should end.
        """
        done = False
        step_reward_sum = 0
        for i in range(self.num_vessels):
            obs = obs_multi[i]
            action = actions_multi
            # print("Action", action) #Debugging
            reward_roll = self.vessels[i].roll**2*self.reward_roll + self.vessels[i].angular_velocity[0]**2*self.reward_rollrate
            reward_control = action[1]**2*self.reward_use_rudder + action[2]**2*self.reward_use_elevator
            reward_path_following = self.chi_error_multi[i]**2*self.reward_heading_error + self.upsilon_error_multi[i]**2*self.reward_pitch_error
            reward_collision_avoidance = self.penalize_obstacle_closeness(i)
            # print("Reward collision avoidance for vessel {}: {}".format(i+1, reward_collision_avoidance))  # Debugging: Print collision avoidance reward for each vessel
            step_reward = self.lambda_reward*reward_path_following + (1-self.lambda_reward)*reward_collision_avoidance \
                        + reward_roll + reward_control
            step_reward_sum += step_reward

            # Check collision for each vessel
            for obstacle in self.nearby_obstacles_multi[i]:
                # print("O_R_{}".format(i),np.linalg.norm(obstacle.position - self.vessels[i].position))
                if np.linalg.norm(obstacle.position - self.vessels[i].position) <= obstacle.radius + self.vessels[i].safety_radius:
                    self.multivessels_collided[i] = True
        self.reward += step_reward_sum
        # print("Nearby obstacles multi-vessel:", self.nearby_obstacles_multi)  # Debugging: Print nearby obstacles for each vessel
        end_cond_1 = self.reward < self.min_reward*self.num_vessels  # Adjusted for multiple vessels
        end_cond_2 = self.total_t_steps >= self.max_t_steps
        end_cond_3 = all(np.linalg.norm(self.path_multi[i].get_endpoint()-self.vessels[i].position) < self.accept_rad and self.waypoint_index_multi[i] == self.n_waypoints-2 for i in range(self.num_vessels))
        end_cond_4 = all(abs(self.prog_multi[i] - self.path_multi[i].length) <= self.accept_rad/2.0 for i in range(self.num_vessels))
        if end_cond_1 or end_cond_2 or end_cond_3 or end_cond_4:
            if end_cond_3:
                print("All vessels reached target!")
                self.success_multi = [True for _ in range(self.num_vessels)]
            elif any(self.multivessels_collided):
                print("At least one vessel collided!")
                self.success_multi = [False for _ in range(self.num_vessels)]
            print("Episode finished after {} timesteps with total reward: {}".format(self.total_t_steps, self.reward.round(1)))
            done = True
        return done, step_reward_sum


    def update_control_errors(self):
        # Update cruise speed error
        self.u_error = np.clip((self.cruise_speed - self.vessel.relative_velocity[0])/2, -1, 1)
        self.chi_error = 0.0
        self.e = 0.0,
        # Get path course and elevation
        s = self.prog
        chi_p, upsilon_p = self.path.get_direction_angles(s)

        # Calculate tracking errors
        SF_rotation = geom.Rzyx(0,upsilon_p,chi_p)
        epsilon = np.transpose(SF_rotation).dot(self.vessel.position-self.path(self.prog))
        e = epsilon[1]
        h = epsilon[2]

        # Calculate course and elevation errors from tracking errors
        chi_r = np.arctan2(-e, self.la_dist)
        upsilon_r = np.arctan2(h, np.sqrt(e**2 + self.la_dist**2))
        chi_d = chi_p + chi_r
        upsilon_d = upsilon_p + upsilon_r
        self.chi_error = np.clip(geom.ssa(self.vessel.chi - chi_d)/np.pi, -1, 1)
        #self.e = np.clip(e/12, -1, 1)
        self.e = e
        self.upsilon_error = np.clip(geom.ssa(self.vessel.upsilon - upsilon_d)/np.pi, -1, 1)
        #self.h = np.clip(h/12, -1, 1)
        self.h = h

    def update_control_errors_multi(self):
        """
        For multiple vessels support, updates the control errors for each vessel in the environment.
        """
        for i, vessel in enumerate(self.vessels):
            # Update cruise speed error
            self.u_error_multi[i] = np.clip((self.cruise_speed - vessel.relative_velocity[0])/2, -1, 1)
            self.chi_error_multi[i] = 0.0
            self.e_multi[i] = 0.0
            self.upsilon_error_multi[i] = 0.0
            self.h_multi[i] = 0.0

            # Get path course and elevation
            s = self.prog_multi[i]
            chi_p, upsilon_p = self.path_multi[i].get_direction_angles(s)
            # Calculate tracking errors
            SF_rotation = geom.Rzyx(0, upsilon_p, chi_p)
            epsilon = np.transpose(SF_rotation).dot(vessel.position - self.path_multi[i](s))  # Note: self.path_multi[i] should be the path for the i-th vessel
            e = epsilon[1]
            h = epsilon[2]

            # Calculate course and elevation errors from tracking errors
            chi_r = np.arctan2(-e, self.la_dist)
            upsilon_r = np.arctan2(h, np.sqrt(e**2 + self.la_dist**2))
            chi_d = chi_p + chi_r
            upsilon_d = upsilon_p + upsilon_r
            self.chi_error_multi[i] = np.clip(geom.ssa(vessel.chi - chi_d)/np.pi, -1, 1)
            #self.e_multi[i] = np.clip(e/12, -1, 1)
            self.e_multi[i] = e  # Keep it as is for now, to match the single vessel case
            self.upsilon_error_multi[i] = np.clip(geom.ssa(vessel.upsilon - upsilon_d)/np.pi, -1, 1)
            #self.h_multi[i] = np.clip(h/12, -1, 1)
            self.h_multi[i] = h  # Keep it as is for now, to match the single vessel case



    def update_nearby_obstacles(self):
        """
        Updates the nearby_obstacles array.
        """
        self.nearby_obstacles = []
        for obstacle in self.obstacles:
            distance_vec_NED = obstacle.position - self.vessel.position
            distance = np.linalg.norm(distance_vec_NED)
            distance_vec_BODY = np.transpose(geom.Rzyx(*self.vessel.attitude)).dot(distance_vec_NED)
            heading_angle_BODY = np.arctan2(distance_vec_BODY[1], distance_vec_BODY[0])
            pitch_angle_BODY = np.arctan2(distance_vec_BODY[2], np.sqrt(distance_vec_BODY[0]**2 + distance_vec_BODY[1]**2))
            # check if the obstacle is inside the sonar window
            if distance - self.vessel.safety_radius - obstacle.radius <= self.sonar_range and abs(heading_angle_BODY) <= self.sensor_span[0]*np.pi/180 \
            and abs(pitch_angle_BODY) <= self.sensor_span[1]*np.pi/180:
                self.nearby_obstacles.append(obstacle)
            elif distance <= obstacle.radius + self.vessel.safety_radius:
                self.nearby_obstacles.append(obstacle)

    def update_nearby_obstacles_multi(self):
        """
        Updates the nearby_obstacles array for multiple vessels.
        """
        for i, vessel in enumerate(self.vessels):
            self.nearby_obstacles_multi[i] = []
            for obstacle in self.obstacles:
                distance_vec_NED = obstacle.position - vessel.position
                distance = np.linalg.norm(distance_vec_NED)
                distance_vec_BODY = np.transpose(geom.Rzyx(*vessel.attitude)).dot(distance_vec_NED)
                heading_angle_BODY = np.arctan2(distance_vec_BODY[1], distance_vec_BODY[0])
                pitch_angle_BODY = np.arctan2(distance_vec_BODY[2], np.sqrt(distance_vec_BODY[0]**2 + distance_vec_BODY[1]**2))
                # check if the obstacle is inside the sonar window
                if distance - vessel.safety_radius - obstacle.radius <= self.sonar_range and abs(heading_angle_BODY) <= self.sensor_span[0]*np.pi/180 \
                and abs(pitch_angle_BODY) <= self.sensor_span[1]*np.pi/180:
                    self.nearby_obstacles_multi[i].append(obstacle)
                elif distance <= obstacle.radius + vessel.safety_radius:
                    self.nearby_obstacles_multi[i].append(obstacle)


    def update_sensor_readings(self):
        """
        Updates the sonar data closeness array.
        """
        self.sensor_readings = np.zeros(shape=self.sensor_suite, dtype=float)
        for obstacle in self.nearby_obstacles:
            for i in range(self.sensor_suite[0]):
                alpha = self.vessel.heading + self.sectors_horizontal[i]
                for j in range(self.sensor_suite[1]):
                    beta = self.vessel.pitch + self.sectors_vertical[j]
                    _, closeness = self.calculate_object_distance(alpha, beta, obstacle)
                    self.sensor_readings[j,i] = max(closeness, self.sensor_readings[j,i]) 

    def update_sensor_readings_multi(self):
        """
        Updates the sonar data closeness array for multiple vessels.
        """
        for i, vessel in enumerate(self.vessels):
            self.multivessels_sensor_readings[i] = np.zeros(shape=self.sensor_suite, dtype=float)
            for obstacle in self.nearby_obstacles_multi[i]:
                for j in range(self.sensor_suite[0]):
                    alpha = vessel.heading + self.sectors_horizontal[j]
                    for k in range(self.sensor_suite[1]):
                        beta = vessel.pitch + self.sectors_vertical[k]
                        _, closeness = self.calculate_object_distance(vessel, alpha, beta, obstacle)
                        self.multivessels_sensor_readings[i][k,j] = max(closeness, self.multivessels_sensor_readings[i][k,j])
        # print("Updated multivessels_sensor_readings:")
        # for i in range(self.num_vessels):
            # print("Vessel {}:\n".format(i+1), np.round(self.multivessels_sensor_readings[i], 3))

    def update_sensor_readings_with_plots(self):
        """
        Updates the sonar data array and renders the simulations as 3D plot. Used for debugging.
        """
        print("Time: {}, Nearby Obstacles: {}".format(self.total_t_steps, len(self.nearby_obstacles)))
        self.sensor_readings = np.zeros(shape=self.sensor_suite, dtype=float)
        ax = self.plot3D()
        ax2 = self.plot3D()
        for obstacle in self.nearby_obstacles:
            for i in range(self.sensor_suite[0]):
                alpha = self.vessel.heading + self.sectors_horizontal[i]
                for j in range(self.sensor_suite[1]):
                    beta = self.vessel.pitch + self.sectors_vertical[j]
                    s, closeness = self.calculate_object_distance(alpha, beta, obstacle)
                    self.sensor_readings[j,i] = max(closeness, self.sensor_readings[j,i])              
                    color = "#05f07a" if s >= self.sonar_range else "#a61717"
                    s = np.linspace(0, s, 100)
                    x = self.vessel.position[0] + s*np.cos(alpha)*np.cos(beta)
                    y = self.vessel.position[1] + s*np.sin(alpha)*np.cos(beta)
                    z = self.vessel.position[2] - s*np.sin(beta)
                    ax.plot3D(x, y, z, color=color)
                    if color == "#a61717": ax2.plot3D(x, y, z, color=color)
                plt.rc('lines', linewidth=3)
        ax.set_xlabel(xlabel="North [m]", fontsize=14)
        ax.set_ylabel(ylabel="East [m]", fontsize=14)
        ax.set_zlabel(zlabel="Down [m]", fontsize=14)
        ax.xaxis.set_tick_params(labelsize=12)
        ax.yaxis.set_tick_params(labelsize=12)
        ax.zaxis.set_tick_params(labelsize=12)
        ax.scatter3D(*self.vessel.position, color="y", s=40, label="AUV")
        print(np.round(self.sensor_readings,3))
        self.axis_equal3d(ax)
        ax.legend(fontsize=14)
        ax2.set_xlabel(xlabel="North [m]", fontsize=14)
        ax2.set_ylabel(ylabel="East [m]", fontsize=14)
        ax2.set_zlabel(zlabel="Down [m]", fontsize=14)
        ax2.xaxis.set_tick_params(labelsize=12)
        ax2.yaxis.set_tick_params(labelsize=12)
        ax2.zaxis.set_tick_params(labelsize=12)
        ax2.scatter3D(*self.vessel.position, color="y", s=40, label="AUV")
        self.axis_equal3d(ax2)
        ax2.legend(fontsize=14)
        plt.show()

    def update_sensor_readings_with_plots_multi(self):
        """
        Updates the sonar data array for multiple vessels and renders the simulations as 3D plots. Used for debugging.
        """
        print("Time: {}, Nearby Obstacles: {}".format(self.total_t_steps, [len(nearby) for nearby in self.nearby_obstacles_multi]))
        for i, vessel in enumerate(self.vessels):
            self.multivessels_sensor_readings[i] = np.zeros(shape=self.sensor_suite, dtype=float)
            axes = self.plot3D_multi(wps_on=False)  # Get the axes for multiple vessels

    def calculate_object_distance(self, vessel, alpha, beta, obstacle):
        """
        Searches along a sonar ray for an object
        """
        s = 0
        while s < self.sonar_range:
            x = vessel.position[0] + s*np.cos(alpha)*np.cos(beta)
            y = vessel.position[1] + s*np.sin(alpha)*np.cos(beta)
            z = vessel.position[2] - s*np.sin(beta)
            if np.linalg.norm(obstacle.position - [x,y,z]) <= obstacle.radius:
                break
            else:
                s += 1
        closeness = np.clip(1-(s/self.sonar_range), 0, 1)
        return s, closeness


    def penalize_obstacle_closeness(self, k):
        """
        Calculates the colav reward
        """
        reward_colav = 0
        sensor_suite_correction = 0
        gamma_c = self.sonar_range/2
        epsilon = 0.05
        epsilon_closeness = 0.05
        horizontal_angles = np.linspace(-self.sensor_span[0]/2, self.sensor_span[0]/2, self.sensor_suite[0])
        vertical_angles = np.linspace(-self.sensor_span[1]/2, self.sensor_span[1]/2, self.sensor_suite[1])
        for i, horizontal_angle in enumerate(horizontal_angles):
            horizontal_factor = (1-(abs(horizontal_angle)/horizontal_angles[-1]))
            for j, vertical_angle in enumerate(vertical_angles):
                vertical_factor = (1-(abs(vertical_angle)/vertical_angles[-1]))
                beta = vertical_factor*horizontal_factor + epsilon
                sensor_suite_correction += beta
                reward_colav += (beta*(1/(gamma_c*max(1-self.multivessels_sensor_readings[k][j,i], epsilon_closeness)**2)))**2
        return - reward_colav / sensor_suite_correction

    
    def plot3D(self, wps_on=True):
        """
        Returns 3D plot of path and obstacles.
        """
        ax = self.path.plot_path(wps_on)
        for obstacle in self.obstacles:    
            ax.plot_surface(*obstacle.return_plot_variables(), color='r')
        return self.axis_equal3d(ax)

    def plot3D_multi(self, ax, wps_on=True):
        """
        Returns 3D plot of path and obstacles for multiple vessels.
        """
        axes = []
        for i in range(self.num_vessels):
            ax1 = self.path_multi[i].plot_path(ax, i, wps_on)
            for obstacle in self.obstacles:
                # Plot obstacles on each vessel's path
                ax1.plot_surface(*obstacle.return_plot_variables(), color='r')
            # Set the axes for each vessel
            axes.append(self.axis_equal3d(ax1))
        return axes  # Return the list of axes for multiple vessels

    def axis_equal3d(self, ax):
        """
        Shifts axis in 3D plots to be equal. Especially useful when plotting obstacles, so they appear spherical.
        
        Parameters:
        ----------
        ax : matplotlib.axes
            The axes to be shifted. 
        """
        extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
        sz = extents[:,1] - extents[:,0]
        centers = np.mean(extents, axis=1)
        maxsize = max(abs(sz))
        r = maxsize/2
        for ctr, dim in zip(centers, 'xyz'):
            getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
        return ax


    def check_object_overlap(self, new_obstacle):
        """
        Checks if a new obstacle is overlapping one that already exists or the target position.
        """
        overlaps = False
        # check if it overlaps target:
        for path in self.path_multi:
            if np.linalg.norm(path.get_endpoint() - new_obstacle.position) < new_obstacle.radius + 5:
                return True
        # check if it overlaps already placed objects
        for obstacle in self.obstacles:
            if np.linalg.norm(obstacle.position - new_obstacle.position) < new_obstacle.radius + obstacle.radius + 5:
                overlaps = True
        return overlaps


    def scenario_beginner(self):
        initial_state = np.zeros(6)
        self.current = Current(mu=0, Vmin=0, Vmax=0, Vc_init=0, alpha_init=0, beta_init=0, t_step=0) #Current object with zero velocity
        waypoints = generate_random_waypoints(self.n_waypoints)
        self.path = QPMI(waypoints)
        init_pos = [np.random.uniform(0,2)*(-5), np.random.normal(0,1)*5, np.random.normal(0,1)*5]
        init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos, init_attitude])
        return initial_state

    def scenario_beginner_multi(self):
        """
        For multiple vessels, initialize the beginner scenario.
        """
        initial_state_multi = []
        self.current = Current(mu=0, Vmin=0, Vmax=0, Vc_init=0, alpha_init=0, beta_init=0, t_step=0)
        for i in range(self.num_vessels):
            waypoints = generate_random_waypoints(self.n_waypoints)
            self.path_multi[i] = QPMI(waypoints)
            # Random initial position for each vessel
            init_pos = [np.random.uniform(0,2)*(-5), np.random.normal(0,1)*5, np.random.normal(0,1)*5]
            # Get the initial attitude based on the path direction at the start
            init_attitude = np.array([0, self.path_multi[i].get_direction_angles(0)[1], self.path_multi[i].get_direction_angles(0)[0]])
            # Combine position and attitude for the initial state of the i-th vessel
            initial_state = np.hstack([init_pos, init_attitude])
            initial_state_multi.append(initial_state)  # Append the initial state for the i-th vessel
        # Return the initial state for all vessels
        return initial_state_multi


    def scenario_intermediate(self):
        #print("\t\t\tfunc scenario_intermediate init")
        initial_state = self.scenario_beginner()
        #print("\t\t\tfunc scenario_intermediate got beginner")
        rad = np.random.uniform(4, 10)
        pos = self.path(self.path.length/2)
        self.obstacles.append(Obstacle(radius=rad, position=pos))
        lengths = np.linspace(self.path.length*1/3, self.path.length*2/3, self.n_int_obstacles)
        for l in lengths:
            obstacle_radius = np.random.uniform(low=4,high=10)
            obstacle_coords = self.path(l)
            obstacle = Obstacle(obstacle_radius, obstacle_coords)
            if self.check_object_overlap(obstacle):
                continue
            else:
                self.obstacles.append(obstacle)
        #print("\n\t\tfunc scenario_intermediate generated", len(self.obstacles), "obstacles")
        #print("\t\t\tfunc scenario_intermediate exit")
        return initial_state
    
    def scenario_intermediate_multi(self):
        """
        For multiple vessels, initialize the intermediate scenario.
        """
        initial_state_multi = self.scenario_beginner_multi()
        #print("\t\t\tfunc scenario_intermediate_multi got beginner multi")
        rad = np.random.uniform(4, 10)
        obst_pos = []
        lengths = [[] for _ in range(self.num_vessels)]  # Initialize a list to store lengths for each vessel
        for path in self.path_multi:  # For each vessel's path, place an obstacle at the midpoint
            pos = path(path.length/2)
            obst_pos.append(pos)  # Store the positions for the obstacles
            self.obstacles.append(Obstacle(radius=rad, position=pos))  # Place the obstacle at the midpoint of each path
            lengths.append(np.linspace(path.length*1/3, path.length*2/3, self.n_int_obstacles))  # Generate lengths for each vessel's path
        for i in range(self.num_vessels):
            # For each vessel, place obstacles along its path
            for l in lengths[i]:
                obstacle_radius = np.random.uniform(low=4,high=10)
                obstacle_coords = self.path_multi[i](l)
                obstacle = Obstacle(obstacle_radius, obstacle_coords)
                if self.check_object_overlap(obstacle):
                    continue
                else:
                    self.obstacles.append(obstacle)
        #print("\n\t\tfunc scenario_intermediate_multi generated", len(self.obstacles), "obstacles")
        #print("\t\t\tfunc scenario_intermediate_multi exit")
        return initial_state_multi

    def scenario_proficient(self):
        #print("\t\tfunc scenario_proficient init")
        initial_state = self.scenario_intermediate()
        #print("\t\t\tgot intermediate (", len(self.obstacles), " obstacles)", sep="")
        lengths = np.random.uniform(self.path.length*1/3, self.path.length*2/3, self.n_pro_obstacles)
        #print("\t\t\tgot", len(lengths), "lengths")
        print("")
        n_checks = 0
        while len(self.obstacles) < self.n_pro_obstacles and n_checks < 1000:
            for l in lengths:
                obstacle_radius = np.random.uniform(low=4,high=10)
                obstacle_coords = self.path(l)
                obstacle = Obstacle(obstacle_radius, obstacle_coords)
                if self.check_object_overlap(obstacle):
                    n_checks += 1
                    #print("\r\t\t\tOVERLAP CHECK TRIGGERED", n_checks, "TIMES", end="", flush=True)
                    continue

                else:
                    self.obstacles.append(obstacle)
        print("\t\tfunc scenario_proficient() --> OVERLAP CHECK TRIGGERED", n_checks, "TIMES") if n_checks > 1 else None
        #print("\n\t\t\t", len(self.obstacles), " obstacles in total", sep="")
        #print("\t\tfunc scenario_proficient exit")
        return initial_state

    def scenario_proficient_multi(self):
        """
        For multiple vessels, initialize the proficient scenario.
        """
        initial_state_multi = self.scenario_intermediate_multi()
        #print("\t\t\tgot intermediate multi (", len(self.obstacles), " obstacles)", sep="")
        lengths = [np.random.uniform(self.path_multi[i].length*1/3, self.path_multi[i].length*2/3, self.n_pro_obstacles) for i in range(self.num_vessels)]
        #print("\t\t\tgot", len(lengths), "lengths for each vessel")
        n_checks = 0
        while len(self.obstacles) < self.n_pro_obstacles*self.num_vessels and n_checks < 1000:
            for i in range(self.num_vessels):
                for l in lengths[i]:
                    obstacle_radius = np.random.uniform(low=4,high=10)
                    obstacle_coords = self.path_multi[i](l)
                    obstacle = Obstacle(obstacle_radius, obstacle_coords)
                    if self.check_object_overlap(obstacle):
                        n_checks += 1
                        continue
                    else:
                        self.obstacles.append(obstacle)
        print("\t\tfunc scenario_proficient_multi() --> OVERLAP CHECK TRIGGERED", n_checks, "TIMES") if n_checks > 1 else None
        #print("\n\t\t\t", len(self.obstacles), " obstacles in total", sep="")
        return initial_state_multi

    def scenario_advanced(self):
        initial_state = self.scenario_proficient()
        while len(self.obstacles) < self.n_adv_obstacles: # Place the rest of the obstacles randomly
            s = np.random.uniform(self.path.length*1/3, self.path.length*2/3)
            obstacle_radius = np.random.uniform(low=4,high=10)
            obstacle_coords = self.path(s) + np.random.uniform(low=-(obstacle_radius+10), high=(obstacle_radius+10), size=(1,3))
            obstacle = Obstacle(obstacle_radius, obstacle_coords[0])
            if self.check_object_overlap(obstacle):
                continue
            else:
                self.obstacles.append(obstacle)
        return initial_state
    
    def scenario_advanced_multi(self):
        """
        For multiple vessels, initialize the advanced scenario.
        """
        initial_state_multi = self.scenario_proficient_multi()
        while len(self.obstacles) < self.n_adv_obstacles*self.num_vessels:
            # Place the rest of the obstacles randomly for each vessel
            for i in range(self.num_vessels):
                s = np.random.uniform(self.path_multi[i].length*1/3, self.path_multi[i].length*2/3)
                obstacle_radius = np.random.uniform(low=4,high=10)
                obstacle_coords = self.path_multi[i](s) + np.random.uniform(low=-(obstacle_radius+10), high=(obstacle_radius+10), size=(1,3))
                obstacle = Obstacle(obstacle_radius, obstacle_coords[0])
                if self.check_object_overlap(obstacle):
                    continue
                else:
                    self.obstacles.append(obstacle)
        return initial_state_multi


    def scenario_expert(self):
        initial_state = self.scenario_advanced()
        self.current = Current(mu=0.2, Vmin=0.5, Vmax=1.0, Vc_init=np.random.uniform(0.5, 1), \
                                    alpha_init=np.random.uniform(-np.pi, np.pi), beta_init=np.random.uniform(-np.pi/4, np.pi/4), t_step=self.step_size)
        self.penalize_control = 1.0
        return initial_state
    
    def scenario_expert_multi(self):
        """
        For multiple vessels, initialize the expert scenario.
        """
        initial_state_multi = self.scenario_advanced_multi()
        # Set a more complex current for multiple vessels
        self.current = Current(mu=0.2, Vmin=0.5, Vmax=1.0, Vc_init=np.random.uniform(0.5, 1), \
                                    alpha_init=np.random.uniform(-np.pi, np.pi), beta_init=np.random.uniform(-np.pi/4, np.pi/4), t_step=self.step_size)
        self.penalize_control = 1.0
        return initial_state_multi


    def scenario_test_path(self):
        self.n_waypoints = len(test_waypoints)
        self.path = QPMI(test_waypoints)
        self.current = Current(mu=0, Vmin=0, Vmax=0, Vc_init=0, alpha_init=0, beta_init=0, t_step=0)
        init_pos = [0,0,0]
        init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos, init_attitude])
        return initial_state
    
    def scenario_test_path_multi(self):
        """
        For multiple vessels, initialize the test path scenario.
        """
        self.n_waypoints = len(test_waypoints)
        self.path_multi = [QPMI(test_waypoints) for _ in range(self.num_vessels)]
        self.current = Current(mu=0, Vmin=0, Vmax=0, Vc_init=0, alpha_init=0, beta_init=0, t_step=0)
        initial_state_multi = []
        for i in range(self.num_vessels):
            init_pos = [0, 0, 0]
            # Get the initial attitude based on the path direction at the start for each vessel
            init_attitude = np.array([0, self.path_multi[i].get_direction_angles(0)[1], self.path_multi[i].get_direction_angles(0)[0]])
            # Combine position and attitude for the initial state of the i-th vessel
            initial_state = np.hstack([init_pos, init_attitude])
            initial_state_multi.append(initial_state)  # Append the initial state for the i-th vessel
        # Return the initial state for all vessels
        return initial_state_multi
        

    def scenario_test_path_current(self):
        initial_state = self.scenario_test_path()
        self.current = Current(mu=0, Vmin=0.75, Vmax=0.75, Vc_init=0.75, alpha_init=np.pi/4, beta_init=np.pi/6, t_step=0)
        return initial_state 


    def scenario_test(self):
        initial_state = self.scenario_test_path()
        points = np.linspace(self.path.length/4, 3*self.path.length/4, 3)
        self.obstacles.append(Obstacle(radius=10, position=self.path(self.path.length/2)))
        return initial_state
        """
        radius = 6
        for p in points:
            pos = self.path(p)
            self.obstacles.append(Obstacle(radius=radius, position=pos))
        return initial_state
        """

    def scenario_test_multi(self):
        """
        For multiple vessels, initialize the test scenario.
        """
        initial_state_multi = self.scenario_test_path_multi()
        points = np.linspace(self.path_multi[0].length/4, 3*self.path_multi[0].length/4, 3)
        self.obstacles.append(Obstacle(radius=10, position=self.path_multi[0](self.path_multi[0].length/2)))
        for i in range(1, self.num_vessels):
            self.obstacles.append(Obstacle(radius=10, position=self.path_multi[i](self.path_multi[i].length/2)))
        return initial_state_multi

    """
    原论文的测试场景，保持不变
    """
    def scenario_test_current(self):
        initial_state = self.scenario_test()
        self.current = Current(mu=0, Vmin=0.75, Vmax=0.75, Vc_init=0.75, alpha_init=np.pi/4, beta_init=np.pi/6, t_step=0) # Constant velocity current (reproducability for report)
        return initial_state


    def scenario_horizontal_test(self):
        waypoints = [(0,0,0), (50,0,0), (100,0,0)]
        self.path = QPMI(waypoints)
        self.current = Current(mu=0, Vmin=0, Vmax=0, Vc_init=0, alpha_init=0, beta_init=0, t_step=0)
        self.obstacles = []
        for i in range(7):
            y = -30+10*i
            self.obstacles.append(Obstacle(radius=5, position=[50,y,0]))
        init_pos = [0,0,0]
        init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos, init_attitude])
        return initial_state


    def scenario_vertical_test(self):
        waypoints = [(0,0,0), (50,0,0), (100,0,0)]
        self.path = QPMI(waypoints)
        self.current = Current(mu=0, Vmin=0, Vmax=0, Vc_init=0, alpha_init=0, beta_init=0, t_step=0)
        self.obstacles = []
        for i in range(7):
            z = -30+10*i
            self.obstacles.append(Obstacle(radius=5, position=[50,0,z]))
        init_pos = [0,0,0]
        init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos, init_attitude])
        return initial_state


    def scenario_deadend_test(self):
        waypoints = [(0,0,0), (50,0,0), (100,0,0)]
        self.path = QPMI(waypoints)
        self.current = Current(mu=0, Vmin=0, Vmax=0, Vc_init=0, alpha_init=0, beta_init=0, t_step=0)
        radius = 25
        angles = np.linspace(-90, 90, 10)*np.pi/180
        obstalce_radius = (angles[1]-angles[0])*radius/2
        for ang1 in angles:
            for ang2 in angles:
                x = 30+radius*np.cos(ang1)*np.cos(ang2)
                y = radius*np.cos(ang1)*np.sin(ang2)
                z = -radius*np.sin(ang1)
                self.obstacles.append(Obstacle(obstalce_radius, [x, y, z]))
        init_pos = [0,0,0]
        init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos, init_attitude])
        return initial_state
    


if __name__ == "__main__":
    """
    Test the environment by creating an instance and plotting the 3D path with obstacles.
    """
    env_config = {
        "n_waypoints": 5,
        "n_int_obstacles": 2,
        "n_pro_obstacles": 5,
        "n_adv_obstacles": 10,
        "max_t_steps": 1000,
        "step_size": 1.0,
        "sensor_suite": (10,10),
        "sensor_span": (60,30), # degrees
        "sonar_range": 50,
        "reward_roll": -0.1,
        "reward_rollrate": -0.1,
        "reward_use_rudder": -0.1,
        "reward_use_elevator": -0.1,
        "reward_heading_error": -1.0,
        "reward_pitch_error": -1.0,
        "lambda_reward": 0.5,
        "min_reward": -100
    }
    env = PathColav3d(env_config, scenario="test_path")
    env.plot_section3()

    """
    Test the environment with multiple vessels
    """
    env_config_multi = env_config.copy()
    env_config_multi["num_vessels"] = 2  # Set number of vessels
    env_multi = PathColav3d(env_config_multi, scenario="test_path")
    env_multi.plot_section3()  # Plot for multiple vessels