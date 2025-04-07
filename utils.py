import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

from pandas import DataFrame
from cycler import cycler
from gym_auv.utils.controllers import PI, PID

PI = PI()
PID_cross = PID(Kp=1.8, Ki=0.01, Kd=0.035)
PID_cross = PID(Kp=1.8, Ki=0.01, Kd=0.035)


def parse_experiment_info():
    """Parser for the flags that can be passed with the run/train/test scripts."""
    """翻译 用于解析可以与运行/训练/测试脚本一起传递的标志的解析器。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", type=int, help="Which experiment number to run/train/test")
    parser.add_argument("--scenario", default="m_advanced", type=str, help="Which scenario to run")
    parser.add_argument("--controller_scenario", default="expert", type=str, help="Which scenario the agent was trained in")
    parser.add_argument("--controller", default=None, type=int, help="Which model to load as main controller. Requires only integer")
    args = parser.parse_args()
    
    experiment_dir = os.path.join(r"./log", r"Experiment {}".format(args.exp_id))

    if args.controller_scenario is not None:
        agent_path = os.path.join(experiment_dir, args.controller_scenario, "agents")
    else:
        agent_path = os.path.join(experiment_dir, args.scenario, "agents")
    if args.controller is not None:
        agent_path = os.path.join(agent_path, "model_" + str(args.controller) + ".pkl")
    else:
        agent_path = os.path.join(agent_path, "last_model.pkl")
    return experiment_dir, agent_path, args.scenario


def calculate_IAE(sim_df):
    """
    Calculates and prints the integral absolute error provided an environment id and simulation data
    """
    """
    计算并打印给定环境id和模拟数据的积分绝对误差
    """
    IAE_cross = sim_df[r"e"].abs().sum()
    IAE_vertical = sim_df[r"h"].abs().sum()
    print("IAE Cross track: {}, IAE Vertical track: {}".format(IAE_cross, IAE_vertical))
    return IAE_cross, IAE_vertical

def calculate_IAE_multi_vessels(sim_df, num_vessels):
    """
    Calculates and prints the integral absolute error for multiple vessels provided a simulation dataframe
    """
    IAE_cross_total = 0
    IAE_vertical_total = 0
    for i in range(num_vessels):
        """
        Loop through each vessel and calculate the IAE for each one
        """
        IAE_cross = sim_df[r"e_{}".format(i)].abs().sum()
        IAE_vertical = sim_df[r"h_{}".format(i)].abs().sum()
        print("IAE Cross track for vessel {}: {}, IAE Vertical track for all vessel {}: {}".format(i+1, IAE_cross, i+1, IAE_vertical))
        IAE_vertical_total += IAE_vertical
        IAE_cross_total += IAE_cross
    return IAE_cross_total, IAE_vertical_total



def simulate_environment(env, agent):
    global error_labels, current_labels, input_labels, state_labels
    state_labels = [r"$N$", r"$E$", r"$D$", r"$\phi$", r"$\theta$", r"$\psi$", r"$u$", r"$v$", r"$w$", r"$p$", r"$q$", r"$r$"]
    current_labels = [r"$u_c$", r"$v_c$", r"$w_c$"]
    input_labels = [r"$\eta$", r"$\delta_r$", r"$\delta_s$"]
    error_labels = [r"$\tilde{u}$", r"$\tilde{\chi}$", r"e", r"$\tilde{\upsilon}$", r"h"]
    labels = np.hstack(["Time", state_labels, input_labels, error_labels, current_labels])
    
    done = False
    env.reset()
    # env.render(mode='human')  # Optional: render the environment for visual feedback
    while not done:
        action = agent.predict(env.observation, deterministic=True)[0]
        _, _, done, _ = env.step(action)
    errors = np.array(env.past_errors)
    time = np.array(env.time).reshape((env.total_t_steps,1))
    sim_data = np.hstack([time, env.past_states, env.past_actions, errors, env.current_history])
    df = DataFrame(sim_data, columns=labels)
    error_labels = [r"e", r"h"]
    return df

def simulate_environment_multi_vessels(env, agent):
    global error_labels_multi, current_labels_multi, input_labels_multi, state_labels_multi
    labels = []
    for i in range(env.num_vessels):
        """
        Define labels for each vessel in the multi-vessel simulation
        """
        state_labels_multi = []
        current_labels_multi = []
        input_labels_multi = []
        error_labels_multi = []
        state_labels_multi.append([r"$N_{}$".format(i), r"$E_{}$".format(i), r"$D_{}$".format(i),
                                  r"$\phi_{}$".format(i), r"$\theta_{}$".format(i), r"$\psi_{}$".format(i),
                                  r"$u_{}$".format(i), r"$v_{}$".format(i), r"$w_{}$".format(i),
                                  r"$p_{}$".format(i), r"$q_{}$".format(i), r"$r_{}$".format(i)])
        current_labels_multi.append([r"$u_c{}$".format(i), r"$v_c{}$".format(i), r"$w_c{}".format(i)])
        input_labels_multi.append([r"$\eta_{"+str(i)+"}$", r"$\delta_r_{"+str(i)+"}$", r"$\delta_s_{"+str(i)+"}$"])
        error_labels_multi.append([r"$\tilde{u}_{i}$".format(u = 'u', i= i), 
                                   r"$\tilde{chi}_{i}$".format(chi = "chi", i = i), r"e_{}".format(i),
                                 r"$\tilde{upsilon}_{i}$".format(upsilon = "\\upsilon", i = i), r"h_{}".format(i)])
        state_labels_multi = np.array(state_labels_multi)
        current_labels_multi = np.array(current_labels_multi)
        input_labels_multi = np.array(input_labels_multi)
        error_labels_multi = np.array(error_labels_multi)
        labels.extend(state_labels_multi[-1])
        labels.extend(input_labels_multi[-1])  # Append input labels for the current vessel
        labels.extend(error_labels_multi[-1])  # Append error labels for the current vessel
        labels.extend(current_labels_multi[-1])  # Append current labels for the current vessel

    labels = np.hstack(["Time", labels])
    # print("Labels", labels) #Debugging
    done = False
    env.reset()
    # env.render(mode='human')  # Optional: render the environment for visual feedback
    while not done:
        # actions = [None for _ in range(env.num_vessels)]  # Initialize actions for each vessel
        for i in range(env.num_vessels):
            actions = agent.predict(env.observation_multi[i], deterministic=True)[0]  # Predict for each vessel
            # actions.append(action)
        # print("Actions for all vessels: ", actions) #Debugging line to see the actions for each vessel
        _, _, done, _ = env.step_multi(actions)  # Step the environment with actions for all vessels
        # print(actions) # Debugging line to see the actions taken by each vessel
    errors = np.array(env.past_errors_multi)  # Collect errors for all vessels
    time = np.array(env.time).reshape((env.total_t_steps, 1))
    sim_data = [[] for _ in range(env.num_vessels)]  # Initialize a list to hold simulation data for each vessel
    for i in range(env.num_vessels):
        # Collect simulation data for each vessel
        # print("Past states for vessel {}: {}".format(i, env.past_states_multi[i][-1:])) # Debugging line to see the last row of past states for each vessel
        # print("Past actions for vessel {}: {}".format(i, env.past_actions_multi[i][-1:])) # Debugging line to see the last row of past actions for each vessel
        # print("Errors for vessel {}: {}".format(i, errors[i][-1:])) # Debugging line to see the last row of errors for each vessel
        # print("Current history for vessel {}: {}".format(i, env.multivessels_current_history[i][-1:])) # Debugging line to see the current history for each vessel
        sim_data[i] = np.hstack([env.past_states_multi[i],
                                env.past_actions_multi[i],
                                errors[i],
                                env.multivessels_current_history[i]])
    # print(sim_data[2][-1:]) # Debugging line to see the last row of the first vessel's simulation data
    # Combine all vessel data into a single array
    sim_data_combined = []
    for i in range(env.num_vessels):
        if i == 0:
            sim_data_combined = sim_data[i]
        else:
            sim_data_combined = np.hstack((sim_data_combined, sim_data[i]))
    # Combine time with the simulation data
    # print("Time", time[-1:])
    sim_data_combined = np.hstack([time, sim_data_combined])  # Combine time with the simulation data
    sim_data_combined = np.array(sim_data_combined)  # Ensure it's a numpy array for consistency
    # print(sim_data_combined[-1:])
    # Create a DataFrame with the combined simulation data
    df = DataFrame(sim_data_combined, columns=labels)
    return df





def set_default_plot_rc():
    """Sets the style for the plots report-ready"""
    colors = (cycler(color= ['#EE6666', '#3388BB', '#88DD89', '#EECC55', '#88BB44', '#FFBBBB']) +
                cycler(linestyle=['-',       '-',      '-',     '--',      ':',       '-.']))
    plt.rc('axes', facecolor='#ffffff', edgecolor='black',
        axisbelow=True, grid=True, prop_cycle=colors)
    plt.rc('grid', color='gray', linestyle='--')
    plt.rc('xtick', direction='out', color='black', labelsize=14)
    plt.rc('ytick', direction='out', color='black', labelsize=14)
    plt.rc('patch', edgecolor='#ffffff')
    plt.rc('lines', linewidth=4)


def plot_attitude(sim_df):
    """Plots the state trajectories for the simulation data"""
    set_default_plot_rc()
    ax = sim_df.plot(x="Time", y=[r"$\phi$",r"$\theta$", r"$\psi$"], kind="line")
    ax.set_xlabel(xlabel="Time [s]",fontsize=14)
    ax.set_ylabel(ylabel="Angular position [rad]",fontsize=14)
    ax.legend(loc="lower right", fontsize=14)
    ax.set_ylim([-np.pi,np.pi])
    plt.show()

def plot_attitude_multi_vessels(sim_df, num_vessels):
    """
    Plots the state trajectories for multiple vessels in the simulation data
    Plot for each vessel's attitude (phi, theta, psi) in a multi-vessel simulation
    """
    for i in range(num_vessels):
        """
        Loop through each vessel and plot their attitude
        """
        set_default_plot_rc()
        ax = sim_df.plot(x="Time", y=[r"$\phi_{}$".format(i), r"$\theta_{}$".format(i), r"$\psi_{}$".format(i)], kind="line")
        ax.set_xlabel(xlabel="Time [s]", fontsize=14)
        ax.set_ylabel(ylabel="Angular position [rad]", fontsize=14)
        ax.legend(loc="lower right", fontsize=14)
        ax.set_ylim([-np.pi, np.pi])
        plt.title("Attitude for Vessel {}".format(i+1))
        plt.show()

def plot_velocity(sim_df):
    """Plots the velocity trajectories for the simulation data"""
    set_default_plot_rc()
    ax = sim_df.plot(x="Time", y=[r"$u$",r"$v$"], kind="line")
    ax.plot(sim_df["Time"], sim_df[r"$w$"], dashes=[3,3], color="#88DD89", label=r"$w$")
    ax.plot([0,sim_df["Time"].iloc[-1]], [1.5,1.5], label=r"$u_d$")
    ax.set_xlabel(xlabel="Time [s]", fontsize=14)
    ax.set_ylabel(ylabel="Velocity [m/s]", fontsize=14)
    ax.legend(loc="lower right", fontsize=14)
    ax.set_ylim([-0.25,2.25])
    plt.show()

def plot_velocity_multi_vessels(sim_df, num_vessels):
    """
    Plots the velocity trajectories for multiple vessels in the simulation data
    Plot for each vessel's velocity (u, v, w) in a multi-vessel simulation
    """
    for i in range(num_vessels):
        set_default_plot_rc()
        ax = sim_df.plot(x="Time", y=[r"$u_{}$".format(i), r"$v_{}$".format(i)], kind="line")
        ax.plot(sim_df["Time"], sim_df[r"$w_{}$".format(i)], dashes=[3,3], color="#88DD89", label=r"$w_{}$".format(i))
        ax.plot([0,sim_df["Time"].iloc[-1]], [1.5,1.5], label=r"$u_d$")
        ax.set_xlabel(xlabel="Time [s]", fontsize=14)
        ax.set_ylabel(ylabel="Velocity [m/s]", fontsize=14)
        ax.legend(loc="lower right", fontsize=14)
        ax.set_ylim([-0.25,2.25])
        plt.title("Velocity for Vessel {}".format(i+1))
        plt.show()

def plot_angular_velocity(sim_df):
    """Plots the angular velocity trajectories for the simulation data"""
    set_default_plot_rc()
    ax = sim_df.plot(x="Time", y=[r"$p$",r"$q$", r"$r$"], kind="line")
    ax.set_xlabel(xlabel="Time [s]", fontsize=14)
    ax.set_ylabel(ylabel="Angular Velocity [rad/s]", fontsize=14)
    ax.legend(loc="lower right", fontsize=14)
    ax.set_ylim([-1,1])
    plt.show()

def plot_angular_velocity_vessels(sim_df, num_vessels):
    """
    Plots the angular velocity trajectories for multiple vessels in the simulation data
    Plot for each vessel's angular velocity (p, q, r) in a multi-vessel simulation
    """
    for i in range(num_vessels):
        set_default_plot_rc()
        ax = sim_df.plot(x="Time", y=[r"$p_{}$".format(i), r"$q_{}$".format(i), r"$r_{}$".format(i)], kind="line")
        ax.set_xlabel(xlabel="Time [s]", fontsize=14)
        ax.set_ylabel(ylabel="Angular Velocity [rad/s]", fontsize=14)
        ax.legend(loc="lower right", fontsize=14)
        ax.set_ylim([-1, 1])
        plt.title("Angular Velocity for Vessel {}".format(i+1))
        plt.show()

def plot_control_inputs(sim_dfs):
    """ Plot control inputs from simulation data"""
    set_default_plot_rc()
    c = ['#EE6666', '#88BB44', '#EECC55']
    for i, sim_df in enumerate(sim_dfs):
        control = np.sqrt(sim_df[r"$\delta_r$"]**2+sim_df[r"$\delta_s$"]**2)
        plt.plot(sim_df["Time"], sim_df[r"$\delta_s$"], linewidth=4, color=c[i])
    plt.xlabel(xlabel="Time [s]", fontsize=14)
    plt.ylabel(ylabel="Normalized Input", fontsize=14)
    plt.legend(loc="lower right", fontsize=14)
    plt.legend([r"$\lambda_r=0.9$", r"$\lambda_r=0.5$", r"$\lambda_r=0.1$"], loc="upper right", fontsize=14)
    plt.ylim([-1.25,1.25])
    plt.show()

def plot_control_inputs_multi_vessels(sim_dfs, num_vessels):
    """
    Plot control inputs from simulation data for multiple vessels
    """
    set_default_plot_rc()
    c = ['#EE6666', '#88BB44', '#EECC55']
    for i in range(num_vessels):
        for j, sim_df in enumerate(sim_dfs):
            control = np.sqrt(sim_df[r"$\delta_r_{"+str(i)+"}$"]**2 + sim_df[r"$\delta_s_{"+str(i)+"}$"]**2) # Calculate control input for each vessel
            plt.plot(sim_df["Time"], sim_df[r"$\delta_s_{"+str(i)+"}$"], linewidth=4, color=c[i], label=r"$\lambda_r={}$".format([0.9, 0.5, 0.1][j]))
    plt.xlabel(xlabel="Time [s]", fontsize=14)
    plt.ylabel(ylabel="Normalized Input", fontsize=14)
    # plt.legend(loc="lower right", fontsize=14)
    plt.legend([r"$\lambda_r=0.9$", r"$\lambda_r=0.5$", r"$\lambda_r=0.1$"], loc="upper right", fontsize=14)
    plt.ylim([-1.25, 1.25])  # Set y-limits for better visibility
    plt.title("Control Inputs for Multiple Vessels")
    plt.show()

def plot_control_errors(sim_dfs):
    """
    Plot control inputs from simulation data
    """
    #error_labels = [r'e', r'h']
    set_default_plot_rc()
    c = ['#EE6666', '#88BB44', '#EECC55']
    for i, sim_df in enumerate(sim_dfs):
        error = np.sqrt(sim_df[r"e"]**2+sim_df[r"h"]**2)
        plt.plot(sim_df["Time"], error, linewidth=4, color=c[i])
    plt.xlabel(xlabel="Time [s]", fontsize=12)
    plt.ylabel(ylabel="Tracking Error [m]", fontsize=12)
    #plt.ylim([0,15])
    plt.legend([r"$\lambda_r=0.9$", r"$\lambda_r=0.5$", r"$\lambda_r=0.1$"], loc="upper right", fontsize=14)
    plt.show()

def plot_control_errors_multi_vessels(sim_dfs, num_vessels):
    """
    Plot control errors from simulation data for multiple vessels
    """
    set_default_plot_rc()
    c = ['#EE6666', '#88BB44', '#EECC55']
    for i in range(num_vessels):
        for j, sim_df in enumerate(sim_dfs):
            error = np.sqrt(sim_df[r"e_{}".format(i)]**2 + sim_df[r"h_{}".format(i)]**2)  # Calculate tracking error for each vessel
            plt.plot(sim_df["Time"], error, linewidth=4, color=c[i], label=r"$vessel_{}$".format(i+1) + r" ($\lambda_r={}$)".format([0.9, 0.5, 0.1][j]))
    plt.xlabel(xlabel="Time [s]", fontsize=14)
    plt.ylabel(ylabel="Tracking Error [m]", fontsize=14)
    # plt.ylim([0, 15])  # Set y-limits for better visibility
    plt.legend(loc="upper right", fontsize=14)
    plt.title("Control Errors for Multiple Vessels")
    plt.show()

def plot_3d(env, sim_df):
    """
    Plots the AUV path in 3D inside the environment provided.
    """
    plt.rcdefaults()
    plt.rc('lines', linewidth=3)
    ax = env.plot3D()#(wps_on=False)
    ax.plot3D(sim_df[r"$N$"], sim_df[r"$E$"], sim_df[r"$D$"], color="#EECC55", label="AUV Path")#, linestyle="dashed")
    ax.set_xlabel(xlabel="North [m]", fontsize=14)
    ax.set_ylabel(ylabel="East [m]", fontsize=14)
    ax.set_zlabel(zlabel="Down [m]", fontsize=14)
    ax.legend(loc="upper right", fontsize=14)
    plt.show()


def plot_multiple_3d(env, sim_dfs, num_vessels):
    """
    Plots the AUV paths in 3D for multiple vessels inside the environment provided.
    """
    plt.rcdefaults()
    plt.rc('lines', linewidth=3)
    ax = plt.axes(projection='3d')
    axes = env.plot3D_multi(ax)   # axes is a list of Axes3D objects for each vessel
    for i in range(num_vessels):
        axes[i].plot3D(
            sim_dfs[r"$N_{}$".format(i)],
            sim_dfs[r"$E_{}$".format(i)],
            sim_dfs[r"$D_{}$".format(i)],
            color="#EECC55",
            label="AUV Path Vessel {}".format(i + 1)
        )  # Plot the path for each vessel
        axes[i].set_xlabel(xlabel="North [m]", fontsize=10)
        axes[i].set_ylabel(ylabel="East [m]", fontsize=10)
        axes[i].set_zlabel(zlabel="Down [m]", fontsize=10)
        axes[i].legend(loc="upper right", fontsize=8)  # Add legend for each vessel
        
    plt.show()
    

def plot_current_data(sim_df):
    set_default_plot_rc()
    #---------------Plot current intensity------------------------------------
    ax1 = sim_df.plot(x="Time", y=current_labels, linewidth=4, style=["-", "-", "-"] )
    ax1.set_title("Current", fontsize=18)
    ax1.set_xlabel(xlabel="Time [s]", fontsize=14)
    ax1.set_ylabel(ylabel="Velocity [m/s]", fontsize=14)
    ax1.set_ylim([-1.25,1.25])
    ax1.legend(loc="right", fontsize=14)
    #ax1.grid(color='k', linestyle='-', linewidth=0.1)
    plt.show()
    #---------------Plot current direction------------------------------------
    """
    ax2 = ax1.twinx()
    ax2 = sim_df.plot(x="Time", y=[r"$\alpha_c$", r"$\beta_c$"], linewidth=4, style=["-", "--"] )
    ax2.set_title("Current", fontsize=18)
    ax2.set_xlabel(xlabel="Time [s]", fontsize=12)
    ax2.set_ylabel(ylabel="Direction [rad]", fontsize=12)
    ax2.set_ylim([-np.pi, np.pi])
    ax2.legend(loc="right", fontsize=12)
    ax2.grid(color='k', linestyle='-', linewidth=0.1)
    plt.show()
    """

def plot_current_data_multi_vessels(sim_df, num_vessels):
    set_default_plot_rc()
    color = ["#EE6666", "#3388BB", "#88DD89"]
    for i in range(num_vessels):
        """
        Plot current data for each vessel in the multi-vessel simulation
        """
        # Plot current intensity
        ax1 = sim_df.plot(x="Time", y=[r"$u_c{}$".format(i), r"$v_c{}$".format(i), r"$w_c{}".format(i)], linewidth=4, style=["-", "-", "-"], color=color[i])
        ax1.set_title("Current Intensity for Vessel {}".format(i + 1), fontsize=18)
        ax1.set_xlabel(xlabel="Time [s]", fontsize=12)
        ax1.set_ylabel(ylabel="Velocity [m/s]", fontsize=12)
        ax1.set_ylim([-1.25, 1.25])
        ax1.legend(loc="upper right", fontsize=8)
    plt.show()


def plot_collision_reward_function():
    horizontal_angles = np.linspace(-70, 70, 300)
    vertical_angles = np.linspace(-70, 70, 300)
    gamma_x = 25
    epsilon = 0.05
    sensor_readings = 0.4*np.ones((300,300))
    image = np.zeros((len(vertical_angles), len(horizontal_angles)))
    for i, horizontal_angle in enumerate(horizontal_angles):
        horizontal_factor = (1-(abs(horizontal_angle)/horizontal_angles[-1]))
        for j, vertical_angle in enumerate(vertical_angles):
            vertical_factor = (1-(abs(vertical_angle)/vertical_angles[-1]))
            beta = horizontal_factor*vertical_factor + epsilon
            image[j,i] = beta*(1/(gamma_x*(sensor_readings[j,i])**4))
    print(image.round(2))
    ax = plt.axes()
    plt.colorbar(plt.imshow(image),ax=ax)
    ax.imshow(image, extent=[-70,70,-70,70])
    ax.set_ylabel("Vertical vessel-relative sensor angle [deg]", fontsize=14)
    ax.set_xlabel("Horizontal vessel-relative sensor angle [deg]", fontsize=14)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    plt.show()


if __name__ == "__main__":
    plot_collision_reward_function()
