import matplotlib.pyplot as plt

# src/utils/plotting.py

import matplotlib.pyplot as plt


def plot_q_traces(q_trace):
    for agent_name in q_trace:
        q1_vals = [q[0] for q in q_trace[agent_name]]
        q2_vals = [q[1] for q in q_trace[agent_name]]

        plt.figure()
        plt.plot(q1_vals, label="Q(action=1)")
        plt.plot(q2_vals, label="Q(action=2)")
        plt.title("Q-values for " + agent_name)
        plt.xlabel("Episode")
        plt.ylabel("Q-value")
        plt.legend()
        plt.grid(True)
        plt.show()


def moving_average(data, window_size):
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window_size + 1)
        window = data[start:i+1]
        smoothed.append(sum(window) / float(len(window)))
    return smoothed


def plot_mean_rewards(mean_reward_X, mean_reward_Y):
    window = 50  # smoothing window

    smooth_X = moving_average(mean_reward_X, window)
    smooth_Y = moving_average(mean_reward_Y, window)

    plt.figure()
    plt.plot(smooth_X, label="Mean reward X (smoothed)")
    plt.plot(smooth_Y, label="Mean reward Y (smoothed)")
    plt.xlabel("Episode")
    plt.ylabel("Mean reward")
    plt.title("Mean reward per type (moving average)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_mean_rewards_comparison(no_X, no_Y, nb_X, nb_Y, window=50):
    smooth_no_X = moving_average(no_X, window)
    smooth_no_Y = moving_average(no_Y, window)
    smooth_nb_X = moving_average(nb_X, window)
    smooth_nb_Y = moving_average(nb_Y, window)
    
    plt.figure(figsize=(10, 6))
    plt.plot(smooth_no_X, label="No-info X", linestyle="--")
    plt.plot(smooth_no_Y, label="No-info Y", linestyle="--")
    plt.plot(smooth_nb_X, label="Neighbor-info X")
    plt.plot(smooth_nb_Y, label="Neighbor-info Y")
    plt.xlabel("Episode")
    plt.ylabel("Mean reward")
    plt.title("No-Info vs Neighbor-Info Learning")
    plt.legend()
    plt.grid(True)
    plt.show()