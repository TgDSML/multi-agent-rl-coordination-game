import matplotlib.pyplot as plt


def moving_average(data, window_size):
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window_size + 1)
        window = data[start:i+1]
        smoothed.append(sum(window) / float(len(window)))
    return smoothed


def plot_q_traces(q_trace, scenario_label):
    for agent_name in q_trace:
        q1_vals = [q[0] for q in q_trace[agent_name]]
        q2_vals = [q[1] for q in q_trace[agent_name]]

        plt.figure()
        plt.plot(q1_vals, label="Q(action=1)")
        plt.plot(q2_vals, label="Q(action=2)")
        plt.title(f"[{scenario_label}] Q-values for {agent_name}")
        plt.xlabel("Episode")
        plt.ylabel("Q-value")
        plt.legend()
        plt.grid(True)
        plt.show()


def plot_mean_rewards(mean_reward_X, mean_reward_Y, scenario_label, window=50):
    smooth_X = moving_average(mean_reward_X, window)
    smooth_Y = moving_average(mean_reward_Y, window)

    plt.figure()
    plt.plot(smooth_X, label="Mean reward X (smoothed)")
    plt.plot(smooth_Y, label="Mean reward Y (smoothed)")
    plt.xlabel("Episode")
    plt.ylabel("Mean reward")
    plt.title(f"[{scenario_label}] Mean reward per type (moving average)")
    plt.legend()
    plt.grid(True)
    plt.show()