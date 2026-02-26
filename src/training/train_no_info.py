from src.graph.network import build_graph_7, build_types_7
from src.environment.coordination_game import CoordinationGame, compute_rewards 
from src.agents.q_agent import QAgent
from src.utils.plotting import plot_q_traces, plot_mean_rewards

def epsilon_schedule(ep, total_episodes, decay_every):
    """
    Step ε-greedy schedule:
    - ε starts at 1.0
    - decreases by 0.01 every 'decay_every' episodes
    - exploration lasts for first 4/5 of total episodes
    - exploitation lasts for last 1/5 of total episodes
    """
    explore_episodes = int(0.8 * total_episodes)

    #exploitation
    if ep >= explore_episodes:
        return 0.0
    
    #exploration
    steps = ep // decay_every
    eps = 1.0 - 0.01 * steps
    if eps < 0.0:
        eps = 0.0
    return eps

def run(total_episodes=5000, alpha=0.1, gamma=0.95, seed=1, verbose_every=250):
    graph = build_graph_7()
    types = build_types_7()
    game = CoordinationGame()
    
    # Decay_every = so that after 0.8 * total_episodes, ε has decayed to 0.0
    # Need about 100 steps of decay (0.01 decrease each step)
    explore_episodes = int(0.8 * total_episodes)
    decay_every = max(1, explore_episodes // 100) # e.g 4000 // 100 = 40 for 5000 total episodes

    # Create agents
    agents = {}
    for name in graph:
        agents[name] = QAgent(name, types[name], alpha=alpha, gamma=gamma, seed=seed)

    #Logs
    mean_reward_X = []
    mean_reward_Y = []
    q_trace = {"Y1": [], "X1": [], "X3": []}

    state = None #No info case: single state

    for ep in range(total_episodes):
        eps = epsilon_schedule(ep, total_episodes, decay_every)

        # 1) All choose actions
        actions = {}
        for name in agents:
            actions[name] = agents[name].act_epsilon_greedy(state, eps) 
        
        # 2) Compute rewards
        rewards = compute_rewards(actions, graph, types, game)

        # 3) Q-update for all agents (next_state = same_state)
        for name in agents:
            agents[name].update_bandit(state, actions[name], rewards[name])
        
        # 4) Track mean reward per type
        rx = []
        ry = []
        for name in rewards:
            if types[name] == "X":
                rx.append(rewards[name])
            else:
                ry.append(rewards[name])

        mean_reward_X.append(sum(rx) / len(rx))
        mean_reward_Y.append(sum(ry) / len(ry))

        # Q values for specific agents asked in the assignment
        for key in ["Y1", "X1", "X3"]:
            q1 = agents[key].get_q(state, 1)
            q2 = agents[key].get_q(state, 2)
            q_trace[key].append((q1, q2))

        if (ep + 1) % verbose_every == 0:
            print(
                "ep", ep + 1,
                "eps", round(eps, 3),
                "decay_every", decay_every,
                "mean_reward_X", round(mean_reward_X[-1], 3),
                "mean_reward_Y", round(mean_reward_Y[-1], 3),
                "sample actions", {k: actions[k] for k in ["Y1", "X1", "X3"]}
            )
    
    return mean_reward_X, mean_reward_Y, q_trace

if __name__ == "__main__":
    run(total_episodes=5000)

    mean_reward_X, mean_reward_Y, q_trace = run(total_episodes=5000)
    plot_mean_rewards(mean_reward_X, mean_reward_Y)
    plot_q_traces(q_trace)

