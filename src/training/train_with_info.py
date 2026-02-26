from src.graph.network import build_graph_7, build_types_7
from src.environment.coordination_game import CoordinationGame, compute_rewards
from src.agents.q_agent import QAgent
from src.utils.plotting import plot_q_traces, plot_mean_rewards

def epsilon_schedule(ep, total_episodes, decay_every):
    explore_episodes = int(0.8 * total_episodes)
    if ep >= explore_episodes:
        return 0.0
    
    steps = ep // decay_every
    eps = 1.0 - 0.01 * steps
    if eps < 0.0:
        eps = 0.0
    return eps

def get_state(agent_name, graph, actions):
    # State is the tuple of neighbor actions
    neighbors = sorted(graph[agent_name])
    return tuple(actions[n] for n in neighbors)

def run(total_episodes=5000, alpha=0.1, gamma=0.95, seed=1, verbose_every=250):
    graph = build_graph_7()
    types = build_types_7()
    game = CoordinationGame()
    
    explore_episodes = int(0.8 * total_episodes)
    decay_every = max(1, explore_episodes // 100)

    # Create agents
    agents = {}
    for name in graph:
        agents[name] = QAgent(name, types[name], alpha=alpha, gamma=gamma, seed=seed)

    # Logs
    mean_reward_X = []
    mean_reward_Y = []
    q_trace = {"Y1": [], "X1": [], "X3": []}

    # Initialize state randomly for each agent
    prev_actions = {}
    for name in agents:
        prev_actions[name] = agents[name].rng.choice([1, 2])

    for ep in range(total_episodes):
        eps = epsilon_schedule(ep, total_episodes, decay_every)

        # 1) each agent forms state based on neighbors' previous actions and chooses action
        states = {}
        for name in agents:
            states[name] = get_state(name, graph, prev_actions)
        
        # 2) each agent chooses action ε-greedy based on its state
        actions = {}
        for name in agents:
            actions[name] = agents[name].act_epsilon_greedy(states[name], eps)
        
        # 3) envorironment gives rewards based on current actions
        rewards = compute_rewards(actions, graph, types, game)

        # 4) next_state is based on current actions for q-update
        next_states = {}
        for name in agents:
            next_states[name] = get_state(name, graph, actions)

        # 5) Q-learning update for all agents
        for name in agents:
            agents[name].update_q(states[name], actions[name], rewards[name], next_states[name])

        # 6) Track mean reward per type
        rx = []
        ry = []
        for name in rewards:
            if types[name] == "X":
                rx.append(rewards[name])
            else:
                ry.append(rewards[name])
        
        mean_reward_X.append(sum(rx) / len(rx))
        mean_reward_Y.append(sum(ry) / len(ry))

        # Track Q-values for specific agents asked in the assignment
        # Q depends on state, so we record Q for the state at this episode
        for key in ["Y1", "X1", "X3"]:
            s = states[key]
            q1 = agents[key].get_q(s, 1)
            q2 = agents[key].get_q(s, 2)
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
            
        prev_actions = actions
        
    return mean_reward_X, mean_reward_Y, q_trace
    
if __name__ == "__main__":
    mean_X, mean_Y, q_trace = run(total_episodes=5000)
    plot_q_traces(q_trace)
    plot_mean_rewards(mean_X, mean_Y)
            
