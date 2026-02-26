class CoordinationGame:
    """
    Coordination game (2 actions: 1 or 2)
    
    Intuition:
    -If both play 1: X agent are happier (get 2), Y agent gets 1
    -If both play 2: Y agent are happier (get 2), X agent gets 1
    -If mismatch: both get 1
    
    Rewards are per EDGE (i with each neighbor j). Total rewward of an agent
    at a timestep is the sum over its neighbors.
    """

    def payoff_per_agent(self, my_action, other_action, my_type):
        #my type is either X or Y
        if my_action == 1 and other_action == 1:
            if my_type == "X":
                return 2
            else:
                return 1
        
        if my_action == 2 and other_action == 2:
            if my_type == "Y":
                return 2
            else:
                return 1
        
        return 1
    
def compute_rewards(actions, graph, types, game):
    """
    actions: dict of agent_id -> action (1 or 2)
    graph: dict of agent_id -> list of neighbor ids (undirected)
    types: dict of agent_id -> type (X or Y)
    game: CoordinationGame
        
    returns: dict of agent_id -> total reward (sum over neighbors)
    """
    rewards = {}
    for i in graph:
        rewards[i] = 0

    for i in graph:
        for j in graph[i]:
            rewards[i] += game.payoff_per_agent(actions[i], actions[j], types[i])
        
    return rewards