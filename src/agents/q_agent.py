import random

class QAgent:
    def __init__(self, agent_id, agent_type, alpha=0.1, gamma=0.95, seed=None):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.alpha = alpha
        self.gamma = gamma
        self.rng = random.Random(seed)

        # Q table stored as dict with keys (state, action)
        self.Q = {}

    def get_q(self, state, action):
        key = (state, action)
        if key not in self.Q:
            self.Q[key] = 0.0
        return self.Q[key]
        
    def greedy_action(self, state):
        q1 = self.get_q(state, 1)
        q2 = self.get_q(state, 2)

        if q1 > q2:
            return 1
        if q2 > q1:
            return 2
            
        # If tie, break randomly
        return self.rng.choice([1, 2])
        
    def act_epsilon_greedy(self, state, epsilon):
        #exploration
        if self.rng.random() < epsilon:
            return self.rng.choice([1, 2])
        #exploitation
        return self.greedy_action(state)
        
    def update_q(self, state, action, reward, next_state):
        current = self.get_q(state, action)
        next_best = max(self.get_q(next_state, 1), self.get_q(next_state, 2))
        target = reward + self.gamma * next_best
        self.Q[(state, action)] = current + self.alpha * (target - current)

    def update_bandit(self, state, action, reward):
        current = self.get_q(state, action)
        self.Q[(state, action)] = current + self.alpha * (reward - current)
