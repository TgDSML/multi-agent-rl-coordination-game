# Multi-Agent Reinforcement Learning in a Coordination Game

This project implements a multi-agent Q-learning framework for studying coordination dynamics in a sparse graphical game.

It was developed as part of the MSc course **Machine Learning** (MSc in Artificial Intelligence).

---

##  Problem Description

We study a repeated coordination game played by 7 agents connected through a sparse interaction graph.

- Each agent belongs to one of two types:
  - **Type X** (column players) prefer action 1
  - **Type Y** (row players) prefer action 2
- Agents do not know the payoff structure.
- Agents learn through **self-play** using Q-learning with ε-greedy exploration.

Two scenarios are examined:

1. **No Information (Bandit Case)**  
   Agents do not observe neighbors’ actions.

2. **Neighbor Information (State-Based Q-learning)**  
   Each agent’s state is defined by the actions of its neighbors.

The objective is to investigate:
- Whether agents converge to equilibrium
- How information availability affects convergence
- How graph topology influences coordination outcomes

---

##  Methodology

- Q-learning with learning rate α
- Discount factor γ = 0.95
- Stepwise ε-decay:
  - ε starts at 1.0
  - Decreases by 0.01 every `decay_every` episodes
  - Exploration lasts for 80% of total episodes
  - Final 20% runs pure exploitation

Training duration: **5000 episodes**

---

## Experiments & Outputs

For both scenarios, the following are plotted:

- Q-values over episodes for agents:
  - Y1
  - X1
  - X3
- Mean total reward per agent type (X vs Y)

These plots allow us to analyze convergence behavior and equilibrium selection.

---
##  Project Structure

multi-agent-rl-coordination-game/
│
├── src/
│ ├── agents/
│ │ └── q_agent.py
│ ├── graph/
│ │ └── network.py
│ ├── training/
│ │ ├── train_no_info.py
│ │ └── train_with_info.py
│ └── utils/
│ └── plotting.py
│
├── report/
│ └── Reinforcement_Learning_Assignment.pdf


---

##  How to Run

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

Run No-Information scenario:
python -m src.training.train_no_info

Run Neighbor-Information scenario:
python -m src.training.train_with_info

Plots will be displayed automatically.
