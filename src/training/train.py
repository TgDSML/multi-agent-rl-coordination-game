from src.training.train_no_info import run as run_no_info
from src.training.train_with_info import run as run_neighbor

if __name__ == "__main__":
    print("=== NO-INFORMATION (Bandit) ===")
    mx_no, my_no, _ = run_no_info(total_episodes=5000)
    
    print("\n=== NEIGHBOR-INFORMATION (Q-Learning) ===")
    mx_nb, my_nb, _ = run_neighbor(total_episodes=5000)
    
    # Plot comparison
    from src.utils.plotting import plot_mean_rewards_comparison  
    plot_mean_rewards_comparison(mx_no, my_no, mx_nb, my_nb)
