from matplotlib import pyplot as plt
import os

# Define the directory where figures will be saved
SAVE_DIR = os.path.join("experiments", "results", "figures")

# Create the directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

def save_figure(fig, filename):
    """
    Saves the given figure to the specified filename within the SAVE_DIR.

    Parameters:
        fig (matplotlib.figure.Figure): The figure object to save.
        filename (str): The name for the saved file (e.g., 'plot.png').
    """
    save_path = os.path.join(SAVE_DIR, filename)
    fig.savefig(save_path)
    print(f"Figure saved to {save_path}")

def plot_time_comparison(game_type, rollouts_list, results):
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    axs = axs.ravel()  # Flatten the 2x2 array to make indexing easier

    alpha_values = sorted(set(r['alpha'] for r in results))  # Get unique alpha values

    for i, alpha in enumerate(alpha_values):
        alpha_results = [r for r in results if r['alpha'] == alpha]
        rollouts = [r['rollouts'] for r in alpha_results]
        times_dr = [r['avg_time_dr'] for r in alpha_results]
        times_naive = [r['avg_time_naive'] for r in alpha_results]

        axs[i].plot(rollouts, times_dr, '-o', label=f'DR (α={alpha})', color='blue')
        axs[i].plot(rollouts, times_naive, '--s', label='Naive MCTS', color='red')

        axs[i].set_xlabel('Number of Rollouts')
        axs[i].set_ylabel('Average Time per Game (seconds)')
        axs[i].set_title(f'Execution Time Comparison (α={alpha})')
        axs[i].legend()
        axs[i].set_xscale('log')
        axs[i].set_yscale('log')

        for x, y in zip(rollouts, times_dr):
            axs[i].annotate(f'{y:.2f}s', (x, y), textcoords="offset points", xytext=(0, 10), ha='center')
        for x, y in zip(rollouts, times_naive):
            axs[i].annotate(f'{y:.2f}s', (x, y), textcoords="offset points", xytext=(0, -15), ha='center')

    plt.tight_layout()
    save_figure(fig, f"{game_type}_time_comparison_all_alphas.png")
    plt.close()


def plot_results(game_type, rollouts_list, alpha_list, results):
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))

    for alpha in alpha_list:
        alpha_results = [r for r in results if r['alpha'] == alpha]
        rollouts = [r['rollouts'] for r in alpha_results]
        success_rates_dr = [r['success_rate_dr'] for r in alpha_results]
        success_rates_naive = [r['success_rate_naive'] for r in alpha_results]

        axes[0, 0].plot(rollouts, success_rates_dr, '-o', label=f'DR (α={alpha})')
        axes[0, 0].plot(rollouts, success_rates_naive, '--s', label=f'Naive (α={alpha})')

    axes[0, 0].set_xlabel('Number of Rollouts')
    axes[0, 0].set_ylabel('Success Rate')
    axes[0, 0].set_title('Success Rate Comparison')
    axes[0, 0].legend()

    if game_type == 'tictactoe':
        for alpha in alpha_list:
            alpha_results = [r for r in results if r['alpha'] == alpha]
            rollouts = [r['rollouts'] for r in alpha_results]
            draw_rates = [r['draw_rate'] for r in alpha_results]
            axes[0, 1].plot(rollouts, draw_rates, '-^', label=f'α={alpha}')

        axes[0, 1].set_xlabel('Number of Rollouts')
        axes[0, 1].set_ylabel('Draw Rate')
        axes[0, 1].set_title('Draw Rate Comparison')
        axes[0, 1].legend()
    else:  # Gridworld
        for alpha in alpha_list:
            alpha_results = [r for r in results if r['alpha'] == alpha]
            rollouts = [r['rollouts'] for r in alpha_results]
            avg_rewards_dr = [r['avg_reward_dr'] for r in alpha_results]
            avg_rewards_naive = [r['avg_reward_naive'] for r in alpha_results]

            axes[0, 1].plot(rollouts, avg_rewards_dr, '-o', label=f'DR (α={alpha})')
            axes[0, 1].plot(rollouts, avg_rewards_naive, '--s', label=f'Naive (α={alpha})')

        axes[0, 1].set_xlabel('Number of Rollouts')
        axes[0, 1].set_ylabel('Average Reward')
        axes[0, 1].set_title('Average Reward Comparison')
        axes[0, 1].legend()

    for alpha in alpha_list:
        alpha_results = [r for r in results if r['alpha'] == alpha]
        rollouts = [r['rollouts'] for r in alpha_results]
        success_diff = [r['success_rate_dr'] - r['success_rate_naive'] for r in alpha_results]

        axes[1, 0].plot(rollouts, success_diff, '-o', label=f'α={alpha}')

    axes[1, 0].set_xlabel('Number of Rollouts')
    axes[1, 0].set_ylabel('Success Rate Difference (DR - Naive)')
    axes[1, 0].set_title('Performance Difference')
    axes[1, 0].legend()

    axes[1, 1].axis('off')  # This subplot is left empty

    plt.tight_layout()
    save_figure(fig, f"{game_type}_comparison.png")
    plt.close()
