import matplotlib.pyplot as plt
import numpy as np

# Data
rollouts = [20, 40, 60, 80, 100]
mcts_dr_diff = [0.26, 0.52, 0.69, 0.59, 0.79]
mcts_is_diff = [0.16, 0.47, 0.51, 0.65, 0.61]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(rollouts, mcts_dr_diff, color='#1f77b4', marker='o', linestyle='-', linewidth=2, markersize=8, label='DR-MCTS - MCTS')
plt.plot(rollouts, mcts_is_diff, color='#ff7f0e', marker='s', linestyle='--', linewidth=2, markersize=8, label='IS-MCTS - MCTS')

# Customize the plot
#plt.title('Difference in Win Rates Compared to MCTS for Tic-Tac-Toe', fontsize=14)
plt.xlabel('Number of Rollouts', fontsize=12)
plt.ylabel('Difference in Win Rate', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# Set x-axis ticks to match the rollout numbers
plt.xticks(rollouts, fontsize=10)
plt.yticks(fontsize=10)

# Add value labels on the plot
#for i, v in enumerate(mcts_dr_diff):
#    plt.text(rollouts[i], v, f'{v:.2f}', ha='left', va='bottom', fontsize=9, color='#1f77b4')
#for i, v in enumerate(mcts_is_diff):
#    plt.text(rollouts[i], v, f'{v:.2f}', ha='right', va='top', fontsize=9, color='#ff7f0e')

# Adjust y-axis to start from 0
plt.ylim(bottom=0, top=1.0)

# Show the plot
plt.tight_layout()
plt.savefig("experiments/results/tic-tac-toe-win-rate-difference.png")