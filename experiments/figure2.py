import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import time
from typing import List, Dict, Tuple, Optional

from src.core.Games import TicTacToe, SmallGo
from src.core.MCTS_class import MCTS_naive, MCTS_IS, MCTS_DR
from src.utils.game_helpers import generate_data_tictactoe
from src.visualizations.plot_helpers import plot_results, plot_time_comparison

import sys
from datetime import datetime
from time import time



def run_win_rate_experiments(game_type="go",
                             num_games=20,
                             rollouts_list=[10, 20, 30, 40, 50],
                             beta_values=[0.25, 0.5, 0.75],
                             seed=42,
                             alpha=0.2,  # Increased alpha for more exploration
                             lambda_param=0.05,
                             save_path='experiments/results'):
    """
    Run comparative experiments between MCTS and DR-MCTS with different beta values
    and create a row of three figures for the paper.
    Fixed version for SmallGo experiments.

    Args:
        game_type: Type of game ("tictactoe" or "go")
        num_games: Number of games to play for each pair
        rollouts_list: List of rollout numbers to test
        beta_values: List of beta_base values to test
        seed: Random seed for reproducibility
        alpha: Alpha parameter for DR-MCTS
        lambda_param: Lambda parameter for DR-MCTS
        save_path: Directory to save results
    """
    os.makedirs(save_path, exist_ok=True)

    # Store results for each beta value
    all_results = {}

    # Set game-specific parameters
    board_size = 5  # 5×5 board for SmallGo

    # Run experiments for each beta value
    for beta_base in beta_values:
        print(f"\n{'=' * 50}")
        print(f"Running experiment with β₀={beta_base:.2f}")
        print(f"{'=' * 50}")

        # Initialize the result structure for this beta value
        results = {}
        results['MCTS vs DR-MCTS'] = {'MCTS': [], 'DR-MCTS': []}

        # Run experiment for each rollout value
        for rollout_idx, rollouts in enumerate(rollouts_list):
            # Set a different seed for each combination of beta value and rollouts
            current_seed = seed
            random.seed(current_seed)
            np.random.seed(current_seed)

            print(f"\n{'-' * 40}")
            print(f"β₀={beta_base:.2f}, Running with {rollouts} rollouts (seed: {current_seed})")
            print(f"{'-' * 40}")

            # Important: Use the same exploration parameters for both algorithms
            # to ensure fair comparison
            exploration_weight = 1.4  # Standard PUCT exploration weight

            # Initialize players for this specific beta value and rollout count
            # Create NEW instances for each rollout count to prevent any state leakage
            mcts_player = MCTS_naive(exploration_weight=exploration_weight, max_workers=4, debug=False)
            dr_mcts_player = MCTS_DR(
                exploration_weight=exploration_weight,  # Same as MCTS
                alpha=alpha,  # Exploration mixing parameter
                beta_base=beta_base,  # Adaptive mixing parameter
                lambda_param=lambda_param,  # Decay rate
                max_workers=4,
                debug=False
            )

            # Initialize counters
            wins = {'MCTS': 0, 'DR-MCTS': 0}
            draws = 0

            # Play num_games for this rollout count
            for game_num in range(num_games):
                # Critical: For Go, alternate which algorithm plays Black (player 1)
                # This is important because Black has an advantage in Go
                if game_num % 2 == 0:
                    black_player_name, white_player_name = 'DR-MCTS', 'MCTS'
                    black_player, white_player = dr_mcts_player, mcts_player
                else:
                    black_player_name, white_player_name = 'MCTS', 'DR-MCTS'
                    black_player, white_player = mcts_player, dr_mcts_player

                # Display progress
                if game_num % 5 == 0 or game_num == num_games - 1:
                    print(f"Game {game_num + 1}/{num_games} - Black: {black_player_name}, White: {white_player_name}")

                # Create a fresh game instance for each game
                game = TicTacToe() if game_type == "tictactoe" else SmallGo(board_size=5)

                # Play the game
                move_count = 0
                max_moves = 75  # Prevent excessively long games

                while not game.game_over() and move_count < max_moves:
                    current_player = game.current_player  # 1 is Black, 2 is White

                    if current_player == 1:  # Black's turn
                        action, _ = black_player.mcts_search(game, rollouts)
                    else:  # White's turn
                        action, _ = white_player.mcts_search(game, rollouts)

                    # Make the move and check if it's valid
                    reward = game.make_move(action)

                    # If move was invalid, print a warning and use a random valid move instead
                    if reward < 0:
                        #print(f"Warning: Invalid move detected! Player: {current_player}")
                        valid_moves = game.available_moves()
                        if valid_moves:
                            action = random.choice(valid_moves)
                            game.make_move(action)

                    move_count += 1

                # Determine the winner
                score = game.calculate_score()

                # In Go, score > 0 means Black (player 1) won
                if score > 0:  # Black won
                    wins[black_player_name] += 1
                    print(f"Game {game_num + 1}: {black_player_name} (Black) won, score: {score:.1f}")
                elif score < 0:  # White won
                    wins[white_player_name] += 1
                    print(f"Game {game_num + 1}: {white_player_name} (White) won, score: {-score:.1f}")
                else:  # Draw (rare in Go)
                    draws += 1
                    print(f"Game {game_num + 1}: Draw")

                # Print board state for the last game to help debug
                if game_num == num_games - 1:
                    print("\nFinal board state of last game:")
                    print(game.print_board_str())

            # Calculate and record win rates for this rollout count
            total_games = wins['MCTS'] + wins['DR-MCTS'] + draws
            print(f"\nResults for β₀={beta_base:.2f} with {rollouts} rollouts:")
            print(f"  MCTS: {wins['MCTS']} wins ({wins['MCTS'] / total_games * 100:.1f}%)")
            print(f"  DR-MCTS: {wins['DR-MCTS']} wins ({wins['DR-MCTS'] / total_games * 100:.1f}%)")
            print(f"  Draws: {draws} ({draws / total_games * 100:.1f}%)")

            # Calculate win rates
            mcts_win_rate = (wins['MCTS'] + 0.5 * draws) / total_games
            drmcts_win_rate = (wins['DR-MCTS'] + 0.5 * draws) / total_games

            # Add to results
            results['MCTS vs DR-MCTS']['MCTS'].append(mcts_win_rate)
            results['MCTS vs DR-MCTS']['DR-MCTS'].append(drmcts_win_rate)

        # Store the results for this beta value
        all_results[beta_base] = results

    # Create the combined figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Colors and styles
    mcts_color = '#66b3ff'  # Blue for MCTS
    drmcts_color = '#ff9999'  # Red for DR-MCTS
    mcts_style = 'dotted'
    drmcts_style = 'solid'

    # Plot results for each beta value
    for i, beta_base in enumerate(beta_values):
        results = all_results[beta_base]
        pair_key = 'MCTS vs DR-MCTS'
        pair_results = results[pair_key]

        ax = axes[i]

        # Plot MCTS results (dotted line)
        mcts_means = np.array(pair_results['MCTS'])
        ax.plot(rollouts_list, mcts_means, marker='o',
                color=mcts_color, linestyle=mcts_style, linewidth=2)

        # Plot DR-MCTS results (solid line)
        drmcts_means = np.array(pair_results['DR-MCTS'])
        ax.plot(rollouts_list, drmcts_means, marker='s',
                color=drmcts_color, linestyle=drmcts_style, linewidth=2)

        # Set axis labels
        ax.set_xlabel('Number of Rollouts', fontsize=12)
        if i == 0:  # Only add y-label to the first subplot
            ax.set_ylabel('Win Rate', fontsize=12)

        # Set y-axis limits to be consistent across subplots
        ax.set_ylim(0, 1)

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)

        # Set x-axis ticks to match rollout numbers
        ax.set_xticks(rollouts_list)

        # No legend as requested

    plt.tight_layout()

    # Save the combined figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = os.path.join(save_path, f"{game_type}_beta_comparison_{timestamp}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Combined figure saved to {fig_path}")

    # Save results to CSV for future reference
    for beta_base in beta_values:
        results = all_results[beta_base]
        pair_key = 'MCTS vs DR-MCTS'
        pair_results = results[pair_key]

        csv_data = {
            'Rollouts': rollouts_list,
            'MCTS': pair_results['MCTS'],
            'DR-MCTS': pair_results['DR-MCTS']
        }

        df = pd.DataFrame(csv_data)
        csv_filename = f"{game_type}_beta{beta_base:.2f}_comparison_{timestamp}.csv"
        csv_path = os.path.join(save_path, csv_filename)
        df.to_csv(csv_path, index=False)
        print(f"Results for β₀={beta_base:.2f} saved to {csv_path}")

    return fig_path, all_results


def run_test_small():
    """Run a test experiment with the SmallGo game."""
    game_type = "go"
    num_games = 50  # Reduced for testing
    rollouts_list = [20, 40, 60, 80, 100]  # Reduced for testing
    beta_values = [0.25, 0.5, 0.75]
    seed = 42
    alpha = 0.1
    lambda_param = 0.1
    save_path = 'experiments/results/go_win_rate'

    # Create save directory
    os.makedirs(save_path, exist_ok=True)

    # Then run the full experiment
    print("\n" + "=" * 60)
    print(f"Running Win Rate Comparison Experiments for SmallGo")
    print("=" * 60)

    fig_path, results = run_win_rate_experiments(
        game_type=game_type,
        num_games=num_games,
        rollouts_list=rollouts_list,
        beta_values=beta_values,
        seed=seed,
        alpha=alpha,
        lambda_param=lambda_param,
        save_path=save_path
    )

    print(f"\nWin rate comparison experiments completed.")
    print(f"Combined figure saved to: {fig_path}")

    # Print the actual results to verify they're different
    print("\nWin Rate Results Summary:")
    for beta in beta_values:
        mcts_rates = results[beta]['MCTS vs DR-MCTS']['MCTS']
        drmcts_rates = results[beta]['MCTS vs DR-MCTS']['DR-MCTS']
        print(f"β₀={beta:.2f}:")
        print(f"  MCTS win rates:    {[f'{rate:.3f}' for rate in mcts_rates]}")
        print(f"  DR-MCTS win rates: {[f'{rate:.3f}' for rate in drmcts_rates]}")

    return results


if __name__ == "__main__":
    run_test_small()
