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


def run_experiment(game_type: str = "tictactoe", num_games: int = 50,
                   rollouts_list: List[int] = [20, 50, 100],
                   seed: int = 42, alpha: float = 0.0,
                   beta_base: float = 0.5, lambda_param: float = 0.05):
    """
    Run comparative experiments between MCTS variants.
    """
    random.seed(seed)
    np.random.seed(seed)

    results = {
        'MCTS vs IS-MCTS': {'MCTS': [], 'IS-MCTS': []},
        'MCTS vs DR-MCTS': {'MCTS': [], 'DR-MCTS': []},
        'IS-MCTS vs DR-MCTS': {'IS-MCTS': [], 'DR-MCTS': []}
    }

    timing_data = {
        'MCTS': [],
        'IS-MCTS': [],
        'DR-MCTS': []
    }

    for rollouts in rollouts_list:
        wins = {
            'MCTS vs IS-MCTS': {'MCTS': 0, 'IS-MCTS': 0},
            'MCTS vs DR-MCTS': {'MCTS': 0, 'DR-MCTS': 0},
            'IS-MCTS vs DR-MCTS': {'IS-MCTS': 0, 'DR-MCTS': 0}
        }
        draws = {
            'MCTS vs IS-MCTS': 0,
            'MCTS vs DR-MCTS': 0,
            'IS-MCTS vs DR-MCTS': 0
        }

        timing = {
            'MCTS': [],
            'IS-MCTS': [],
            'DR-MCTS': []
        }

        # Display experiment configuration
        print(f"\n{'=' * 50}")
        print(f"Starting experiment: {game_type.upper()} with {rollouts} rollouts")
        print(f"Parameters: α={alpha}, β₀={beta_base}, λ={lambda_param}")
        print(f"{'=' * 50}")

        players = {
            'MCTS': MCTS_naive(max_workers=4),
            'IS-MCTS': MCTS_IS(alpha=alpha, beta_base=beta_base, lambda_param=lambda_param, max_workers=4),
            'DR-MCTS': MCTS_DR(alpha=alpha, beta_base=beta_base, lambda_param=lambda_param, max_workers=4)
        }

        player_pairs = [('MCTS', 'IS-MCTS'), ('MCTS', 'DR-MCTS'), ('IS-MCTS', 'DR-MCTS')]
        for pair in player_pairs:
            pair_key = f"{pair[0]} vs {pair[1]}"
            print(f"\nRunning match: {pair_key}")

            for game_num in tqdm(range(num_games), desc=f"Rollouts: {rollouts}, {game_type}, Pair: {pair_key}"):
                if game_num % 2 == 0:
                    player_x_name, player_o_name = pair
                else:
                    player_o_name, player_x_name = pair
                player_x = players[player_x_name]
                player_o = players[player_o_name]

                # Show more detailed progress for Go games
                sys.stdout.write(f"\rGame {game_num + 1}/{num_games} - {player_x_name} vs {player_o_name} - ")
                sys.stdout.flush()

                # Time the game playing
                start_time = time()

                # Add progress tracking inside play_game function
                result, move_count = play_game_with_progress(player_x, player_o, rollouts, game_type)

                game_time = time() - start_time

                # Show result immediately
                if result == 1:  # Player X won
                    wins[pair_key][player_x_name] += 1
                    outcome = f"{player_x_name} won"
                elif result == -1:  # Player O won
                    wins[pair_key][player_o_name] += 1
                    outcome = f"{player_o_name} won"
                else:  # Draw
                    draws[pair_key] += 1
                    outcome = "Draw"

                print(f"{outcome} in {move_count} moves ({game_time:.2f}s)")

                # Record timing
                timing[player_x_name].append(game_time / 2)  # Approximate time per player
                timing[player_o_name].append(game_time / 2)  # Approximate time per player

            # Display intermediate results after each pair
            print(f"\nResults for {pair_key}:")
            total = wins[pair_key][pair[0]] + wins[pair_key][pair[1]] + draws[pair_key]
            print(f"  {pair[0]}: {wins[pair_key][pair[0]]} wins ({wins[pair_key][pair[0]] / total * 100:.1f}%)")
            print(f"  {pair[1]}: {wins[pair_key][pair[1]]} wins ({wins[pair_key][pair[1]] / total * 100:.1f}%)")
            print(f"  Draws: {draws[pair_key]} ({draws[pair_key] / total * 100:.1f}%)")
            print(
                f"  Average time per move: {sum(timing[pair[0]]) / len(timing[pair[0]]):.3f}s vs {sum(timing[pair[1]]) / len(timing[pair[1]]):.3f}s")

        # Calculate win rates
        for pair_key in wins:
            pair = pair_key.split(' vs ')
            total_games = wins[pair_key][pair[0]] + wins[pair_key][pair[1]] + draws[pair_key]

            win_rate1 = (wins[pair_key][pair[0]] + 0.5 * draws[pair_key]) / total_games
            win_rate2 = (wins[pair_key][pair[1]] + 0.5 * draws[pair_key]) / total_games

            results[pair_key][pair[0]].append(win_rate1)
            results[pair_key][pair[1]].append(win_rate2)

        # Record timing data
        for player in timing:
            if timing[player]:  # Check if any games were played
                timing_data[player].append(sum(timing[player]) / len(timing[player]))
            else:
                timing_data[player].append(0)

    return results, timing_data, rollouts_list


def play_game_with_progress(player_x, player_o, num_rollouts, game_type="tictactoe"):
    """Play a game between two MCTS players with detailed progress tracking."""
    # Create a new game
    if game_type == "tictactoe":
        game = TicTacToe()
    elif game_type == "go":
        game = SmallGo(board_size=5)  # 5x5 Go board
    else:
        raise ValueError(f"Unknown game type: {game_type}")

    move_count = 0
    move_times = []

    while not game.game_over():
        move_start = time()
        if game.current_player == 1:  # X's turn
            action, _ = player_x.mcts_search(game, num_rollouts)
        else:  # O's turn
            action, _ = player_o.mcts_search(game, num_rollouts)

        game.make_move(action)
        move_count += 1

        move_time = time() - move_start
        move_times.append(move_time)

        # For Go, display detailed board state periodically
        if game_type == "go" and move_count % 5 == 0:
            # Calculate average move time
            avg_time = sum(move_times[-5:]) / min(5, len(move_times[-5:]))

            # Get board representation
            if hasattr(game, 'print_board_str'):
                board_str = game.print_board_str()
            else:
                board_str = f"Move {move_count} (no board visualization available)"

            print(f"\nAfter move {move_count} - Avg time: {avg_time:.2f}s\n{board_str}")

        # For all games, show periodic updates
        if move_count % 10 == 0 or (game_type == "go" and move_count % 5 == 0):
            sys.stdout.write(f"\rMove {move_count} - Avg time per move: {sum(move_times) / len(move_times):.2f}s")
            sys.stdout.flush()

    # Determine winner
    if game_type == "tictactoe":
        if hasattr(game, 'is_winner'):
            if game.is_winner('X'):
                return 1, move_count  # Player X won
            elif game.is_winner('O'):
                return -1, move_count  # Player O won
            else:
                return 0, move_count  # Draw
        else:
            return 0, move_count  # Assume draw if can't determine winner
    elif game_type == "go":
        score = game.calculate_score()
        if score > 0:  # Black won (first player)
            return 1, move_count
        elif score < 0:  # White won (second player)
            return -1, move_count
        else:
            return 0, move_count  # Draw (unlikely in Go)


def plot_results(rollouts_list: List[int], results: Dict, time_results: Dict,
                 game_type: str, alpha: float, beta_base: float, lambda_param: float,
                 save_path: str = 'experiments/results'):
    """Plot performance results and save to file."""
    os.makedirs(save_path, exist_ok=True)

    # Performance plots
    for pair, pair_results in results.items():
        plt.figure(figsize=(10, 6))
        csv_data = {'Rollouts': rollouts_list}

        for player, win_rates in pair_results.items():
            means = np.array(win_rates)
            plt.plot(rollouts_list, means, label=player, marker='o')
            csv_data[f'{player}_mean'] = means

        plt.xlabel('Number of Simulations')
        plt.ylabel('Win Rate')
        plt.title(
            f'{pair} - {game_type.capitalize()} Win Rates\n(α={alpha:.2f}, β₀={beta_base:.2f}, λ={lambda_param:.4f})')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        plot_filename = f'{game_type}_{pair.replace(" ", "_")}_performance_alpha_{alpha:.2f}_beta0_{beta_base:.2f}_lambda_{lambda_param:.4f}.png'
        plt.savefig(os.path.join(save_path, plot_filename))
        print(f"Plot saved to {os.path.join(save_path, plot_filename)}")
        plt.close()

        df = pd.DataFrame(csv_data)
        csv_filename = f'{game_type}_{pair.replace(" ", "_")}_results_alpha_{alpha:.2f}_beta0_{beta_base:.2f}_lambda_{lambda_param:.4f}.csv'
        csv_path = os.path.join(save_path, csv_filename)
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")


def run_experiments_for_game(game_type: str, **kwargs):
    """Run experiments for a specific game type with given parameters."""
    results, time_results, rollouts_list = run_experiment(game_type=game_type, **kwargs)
    plot_results(rollouts_list, results, time_results, game_type,
                 kwargs.get('alpha', 0),
                 kwargs.get('beta_base', 0.5),
                 kwargs.get('lambda_param', 0.05),
                 kwargs.get('save_path', 'experiments/results'))
    return results, time_results


import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional

from src.core.Games import TicTacToe, SmallGo
from src.core.MCTS_class import MCTS_naive, MCTS_IS, MCTS_DR
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional

from src.core.Games import TicTacToe, SmallGo
from src.core.MCTS_class import MCTS_naive, MCTS_IS, MCTS_DR


def run_win_rate_experiments(game_type="tictactoe",
                             num_games=50,
                             rollouts_list=[10, 20, 30, 40, 50],
                             beta_values=[0.25, 0.5, 0.75],
                             seed=42,
                             alpha=0.0,
                             lambda_param=0.05,
                             save_path='experiments/results'):
    """
    Run comparative experiments between MCTS and DR-MCTS with different beta values
    and create a row of three figures for the paper.

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
            # This ensures different game sequences for each experiment condition
            current_seed = seed + int(beta_base * 100) + rollout_idx * 1000
            random.seed(current_seed)
            np.random.seed(current_seed)

            print(f"\n{'-' * 40}")
            print(f"β₀={beta_base:.2f}, Running with {rollouts} rollouts (seed: {current_seed})")
            print(f"{'-' * 40}")

            # Initialize players for this specific beta value and rollout count
            # Create NEW instances for each rollout count to prevent any state leakage
            mcts_player = MCTS_naive(max_workers=4, debug=False)
            dr_mcts_player = MCTS_DR(alpha=alpha, beta_base=beta_base, lambda_param=lambda_param, max_workers=4,
                                     debug=False)

            # Initialize counters
            wins = {'MCTS': 0, 'DR-MCTS': 0}
            draws = 0

            # Play num_games for this rollout count
            for game_num in range(num_games):
                # Alternate which player goes first
                if game_num % 2 == 0:
                    player_x_name, player_o_name = 'MCTS', 'DR-MCTS'
                    player_x, player_o = mcts_player, dr_mcts_player
                else:
                    player_o_name, player_x_name = 'MCTS', 'DR-MCTS'
                    player_x, player_o = dr_mcts_player, mcts_player

                # Display progress
                if game_num % 5 == 0 or game_num == num_games - 1:
                    print(f"Game {game_num + 1}/{num_games} - {player_x_name} vs {player_o_name}")

                # Create a fresh game instance for each game
                game = TicTacToe() if game_type == "tictactoe" else SmallGo(board_size=5)

                # Play the game
                move_count = 0
                while not game.game_over():
                    current_player = game.current_player  # 1 is X, other is O

                    if current_player == 1:  # X's turn
                        action, _ = player_x.mcts_search(game, rollouts)
                    else:  # O's turn
                        action, _ = player_o.mcts_search(game, rollouts)

                    game.make_move(action)
                    move_count += 1

                # Determine the winner
                if game_type == "tictactoe":
                    if game.is_winner('X'):
                        result = 1  # X won
                    elif game.is_winner('O'):
                        result = -1  # O won
                    else:
                        result = 0  # Draw
                else:  # Go
                    score = game.calculate_score()
                    if score > 0:
                        result = 1  # Black won
                    elif score < 0:
                        result = -1  # White won
                    else:
                        result = 0  # Draw

                # Record the result
                if result == 1:  # Player X won
                    wins[player_x_name] += 1
                elif result == -1:  # Player O won
                    wins[player_o_name] += 1
                else:  # Draw
                    draws += 1

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


def play_game(player_x, player_o, num_rollouts, game_type="tictactoe"):
    """
    Play a game between two MCTS players.
    Simplified version of play_game_with_progress.
    """
    # Create a new game
    if game_type == "tictactoe":
        game = TicTacToe()
    elif game_type == "go":
        game = SmallGo(board_size=5)  # 5x5 Go board
    else:
        raise ValueError(f"Unknown game type: {game_type}")

    move_count = 0

    while not game.game_over():
        if game.current_player == 1:  # X's turn
            action, _ = player_x.mcts_search(game, num_rollouts)
        else:  # O's turn
            action, _ = player_o.mcts_search(game, num_rollouts)

        game.make_move(action)
        move_count += 1

    # Determine winner
    if game_type == "tictactoe":
        if hasattr(game, 'is_winner'):
            if game.is_winner('X'):
                return 1, move_count  # Player X won
            elif game.is_winner('O'):
                return -1, move_count  # Player O won
            else:
                return 0, move_count  # Draw
        else:
            return 0, move_count  # Assume draw if can't determine winner
    elif game_type == "go":
        score = game.calculate_score()
        if score > 0:  # Black won (first player)
            return 1, move_count
        elif score < 0:  # White won (second player)
            return -1, move_count
        else:
            return 0, move_count  # Draw (unlikely in Go)


def run_test():
    """Run the win rate experiments with the fixed code."""
    game_type = "tictactoe"  # Changed to SmallGo for testing
    num_games = 50  # Reduced for testing
    rollouts_list = [20, 40, 60, 80, 100]
    beta_values = [0.25, 0.5, 0.75]
    seed = 42
    alpha = 0.0
    lambda_param = 0.01
    save_path = 'experiments/results/win_rate'

    # Create save directory
    os.makedirs(save_path, exist_ok=True)

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


#if __name__ == "__main__":
#    run_test()
'''
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


def run_test_smallgo():
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

    # First run a detailed analysis to debug algorithm behavior
    # analyze_smallgo_algorithm_performance()

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


#if __name__ == "__main__":
#    run_test_smallgo()

'''
def run_win_rate_experiments(game_type="go",
                             num_games=20,
                             rollouts_list=[10, 20, 30, 40, 50],
                             beta_values=[0.25, 0.5, 0.75],
                             seed=42,
                             alpha=0.2,  # Increased alpha for more exploration
                             lambda_param=0.05,
                             save_path='experiments/results'):
    """
    Run comparative experiments between MCTS and IS-MCTS with different beta values
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
        results['MCTS vs IS-MCTS'] = {'MCTS': [], 'IS-MCTS': []}

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
            dr_mcts_player = MCTS_IS(
                exploration_weight=exploration_weight,  # Same as MCTS
                alpha=alpha,  # Exploration mixing parameter
                beta_base=beta_base,  # Adaptive mixing parameter
                lambda_param=lambda_param,  # Decay rate
                max_workers=4,
                debug=False
            )

            # Initialize counters
            wins = {'MCTS': 0, 'IS-MCTS': 0}
            draws = 0

            # Play num_games for this rollout count
            for game_num in range(num_games):
                # Critical: For Go, alternate which algorithm plays Black (player 1)
                # This is important because Black has an advantage in Go
                if game_num % 2 == 0:
                    black_player_name, white_player_name = 'IS-MCTS', 'MCTS'
                    black_player, white_player = dr_mcts_player, mcts_player
                else:
                    black_player_name, white_player_name = 'MCTS', 'IS-MCTS'
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
            total_games = wins['MCTS'] + wins['IS-MCTS'] + draws
            print(f"\nResults for β₀={beta_base:.2f} with {rollouts} rollouts:")
            print(f"  MCTS: {wins['MCTS']} wins ({wins['MCTS'] / total_games * 100:.1f}%)")
            print(f"  IS-MCTS: {wins['IS-MCTS']} wins ({wins['IS-MCTS'] / total_games * 100:.1f}%)")
            print(f"  Draws: {draws} ({draws / total_games * 100:.1f}%)")

            # Calculate win rates
            mcts_win_rate = (wins['MCTS'] + 0.5 * draws) / total_games
            ismcts_win_rate = (wins['IS-MCTS'] + 0.5 * draws) / total_games

            # Add to results
            results['MCTS vs IS-MCTS']['MCTS'].append(mcts_win_rate)
            results['MCTS vs IS-MCTS']['IS-MCTS'].append(ismcts_win_rate)

        # Store the results for this beta value
        all_results[beta_base] = results

    # Create the combined figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Colors and styles
    mcts_color = '#66b3ff'  # Blue for MCTS
    # change color to pink
    ismcts_color = '#f9a9a9'
    mcts_style = 'dotted'
    ismcts_style = 'solid'

    # Plot results for each beta value
    for i, beta_base in enumerate(beta_values):
        results = all_results[beta_base]
        pair_key = 'MCTS vs IS-MCTS'
        pair_results = results[pair_key]

        ax = axes[i]

        # Plot MCTS results (dotted line)
        mcts_means = np.array(pair_results['MCTS'])
        ax.plot(rollouts_list, mcts_means, marker='o',
                color=mcts_color, linestyle=mcts_style, linewidth=2)

        # Plot DR-MCTS results (solid line)
        ismcts_means = np.array(pair_results['IS-MCTS'])
        ax.plot(rollouts_list, ismcts_means, marker='s',
                color=ismcts_color, linestyle=ismcts_style, linewidth=2)

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
        pair_key = 'MCTS vs IS-MCTS'
        pair_results = results[pair_key]

        csv_data = {
            'Rollouts': rollouts_list,
            'MCTS': pair_results['MCTS'],
            'IS-MCTS': pair_results['IS-MCTS']
        }

        df = pd.DataFrame(csv_data)
        csv_filename = f"{game_type}_beta{beta_base:.2f}_comparison_{timestamp}.csv"
        csv_path = os.path.join(save_path, csv_filename)
        df.to_csv(csv_path, index=False)
        print(f"Results for β₀={beta_base:.2f} saved to {csv_path}")

    return fig_path, all_results


def run_test_smallgo():
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

    # First run a detailed analysis to debug algorithm behavior
    # analyze_smallgo_algorithm_performance()

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
        mcts_rates = results[beta]['MCTS vs IS-MCTS']['MCTS']
        ismcts_rates = results[beta]['MCTS vs IS-MCTS']['IS-MCTS']
        print(f"β₀={beta:.2f}:")
        print(f"  MCTS win rates:    {[f'{rate:.3f}' for rate in mcts_rates]}")
        print(f"  DR-MCTS win rates: {[f'{rate:.3f}' for rate in ismcts_rates]}")

    return results


if __name__ == "__main__":
    run_test_smallgo()


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import random
import csv
import time

from src.core.Games import SmallGo
from src.core.MCTS_class import MCTS_naive, MCTS_DR


def find_drmcts_win_at_8_moves(max_attempts=100, save_path='experiments/dr_mcts_paper'):
    """
    Run games until finding one where DR-MCTS (White) wins at exactly move 8
    with MCTS playing as Black (first player).

    Uses fixed parameters: rollouts=100, alpha=0, beta_base=0.75, lambda_param=0.01

    Win condition: DR-MCTS (White) must have captured more stones than MCTS (Black)
    after exactly 8 moves. The function ignores territory and board position scores,
    and only considers the capture count difference.

    Args:
        max_attempts: Maximum number of games to try
        save_path: Directory to save visualizations
    """
    os.makedirs(save_path, exist_ok=True)

    # Fixed parameters as specified
    rollouts = 10
    alpha = 0.0
    beta_base = 0.5
    lambda_param = 0.1
    exploration_weight = 1.4

    print(f"Looking for a game where DR-MCTS (White) wins by capturing more stones than MCTS at exactly move 8")
    print(f"Fixed parameters: rollouts={rollouts}, alpha={alpha}, beta_base={beta_base}, lambda_param={lambda_param}")
    print(f"Win condition: DR-MCTS (White) must have more captures than MCTS (Black) after 8 moves")
    print(f"Will try up to {max_attempts} games with different random seeds")

    for attempt in range(max_attempts):
        # Different seed for each attempt
        seed = 42 + attempt
        random.seed(seed)
        np.random.seed(seed)

        print(f"\nAttempt {attempt + 1}/{max_attempts} - Seed: {seed}")

        # Create players with fixed parameters
        mcts_player = MCTS_naive(
            exploration_weight=exploration_weight,
            max_workers=4,
            debug=True  # Enable debug mode
        )

        dr_mcts_player = MCTS_DR(
            exploration_weight=exploration_weight,
            alpha=alpha,
            beta_base=beta_base,
            lambda_param=lambda_param,
            max_workers=4,
            debug=False  # Enable debug mode
        )

        # Create new game
        game = SmallGo(board_size=5)

        # Track game states
        game_states = [(game.clone(), None, 0)]  # (game, last_move, move_number)

        # Track captures
        captures_data = [{"move": 0, "black_captures": 0, "white_captures": 0}]

        # Play for exactly 8 moves
        target_moves = 8
        game_ended_early = False

        for move in range(target_moves):
            current_player = game.current_player

            # Check action-value estimate differences by generating both
            mcts_action, mcts_value = mcts_player.mcts_search(game, rollouts)
            drmcts_action, drmcts_value = dr_mcts_player.mcts_search(game, rollouts)

            if current_player == 1:  # Black's turn (MCTS)
                action, value = mcts_action, mcts_value
                player_name = "MCTS"
                print(
                    f"  Move {move + 1}: MCTS plays at ({action % game.board_size}, {action // game.board_size}) - value: {value:.3f}")
                print(
                    f"    For comparison, DR-MCTS would choose: ({drmcts_action % game.board_size}, {drmcts_action // game.board_size}) - value: {drmcts_value:.3f}")
            else:  # White's turn (DR-MCTS)
                action, value = drmcts_action, drmcts_value
                player_name = "DR-MCTS"
                print(
                    f"  Move {move + 1}: DR-MCTS plays at ({action % game.board_size}, {action // game.board_size}) - value: {value:.3f}")
                print(
                    f"    For comparison, MCTS would choose: ({mcts_action % game.board_size}, {mcts_action // game.board_size}) - value: {mcts_value:.3f}")

            # Calculate coordinates of the move
            if action < game.board_size * game.board_size:  # Not a pass move
                x, y = action % game.board_size, action // game.board_size
                move_coords = (x, y)
            else:
                move_coords = None  # Pass move

            # Make the move
            reward = game.make_move(action)

            # Record the state
            game_states.append((game.clone(), move_coords, move + 1))

            # Track captures
            captures_data.append({
                "move": move + 1,
                "black_captures": game.captured_stones[1],
                "white_captures": game.captured_stones[2]
            })

            # Print capture info
            print(f"    Captures - Black: {game.captured_stones[1]}, White: {game.captured_stones[2]}")

            # Check for a white win by capture at move 8
            if move + 1 == target_moves and current_player == 2:  # Move 8 and White just played
                # Calculate capture difference
                capture_difference = game.captured_stones[2] - game.captured_stones[1]

                # Modified win condition: DR-MCTS (White) has captured more stones than MCTS (Black)
                if capture_difference > 1:  # White has more captures
                    print(f"\n✓ Success! DR-MCTS (White) won by capturing more stones at move {move + 1}!")
                    print(f"   Captures - Black: {game.captured_stones[1]}, White: {game.captured_stones[2]}")
                    print(f"   Capture difference: {capture_difference}")

                    # Save captures data
                    save_captures_to_csv(captures_data, save_path)

                    # Visualize all 8 moves with numbering from 1
                    visualize_8_moves(game_states[1:], save_path)  # Skip initial state

                    return True, game_states, captures_data

            # Check if game ended early
            if game.game_over():
                game_ended_early = True
                break

        # Standard check for game end and winner
        if game.game_over() or move == target_moves - 1:
            # We only care about capture difference
            capture_difference = game.captured_stones[2] - game.captured_stones[1]

            # Don't consider territory or board score wins - only capture advantages
            if capture_difference > 0:  # White has more captures
                if game_ended_early and move + 1 != target_moves:
                    print(
                        f"\nDR-MCTS won by captures, but game ended early at move {move + 1} instead of move {target_moves}")
                else:
                    print(f"\n✓ Success! DR-MCTS (White) won by capturing more stones at exactly move {move + 1}!")
                    print(f"   Captures - Black: {game.captured_stones[1]}, White: {game.captured_stones[2]}")
                    print(f"   Capture difference: {capture_difference}")

                    # Save captures data
                    save_captures_to_csv(captures_data, save_path)

                    # Visualize all 8 moves with numbering from 1
                    visualize_8_moves(game_states[1:], save_path)  # Skip initial state

                    return True, game_states, captures_data
            else:
                if game.captured_stones[1] > game.captured_stones[2]:
                    print(
                        f"Game ended, but MCTS (Black) won with more captures ({game.captured_stones[1]} vs {game.captured_stones[2]})")
                else:
                    print(
                        f"Game ended with no capture advantage (Black: {game.captured_stones[1]}, White: {game.captured_stones[2]})")
        else:
            print(f"Game did not end at move 8 - continuing to next attempt")

    print(f"\nNo success. Could not find a game where DR-MCTS wins at exactly move 8 after {max_attempts} attempts.")
    return False, None, None

def save_captures_to_csv(captures_data, save_path):
    """
    Save capture data to a CSV file.

    Args:
        captures_data: List of dictionaries with capture information
        save_path: Directory to save the CSV
    """
    csv_path = os.path.join(save_path, "drmcts_win_captures.csv")

    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['move', 'black_captures', 'white_captures']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for data in captures_data:
            writer.writerow(data)

    print(f"Captures data saved to {csv_path}")


def visualize_8_moves(game_states, save_path):
    """
    Visualize exactly 8 moves of a SmallGo game with move numbers from 1-8.

    Args:
        game_states: List of (game, last_move, move_number) tuples for moves 1-8
        save_path: Directory to save visualizations
    """
    # Create individual images for each move
    for i, (game, last_move, _) in enumerate(game_states):
        # Create figure
        fig, ax = plt.subplots(figsize=(6, 6))

        # Use 1-indexed move numbers
        move_number = i + 1

        # Visualize board with consistent numbering
        visualize_go_board_with_numbers(ax, game, game_states[:i + 1], renumber=True)

        # Save figure
        fig_path = os.path.join(save_path, f"drmcts_win_move_{move_number}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved Move {move_number} board to {fig_path}")

    # Create combined 2×4 grid with all 8 moves
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, (game, last_move, _) in enumerate(game_states):
        if i < len(axes):
            visualize_go_board_with_numbers(axes[i], game, game_states[:i + 1], renumber=True)

    # Hide any unused subplots
    for j in range(len(game_states), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()

    # Save combined figure
    combined_path = os.path.join(save_path, f"drmcts_win_all_moves.png")
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved combined visualization to {combined_path}")


def visualize_go_board_with_numbers(ax, game, previous_states, renumber=True):
    """
    Visualize a Go board with numbered stones showing the move sequence.

    Args:
        ax: Matplotlib axis
        game: Current game state
        previous_states: List of all previous states including current
        renumber: If True, use 1-indexed numbering regardless of original move numbers
    """
    board_size = game.board_size

    # Create background grid
    for i in range(board_size):
        ax.axhline(i, color='black', lw=1)
        ax.axvline(i, color='black', lw=1)

    # Set green background
    ax.set_facecolor('#4c7e30')

    # Create dictionary to track move numbers for each position
    move_numbers = {}

    # Process previous states to get move numbers
    for i, (prev_game, last_move, _) in enumerate(previous_states):
        if last_move is not None:
            x, y = last_move
            # Use 1-indexed numbering if renumbering
            move_number = i + 1 if renumber else prev_game.move_count
            move_numbers[(x, y)] = move_number

    # Add stones
    for y in range(board_size):
        for x in range(board_size):
            if game.board[y, x] == 1:  # Black stone (MCTS)
                circle = Circle((x, y), 0.4, color='black', zorder=2)
                ax.add_patch(circle)

                # Add move number if this position has one
                if (x, y) in move_numbers:
                    ax.text(x, y, str(move_numbers[(x, y)]), color='white',
                            ha='center', va='center', fontsize=12, fontweight='bold')

            elif game.board[y, x] == 2:  # White stone (DR-MCTS)
                circle = Circle((x, y), 0.4, facecolor='white', edgecolor='black', lw=1, zorder=2)
                ax.add_patch(circle)

                # Add move number if this position has one
                if (x, y) in move_numbers:
                    ax.text(x, y, str(move_numbers[(x, y)]), color='black',
                            ha='center', va='center', fontsize=12, fontweight='bold')

    # Add coordinate labels
    ax.set_xticks(range(board_size))
    ax.set_yticks(range(board_size))
    ax.set_xticklabels([chr(65 + i) for i in range(board_size)])
    ax.set_yticklabels(range(board_size, 0, -1))

    # Set limits
    ax.set_xlim(-0.5, board_size - 0.5)
    ax.set_ylim(-0.5, board_size - 0.5)

    # Simple legend - MCTS is Black and DR-MCTS is White
    black_circle = Circle((0, 0), 0.4, color='black')
    white_circle = Circle((0, 0), 0.4, facecolor='white', edgecolor='black', lw=1)
    ax.legend([black_circle, white_circle], ['MCTS (Black)', 'DR-MCTS (White)'],
              loc='upper right', fontsize=8, framealpha=0.7)


#if __name__ == "__main__":
#    find_drmcts_win_at_8_moves()