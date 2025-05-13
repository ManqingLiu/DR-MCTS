
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import random
import csv

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


if __name__ == "__main__":
   find_drmcts_win_at_8_moves()