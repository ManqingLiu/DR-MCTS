import numpy as np
import random
from tqdm import tqdm
from src.core.Games import TicTacToe
from src.core.MCTS_class import MCTS_naive, MCTS_DR

def play_game_tic_tac_toe(mcts, game, num_rollouts: int = 100):
    while not game.game_over():
        if game.current_player == 'X':
            action = mcts.mcts_search(game, num_rollouts)
            if isinstance(action, tuple):
                action = action[0]  # Take the first element if it's a tuple
        else:
            action = random.choice(game.available_moves())
        game.make_move(action)

    if game.is_winner('X'):
        return 1  # MCTS (X) wins
    elif game.is_winner('O'):
        return -1  # Random player (O) wins
    else:
        return 0  # Draw


def play_single_player_game(mcts, game, num_rollouts):
    total_reward = 0
    while not game.is_terminal():
        action = mcts.mcts_search(game, num_simulations=num_rollouts)
        if isinstance(action, tuple):
            action = action[0]
        _, reward, _ = game.step(action)
        total_reward += reward
    return total_reward, game.is_winner('X')  # Return both total reward and whether the goal was reached



def play_game_mcts_vs_mcts(mcts1, mcts2, game, num_rollouts: int = 100):
    while not game.game_over():
        if isinstance(game, TicTacToe):
            if game.current_player == 'X':
                action = mcts1.mcts_search(game, num_simulations=num_rollouts)
            else:
                action = mcts2.mcts_search(game, num_simulations=num_rollouts)
        else:  # Gridworld
            action = mcts1.mcts_search(game, num_simulations=num_rollouts)

        if isinstance(action, tuple):
            action = action[0]  # Take the first element if it's a tuple
        game.make_move(action)

    if isinstance(game, TicTacToe):
        if game.is_winner('X'):
            return 1  # MCTS1 (X) wins
        elif game.is_winner('O'):
            return -1  # MCTS2 (O) wins
        else:
            return 0  # Draw
    else:  # Gridworld
        if game.is_winner('X'):
            return 1  # MCTS1 wins
        else:
            return -1  # MCTS1 loses

def calculate_win_rates_naive_vs_dr(num_games: int = 1000, num_rollouts: int = 100):
    naive_wins = 0
    dr_wins = 0
    draws = 0

    for game_num in tqdm(range(num_games), desc="Games Played"):
        game = TicTacToe()
        mcts_naive = MCTS_naive()
        mcts_dr = MCTS_DR()

        # Alternate which MCTS plays as X
        if game_num % 2 == 0:
            result = play_game_mcts_vs_mcts(mcts_dr, mcts_naive, game, num_rollouts)
            if result == 1:
                dr_wins += 1
            elif result == -1:
                naive_wins += 1
            else:
                draws += 1
        else:
            result = play_game_mcts_vs_mcts(mcts_naive, mcts_dr, game, num_rollouts)
            if result == 1:
                naive_wins += 1
            elif result == -1:
                dr_wins += 1
            else:
                draws += 1

    naive_win_rate = naive_wins / num_games
    dr_win_rate = dr_wins / num_games
    draw_rate = draws / num_games

    print(f"\nResults after {num_games} games:")
    print(f"DR MCTS - Win rate: {dr_win_rate:.2f}")
    print(f"Naive MCTS - Win rate: {naive_win_rate:.2f}")
    print(f"Draw rate: {draw_rate:.2f}")

    return dr_win_rate, naive_win_rate, draw_rate


def calculate_true_values_gridworld(size=4, gamma=0.95):
    V = np.zeros((size, size))
    theta = 0.0001
    while True:
        delta = 0
        for i in range(size):
            for j in range(size):
                if (i, j) == (size - 1, size - 1):
                    continue  # Skip terminal state
                v = V[i, j]
                max_v = float('-inf')
                for action in [0, 1, 2, 3]:
                    next_i, next_j = i, j
                    if action == 0:
                        next_i = max(0, i - 1)
                    elif action == 1:
                        next_j = min(size - 1, j + 1)
                    elif action == 2:
                        next_i = min(size - 1, i + 1)
                    elif action == 3:
                        next_j = max(0, j - 1)

                    r = 10 if (next_i, next_j) == (size - 1, size - 1) else -1
                    max_v = max(max_v, r + gamma * V[next_i, next_j])
                V[i, j] = max_v
                delta = max(delta, abs(v - V[i, j]))
        if delta < theta:
            break
    return V

def generate_data_tictactoe(num_games: int):
    data = []
    for _ in range(num_games):
        game = TicTacToe()
        state = game.get_state()
        while not game.game_over():
            action = random.choice(game.available_moves())
            old_state = state
            game.make_move(action)
            new_state = game.get_state()
            reward = 1 if game.is_winner(game.current_player) else 0
            data.append((old_state, action, reward, new_state))
            state = new_state
    return data


def calculate_win_rates(num_games: int = 1000, num_rollouts: int = 100):
    naive_wins = 0
    naive_draws = 0
    dr_wins = 0
    dr_draws = 0

    for _ in range(num_games):
        # Naive MCTS as X
        game = TicTacToe()
        mcts_naive = MCTS_naive()
        result = play_game_tic_tac_toe(mcts_naive, game, num_rollouts)
        if result == 1:
            naive_wins += 1
        elif result == 0:
            naive_draws += 1

        # DR MCTS as X
        game = TicTacToe()
        mcts_dr = MCTS_DR()
        result = play_game_tic_tac_toe(mcts_dr, game, num_rollouts)
        if result == 1:
            dr_wins += 1
        elif result == 0:
            dr_draws += 1

    naive_win_rate = naive_wins / num_games
    naive_draw_rate = naive_draws / num_games
    dr_win_rate = dr_wins / num_games
    dr_draw_rate = dr_draws / num_games

    print(f"Naive MCTS - Win rate: {naive_win_rate:.2f}, Draw rate: {naive_draw_rate:.2f}")
    print(f"DR MCTS - Win rate: {dr_win_rate:.2f}, Draw rate: {dr_draw_rate:.2f}")

    return naive_win_rate, naive_draw_rate, dr_win_rate, dr_draw_rate
