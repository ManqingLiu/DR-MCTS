from typing import List, Tuple



class TicTacToe:
    def __init__(self):
        self.board = [' ' for _ in range(9)]
        self.current_player = 'X'

    def available_moves(self) -> List[int]:
        return [i for i, spot in enumerate(self.board) if spot == ' ']

    def make_move(self, position: int) -> float:
        if self.board[position] == ' ':
            self.board[position] = self.current_player
            if self.is_winner(self.current_player):
                reward = 1.0
            elif self.is_draw():
                reward = 0.5
            else:
                reward = 0.0
            self.current_player = 'O' if self.current_player == 'X' else 'X'
            return reward
        return -1.0  # Invalid move

    def is_winner(self, player: str) -> bool:
        win_states = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]  # Diagonals
        ]
        return any(all(self.board[i] == player for i in state) for state in win_states)

    def is_draw(self) -> bool:
        return ' ' not in self.board

    def game_over(self) -> bool:
        return self.is_winner('X') or self.is_winner('O') or self.is_draw()

    def get_state(self) -> Tuple[str]:
        return tuple(self.board)

    def clone(self) -> 'TicTacToe':
        new_game = TicTacToe()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        return new_game


import numpy as np
from typing import List, Tuple, Set

import numpy as np
from typing import List, Tuple, Set, Dict, Optional
import functools


class SmallGo:
    def __init__(self, board_size=5):
        self.board_size = board_size
        # Use numpy array for more efficient operations
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.current_player = 1  # Black goes first
        self.passes = 0
        self.previous_board_hash = None  # Hash instead of full board copy
        self.captured_stones = {1: 0, 2: 0}  # Black: 0, White: 0

        # Cache for legal moves
        self._legal_moves_cache = None
        # Cache for group liberties
        self._liberty_cache = {}  # Format: (x, y) -> set of liberty positions
        # Cache for connected stones
        self._connected_stones_cache = {}  # Format: (x, y) -> set of stone positions

        # Pre-compute neighbor positions for each position on the board
        self._neighbor_positions = {}
        for y in range(board_size):
            for x in range(board_size):
                self._neighbor_positions[(x, y)] = [
                    (nx, ny) for nx, ny in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
                    if 0 <= nx < board_size and 0 <= ny < board_size
                ]

        # Zobrist hashing for board positions
        self._init_zobrist_hashing()

    def _init_zobrist_hashing(self):
        """Initialize Zobrist hashing for faster board state comparison."""
        # Create random bitstrings for each position and stone type
        rng = np.random.RandomState(42)  # Fixed seed for reproducibility
        self._zobrist_table = np.zeros((self.board_size, self.board_size, 3), dtype=np.int64)
        for x in range(self.board_size):
            for y in range(self.board_size):
                for stone in range(1, 3):  # 1=black, 2=white
                    self._zobrist_table[y, x, stone] = rng.randint(0, 2 ** 63 - 1)

        # Calculate current board hash
        self.current_hash = 0

    def _calculate_hash(self) -> int:
        """Calculate the Zobrist hash for the current board state."""
        hash_value = 0
        for y in range(self.board_size):
            for x in range(self.board_size):
                stone = self.board[y, x]
                if stone > 0:  # If there's a stone
                    hash_value ^= self._zobrist_table[y, x, stone]
        return hash_value

    def _update_hash(self, x: int, y: int, old_stone: int, new_stone: int):
        """Update the Zobrist hash incrementally when a stone changes."""
        # XOR out the old stone if it exists
        if old_stone > 0:
            self.current_hash ^= self._zobrist_table[y, x, old_stone]
        # XOR in the new stone if it exists
        if new_stone > 0:
            self.current_hash ^= self._zobrist_table[y, x, new_stone]

    def get_state(self) -> Tuple:
        """Return the state as a tuple for hashing in MCTS nodes."""
        # Use the Zobrist hash as part of the state for more efficient comparison
        return (self.current_hash, self.current_player, self.passes)

    def _invalidate_caches(self):
        """Invalidate caches when the board state changes."""
        self._legal_moves_cache = None
        self._liberty_cache = {}
        self._connected_stones_cache = {}

    def clone(self):
        """Create a deep copy of the game state."""
        new_game = SmallGo(self.board_size)
        # Use numpy's copy method for efficient array copying
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.passes = self.passes
        new_game.previous_board_hash = self.previous_board_hash
        new_game.captured_stones = self.captured_stones.copy()
        new_game.current_hash = self.current_hash
        # We reuse the pre-computed neighbor positions and Zobrist table since they're static
        new_game._neighbor_positions = self._neighbor_positions
        new_game._zobrist_table = self._zobrist_table
        return new_game

    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get the orthogonal neighbors of a position using pre-computed values."""
        return self._neighbor_positions[(x, y)]

    def get_connected_stones(self, x: int, y: int) -> Set[Tuple[int, int]]:
        """Get all connected stones of the same color."""
        cache_key = (x, y)
        if cache_key in self._connected_stones_cache:
            return self._connected_stones_cache[cache_key].copy()

        stone_type = self.board[y, x]
        if stone_type == 0:
            return set()

        visited = set()
        to_check = [(x, y)]

        while to_check:
            cx, cy = to_check.pop(0)
            visited.add((cx, cy))

            for nx, ny in self.get_neighbors(cx, cy):
                if self.board[ny, nx] == stone_type and (nx, ny) not in visited:
                    to_check.append((nx, ny))

        # Cache the result
        self._connected_stones_cache[cache_key] = visited.copy()
        return visited

    def get_group_liberties(self, x: int, y: int) -> Set[Tuple[int, int]]:
        """Get all liberty positions for a group of stones."""
        cache_key = (x, y)
        if cache_key in self._liberty_cache:
            return self._liberty_cache[cache_key].copy()

        stone_type = self.board[y, x]
        if stone_type == 0:
            return set()

        stones = self.get_connected_stones(x, y)
        liberties = set()

        for sx, sy in stones:
            for nx, ny in self.get_neighbors(sx, sy):
                if self.board[ny, nx] == 0:
                    liberties.add((nx, ny))

        # Cache the result
        self._liberty_cache[cache_key] = liberties.copy()
        return liberties

    def group_has_liberties(self, x: int, y: int) -> bool:
        """Check if a group has any liberties."""
        # Quick check: if any neighbor is empty, group has liberties
        for nx, ny in self.get_neighbors(x, y):
            if self.board[ny, nx] == 0:
                return True

        # More thorough check using cached group liberties
        liberties = self.get_group_liberties(x, y)
        return len(liberties) > 0

    # Use LRU cache to avoid recalculating expensive checks
    @functools.lru_cache(maxsize=128)
    def _would_be_suicide(self, x: int, y: int, player: int) -> bool:
        """Check if placing a stone would be suicide."""
        # Convert board to tuple for caching
        board_tuple = tuple(map(tuple, self.board))

        # Place stone temporarily
        old_value = self.board[y, x]
        self.board[y, x] = player

        # Check if this stone has liberties
        has_liberties = False

        # First check if any adjacent position is empty (quick liberty check)
        for nx, ny in self.get_neighbors(x, y):
            if self.board[ny, nx] == 0:
                has_liberties = True
                break

        # If no immediate liberties, check if part of a group with liberties
        if not has_liberties:
            group = self.get_connected_stones(x, y)
            for gx, gy in group:
                for nx, ny in self.get_neighbors(gx, gy):
                    if self.board[ny, nx] == 0:
                        has_liberties = True
                        break
                if has_liberties:
                    break

        # Check if this move captures opponent stones (which makes it not suicide)
        captures_opponent = False
        if not has_liberties:
            opponent = 3 - player
            for nx, ny in self.get_neighbors(x, y):
                if self.board[ny, nx] == opponent:
                    opponent_group = self.get_connected_stones(nx, ny)
                    opponent_has_liberties = False
                    for ox, oy in opponent_group:
                        for onx, ony in self.get_neighbors(ox, oy):
                            if (onx, ony) != (x, y) and self.board[ony, onx] == 0:
                                opponent_has_liberties = True
                                break
                        if opponent_has_liberties:
                            break
                    if not opponent_has_liberties:
                        captures_opponent = True
                        break

        # Restore the board
        self.board[y, x] = old_value

        # If it has liberties or captures opponent stones, it's not suicide
        return not (has_liberties or captures_opponent)

    def check_captures(self, x: int, y: int, apply: bool = True) -> int:
        """Check and remove captured stones after placing at (x,y)."""
        captured = 0
        opponent = 3 - self.current_player

        # Track which groups we've already checked to avoid redundant processing
        checked_groups = set()

        for nx, ny in self.get_neighbors(x, y):
            if self.board[ny, nx] == opponent:
                group_key = self._get_group_key(nx, ny)
                if group_key in checked_groups:
                    continue

                checked_groups.add(group_key)

                # Check if this opponent group has no liberties
                if not self.group_has_liberties(nx, ny):
                    # Get all stones in the captured group
                    stones = self.get_connected_stones(nx, ny)
                    captured += len(stones)

                    if apply:
                        # Remove captured stones
                        for sx, sy in stones:
                            old_stone = self.board[sy, sx]
                            self.board[sy, sx] = 0
                            # Update hash
                            self._update_hash(sx, sy, old_stone, 0)

                        # Invalidate caches after removing stones
                        self._invalidate_caches()

        return captured

    def _get_group_key(self, x: int, y: int) -> int:
        """Generate a unique key for a stone group to avoid redundant checks."""
        # Find the minimum position (lexicographically) in the group to use as key
        group = self.get_connected_stones(x, y)
        return min(group) if group else (x, y)

    def is_legal_move(self, x: int, y: int) -> bool:
        """Check if placing a stone at (x,y) is a legal move."""
        # Check if position is on the board
        if not (0 <= x < self.board_size and 0 <= y < self.board_size):
            return False

        # Check if position is empty
        if self.board[y, x] != 0:
            return False

        # Check for suicide rule
        if self._would_be_suicide(x, y, self.current_player):
            return False

        # Check for ko rule
        if self.previous_board_hash is not None:
            # Place stone temporarily
            old_value = self.board[y, x]
            self.board[y, x] = self.current_player
            old_hash = self.current_hash
            self._update_hash(x, y, 0, self.current_player)

            # Check if any captures would occur
            opponent = 3 - self.current_player
            captured_stones = []
            for nx, ny in self.get_neighbors(x, y):
                if self.board[ny, nx] == opponent:
                    if not self.group_has_liberties(nx, ny):
                        group = self.get_connected_stones(nx, ny)
                        captured_stones.extend(group)

            # Temporarily remove captured stones
            for sx, sy in captured_stones:
                old_stone = self.board[sy, sx]
                self.board[sy, sx] = 0
                self._update_hash(sx, sy, old_stone, 0)

            # Check if resulting board hash matches previous board hash
            if self.current_hash == self.previous_board_hash:
                # Restore captured stones
                for sx, sy in captured_stones:
                    self.board[sy, sx] = opponent
                    self._update_hash(sx, sy, 0, opponent)

                # Restore the original position
                self.board[y, x] = old_value
                self.current_hash = old_hash
                return False

            # Restore captured stones
            for sx, sy in captured_stones:
                self.board[sy, sx] = opponent
                self._update_hash(sx, sy, 0, opponent)

            # Restore the original position
            self.board[y, x] = old_value
            self.current_hash = old_hash

        return True

    def available_moves(self) -> List[int]:
        """Return list of legal moves as flattened indices (0 to board_size^2)."""
        # Use cached legal moves if available
        if self._legal_moves_cache is not None:
            return self._legal_moves_cache

        legal_moves = []
        for y in range(self.board_size):
            for x in range(self.board_size):
                if self.is_legal_move(x, y):
                    legal_moves.append(y * self.board_size + x)

        # Pass is always a legal move (represented as board_size^2)
        legal_moves.append(self.board_size * self.board_size)

        # Cache the result
        self._legal_moves_cache = legal_moves
        return legal_moves

    def make_move(self, position: int) -> float:
        """Make a move given by flattened index. Returns reward."""
        # Increment move counter
        if not hasattr(self, 'move_count'):
            self.move_count = 0
        self.move_count += 1

        # Check if the game is already over
        if self.game_over():
            return 0.0

        # Pass move
        if position == self.board_size * self.board_size:
            self.passes += 1
            self.current_player = 3 - self.current_player  # Switch player
            self._invalidate_caches()

            # Check if the game is over (two consecutive passes)
            if self.passes >= 2:
                score = self.calculate_score()
                return 1.0 if score > 0 else 0.0

            return 0.0  # Game continues after pass

        # Regular move
        x, y = position % self.board_size, position // self.board_size

        if not self.is_legal_move(x, y):
            return -1.0  # Illegal move

        # Store the hash of the current board state for ko rule checking
        self.previous_board_hash = self.current_hash

        # Place the stone
        old_value = self.board[y, x]
        self.board[y, x] = self.current_player
        self._update_hash(x, y, old_value, self.current_player)

        # Check for captures
        captured = self.check_captures(x, y)
        self.captured_stones[self.current_player] += captured

        # Reset passes counter
        self.passes = 0

        # Switch player
        self.current_player = 3 - self.current_player

        # Invalidate caches after making a move
        self._invalidate_caches()

        # Return reward
        if self.game_over():
            score = self.calculate_score()
            return 1.0 if score > 0 else 0.0
        else:
            # Small reward for capturing stones
            return min(0.1 * captured, 0.5)

    def calculate_score(self) -> float:
        """Calculate the score (positive if current player is winning)."""
        # Simple scoring: stones on board + captures
        black_stones = np.sum(self.board == 1)
        white_stones = np.sum(self.board == 2)

        black_score = black_stones + self.captured_stones[1]
        white_score = white_stones + self.captured_stones[2] + 6.5  # komi (6.5 points)

        score = black_score - white_score

        # Adjust for perspective of current player
        if self.current_player == 2:  # white
            score = -score

        return score

    def game_over(self) -> bool:
        """
        Check if the game is over with additional termination conditions:
        1. Two consecutive passes (traditional Go rule)
        2. Maximum move count reached
        3. Board is nearly full with minimal empty spaces
        """
        # Traditional termination condition - two consecutive passes
        if self.passes >= 2:
            return True

        # Maximum move count (reasonable limit for 5×5 Go)
        MAX_MOVES = 75  # Adjust this value as needed
        if hasattr(self, 'move_count') and self.move_count >= MAX_MOVES:
            if not hasattr(self, '_reported_max_moves'):
                self._reported_max_moves = True
                #print(f"Game terminated due to reaching maximum moves: {MAX_MOVES}")
            return True

        # Board fullness check (terminate if board is >90% full)
        filled_positions = np.sum(self.board > 0) if isinstance(self.board, np.ndarray) else sum(
            1 for y in range(self.board_size)
            for x in range(self.board_size)
            if self.board[y][x] != 0
        )
        board_fullness = filled_positions / (self.board_size ** 2)

        if board_fullness > 0.9:  # 90% full
            if not hasattr(self, '_reported_fullness'):
                self._reported_fullness = True
                #print(f"Game terminated due to board fullness: {board_fullness:.1%}")
            return True

        # Game continues
        return False

    def print_board_str(self):
        """Return a string representation of the current board state."""
        symbols = {0: '.', 1: '●', 2: '○'}
        board_str = '  ' + ' '.join(chr(ord('A') + i) for i in range(self.board_size)) + '\n'
        for i, row in enumerate(self.board):
            board_str += f"{self.board_size - i:2d} " + ' '.join(symbols[cell] for cell in row) + '\n'
        board_str += f"Current player: {'Black' if self.current_player == 1 else 'White'}\n"
        board_str += f"Captures - Black: {self.captured_stones[1]}, White: {self.captured_stones[2]}\n"
        return board_str

# Example usage
if __name__ == "__main__":
    game = SmallGo()
    game.print_board()

    # Make some moves
    game.make_move(12)  # Center point (2,2)
    game.make_move(7)  # Point (2,1)
    game.print_board()

    # Available moves
    print(f"Available moves: {game.available_moves()}")

    # Pass move
    game.make_move(25)  # Pass (for 5x5 board)