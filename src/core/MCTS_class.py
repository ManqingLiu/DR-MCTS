import math
import random
from typing import List, Tuple
import numpy as np
from sklearn.model_selection import KFold
import concurrent.futures
from functools import partial


class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0
        self.q_hat = {}  # Action-value estimates
        self.v_hat = 0  # State-value estimate
        self.total_return = 0
        self.action_visits = {}  # Count of visits for each action
        self.rewards = {}  # List of rewards for each action


class MCTS_base:
    """Base class for MCTS algorithms with parallelization support"""

    def __init__(self, max_workers: int = 4, debug: bool = False):
        self.root = None
        self.max_workers = max_workers  # Number of parallel workers
        self.debug = debug

    def _run_single_simulation(self, game, sim_id):
        """Run a single MCTS simulation - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _run_single_simulation")

    def mcts_search(self, game, num_simulations: int) -> Tuple[int, float]:
        """Main MCTS search algorithm with parallelization support"""
        self.root = MCTSNode(game.get_state())
        self.root.visits = 1  # Initialize root visits to 1
        self.root.action_visits = {}  # Initialize action_visits

        # For small numbers of simulations, do them sequentially
        if num_simulations <= 1 or self.max_workers <= 1:
            for i in range(num_simulations):
                self._run_single_simulation(game, i)
        else:
            # Run simulations in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Create a partial function with the game parameter
                sim_func = partial(self._run_single_simulation, game)

                # Submit all simulation jobs
                futures = [executor.submit(sim_func, i) for i in range(num_simulations)]

                # Wait for all to complete
                concurrent.futures.wait(futures)

                # Check for exceptions
                for future in futures:
                    try:
                        future.result()  # Will raise any exceptions that occurred
                    except Exception as e:
                        if self.debug:
                            print(f"Error in simulation: {e}")

        # Find best action after all simulations
        best_action = self.get_best_action(self.root)

        # Calculate the value of the best action
        best_value = 0.0
        if best_action in self.root.children:
            best_child = self.root.children[best_action]
            best_value = best_child.value / max(best_child.visits, 1)

        if self.debug:
            print(f"Root visits: {self.root.visits}")
            print(f"Best action: {best_action} with value {best_value}")
            for action, child in self.root.children.items():
                print(f"Action {action}: visits={child.visits}, value={child.value / max(child.visits, 1)}")

        return best_action, best_value

    def get_best_action(self, node: MCTSNode) -> int:
        """Get the best action according to the algorithm's policy"""
        raise NotImplementedError("Subclasses must implement get_best_action")


class MCTS_naive(MCTS_base):
    def __init__(self, exploration_weight: float = 1.4, max_workers: int = 4, debug: bool = False):
        super().__init__(max_workers, debug)
        self.exploration_weight = exploration_weight

    def choose_action(self, node: MCTSNode, game) -> int:
        """Select an action during tree traversal using PUCT formula"""
        if not node.children:
            return random.choice(game.available_moves())

        def score(action: int, child: MCTSNode) -> float:
            if child.visits == 0:
                return float('inf')

            Q = child.value / max(child.visits, 1)

            # Uniform random policy
            policy_score = 1.0 / len(node.children)

            N = node.visits
            n_a = node.action_visits.get(action, 0)  # Use action visits from parent node

            puct_score = Q + self.exploration_weight * policy_score * (math.sqrt(N) / (1 + n_a))
            return puct_score

        return max(node.children.keys(), key=lambda action: score(action, node.children[action]))

    def _run_single_simulation(self, game, sim_id):
        """Run a single MCTS simulation"""
        # Clone the game to avoid state interference
        sim_game = game.clone()
        node = self.root

        # Selection
        while node.children and not sim_game.game_over():
            action = self.choose_action(node, sim_game)

            # Update action visits before taking the action
            node.action_visits[action] = node.action_visits.get(action, 0) + 1

            sim_game.make_move(action)
            node = node.children[action]

        # Expansion
        if not sim_game.game_over():
            for action in sim_game.available_moves():
                if action not in node.children:
                    node.children[action] = MCTSNode(sim_game.get_state(), parent=node)
                    node.children[action].action_visits = {}  # Initialize action_visits

            # Choose an action for expansion
            action = self.get_rollout_action(sim_game)

            # Update action visits
            node.action_visits[action] = node.action_visits.get(action, 0) + 1

            sim_game.make_move(action)
            node = node.children[action]

        # Simulation (rollout)
        while not sim_game.game_over():
            action = self.get_rollout_action(sim_game)
            sim_game.make_move(action)

        # Backpropagation
        value = self.calculate_terminal_value(sim_game)

        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent

        return value  # Return value for potential future use

    def calculate_terminal_value(self, game) -> float:
        """Calculate the terminal state value based on game type"""
        if hasattr(game, 'calculate_score'):
            # For Go-like games
            score = game.calculate_score()
            return 1.0 if score > 0 else 0.0
        elif hasattr(game, 'is_winner'):
            # For TicTacToe-like games
            if hasattr(self.root.state, '__getitem__') and len(self.root.state) > 0:
                # Try to use the first player's symbol
                first_player_symbol = self.root.state[0]
                return 1.0 if game.is_winner(first_player_symbol) else 0.0
            else:
                # Default to checking if 'X' won (for TicTacToe)
                return 1.0 if game.is_winner('X') else 0.0
        else:
            # Generic fallback
            return 0.5  # Draw/neutral outcome

    def get_best_action(self, node: MCTSNode) -> int:
        """Select the best action after search is complete"""
        if not node.children:
            # If no children (should not happen after search), fall back to random
            return random.choice(game.available_moves())

        # Use action visits from the parent node
        return max(node.children.keys(), key=lambda action: node.action_visits.get(action, 0))

    def get_rollout_action(self, game):
        """Get rollout action based on game type"""
        if hasattr(game, 'calculate_score'):  # SmallGo
            return self.go_rollout_policy(game)
        else:  # TicTacToe or others
            return self.tictactoe_rollout_policy(game)

    def tictactoe_rollout_policy(self, game):
        """Rollout policy for TicTacToe"""
        preferred_moves = [4, 0, 2, 6, 8, 1, 3, 5, 7]
        available = game.available_moves()
        for move in preferred_moves:
            if move in available:
                return move
        return random.choice(available)

    def go_rollout_policy(self, game):
        """Simple rollout policy for Go"""
        available = game.available_moves()

        # In Go, avoid playing on the edges in rollouts
        if len(available) > 1:
            board_size = game.board_size
            center_region = []
            for move in available:
                # Skip pass move
                if move == board_size * board_size:
                    continue

                x, y = move % board_size, move // board_size
                # Check if it's not on the edge
                if 0 < x < board_size - 1 and 0 < y < board_size - 1:
                    center_region.append(move)

            if center_region:
                return random.choice(center_region)

        # Fallback to random move
        return random.choice(available)


class MCTS_IS(MCTS_base):
    def __init__(self, exploration_weight: float = 1.4, gamma: float = 1.0, alpha: float = 0.5,
                 beta_base: float = 0.5, lambda_param: float = 0.05, temperature: float = 1.0,
                 max_workers: int = 4, debug: bool = False):
        super().__init__(max_workers, debug)
        self.exploration_weight = exploration_weight
        self.gamma = gamma
        self.alpha = alpha  # Added alpha parameter which was missing
        self.beta_base = beta_base
        self.lambda_param = lambda_param
        self.temperature = temperature

    def choose_action(self, node: MCTSNode, game) -> int:
        if not node.children:
            return random.choice(game.available_moves())

        def score(action: int, child: MCTSNode) -> float:
            if child.visits == 0:
                return float('inf')

            Q = node.q_hat.get(action, 0)

            # Uniform random policy
            policy_score = 1.0 / len(node.children)

            N = node.visits
            n_a = node.action_visits.get(action, 0)

            puct_score = Q + self.exploration_weight * policy_score * (math.sqrt(N) / (1 + n_a))
            return puct_score

        return max(node.children.keys(), key=lambda action: score(action, node.children[action]))

    def get_adaptive_beta(self, node: MCTSNode, action: int) -> float:
        """Adaptively adjust beta based on visits to a state-action pair."""
        visits = node.action_visits.get(action, 0)
        beta = self.beta_base * math.exp(-self.lambda_param * visits)

        if self.debug:
            print(f"State: {node.state}, Action: {action}, Visits: {visits}, Beta: {beta}")

        return beta

    def mc_is_update(self, trajectory: List[Tuple[MCTSNode, int, float, MCTSNode]]):
        """Update values along trajectory using Monte Carlo with Importance Sampling."""
        cumulative_return = 0
        cumulative_importance_ratio = 1.0

        for t, (node, action, reward, next_node) in enumerate(reversed(trajectory)):
            node.visits += 1

            # Initialize attributes if needed
            if not hasattr(node, 'action_visits'):
                node.action_visits = {}
            if not hasattr(node, 'total_return'):
                node.total_return = 0

            # Update cumulative return with discount
            cumulative_return = reward + self.gamma * cumulative_return

            # Monte Carlo update for q_hat
            if action not in node.q_hat:
                node.q_hat[action] = 0
            node.q_hat[action] += (cumulative_return - node.q_hat[action]) / node.action_visits[action]

            # Calculate importance ratio
            try:
                target_prob, behavior_prob = self.get_target_policy_probability(node, action)
                importance_ratio = min(target_prob / (behavior_prob + 1e-10), 10)  # Clipping for stability
            except Exception as e:
                if self.debug:
                    print(f"Policy calculation error: {e}. Using default values.")
                importance_ratio = 1.0

            # Update cumulative importance ratio
            cumulative_importance_ratio *= importance_ratio
            cumulative_importance_ratio = min(cumulative_importance_ratio, 10)  # Clipping for stability

            # IS estimate
            is_estimate = cumulative_importance_ratio * cumulative_return

            # Get adaptive beta based on state-action visits
            adaptive_beta = self.get_adaptive_beta(node, action)

            # Combine MC and IS estimates for v_hat using adaptive beta
            mc_estimate = node.total_return / node.visits
            node.v_hat = adaptive_beta * mc_estimate + (1 - adaptive_beta) * is_estimate

            # Update total return for future MC estimates
            node.total_return += cumulative_return

        return node.v_hat

    def get_target_policy_probability(self, node: MCTSNode, action: int) -> Tuple[float, float]:
        """Calculate target and behavior policy probabilities."""
        try:
            q_values = np.array([node.q_hat.get(a, 0) for a in node.children.keys()])
            softmax_probs = self.softmax(q_values)
            mcts_prob = softmax_probs[list(node.children.keys()).index(action)]
            uniform_prob = 1.0 / len(node.children)
            target_prob = (1 - self.alpha) * mcts_prob + self.alpha * uniform_prob

            # Simple uniform behavior probability
            behavior_prob = uniform_prob

            return target_prob, behavior_prob
        except Exception as e:
            if self.debug:
                print(f"Error in policy probability calculation: {e}")
            # Default to uniform probabilities
            return 1.0 / len(node.children), 1.0 / len(node.children)

    def softmax(self, x):
        """Compute softmax values for vector x with temperature control."""
        try:
            e_x = np.exp((x - np.max(x)) / self.temperature)
            return e_x / (e_x.sum() + 1e-10)  # Avoid division by zero
        except:
            # Return uniform distribution if there's an error
            return np.ones_like(x) / len(x)

    def _run_single_simulation(self, game, sim_id):
        """Run a single MCTS simulation for IS-MCTS."""
        # Clone the game to avoid state interference
        sim_game = game.clone()
        node = self.root

        # Store trajectory for backpropagation
        trajectory = []

        # Selection
        while node.children and not sim_game.game_over():
            action = self.choose_action(node, sim_game)

            # Initialize or update action_visits for this node
            if not hasattr(node, 'action_visits'):
                node.action_visits = {}
            node.action_visits[action] = node.action_visits.get(action, 0) + 1

            # Store the current node, action, and reward before transitioning
            prev_node = node
            reward = sim_game.make_move(action)

            # Get the next node
            if action in node.children:
                node = node.children[action]
            else:
                # This should not happen if choose_action is working correctly
                if self.debug:
                    print(f"Warning: Selected action {action} not in children.")
                break

            # Store the transition in the trajectory
            trajectory.append((prev_node, action, reward, node))

        # Expansion (if needed)
        if not sim_game.game_over():
            # Add all possible actions as children
            for action in sim_game.available_moves():
                if action not in node.children:
                    # Create child node and initialize its attributes
                    node.children[action] = MCTSNode(sim_game.get_state(), parent=node)
                    child_node = node.children[action]
                    child_node.action_visits = {}
                    child_node.q_hat = {}

            # Choose an action for expansion
            action = self.get_rollout_action(sim_game)

            # Update action visits
            node.action_visits[action] = node.action_visits.get(action, 0) + 1

            # Store the current node before transitioning
            prev_node = node

            # Make the move and get the reward
            reward = sim_game.make_move(action)

            # Get the next node
            if action in node.children:
                node = node.children[action]
            else:
                # Create a new node if it doesn't exist
                node.children[action] = MCTSNode(sim_game.get_state(), parent=node)
                node = node.children[action]
                node.action_visits = {}
                node.q_hat = {}

            # Store the transition in the trajectory
            trajectory.append((prev_node, action, reward, node))

        # Simulation (rollout)
        while not sim_game.game_over():
            action = self.get_rollout_action(sim_game)
            reward = sim_game.make_move(action)

        # Check terminal state value
        terminal_value = 0.0
        if hasattr(sim_game, 'calculate_score'):
            # For Go, value is 1 if player 1 (Black) wins, 0 otherwise
            score = sim_game.calculate_score()
            terminal_value = 1.0 if score > 0 else 0.0
        elif hasattr(sim_game, 'is_winner'):
            # For TicTacToe, value is 1 if first player wins, 0 otherwise
            if hasattr(self.root.state, '__getitem__') and len(self.root.state) > 0:
                first_player_symbol = self.root.state[0]
                terminal_value = 1.0 if sim_game.is_winner(first_player_symbol) else 0.0
            else:
                terminal_value = 1.0 if sim_game.is_winner('X') else 0.0

        # Backpropagation using Importance Sampling
        if trajectory:
            # Update values along the trajectory using IS update
            return self.mc_is_update(trajectory)
        else:
            # If no trajectory (e.g., game was already over), update root directly
            self.root.visits += 1
            self.root.v_hat = terminal_value
            return terminal_value

    def get_best_action(self, node: MCTSNode) -> int:
        """Select the best action after search is complete."""
        if not node.children:
            # Should not happen after search, but just in case
            available = []
            if hasattr(node, 'state'):
                # Try to infer game type from state
                if isinstance(node.state, tuple) and len(node.state) == 9:  # TicTacToe
                    available = [i for i, spot in enumerate(node.state) if spot == ' ']
                else:  # Default
                    available = list(range(9))  # TicTacToe board size
            return random.choice(available)

        # Use action visits from the parent node
        return max(node.children.keys(), key=lambda action: node.action_visits.get(action, 0))

    def get_rollout_action(self, game):
        """Get rollout action based on game type."""
        if hasattr(game, 'calculate_score'):  # SmallGo
            return self.go_rollout_policy(game)
        else:  # TicTacToe or others
            return self.tictactoe_rollout_policy(game)

    def tictactoe_rollout_policy(self, game):
        """Rollout policy for TicTacToe."""
        preferred_moves = [4, 0, 2, 6, 8, 1, 3, 5, 7]
        available = game.available_moves()
        for move in preferred_moves:
            if move in available:
                return move
        return random.choice(available)

    def go_rollout_policy(self, game):
        """Simple rollout policy for Go."""
        available = game.available_moves()

        # In Go, avoid playing on the edges in rollouts
        if len(available) > 1:
            board_size = game.board_size
            center_region = []
            for move in available:
                # Skip pass move
                if move == board_size * board_size:
                    continue

                x, y = move % board_size, move // board_size
                # Check if it's not on the edge
                if 0 < x < board_size - 1 and 0 < y < board_size - 1:
                    center_region.append(move)

            if center_region:
                return random.choice(center_region)

        # Fallback to random move
        return random.choice(available)


class MCTS_DR(MCTS_base):
    def __init__(self, exploration_weight: float = 1.4, gamma: float = 0.95, alpha: float = 0.5,
                 beta_base: float = 0.5, lambda_param: float = 0.05,
                 max_workers: int = 4, debug: bool = False):
        super().__init__(max_workers, debug)
        self.exploration_weight = exploration_weight
        self.gamma = gamma
        self.alpha = alpha  # Adding alpha parameter that was missing
        self.beta_base = beta_base
        self.lambda_param = lambda_param
        self.temperature = 1.0
        self.kfold = 2  # 2-fold cross-validation

    def choose_action(self, node: MCTSNode, game) -> int:
        if not node.children:
            return random.choice(game.available_moves())

        def score(action: int, child: MCTSNode) -> float:
            if child.visits == 0:
                return float('inf')

            Q = node.q_hat.get(action, 0)  # Use parent node's Q-value for the action

            # Uniform random policy
            policy_score = 1.0 / len(node.children)

            N = node.visits
            n_a = node.action_visits.get(action, 0)  # Use action visits from parent node

            puct_score = Q + self.exploration_weight * policy_score * (math.sqrt(N) / (1 + n_a))
            return puct_score

        return max(node.children.keys(), key=lambda action: score(action, node.children[action]))

    def get_adaptive_beta(self, node: MCTSNode, action: int) -> float:
        """
        Adaptively adjust beta based on the number of visits to a state-action pair.
        As we get more visits, we reduce beta to favor the DR estimator.
        """
        visits = node.action_visits.get(action, 0)

        # Exponential decay of beta as visits increase
        beta = self.beta_base * math.exp(-self.lambda_param * visits)

        # Debug logging if enabled
        if self.debug:
            print(f"State: {node.state}, Action: {action}, Visits: {visits}, Beta: {beta}")

        return beta

    def mc_dr_update(self, trajectory: List[Tuple[MCTSNode, int, float, MCTSNode]]):
        """Update values along trajectory using Doubly Robust estimator."""
        v_next = 0
        cumulative_weight = 1

        for t, (node, action, reward, next_node) in enumerate(reversed(trajectory)):
            node.visits += 1

            # Initialize data structures if needed
            if not hasattr(node, 'rewards'):
                node.rewards = {}
            if not hasattr(node, 'q_hat'):
                node.q_hat = {}

            # Update rewards for this action
            node.rewards[action] = node.rewards.get(action, []) + [reward + self.gamma * v_next]

            # Estimate V(h) using weighted average of rewards from child nodes
            v_hat = self.estimate_v_hat(node)

            # Estimate Q(h,a) using k-fold cross-validation
            q_hat = self.estimate_q_hat(node, action)

            # Store q_hat for use in PUCT
            node.q_hat[action] = q_hat

            # DR estimator
            try:
                target_prob, behavior_prob = self.get_target_policy_probability(node, action)
                step_importance_weight = min(target_prob / (behavior_prob + 1e-10), 10)  # Clipping for stability
            except Exception as e:
                # Fallback if there's an issue with policy calculation
                if self.debug:
                    print(f"Policy calculation error: {e}. Using default values.")
                step_importance_weight = 1.0

            # Update cumulative weight
            cumulative_weight *= step_importance_weight
            cumulative_weight = min(cumulative_weight, 10)  # Clipping cumulative weight

            # DR estimate
            dr_estimate = v_hat + cumulative_weight * (reward + self.gamma * v_next - q_hat)

            # Get adaptive beta based on state-action visits
            adaptive_beta = self.get_adaptive_beta(node, action)

            # For debugging
            if self.debug and t == 0:  # Only for last state in trajectory
                print(
                    f"State: {node.state}, Action: {action}, Visits: {node.action_visits.get(action, 0)}, Beta: {adaptive_beta}")

            # Combine estimates with adaptive beta
            v_next = adaptive_beta * v_hat + (1 - adaptive_beta) * dr_estimate

            # Update node's v_hat
            node.v_hat = v_next

        return self.root.v_hat  # Return the estimated value of the root node

    def estimate_v_hat(self, node: MCTSNode) -> float:
        """Estimate state value based on weighted average of action values."""
        if not node.children:
            return 0.0

        v_hat = 0.0
        total_visits = sum(node.action_visits.values())

        if total_visits == 0:
            return 0.0

        for action, child in node.children.items():
            action_prob = node.action_visits.get(action, 0) / total_visits
            action_value = np.mean(node.rewards.get(action, [0]))
            v_hat += action_prob * action_value

        return v_hat

    def estimate_q_hat(self, node: MCTSNode, action: int) -> float:
        """Estimate action value using k-fold cross-validation."""
        rewards = np.array(node.rewards.get(action, [0]))

        if len(rewards) < self.kfold:
            return np.mean(rewards)

        try:
            kf = KFold(n_splits=self.kfold, shuffle=True, random_state=42)
            q_estimates = []

            for train_index, test_index in kf.split(rewards):
                train_rewards = rewards[train_index]
                test_rewards = rewards[test_index]

                q_estimate = np.mean(train_rewards)
                q_estimates.append(np.mean((test_rewards - q_estimate) ** 2))

            return np.mean(q_estimates)
        except Exception as e:
            if self.debug:
                print(f"K-fold estimation error: {e}")
            # Fallback if k-fold fails
            return np.mean(rewards)

    def get_target_policy_probability(self, node: MCTSNode, action: int) -> Tuple[float, float]:
        """Calculate target and behavior policy probabilities."""
        try:
            q_values = np.array([node.q_hat.get(a, 0) for a in node.children.keys()])
            softmax_probs = self.softmax(q_values)
            mcts_prob = softmax_probs[list(node.children.keys()).index(action)]
            uniform_prob = 1.0 / len(node.children)
            target_prob = (1 - self.alpha) * mcts_prob + self.alpha * uniform_prob

            # Simple uniform behavior probability as fallback
            behavior_prob = uniform_prob

            return target_prob, behavior_prob
        except Exception as e:
            if self.debug:
                print(f"Error in policy probability calculation: {e}")
            # Default to uniform probabilities if there's an issue
            return 1.0 / len(node.children), 1.0 / len(node.children)

    def softmax(self, x):
        """Compute softmax values with temperature control."""
        try:
            e_x = np.exp((x - np.max(x)) / self.temperature)
            return e_x / (e_x.sum() + 1e-10)  # Avoid division by zero
        except Exception as e:
            if self.debug:
                print(f"Softmax calculation error: {e}")
            # Return uniform distribution if there's an error
            return np.ones_like(x) / len(x)

    def get_best_action(self, node: MCTSNode) -> int:
        """Select best action based on visitation counts."""
        if not node.children:
            # Fallback to a default action if somehow we have no children
            available = []
            if hasattr(node, 'state'):
                # Try to infer game type from state
                if isinstance(node.state, tuple) and len(node.state) == 9:  # TicTacToe
                    available = [i for i, spot in enumerate(node.state) if spot == ' ']
                else:  # Default
                    available = list(range(9))  # TicTacToe board size
            return random.choice(available)

        # Use action visits from the parent node
        return max(node.children.keys(), key=lambda action: node.action_visits.get(action, 0))

    def get_rollout_action(self, game):
        """Get rollout action based on game type"""
        if hasattr(game, 'calculate_score'):  # SmallGo
            return self.go_rollout_policy(game)
        else:  # TicTacToe or others
            return self.tictactoe_rollout_policy(game)

    def tictactoe_rollout_policy(self, game):
        """Rollout policy for TicTacToe"""
        preferred_moves = [4, 0, 2, 6, 8, 1, 3, 5, 7]
        available = game.available_moves()
        for move in preferred_moves:
            if move in available:
                return move
        return random.choice(available)

    def go_rollout_policy(self, game):
        """Simple rollout policy for Go"""
        available = game.available_moves()

        # In Go, avoid playing on the edges in rollouts
        if len(available) > 1:
            board_size = game.board_size
            center_region = []
            for move in available:
                # Skip pass move
                if move == board_size * board_size:
                    continue

                x, y = move % board_size, move // board_size
                # Check if it's not on the edge
                if 0 < x < board_size - 1 and 0 < y < board_size - 1:
                    center_region.append(move)

            if center_region:
                return random.choice(center_region)

        # Fallback to random move
        return random.choice(available)

    '''
    def _run_single_simulation(self, game, sim_id):
        """Run a single MCTS simulation for DR-MCTS."""
        # Clone the game to avoid state interference
        sim_game = game.clone()
        node = self.root

        # Store trajectory for backpropagation
        trajectory = []

        # Selection
        while node.children and not sim_game.game_over():
            action = self.choose_action(node, sim_game)

            # Initialize or update action_visits for this node
            if not hasattr(node, 'action_visits'):
                node.action_visits = {}
            node.action_visits[action] = node.action_visits.get(action, 0) + 1

            # Store the current node, action, and reward before transitioning
            prev_node = node
            reward = sim_game.make_move(action)

            # Get the next node
            if action in node.children:
                node = node.children[action]
            else:
                # This should not happen if choose_action is working correctly
                if self.debug:
                    print(f"Warning: Selected action {action} not in children.")
                break

            # Store the transition in the trajectory
            trajectory.append((prev_node, action, reward, node))

        # Expansion (if needed)
        if not sim_game.game_over():
            # Initialize rewards dictionary if needed
            if not hasattr(node, 'rewards'):
                node.rewards = {}

            # Add all possible actions as children
            for action in sim_game.available_moves():
                if action not in node.children:
                    # Create child node and initialize its attributes
                    node.children[action] = MCTSNode(sim_game.get_state(), parent=node)
                    child_node = node.children[action]
                    child_node.action_visits = {}
                    child_node.rewards = {}
                    child_node.q_hat = {}

            # Choose an action for expansion
            action = self.get_rollout_action(sim_game)

            # Update action visits
            node.action_visits[action] = node.action_visits.get(action, 0) + 1

            # Store the current node before transitioning
            prev_node = node

            # Make the move and get the reward
            reward = sim_game.make_move(action)

            # Get the next node
            if action in node.children:
                node = node.children[action]
            else:
                # Create a new node if it doesn't exist
                node.children[action] = MCTSNode(sim_game.get_state(), parent=node)
                node = node.children[action]
                node.action_visits = {}
                node.rewards = {}
                node.q_hat = {}

            # Store the transition in the trajectory
            trajectory.append((prev_node, action, reward, node))

        # Simulation (rollout)
        while not sim_game.game_over():
            action = self.get_rollout_action(sim_game)
            reward = sim_game.make_move(action)

        # Check terminal state value
        terminal_value = 0.0
        if hasattr(sim_game, 'calculate_score'):
            # For Go, value is 1 if player 1 (Black) wins, 0 otherwise
            score = sim_game.calculate_score()
            terminal_value = 1.0 if score > 0 else 0.0
        elif hasattr(sim_game, 'is_winner'):
            # For TicTacToe, value is 1 if first player wins, 0 otherwise
            if hasattr(self.root.state, '__getitem__') and len(self.root.state) > 0:
                first_player_symbol = self.root.state[0]
                terminal_value = 1.0 if sim_game.is_winner(first_player_symbol) else 0.0
            else:
                terminal_value = 1.0 if sim_game.is_winner('X') else 0.0

        # Backpropagation using DR estimator
        if trajectory:
            # Add dummy reward and next_node for the terminal state
            terminal_node = node
            terminal_node.v_hat = terminal_value

            # Update values along the trajectory
            return self.mc_dr_update(trajectory)
        else:
            # If no trajectory (e.g., game was already over), update root directly
            self.root.visits += 1
            self.root.v_hat = terminal_value
            return terminal_value
        '''
