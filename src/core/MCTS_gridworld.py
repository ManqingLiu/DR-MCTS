import math
import random
import matplotlib.pyplot as plt
import time
class Gridworld:
    def __init__(self, size=4):
        self.size = size
        self.state = (0, 0)  # Start at top-left corner
        self.terminal_states = {(size - 1, size - 1): 10}  # Bottom-right corner is terminal with reward 10
        self.max_steps = 100
        self.steps = 0

    def reset(self):
        self.state = (0, 0)
        self.steps = 0
        return self.state

    def step(self, action):
        if self.is_terminal():
            return self.state, 0, True

        self.steps += 1
        x, y = self.state
        if action == 0:  # Up
            x = max(0, x - 1)
        elif action == 1:  # Right
            y = min(self.size - 1, y + 1)
        elif action == 2:  # Down
            x = min(self.size - 1, x + 1)
        elif action == 3:  # Left
            y = max(0, y - 1)

        self.state = (x, y)

        if self.state in self.terminal_states:
            return self.state, self.terminal_states[self.state], True
        elif self.steps >= self.max_steps:
            return self.state, -1, True
        else:
            return self.state, -1, False

    def is_terminal(self):
        return self.state in self.terminal_states or self.steps >= self.max_steps

    def get_state(self):
        return self.state

    def available_actions(self):
        return [0, 1, 2, 3]  # All actions are always available

    def clone(self):
        new_env = Gridworld(self.size)
        new_env.state = self.state
        new_env.steps = self.steps
        return new_env


def calculate_true_values(size=4, gamma=0.95):
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

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.q_value = 0.0

class MCTS_Naive:
    def __init__(self, exploration_weight=1.4):
        self.exploration_weight = exploration_weight
        self.root = None

    def choose_action(self, node, game):
        if not node.children:
            return random.choice(game.available_actions())

        def score(n):
            if n.visits == 0:
                return float('inf')
            return n.value / n.visits + self.exploration_weight * math.sqrt(math.log(node.visits) / n.visits)

        return max(node.children, key=lambda c: score(node.children[c]))

    def rollout(self, game):
        total_reward = 0
        discount = 1.0
        while not game.is_terminal():
            action = random.choice(game.available_actions())
            _, reward, _ = game.step(action)
            total_reward += discount * reward
            discount *= 0.95  # Assuming a discount factor of 0.95
        return total_reward

    def mcts_search(self, game, num_simulations):
        self.root = MCTSNode(game.get_state())

        for _ in range(num_simulations):
            node = self.root
            sim_game = game.clone()

            # Selection
            while node.children and not sim_game.is_terminal():
                action = self.choose_action(node, sim_game)
                sim_game.step(action)
                if action not in node.children:
                    node.children[action] = MCTSNode(sim_game.get_state(), parent=node)
                node = node.children[action]

            # Expansion
            if not sim_game.is_terminal():
                action = random.choice(sim_game.available_actions())
                sim_game.step(action)
                node.children[action] = MCTSNode(sim_game.get_state(), parent=node)
                node = node.children[action]

            # Simulation
            result = self.rollout(sim_game)

            # Backpropagation
            while node:
                node.visits += 1
                node.value += result
                node = node.parent

        best_action = max(self.root.children, key=lambda c: self.root.children[c].visits)
        return self.root.children[best_action].value / self.root.children[best_action].visits


import numpy as np
from typing import List, Tuple


class MCTS_IS:
    def __init__(self):
        self.Q = {}
        self.N = {}
        self.env = None
        self.samples = []

    def mcts_search(self, env, num_rollouts):
        self.env = env
        self.samples = []
        root = env.get_state()
        for _ in range(num_rollouts):
            self.simulate(env.clone(), root)
        return self.estimate_value(root)

    def simulate(self, env, state, depth=0):
        if env.is_terminal():
            return env.step(0)[1]  # Return the reward

        if state not in self.Q:
            self.Q[state] = {a: 0 for a in env.available_actions()}
            self.N[state] = {a: 0 for a in env.available_actions()}
            return self.rollout(env)

        action = self.select_action(state)
        next_state, reward, _ = env.step(action)
        q = reward + self.simulate(env, next_state, depth + 1)

        self.N[state][action] += 1
        self.Q[state][action] += (q - self.Q[state][action]) / self.N[state][action]
        return q

    def select_action(self, state):
        return max(self.Q[state], key=self.Q[state].get)

    def rollout(self, env):
        cumulative_reward = 0
        discount = 1.0
        while not env.is_terminal():
            action = np.random.choice(env.available_actions())
            _, reward, _ = env.step(action)
            cumulative_reward += discount * reward
            discount *= 0.99  # Add a discount factor
        self.samples.append(cumulative_reward)
        return cumulative_reward

    def estimate_value(self, state):
        actions = self.env.available_actions()
        target_probs = [self.get_target_policy_probability(state, a) for a in actions]
        behavior_probs = [1.0 / len(actions) for _ in actions]
        importance_weights = [t / b for t, b in zip(target_probs, behavior_probs)]

        weighted_samples = [w * s for w, s in zip(importance_weights, self.samples)]
        return np.mean(weighted_samples)

    def get_target_policy_probability(self, state, action):
        if state not in self.N:
            return 1.0 / len(self.env.available_actions())
        total_visits = sum(self.N[state].values())
        return self.N[state][action] / max(total_visits, 1)


class MCTS_WIS:
    def __init__(self):
        self.Q = {}
        self.N = {}
        self.env = None
        self.samples = []

    def mcts_search(self, env, num_rollouts):
        self.env = env
        self.samples = []
        root = env.get_state()
        for _ in range(num_rollouts):
            self.simulate(env.clone(), root)
        return self.estimate_value(root)

    def simulate(self, env, state, depth=0):
        if env.is_terminal():
            return env.step(0)[1]  # Return the reward

        if state not in self.Q:
            self.Q[state] = {a: 0 for a in env.available_actions()}
            self.N[state] = {a: 0 for a in env.available_actions()}
            return self.rollout(env)

        action = self.select_action(state)
        next_state, reward, _ = env.step(action)
        q = reward + self.simulate(env, next_state, depth + 1)

        self.N[state][action] += 1
        self.Q[state][action] += (q - self.Q[state][action]) / self.N[state][action]
        return q

    def select_action(self, state):
        return max(self.Q[state], key=self.Q[state].get)

    def rollout(self, env):
        cumulative_reward = 0
        discount = 1.0
        while not env.is_terminal():
            action = np.random.choice(env.available_actions())
            _, reward, _ = env.step(action)
            cumulative_reward += discount * reward
            discount *= 0.99  # Add a discount factor
        self.samples.append(cumulative_reward)
        return cumulative_reward

    def estimate_value(self, state):
        actions = self.env.available_actions()
        target_probs = [self.get_target_policy_probability(state, a) for a in actions]
        behavior_probs = [1.0 / len(actions) for _ in actions]
        importance_weights = [t / b for t, b in zip(target_probs, behavior_probs)]

        normalized_weights = np.array(importance_weights) / np.sum(importance_weights)
        weighted_samples = [w * s for w, s in zip(normalized_weights, self.samples)]
        return np.sum(weighted_samples)

    def get_target_policy_probability(self, state, action):
        if state not in self.N:
            return 1.0 / len(self.env.available_actions())
        total_visits = sum(self.N[state].values())
        return self.N[state][action] / max(total_visits, 1)
class MCTS_DR:
    def __init__(self, exploration_weight: float = 1.4, gamma: float = 0.95):
        self.exploration_weight = exploration_weight
        self.gamma = gamma
        self.root = None

    def choose_action(self, node: MCTSNode, game: Gridworld) -> int:
        if not node.children:
            return random.choice(game.available_actions())

        def score(n: MCTSNode) -> float:
            if n.visits == 0:
                return float('inf')
            exploitation = n.q_value / n.visits
            exploration = self.exploration_weight * math.sqrt(math.log(node.visits) / n.visits)
            return exploitation + exploration

        return max(node.children, key=lambda c: score(node.children[c]))

    def mcts_search(self, game: Gridworld, num_simulations: int) -> Tuple[int, float]:
        self.root = MCTSNode(game.get_state())
        self.root.visits = 1

        for _ in range(num_simulations):
            node = self.root
            sim_game = game.clone()
            trajectory = []

            while not sim_game.is_terminal():
                if not node.children:
                    for action in sim_game.available_actions():
                        node.children[action] = MCTSNode(sim_game.get_state(), parent=node)
                    action = random.choice(list(node.children.keys()))
                else:
                    action = self.choose_action(node, sim_game)
                _, reward, _ = sim_game.step(action)
                next_node = node.children[action]
                trajectory.append((node, action, reward, next_node))
                node = next_node

            self.sequential_dr_update(trajectory)

        dr_estimate = self.estimate_value(self.root)
        return self.get_best_action(self.root), dr_estimate

    def sequential_dr_update(self, trajectory: List[Tuple[MCTSNode, int, float, MCTSNode]]):
        v_next = 0.0
        for node, action, reward, _ in reversed(trajectory):
            node.visits += 1
            child_node = node.children[action]
            child_node.visits += 1

            logged_propensity = 1.0 / len(node.children)
            target_propensity = self.get_target_policy_probability(node, action)
            importance_weight = min(target_propensity / (logged_propensity + 1e-8), 10)  # Clipped importance weight

            v_estimate = node.value / max(node.visits, 1)
            q_estimate = child_node.q_value / max(child_node.visits, 1)

            v_dr = v_estimate + importance_weight * (reward + self.gamma * v_next - q_estimate)
            v_dr = np.clip(v_dr, -1000, 1000)  # Value clipping

            # Stable averaging
            node.value = (node.value * (node.visits - 1) + v_dr) / node.visits
            child_node.q_value = (child_node.q_value * (child_node.visits - 1) + (reward + self.gamma * v_next)) / child_node.visits

            v_next = v_dr

            #print(f"Node visits: {node.visits}, V estimate: {v_estimate}, Q estimate: {q_estimate}")
            #print(f"Importance weight: {importance_weight}, V_DR: {v_dr}")
            #print(f"Updated node value: {node.value}, Updated Q value: {child_node.q_value}")
            #print("---")

    def get_target_policy_probability(self, node: MCTSNode, action: int) -> float:
        total_visits = sum(child.visits for child in node.children.values())
        return node.children[action].visits / max(total_visits, 1)

    def estimate_value(self, node: MCTSNode) -> float:
        return node.value / max(node.visits, 1)

    def get_best_action(self, node: MCTSNode) -> int:
        return max(node.children, key=lambda c: node.children[c].visits)


class MCTS_WDR:

    def __init__(self, exploration_weight: float = 1.4, gamma: float = 0.95):
        self.exploration_weight = exploration_weight
        self.gamma = gamma
        self.root = None

    def choose_action(self, node: MCTSNode, game: 'Gridworld') -> int:
        if not node.children:
            return np.random.choice(game.available_actions())

        return max(node.children, key=lambda c: node.children[c].visits)

    def rollout(self, game: 'Gridworld') -> float:
        total_reward = 0
        discount = 1.0
        while not game.is_terminal():
            action = np.random.choice(game.available_actions())
            _, reward, _ = game.step(action)
            total_reward += discount * reward
            discount *= self.gamma
        return total_reward

    def get_target_policy_probability(self, node: MCTSNode, action: int) -> float:
        total_visits = sum(child.visits for child in node.children.values())
        return (node.children[action].visits + 1e-8) / max(total_visits, 1)

    def update_tree(self, trajectory: List[Tuple[MCTSNode, int, float, MCTSNode]]):
        for node, action, reward, child_node in trajectory:
            if action is not None:
                node.visits += 1
                child_node.visits += 1
                child_node.value += reward
                child_node.q_value += reward + self.gamma * child_node.value

    def mcts_search(self, game: 'Gridworld', num_simulations: int) -> float:
        self.root = MCTSNode(game.get_state())
        trajectories = []

        for _ in range(num_simulations):
            trajectory = self.simulate(game.clone())
            self.update_tree(trajectory)
            trajectories.append(trajectory)

        return self.compute_wdr_estimate(trajectories)

    def simulate(self, game: 'Gridworld') -> List[Tuple[MCTSNode, int, float, MCTSNode]]:
        node = self.root
        trajectory = []

        while not game.is_terminal():
            if not node.children:
                action = np.random.choice(game.available_actions())
                _, reward, _ = game.step(action)
                new_node = MCTSNode(game.get_state(), parent=node)
                node.children[action] = new_node
                trajectory.append((node, action, reward, new_node))
                trajectory.append((new_node, None, self.rollout(game), None))
                return trajectory

            action = self.choose_action(node, game)
            _, reward, _ = game.step(action)
            child_node = node.children.get(action)
            if child_node is None:
                child_node = MCTSNode(game.get_state(), parent=node)
                node.children[action] = child_node

            trajectory.append((node, action, reward, child_node))
            node = child_node

        trajectory.append((node, None, 0, None))
        return trajectory

    def compute_wdr_estimate(self, trajectories: List[List[Tuple[MCTSNode, int, float, MCTSNode]]]) -> float:
        max_length = max(len(traj) for traj in trajectories)
        num_trajectories = len(trajectories)

        j_step_returns = np.zeros((num_trajectories, max_length))
        importance_weights = np.zeros((num_trajectories, max_length))

        for i, trajectory in enumerate(trajectories):
            cumulative_importance_weight = 1.0
            cumulative_reward = 0.0

            for t, (node, action, reward, _) in enumerate(trajectory[:-1]):
                logged_prob = 1.0 / max(len(node.children), 1)
                target_prob = self.get_target_policy_probability(node, action)
                step_importance_weight = target_prob / logged_prob
                cumulative_importance_weight *= step_importance_weight
                cumulative_importance_weight = np.clip(cumulative_importance_weight, 1e-8, 1e8)

                cumulative_reward = reward + self.gamma * cumulative_reward
                j_step_returns[i, t] = cumulative_reward
                importance_weights[i, t] = cumulative_importance_weight

        # Normalize importance weights
        epsilon = 1e-8
        normalized_weights = importance_weights / (np.sum(importance_weights, axis=0, keepdims=True) + epsilon)

        # Compute WDR estimate
        wdr_estimate = np.sum(normalized_weights * j_step_returns) / num_trajectories
        wdr_estimate = np.clip(wdr_estimate, -1e8, 1e8)

        return wdr_estimate

def run_mcts(mcts, game, num_simulations):
    best_action = mcts.mcts_search(game, num_simulations)
    value = mcts.root.value / mcts.root.visits if mcts.root.visits > 0 else 0
    print(f"MCTS search completed. Root value: {mcts.root.value}, visits: {mcts.root.visits}, returned value: {value}")
    return value


def bootstrap_mse(errors: List[float], num_bootstrap: int = 1000, confidence: float = 0.95) -> Tuple[
    float, float, float]:
    errors = np.array(errors).flatten()  # Ensure errors is a 1-D array
    bootstrapped_mses = []
    for _ in range(num_bootstrap):
        sample = np.random.choice(errors, size=len(errors), replace=True)
        bootstrapped_mses.append(np.mean(sample))

    mean_mse = np.mean(bootstrapped_mses)
    ci_lower = np.percentile(bootstrapped_mses, (1 - confidence) / 2 * 100)
    ci_upper = np.percentile(bootstrapped_mses, (1 + confidence) / 2 * 100)

    return mean_mse, ci_lower, ci_upper
def run_experiment(num_games: int, rollouts_list: List[int], true_values: np.ndarray) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]], List[Tuple[float, float, float]], List[Tuple[float, float, float]], List[Tuple[float, float, float]], List[float], List[float], List[float], List[float], List[float]]:
    results_naive = []
    results_dr = []
    results_wdr = []
    results_is = []
    results_wis = []
    times_naive = []
    times_dr = []
    times_wdr = []
    times_is = []
    times_wis = []

    for rollouts in rollouts_list:
        errors_naive = []
        errors_dr = []
        errors_wdr = []
        errors_is = []
        errors_wis = []
        time_naive = 0
        time_dr = 0
        time_wdr = 0
        time_is = 0
        time_wis = 0

        for _ in range(num_games):
            env = Gridworld()
            state = env.get_state()
            true_value = true_values[state[0], state[1]]

            # Naive MCTS
            mcts_naive = MCTS_Naive()
            start_time = time.time()
            value_naive = mcts_naive.mcts_search(env, rollouts)
            time_naive += time.time() - start_time
            errors_naive.append((value_naive - true_value) ** 2)

            # MCTS with DR
            mcts_dr = MCTS_DR()
            start_time = time.time()
            value_dr = mcts_dr.mcts_search(env, rollouts)
            time_dr += time.time() - start_time
            errors_dr.append((value_dr - true_value) ** 2)

            # MCTS with WDR
            mcts_wdr = MCTS_WDR()
            start_time = time.time()
            value_wdr = mcts_wdr.mcts_search(env, rollouts)
            time_wdr += time.time() - start_time
            errors_wdr.append((value_wdr - true_value) ** 2)

            # MCTS with IS
            mcts_is = MCTS_IS()
            start_time = time.time()
            value_is = mcts_is.mcts_search(env, rollouts)
            time_is += time.time() - start_time
            errors_is.append((value_is - true_value) ** 2)

            # MCTS with WIS
            mcts_wis = MCTS_WIS()
            start_time = time.time()
            value_wis = mcts_wis.mcts_search(env, rollouts)
            time_wis += time.time() - start_time
            errors_wis.append((value_wis - true_value) ** 2)

        results_naive.append(bootstrap_mse(errors_naive))
        results_dr.append(bootstrap_mse(errors_dr))
        results_wdr.append(bootstrap_mse(errors_wdr))
        results_is.append(bootstrap_mse(errors_is))
        results_wis.append(bootstrap_mse(errors_wis))
        times_naive.append(time_naive / num_games)
        times_dr.append(time_dr / num_games)
        times_wdr.append(time_wdr / num_games)
        times_is.append(time_is / num_games)
        times_wis.append(time_wis / num_games)

    return results_naive, results_dr, results_wdr, results_is, results_wis, times_naive, times_dr, times_wdr, times_is, times_wis
def plot_results(rollouts_list: List[int], results_naive: List[Tuple[float, float, float]],
                 results_dr: List[Tuple[float, float, float]], results_wdr: List[Tuple[float, float, float]],
                 results_is: List[Tuple[float, float, float]], results_wis: List[Tuple[float, float, float]],
                 times_naive: List[float], times_dr: List[float], times_wdr: List[float],
                 times_is: List[float], times_wis: List[float], results_dir: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))

    # Log MSE plot
    for results, label, color in zip([results_naive, results_is, results_wis, results_dr, results_wdr],
                                     ['Naive MCTS', 'MCTS with IS', 'MCTS with WIS', 'MCTS with DR', 'MCTS with WDR'],
                                     ['blue', 'orange', 'green', 'red', 'purple']):
        means, ci_lowers, ci_uppers = zip(*results)
        log_means = np.log10(means)
        log_ci_lowers = np.log10(ci_lowers)
        log_ci_uppers = np.log10(ci_uppers)
        ax1.plot(rollouts_list, log_means, label=label, marker='o', color=color)
        ax1.fill_between(rollouts_list, log_ci_lowers, log_ci_uppers, alpha=0.2, color=color)

    ax1.set_xlabel('Number of Rollouts')
    ax1.set_ylabel('Log10 MSE')
    ax1.set_title('Comparison of Log MSE')
    ax1.legend()
    ax1.grid(True)
    ax1.set_xscale('log')

    # Time plot
    ax2.plot(rollouts_list, times_naive, label='Naive MCTS', marker='o', color='blue')
    ax2.plot(rollouts_list, times_dr, label='MCTS with DR', marker='s', color='orange')
    ax2.plot(rollouts_list, times_wdr, label='MCTS with WDR', marker='^', color='green')
    ax2.plot(rollouts_list, times_is, label='MCTS with IS', marker='D', color='red')
    ax2.plot(rollouts_list, times_wis, label='MCTS with WIS', marker='*', color='purple')
    ax2.set_xlabel('Number of Rollouts')
    ax2.set_ylabel('Average Time per Game (seconds)')
    ax2.set_title('Comparison of Execution Time')
    ax2.legend()
    ax2.grid(True)
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(results_dir)
    plt.close()


if __name__ == "__main__":
    true_values = calculate_true_values()
    num_games = 100
    rollouts_list = [10, 100, 500, 1000, 1500, 2000]
    results_naive, results_dr, results_wdr, results_is, results_wis, times_naive, times_dr, times_wdr, times_is, times_wis = run_experiment(
        num_games, rollouts_list, true_values)

    # Print results
    print("Results:")
    for i, rollouts in enumerate(rollouts_list):
        print(f"\nRollouts: {rollouts}")
        print(
            f"Naive MCTS - Log MSE: {np.log10(results_naive[i][0]):.6f} ({np.log10(results_naive[i][1]):.6f}, {np.log10(results_naive[i][2]):.6f}), Avg Time: {times_naive[i]:.6f} seconds")
        print(
            f"MCTS with DR - Log MSE: {np.log10(results_dr[i][0]):.6f} ({np.log10(results_dr[i][1]):.6f}, {np.log10(results_dr[i][2]):.6f}), Avg Time: {times_dr[i]:.6f} seconds")
        print(
            f"MCTS with WDR - Log MSE: {np.log10(results_wdr[i][0]):.6f} ({np.log10(results_wdr[i][1]):.6f}, {np.log10(results_wdr[i][2]):.6f}), Avg Time: {times_wdr[i]:.6f} seconds")
        print(
            f"MCTS with IS - Log MSE: {np.log10(results_is[i][0]):.6f} ({np.log10(results_is[i][1]):.6f}, {np.log10(results_is[i][2]):.6f}), Avg Time: {times_is[i]:.6f} seconds")
        print(
            f"MCTS with WIS - Log MSE: {np.log10(results_wis[i][0]):.6f} ({np.log10(results_wis[i][1]):.6f}, {np.log10(results_wis[i][2]):.6f}), Avg Time: {times_wis[i]:.6f} seconds")

    results_dir = "MCTS_comparison_gridworld_with_log_MSE.png"
    plot_results(rollouts_list, results_naive, results_dr, results_wdr, results_is, results_wis, times_naive, times_dr,
                 times_wdr, times_is, times_wis, results_dir)