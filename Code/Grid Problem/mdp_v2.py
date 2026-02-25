import numpy as np
import time
from typing import Tuple, List, Dict


class GridWorld:
    """
    Grid World MDP Environment

    States: grid cells
    Actions: up, down, left, right
    Rewards: goal state (+1), obstacle (-1), other states (-0.04)
    """

    def __init__(self, rows=4, cols=4, gamma=0.9):
        self.rows = rows
        self.cols = cols
        self.gamma = gamma  # discount factor

        # Define special states (defaults, can be overwritten)
        self.goal_states = [(0, 3)]      # Goal with reward +1
        self.obstacle_states = [(1, 3)]  # Obstacle with reward -1
        self.blocked_states = [(1, 1)]   # Wall (cannot enter)

        # Actions: 0=up, 1=down, 2=left, 3=right
        self.actions = [0, 1, 2, 3]
        self.action_names = ['↑', '↓', '←', '→']

        # State rewards
        self.rewards = self._initialize_rewards()

    def _initialize_rewards(self):
        """Initialize reward for each state"""
        rewards = {}
        for i in range(self.rows):
            for j in range(self.cols):
                if (i, j) in self.goal_states:
                    rewards[(i, j)] = 1.0
                elif (i, j) in self.obstacle_states:
                    rewards[(i, j)] = -1.0
                elif (i, j) in self.blocked_states:
                    rewards[(i, j)] = 0.0
                else:
                    rewards[(i, j)] = -0.04  # Living cost
        return rewards

    def get_next_state(self, state: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Get next state given current state and action"""
        i, j = state

        # Terminal states don't move
        if state in self.goal_states or state in self.obstacle_states:
            return state

        # Apply action
        if action == 0:      # up
            next_state = (max(0, i - 1), j)
        elif action == 1:    # down
            next_state = (min(self.rows - 1, i + 1), j)
        elif action == 2:    # left
            next_state = (i, max(0, j - 1))
        else:                # right
            next_state = (i, min(self.cols - 1, j + 1))

        # Check if next state is blocked
        if next_state in self.blocked_states:
            return state

        return next_state

    def get_transition_prob(
        self,
        state: Tuple[int, int],
        action: int,
        next_state: Tuple[int, int],
        noise: float = 0.2
    ) -> float:
        """
        Get transition probability P(s'|s,a)

        With probability (1-noise), action succeeds
        With probability noise, move in perpendicular directions
        """
        if state in self.goal_states or state in self.obstacle_states:
            return 1.0 if state == next_state else 0.0

        intended_next = self.get_next_state(state, action)

        # Calculate perpendicular actions
        if action in [0, 1]:   # up or down
            perp_actions = [2, 3]  # left, right
        else:                  # left or right
            perp_actions = [0, 1]  # up, down

        perp_states = [self.get_next_state(state, a) for a in perp_actions]

        prob = 0.0
        if next_state == intended_next:
            prob += (1 - noise)
        if next_state in perp_states:
            prob += noise / 2
        return prob

    def get_all_states(self) -> List[Tuple[int, int]]:
        """Get all valid states"""
        states = []
        for i in range(self.rows):
            for j in range(self.cols):
                if (i, j) not in self.blocked_states:
                    states.append((i, j))
        return states


class ValueIteration:
    """Value Iteration Algorithm for MDP"""

    def __init__(self, gridworld: GridWorld, theta=1e-6):
        self.env = gridworld
        self.theta = theta  # convergence threshold
        self.V = {state: 0.0 for state in self.env.get_all_states()}
        self.iterations = 0
        self.convergence_history = []

    def bellman_update(self, state: Tuple[int, int]) -> float:
        """Perform Bellman optimality update for a state"""
        if state in self.env.goal_states or state in self.env.obstacle_states:
            return self.env.rewards[state]

        max_value = float('-inf')
        for action in self.env.actions:
            action_value = 0.0
            for next_state in self.env.get_all_states():
                prob = self.env.get_transition_prob(state, action, next_state)
                if prob > 0:
                    reward = self.env.rewards[next_state]
                    action_value += prob * (reward + self.env.gamma * self.V[next_state])
            max_value = max(max_value, action_value)
        return max_value

    def iterate(self) -> float:
        """Perform one iteration of value iteration"""
        delta = 0.0
        new_V = {}
        for state in self.env.get_all_states():
            old_value = self.V[state]
            new_V[state] = self.bellman_update(state)
            delta = max(delta, abs(new_V[state] - old_value))
        self.V = new_V
        self.iterations += 1
        self.convergence_history.append(delta)
        return delta

    def solve(self) -> Tuple[Dict, Dict, float]:
        """Run value iteration until convergence"""
        start_time = time.time()
        while True:
            delta = self.iterate()
            if delta < self.theta:
                break
        elapsed_time = time.time() - start_time
        policy = self.extract_policy()
        return self.V, policy, elapsed_time

    def extract_policy(self) -> Dict[Tuple[int, int], int]:
        """Extract optimal policy from value function"""
        policy = {}
        for state in self.env.get_all_states():
            if state in self.env.goal_states or state in self.env.obstacle_states:
                policy[state] = None
                continue

            best_action = None
            best_value = float('-inf')
            for action in self.env.actions:
                action_value = 0.0
                for next_state in self.env.get_all_states():
                    prob = self.env.get_transition_prob(state, action, next_state)
                    if prob > 0:
                        reward = self.env.rewards[next_state]
                        action_value += prob * (reward + self.env.gamma * self.V[next_state])
                if action_value > best_value:
                    best_value = action_value
                    best_action = action
            policy[state] = best_action
        return policy


class PolicyIteration:
    """Policy Iteration Algorithm for MDP"""

    def __init__(self, gridworld: GridWorld, theta=1e-6):
        self.env = gridworld
        self.theta = theta
        self.V = {state: 0.0 for state in self.env.get_all_states()}
        self.policy = self._initialize_policy()
        self.iterations = 0
        self.policy_changes = []

    def _initialize_policy(self) -> Dict[Tuple[int, int], int]:
        """Initialize random policy"""
        policy = {}
        for state in self.env.get_all_states():
            if state in self.env.goal_states or state in self.env.obstacle_states:
                policy[state] = None
            else:
                policy[state] = np.random.choice(self.env.actions)
        return policy

    def policy_evaluation(self):
        """Evaluate current policy until convergence"""
        while True:
            delta = 0.0
            new_V = {}
            for state in self.env.get_all_states():
                if state in self.env.goal_states or state in self.env.obstacle_states:
                    new_V[state] = self.env.rewards[state]
                    continue

                action = self.policy[state]
                state_value = 0.0
                for next_state in self.env.get_all_states():
                    prob = self.env.get_transition_prob(state, action, next_state)
                    if prob > 0:
                        reward = self.env.rewards[next_state]
                        state_value += prob * (reward + self.env.gamma * self.V[next_state])
                new_V[state] = state_value
                delta = max(delta, abs(new_V[state] - self.V[state]))
            self.V = new_V
            if delta < self.theta:
                break

    def policy_improvement(self) -> bool:
        """Improve policy based on current value function"""
        policy_stable = True
        for state in self.env.get_all_states():
            if state in self.env.goal_states or state in self.env.obstacle_states:
                continue

            old_action = self.policy[state]
            best_action = None
            best_value = float('-inf')

            for action in self.env.actions:
                action_value = 0.0
                for next_state in self.env.get_all_states():
                    prob = self.env.get_transition_prob(state, action, next_state)
                    if prob > 0:
                        reward = self.env.rewards[next_state]
                        action_value += prob * (reward + self.env.gamma * self.V[next_state])
                if action_value > best_value:
                    best_value = action_value
                    best_action = action

            self.policy[state] = best_action
            if old_action != best_action:
                policy_stable = False

        return policy_stable

    def solve(self) -> Tuple[Dict, Dict, float]:
        """Run policy iteration until convergence"""
        start_time = time.time()
        while True:
            self.policy_evaluation()
            policy_stable = self.policy_improvement()
            self.iterations += 1
            self.policy_changes.append(not policy_stable)
            if policy_stable:
                break
        elapsed_time = time.time() - start_time
        return self.V, self.policy, elapsed_time


def visualize_grid(gridworld: GridWorld,
                   values: Dict = None,
                   policy: Dict = None,
                   title: str = ""):
    """Visualize grid world with neatly aligned values and policy."""
    cell_width = 9                 # total characters per cell
    inner_width = cell_width - 2   # leave 1 space padding on each side
    total_width = gridworld.cols * (cell_width + 1) + 1  # +1 for vertical bars

    print(f"\n{'=' * total_width}")
    print(f"{title:^{total_width}}")
    print(f"{'=' * total_width}")

    for i in range(gridworld.rows):
        # horizontal separator
        print("-" * total_width)

        # ------- value row -------
        if values is not None:
            line = "|"
            for j in range(gridworld.cols):
                state = (i, j)
                if state in gridworld.blocked_states:
                    text = "WALL"
                elif state in gridworld.goal_states:
                    text = "GOAL!"
                elif state in gridworld.obstacle_states:
                    text = "BAD!"
                else:
                    text = f"{values[state]:.2f}"
                cell = f" {text:^{inner_width}} "
                line += cell + "|"
            print(line)

        # ------- policy row -------
        if policy is not None:
            line = "|"
            for j in range(gridworld.cols):
                state = (i, j)
                if state in gridworld.blocked_states:
                    text = "█"
                elif state in gridworld.goal_states:
                    text = "★"
                elif state in gridworld.obstacle_states:
                    text = "✗"
                elif policy[state] is not None:
                    text = gridworld.action_names[policy[state]]
                else:
                    text = ""
                cell = f" {text:^{inner_width}} "
                line += cell + "|"
            print(line)

    # bottom border after last row
    print("-" * total_width)



def compare_algorithms(gridworld: GridWorld):
    """Compare Value Iteration and Policy Iteration"""
    print("\n" + "=" * 60)
    print("COMPARING VALUE ITERATION AND POLICY ITERATION")
    print("=" * 60)

    # Run Value Iteration
    print("\n>>> Running Value Iteration...")
    vi = ValueIteration(gridworld)
    vi_values, vi_policy, vi_time = vi.solve()
    print(f"✓ Converged in {vi.iterations} iterations")
    print(f"✓ Time elapsed: {vi_time:.4f} seconds")

    # Run Policy Iteration
    print("\n>>> Running Policy Iteration...")
    pi = PolicyIteration(gridworld)
    pi_values, pi_policy, pi_time = pi.solve()
    print(f"✓ Converged in {pi.iterations} iterations")
    print(f"✓ Time elapsed: {pi_time:.4f} seconds")

    # Visualize results
    visualize_grid(gridworld, vi_values, vi_policy, "VALUE ITERATION RESULT")
    visualize_grid(gridworld, pi_values, pi_policy, "POLICY ITERATION RESULT")

    # Comparison summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<30} {'Value Iteration':<15} {'Policy Iteration':<15}")
    print("-" * 60)
    print(f"{'Iterations to converge':<30} {vi.iterations:<15} {pi.iterations:<15}")
    print(f"{'Time (seconds)':<30} {vi_time:<15.4f} {pi_time:<15.4f}")
    print(f"{'Speed ratio (VI/PI)':<30} {vi_time / pi_time:<15.2f} {'-':<15}")

    policies_match = all(vi_policy.get(s) == pi_policy.get(s)
                         for s in gridworld.get_all_states())
    print(f"{'Optimal policies match':<30} {str(policies_match):<15}")

    max_value_diff = max(abs(vi_values[s] - pi_values[s])
                         for s in gridworld.get_all_states())
    print(f"{'Max value difference':<30} {max_value_diff:<15.6f}")

    return {
        'value_iteration': {
            'iterations': vi.iterations,
            'time': vi_time,
            'values': vi_values,
            'policy': vi_policy
        },
        'policy_iteration': {
            'iterations': pi.iterations,
            'time': pi_time,
            'values': pi_values,
            'policy': pi_policy
        }
    }


def create_large_grid(gamma: float) -> GridWorld:
    """7x7 maze-like grid with two goals and several bad terminals."""
    grid = GridWorld(rows=7, cols=7, gamma=gamma)
    grid.goal_states = [(0, 6), (6, 6)]
    grid.obstacle_states = [(2, 3), (4, 3), (3, 5)]
    grid.blocked_states = [
        (1, 2), (2, 2), (3, 2),
        (4, 2), (5, 2),
        (3, 3), (3, 4),
        (5, 4), (5, 5)
    ]
    grid.rewards = grid._initialize_rewards()
    return grid


if __name__ == "__main__":
    print("=" * 70)
    print("MDP SOLVER: DISCOUNT FACTOR COMPARISON (TWO GRIDS)")
    print("=" * 70)

    # ----------------------------------------------------------
    # GRID 1: 4x4 CLASSIC WORLD WITH TWO DISCOUNT FACTORS
    # ----------------------------------------------------------
    gamma4_low = 0.9
    gamma4_high = 0.99

    print("\n" + "#" * 70)
    print("# GRID 1: 4x4 WORLD WITH γ =", gamma4_low, "AND γ =", gamma4_high)
    print("#" * 70)

    # 4x4 with lower gamma
    print(f"\nInitializing 4x4 Grid World with γ = {gamma4_low} ...")
    grid4_low = GridWorld(rows=4, cols=4, gamma=gamma4_low)
    print("Configuration (same layout for both γ values):")
    print(f"- Size: {grid4_low.rows}x{grid4_low.cols}")
    print(f"- Goal states: {grid4_low.goal_states} (reward: +1)")
    print(f"- Obstacle states: {grid4_low.obstacle_states} (reward: -1)")
    print(f"- Blocked states: {grid4_low.blocked_states} (walls)")
    print(f"- Living cost: -0.04 per step")

    results4_low = compare_algorithms(grid4_low)

    # 4x4 with higher gamma
    print(f"\nInitializing 4x4 Grid World with γ = {gamma4_high} (same layout) ...")
    grid4_high = GridWorld(rows=4, cols=4, gamma=gamma4_high)
    results4_high = compare_algorithms(grid4_high)

    print("\n>>> NOTE: For GRID 1, compare γ = 0.9 vs γ = 0.99 on the same 4x4 layout")
    print("    to see how increasing γ changes values and the optimal policy.")

    # ----------------------------------------------------------
    # GRID 2: LARGER 7x7 WORLD WITH TWO DISCOUNT FACTORS
    # ----------------------------------------------------------
    gamma7_low = 0.9
    gamma7_high = 0.99

    print("\n\n" + "#" * 70)
    print("# GRID 2: 7x7 MAZE WORLD WITH γ =", gamma7_low, "AND γ =", gamma7_high)
    print("#" * 70)

    # 7x7 with lower gamma
    print(f"\nInitializing 7x7 Grid World with γ = {gamma7_low} ...")
    grid7_low = create_large_grid(gamma7_low)
    print("7x7 Layout (used for both γ values):")
    print(f"- Size: {grid7_low.rows}x{grid7_low.cols}")
    print(f"- Goal states: {grid7_low.goal_states} (reward: +1 each)")
    print(f"- Obstacle states: {grid7_low.obstacle_states} (reward: -1 each)")
    print(f"- Blocked states (walls): {len(grid7_low.blocked_states)} cells")
    print(f"- Living cost: -0.04 per step")

    results7_low = compare_algorithms(grid7_low)

    # 7x7 with higher gamma
    print(f"\nInitializing 7x7 Grid World with γ = {gamma7_high} (same layout) ...")
    grid7_high = create_large_grid(gamma7_high)
    results7_high = compare_algorithms(grid7_high)

    print("\n>>> NOTE: For GRID 2, compare γ = 0.9 vs γ = 0.99 on the same 7x7 maze")
    print("    to see how longer-term planning (higher γ) affects the policy.")

    # ----------------------------------------------------------
    # SMALL NUMERICAL SUMMARY
    # ----------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("SUMMARY OF DISCOUNT FACTOR EFFECTS (ITERATIONS ONLY)")
    print("=" * 70)
    print(f"{'Grid / γ':<30} {'VI iters':<10} {'PI iters':<10}")
    print("-" * 70)
    print(f"{'4x4, γ=0.9':<30} {results4_low['value_iteration']['iterations']:<10} "
          f"{results4_low['policy_iteration']['iterations']:<10}")
    print(f"{'4x4, γ=0.99':<30} {results4_high['value_iteration']['iterations']:<10} "
          f"{results4_high['policy_iteration']['iterations']:<10}")
    print(f"{'7x7, γ=0.9':<30} {results7_low['value_iteration']['iterations']:<10} "
          f"{results7_low['policy_iteration']['iterations']:<10}")
    print(f"{'7x7, γ=0.99':<30} {results7_high['value_iteration']['iterations']:<10} "
          f"{results7_high['policy_iteration']['iterations']:<10}")
    print("\n" + "=" * 70)
    print("✓ ALL TWO-GRID DISCOUNT FACTOR EXPERIMENTS COMPLETE!")
    print("=" * 70)
