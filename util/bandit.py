import numpy as np
import numpy.typing as npt
from typing import List, Any

class Bandit:
    def __init__(self, rewards, means: List[float]):
        self.rewards = rewards  # Rewards for each arm
        self.means = means  # True means for each arm (from x)
        self.arms = len(rewards)
        self.valid_runs = len(self.rewards[0])
        self.current_reward = 0
        self.optimal_count = 0
        self.optimal_percentages = [] # Keeps track of the percentage of times the optimal arm was chosen for plotting purposes
        self.optimal_arm = np.argmax(means)

    def get_arms(self):
        return self.arms

    def get_valid_runs(self):
        return self.valid_runs

    def get_optimal_percentages(self):
        return self.optimal_percentages

    def reset(self):
        self.current_reward = 0
        self.optimal_count = 0
        self.optimal_percentages = []

    def get_reward(self, choice: int) -> float:
        if choice < 0 or choice >= self.arms:
            raise ValueError("Invalid choice")

        if self.current_reward >= len(self.rewards[choice]):
            raise ValueError("No more rewards left for this arm")

        learner_reward = self.rewards[choice][self.current_reward]
        self.current_reward += 1

        if choice == self.optimal_arm:
            self.optimal_count += 1

        self.optimal_percentages.append(self.optimal_count / self.current_reward)

        return learner_reward