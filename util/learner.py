from typing import List, Tuple
from util.bandit import Bandit
import random

class Learner:
    def __init__(self, bandit: Bandit, initial_value=0.0, epsilon=0.0,):
        self.bandit = bandit
        self.num_arms = bandit.get_arms()
        self.model: List[Tuple[float, int]] =[(initial_value, 0) for i in range(bandit.get_arms())]
        self.epsilon = epsilon
        self.reward_total = 0
        self.reward_count = 0
        self.avg_rewards = [0.0]

    def __update_estimated_reward(self, reward: float, arm: int):
        old_estimate, step = self.model[arm]
        new_estimate = old_estimate + (1 / (step + 1)) * (reward - old_estimate)
        self.model[arm] = (new_estimate, step + 1)

    def choose_reward(self):
        if (random.random()) < self.epsilon:
            choice = random.randint(0, self.num_arms - 1)
            reward = self.bandit.get_reward(choice)
        else:
            choice_val = max(self.model, key=lambda x: x[0])
            choice_array = [index for index, value in enumerate(self.model) if value[0] == choice_val[0]]
            choice = random.choice(choice_array)
            reward = self.bandit.get_reward(choice)

        self.reward_count += 1
        self.reward_total += reward
        self.avg_rewards.append(self.reward_total / self.reward_count)
        self.__update_estimated_reward(reward, choice)

    def get_model(self):
        return self.model

    def get_avg_rewards(self):
        return self.avg_rewards

    def get_optimal_choice(self):
        return self.bandit.get_optimal_percentages()

    def learn_model(self):
        for _ in range(self.bandit.valid_runs):
            self.choose_reward()