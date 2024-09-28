import random
from typing import Any, Generic, TypeVar  # cast, Callable

import gymnasium as gym

Action = TypeVar("Action")


class EpsilonGreedyWrapper(gym.ActionWrapper, Generic[Action]):
    def __init__(self, env: gym.Env[Any, Action], epsilon: float = 0.1):
        super().__init__(env)
        if 0 <= epsilon < 1:
            self.epsilon = epsilon
        else:
            raise ValueError("epsilon must be in the interval [0, 1[")

    def action(self, action: Action) -> Action:
        if random.random() < self.epsilon:
            action = self.env.action_space.sample()
            print(f"Random action: {action}")
            return action
        return action


if __name__ == "__main__":
    env = EpsilonGreedyWrapper[int](gym.make("CartPole-v1"))
    obs = env.reset()
    total_reward = 0.0

    while True:
        obs, reward, done, _, _ = env.step(0)  # type: ignore
        total_reward += reward  # type: ignore
        if done:
            break

    print(f"Total reward : {total_reward:.2f}")
    env.close()
