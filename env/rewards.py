from abc import ABCMeta, abstractmethod

class RewardScheme(metaclass=ABCMeta):
    """A component to compute the reward at each step of an episode."""

    registered_name = "rewards"

    @abstractmethod
    def reward(self, env: 'TradingEnv') -> float:
        """Computes the reward for the current step of an episode."""
        raise NotImplementedError()

    def reset(self) -> None:
        """Resets the reward scheme."""
        pass

class SimpleProfit(RewardScheme):
    """A simple reward scheme that rewards the agent for incremental increases in net worth."""

    def __init__(self, window_size: int = 1):
        self.window_size = window_size

    def reward(self, env: 'TradingEnv') -> float:
        return self.get_reward(env.observer.portfolio)

    def get_reward(self, portfolio: 'Portfolio') -> float:
        """Rewards the agent for incremental increases in net worth over a sliding window."""
        return portfolio.net_worth
