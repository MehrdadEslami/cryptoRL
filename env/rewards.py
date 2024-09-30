from abc import ABCMeta, abstractmethod

import numpy as np


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

    def __init__(self):
        print('IN SimpleProfit Reward SChema')

    def reward(self, usdt_balance, btc_balance, action, current_price, new_state_price) -> float:
        return self.simpleProfit(action, current_price, new_state_price)
        # return self.logProfit(usdt_balance, btc_balance, current_price, new_state_price)

    def calculate_reward(self, state) -> float:
        """Rewards the agent for incremental increases in net worth over a sliding window."""
        return 0

    def PnL(self, new_net_worth):
        return new_net_worth - self.net_worth

    def SharpeRatio(self, risk_free_rate=0.0):
        returns = np.diff(self.prices) / self.prices[:-1]
        excess_returns = returns - risk_free_rate
        return np.mean(excess_returns) / np.std(excess_returns)

    def RewardClipping(self, reward, clip_value=1.0):
        return np.clip(reward, -clip_value, clip_value)

    def simpleProfit(self, action, current_price, new_state_price):
        print('IN simpleProfit action: %f, current_price : %f, new_state_price: %f' % (
            action, current_price, new_state_price))
        reward = round(action * ((new_state_price - current_price)/current_price*1000), 3)
        if reward == -0.00:
            reward = 0
        return reward

    def logProfit(self, usdt_balance, btc_balance, current_price, new_state_price):
        print('IN LogProfit usdt: %f, btc: %f, current_price : %f, new_state_price: %f' % (
            usdt_balance, btc_balance, current_price, new_state_price))
        return round(np.log( (usdt_balance + btc_balance*new_state_price) / 1000) , 3)
