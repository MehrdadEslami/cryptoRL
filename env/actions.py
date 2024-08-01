from abc import ABCMeta, abstractmethod
from gym.spaces import Discrete, Space
from typing import Any
import numpy as np


class ActionScheme(metaclass=ABCMeta):
    """A component for determining the action to take at each step of an episode."""

    registered_name = "actions"

    @property
    @abstractmethod
    def action_space(self) -> Space:
        """The action space of the `TradingEnv`. (`Space`, read-only)"""
        raise NotImplementedError()

    @abstractmethod
    def perform(self, env: 'TradingEnv', action: Any, usdt_balance: float, btc_balance: float,
                current_price: float) -> None:
        """Performs an action on the environment."""
        raise NotImplementedError()

    def reset(self) -> None:
        """Resets the action scheme."""
        pass


class BSH(ActionScheme):
    """A simple continuous action scheme where the actions are between -1 and 1."""

    registered_name = "bsh"

    def __init__(self,):
        super().__init__()
        self.trading_pair = 'BTC/USDT'
        self._action_space = Discrete(5)
        self.actions = [-0.75, -0.25, 0, 0.25, 0.75]
        # self._action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_n = 5
        print('Action object Created')

    @property
    def action_space(self) -> Space:
        return self._action_space

    def perform(self, action: int, usdt_balance: float, btc_balance: float,
                current_price: float) -> Any:
        """Performs the action on the given environment."""
        print('Here In Action Perform')
        print('action: %d, usdt_balance: %f, btc_balance: %f, current_price:%f' % (
        action, usdt_balance, btc_balance, current_price))
        if action < 0:
            # Sell action
            btc_to_sell = round(abs(action * btc_balance), 5)
            print('btc to sell', btc_to_sell)
            if btc_to_sell < 0:
                print('no BTC to SELL')
            else:
                revenue = btc_to_sell * current_price
                btc_balance -= btc_to_sell
                usdt_balance += revenue
                print(f"Sold {btc_to_sell:.6f} BTC for {revenue:.2f} USDT")
        elif action > 0:
            # Buy action
            usdt_to_spend = round(float(action * usdt_balance), 2)
            print('usdt_to_spend', usdt_to_spend)
            if usdt_balance < usdt_to_spend:
                print('Not enough balance to buy. USDT balance is', usdt_balance)
                print('USDT to spend is', usdt_to_spend)
            else:
                btc_to_buy = usdt_to_spend / current_price
                usdt_balance -= usdt_to_spend
                btc_balance += btc_to_buy
                print(f"Bought {btc_to_buy:.6f} BTC for {usdt_to_spend:.2f} USDT")
                print("New USDT balance:", usdt_balance)
                print("New BTC balance:", btc_balance)
        else:
            print("No action taken")

        return usdt_balance, btc_balance

    def sample(self) -> float:
        """Samples a random action from the action space."""
        return float(self._action_space.sample())

    def reset(self):
        self.action = 0
