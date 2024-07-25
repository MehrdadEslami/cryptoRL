from abc import ABCMeta, abstractmethod
from gym.spaces import Space, Discrete
from typing import Any

class ActionScheme(metaclass=ABCMeta):
    """A component for determining the action to take at each step of an episode."""

    registered_name = "actions"

    @property
    @abstractmethod
    def action_space(self) -> Space:
        """The action space of the `TradingEnv`. (`Space`, read-only)"""
        raise NotImplementedError()

    @abstractmethod
    def perform(self, env: 'TradingEnv', action: Any) -> None:
        """Performs an action on the environment."""
        raise NotImplementedError()

    def reset(self) -> None:
        """Resets the action scheme."""
        pass

class BSH(ActionScheme):
    """A simple discrete action scheme where the only options are to buy, sell, or hold."""

    registered_name = "bsh"

    def __init__(self, trading_pair: str):
        super().__init__()
        self.trading_pair = trading_pair
        print('Action object Created')

    @property
    def action_space(self) -> Space:
        return Discrete(7, start=-1)  # Actions:  -1, -0.5, -0.25, 0, 0.25, 0.5,  1

    def perform(self, env: 'TradingEnv', action: Any) -> None:
        """Performs the action on the given environment."""
        orders = self.get_orders(action, env.observer.portfolio)

        for order in orders:
            if order:
                order.execute()

    def get_orders(self, action: int, portfolio: 'Portfolio') -> 'Order':
        orders = []
        if action == 0:  # Sell all
            orders.append(Order(step=1, side='sell', trade_type='limit', trading_pair=self.trading_pair, quantity=0.1, price=1000))
        elif action == 1:  # Buy 75%
            orders.append(Order(step=1, side='buy', trade_type='limit', trading_pair=self.trading_pair, quantity=0.1, price=1000))
        return orders

    def reset(self):
        self.action = 0
