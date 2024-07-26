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

    def perform(self, env: 'TradingEnv', action: float, usdt_balance, btc_balance, current_price) -> None:
        """Performs the action on the given environment."""
        # orders = self.get_orders(action, env.observer.portfolio)
        #
        # for order in orders:
        #     if order:
        #         order.execute()
        print('Here In Action Perform')
        print('Action: %f, usdt_balance: %f, btc_balance: %f, current_price:%f'%(action, usdt_balance, btc_balance, current_price))
        if action < 0:
            # Sell action
            btc_to_sell = abs(action) * btc_balance
            revenue = btc_to_sell * current_price
            btc_balance -= btc_to_sell
            usdt_balance += revenue
            print(f"Sold {btc_to_sell:.6f} BTC for {revenue:.2f} USDT")
        elif action > 0:
            # Buy action
            usdt_to_spend = action * usdt_balance
            if usdt_balance < usdt_to_spend:
                print('not enought To buy usdt_balance is', usdt_balance)
                print('not enought To buy uusdt_to_spend is', usdt_to_spend)
            btc_to_buy = usdt_to_spend / current_price
            usdt_balance -= usdt_to_spend
            btc_balance += btc_to_buy
            print(f"Bought {btc_to_buy:.6f} BTC for {usdt_to_spend:.2f} USDT")
            print("now usdt_balance is ", usdt_balance)
            print("now btc_balance is ", btc_balance)
        else:
            print("No action taken")

        return [usdt_balance, btc_balance]

    def get_orders(self, action: int, portfolio: 'Portfolio') -> 'Order':
        orders = []
        if action == 0:  # Sell all
            orders.append(Order(step=1, side='sell', trade_type='limit', trading_pair=self.trading_pair, quantity=0.1, price=1000))
        elif action == 1:  # Buy 75%
            orders.append(Order(step=1, side='buy', trade_type='limit', trading_pair=self.trading_pair, quantity=0.1, price=1000))
        return orders

    def reset(self):
        self.action = 0
