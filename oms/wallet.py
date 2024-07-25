# Copyright 2020 The TensorTrade Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

from typing import Dict, Tuple
from collections import namedtuple
from decimal import Decimal

import numpy as np

from .ledger import Ledger
from oms.exchanges.exchange import Exchange
from core.exceptions import InsufficientFunds
Transfer = namedtuple("Transfer", ["quantity", "commission"])


class Wallet:
    """A wallet stores the balance of a specific instrument on a specific exchange.

    Parameters
    ----------
    exchange : `Exchange`
        The exchange associated with this wallet.
    balance : `Quantity`
        The initial balance quantity for the wallet.
    """

    ledger = Ledger()

    def __init__(self, exchange: 'Exchange', instrument: 'Instrument', size: 'float'):
        self.exchange = exchange
        self.balance = size
        self.instrument = instrument

    @property
    def total_balance(self) -> 'float':
        """The total balance of the wallet available for use and locked in orders. (`Quantity`, read-only)"""
        total_balance = self.balance

        # for quantity in self._locked.values():
        #     total_balance += quantity.size

        return total_balance

    def deposit(self, quantity: 'float', reason: str) -> 'Quantity':
        """Deposits funds into the wallet.

        Parameters
        ----------
        quantity : `Quantity`
            The amount to deposit into this wallet.
        reason : str
            The reason for depositing the amount.

        Returns
        -------
        `Quantity`
            The deposited amount.
        """

        self.balance += quantity

        self.ledger.commit(wallet=self,
                           quantity=float,
                           source=self.exchange.name,
                           target="{}:{}/locked".format(self.exchange.name, self.instrument),
                           memo="DEPOSIT ({})".format(reason))

        return quantity

    def withdraw(self, quantity: 'float', reason: str) -> 'Quantity':
        """Withdraws funds from the wallet.

        Parameters
        ----------
        quantity : `Quantity`
            The amount to withdraw from this wallet.
        reason : str
            The reason for withdrawing the amount.

        Returns
        -------
        `Quantity`
            The withdrawn amount.
        """
        if self.balance - quantity <0:
            print('insufficion balance')
            raise InsufficientFunds(self.balance-quantity,quantity)
        else:
            self.balance -= quantity

        self.ledger.commit(wallet=self,
                           quantity=quantity,
                           source="{}:{}/locked".format(self.exchange.name, self.instrument),
                           target=self.exchange.name,
                           memo="WITHDRAWAL ({})".format(reason))

        return quantity

    @classmethod
    def from_tuple(cls, wallet_tuple: 'Tuple[Exchange, Instrument, float]') -> 'Wallet':
        """Creates a wallet from a wallet tuple.

        Parameters
        ----------
        wallet_tuple : `Tuple[Exchange, Instrument, float]`
            A tuple containing an exchange, instrument, and amount.

        Returns
        -------
        `Wallet`
            A wallet corresponding to the arguments given in the tuple.
        """
        exchange, instrument, balance = wallet_tuple
        return cls(exchange, instrument, balance)

    @staticmethod
    def transfer(source: 'Wallet',
                 target: 'Wallet',
                 quantity: 'float',
                 commission: 'float',
                 reason: str) -> 'Transfer':
        """Transfers funds from one wallet to another.

        Parameters
        ----------
        source : `Wallet`
            The wallet in which funds will be transferred from
        target : `Wallet`
            The wallet in which funds will be transferred to
        quantity : `Quantity`
            The quantity to be transferred from the source to the target.
            In terms of the instrument of the source wallet.
        commission :  `Quantity`
            The commission to be taken from the source wallet for performing
            the transfer of funds.
        exchange_pair : `ExchangePair`
            The exchange pair associated with the transfer
        reason : str
            The reason for transferring the funds.

        Returns
        -------
        `Transfer`
            A transfer object describing the transaction.

        Raises
        ------
        Exception
            Raised if an equation that describes the conservation of funds
            is broken.
        """
        commission = source.withdraw(commission, "COMMISSION")
        quantity = source.withdraw(quantity, "FILL ORDER")

        target.deposit(quantity, 'TRADED {} {}@{}'.format(quantity, source.instrument, target.instrument))

        return Transfer(quantity, commission)

    def reset(self) -> None:
        """Resets the wallet."""
        self.balance = 0

    def __str__(self) -> str:
        return '<Wallet: balance={}'.format(self.balance)

    def __repr__(self) -> str:
        return str(self)


    def remove_position(self, symbol: str, quantity: float):
        if symbol in self.positions:
            position = self.positions[symbol]
            position['quantity'] -= quantity
            if position['quantity'] <= 0:
                del self.positions[symbol]

    def net_worth(self):
        return self.balance + sum(pos['quantity'] * pos['price'] for pos in self.positions.values())
