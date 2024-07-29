from abc import ABCMeta, abstractmethod
from gym.spaces import Box, Space
from influxdb_client import InfluxDBClient
import numpy as np
import pandas as pd
import time


def convert_timestamp_to_float(timestamp_str):
    # Parse the timestamp string to a datetime object
    dt = pd.to_datetime(timestamp_str)
    # Convert the datetime object to a UNIX timestamp (float)
    unix_timestamp = dt.timestamp()
    return unix_timestamp


# Define min and max timestamps for normalization (for example purposes)
min_timestamp = convert_timestamp_to_float("2024-01-01T00:00:00.000000+00:00")
max_timestamp = convert_timestamp_to_float("2024-12-31T23:59:59.999999+00:00")


class Observer(metaclass=ABCMeta):
    """A component to generate an observation at each step of an episode."""

    registered_name = "observer"

    @property
    @abstractmethod
    def observation_space(self) -> Space:
        """The observation space of the `TradingEnv`. (`Space`, read-only)"""
        raise NotImplementedError()

    @abstractmethod
    def observe(self, env: 'TradingEnv') -> np.array:
        """Gets the observation at the current step of an episode"""
        raise NotImplementedError()

    def next(self):
        """Retrieve Next Image"""
        pass

    def reset(self, random_start=0):
        """Resets the observer."""
        pass


class ObserverM(Observer):

    def __init__(self, influx_config, symbol, buffer_size: int = 25, **kwargs) -> None:
        super().__init__()

        self.influxdb_config = influx_config
        self.client = InfluxDBClient(**self.influxdb_config)
        self.buffer_size = buffer_size
        self.slice_size = self.buffer_size ** 2
        self.query_api = self.client.query_api()
        self.trades = pd.DataFrame()
        self.state_mean_price = 0
        self.last_trade_time = 0
        self.next_image_i = 0
        self.symbol = symbol
        self.state = None
        self.step = 0

        self._observation_space = Box(
            low=0,
            high=1,
            shape=(self.buffer_size, self.buffer_size, 4),
            dtype=np.uint8
        )

        self.max_prices = 0
        self.min_prices = 0
        self.max_qty = 0
        self.min_qty = 0
        self.mean_qty = 0
        self.var_qty = 0
        self.mean_price = 0
        self.var_price = 0

    @property
    def observation_space(self) -> Space:
        return self._observation_space

    def observe(self) -> np.array:
        self.state, self.state_mean_price = self.next()
        return self.state, self.state_mean_price

    def next(self):
        while len(self.trades) < self.buffer_size ** 2:
            self.query_trades(self.last_trade_time, "now()")
            if len(self.trades) < self.buffer_size ** 2:
                time.wait(600)
        end = self.next_image_i + self.slice_size
        slice_trades = self.trades.iloc[self.next_image_i:end]
        self.step = self.step + 1
        image = self.trades_to_normal_image(slice_trades.reset_index(drop=True))
        self.next_image_i = end
        return image, self.mean_price

    def reset(self) -> None:
        print('ITS RESETING OBSERVER ..... ')
        self.trades = pd.DataFrame()
        self.last_trade_time = 0
        self.next_image_i = 0
        self.step = 0

    def query_trades(self, start, end):
        query = f'''
            from(bucket: "{self.influxdb_config['bucket']}")
              |> range(start: {start}, stop: {end})
              |> filter(fn: (r) => r["_measurement"] == "trades")
              |> filter(fn: (r) => r["symbol"] == "{self.symbol}")
              |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
              |> keep(columns: ["_time", "price", "quantity", "side"])
            '''
        # print(query)
        tables = self.query_api.query(query)
        self.client.close()

        records = []
        for table in tables:
            for record in table.records:
                records.append((record["_time"], record["price"], record["quantity"], record["side"]))

        self.trades = pd.DataFrame(records, columns=['time', 'price', 'quantity', 'side'])

        # Normalize price and quantity
        self.max_prices = self.trades['price'].max()
        self.min_prices = self.trades['price'].min()
        self.max_qty = self.trades['quantity'].max()
        self.min_qty = self.trades['quantity'].min()
        self.mean_qty = self.trades['quantity'].mean()
        self.var_qty = self.trades['quantity'].var()
        self.mean_price = self.trades['price'].mean()
        self.var_price = self.trades['price'].var()

        if not self.trades.empty:
            self.last_trade_time = self.trades.iloc[-1]['time'].isoformat()
            print('last trade time ', self.last_trade_time)
        print('THE INFLUX QUERY DONE and last trade time is:', self.last_trade_time)

    def trades_to_normal_image(self, trades):
        # Initialize image channels
        buy_channel = np.zeros((self.buffer_size, self.buffer_size))
        sell_channel = np.zeros((self.buffer_size, self.buffer_size))
        price_channel = np.zeros((self.buffer_size, self.buffer_size))
        time_channel = np.zeros((self.buffer_size, self.buffer_size))
        for j, trade in trades.iterrows():
            side = trade['side']
            price = trade['price']
            quantity = trade['quantity']
            trade_time = convert_timestamp_to_float(trade['time'])

            # NORMILIZING
            price_norm = (price - self.min_prices) / (self.max_prices - self.min_prices)
            quantity_norm = (quantity - self.mean_qty) / self.var_qty
            time_norm = (trade_time - min_timestamp) / (max_timestamp - min_timestamp)

            row = j // self.buffer_size
            col = j % self.buffer_size

            price_channel[row, col] = price_norm
            time_channel[row, col] = time_norm
            if side == 'buy':
                buy_channel[row, col] = quantity_norm
            else:
                sell_channel[row, col] = quantity_norm

            # print('[%d,%d]' % (row, col))
            # print('side:', side)
            # print("time = %f, norm time=%f" % (trade_time, time_norm))
            # print("quantity = %f, quantity time=%f" % (quantity, quantity_norm))
            # print("price = %f, norm price=%f" % (price, price_norm))

        image = np.stack((buy_channel, sell_channel, price_channel, time_channel), axis=2)
        print('CONVERT trade to image step:', self.step)
        return image
