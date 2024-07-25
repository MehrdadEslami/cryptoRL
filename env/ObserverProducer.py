import numpy as np
import pandas as pd
import requests
import json
import time
import logging
from influxdb_client import InfluxDBClient
from kafka import KafkaProducer

with open("../config.json", "r") as file:
    config = json.load(file)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KafkaProducer")

# Initialize Kafka producer
producer = KafkaProducer(
    bootstrap_servers='localhost:29092',  # Use the external port
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    retries=5,  # Retry configuration
    batch_size=16384,  # Adjust batch size if needed
    linger_ms=10  # Adjust linger time if needed
)


class ObserverProducer:
    def __init__(self):
        self.min_qty = None
        self.min_prices = None
        self.max_qty = None
        self.max_prices = None
        self.influxdb_config = config['influxdb']
        self.buffer_size = int(config['buffer_size'])
        self.symbol = config['symbol']
        self.client = InfluxDBClient(**self.influxdb_config)
        self.query_api = self.client.query_api()
        self.start_query_time = 0
        self.end_query_time = "now()"
        self.trades = pd.DataFrame()

    def query_trades(self, start, end):
        query = f'''
            from(bucket: "{self.influxdb_config['bucket']}")
              |> range(start: {start}, stop: {end})
              |> filter(fn: (r) => r["_measurement"] == "trades")
              |> filter(fn: (r) => r["symbol"] == "{self.symbol}")
              |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
              |> keep(columns: ["_time", "price", "quantity", "side"])
            '''
        print(query)
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
            last_trade_time = self.trades.iloc[-1]['time']
            self.start_query_time = last_trade_time.isoformat()
            print('last time ', last_trade_time)

    def trades_to_normal_image(self, trades):
        # Initialize image channels
        buy_channel = np.zeros((self.buffer_size, self.buffer_size))
        sell_channel = np.zeros((self.buffer_size, self.buffer_size))
        price_channel = np.zeros((self.buffer_size, self.buffer_size))
        time_channel = np.zeros((self.buffer_size, self.buffer_size))
        self.step = self.step + 1
        for j, trade in trades.iterrows():
            side = trade['side']
            price = trade['price']
            quantity = trade['quantity']
            trade_time = trade['time'].isoformat()

            # NORMILIZING
            price_norm = (price - self.min_prices) / (self.max_prices - self.min_prices)
            quantity_norm = (quantity - self.mean_qty) / self.var_qty

            row = j // self.buffer_size
            col = j % self.buffer_size

            price_channel[row, col] = price_norm
            time_channel[row, col] = trade_time
            if side == 'buy':
                buy_channel[row, col] = quantity_norm
            else:
                sell_channel[row, col] = quantity_norm

        image = np.stack((buy_channel, sell_channel, price_channel, time_channel), axis=2)
        return image

    def fetch_and_process_trades(self):
        while True:
            try:
                print('Trades Lenght: ', len(self.trades))
                while len(self.trades)== 0:
                    print('trades is empty wait 10 min to fetch again')
                    self.query_trades(self.start_query_time, "now()")
                    time.sleep(1)

                num_trades = len(self.trades)
                for i in range(0, num_trades, self.buffer_size ** 2):
                    end = i + self.buffer_size ** 2
                    trades_slice = self.trades.iloc[i:end]
                    print('Trade_slice', trades_slice)
                    print('Trade_slice len', len(trades_slice))
                    if len(trades_slice) < self.buffer_size ** 2:
                        break
                    image = self.trades_to_normal_image(trades_slice.reset_index(drop=True))
                    print('image ', image)
                    producer.send('imagesTopic', image.tolist())  # Adjust sleep time as needed to control fetch rate
                    print('image sent to imagesTopic')
                    print('image to list', image.tolist())
            except requests.RequestException as e:
                logger.error(f"Error fetching trades: {e}")
            except Exception as e:
                logger.error(f"Unexpected error: {e}")


try:
    obs_produser = ObserverProducer()
    obs_produser.fetch_and_process_trades()

except KeyboardInterrupt:
    logger.info("Observer Producer stopped by user")
except Exception as e:
    logger.error(f"Error in Observer producer: {e}")
finally:
    producer.close()
