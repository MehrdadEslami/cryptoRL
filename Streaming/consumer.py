import time
from kafka import KafkaConsumer
import json
import numpy as np
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import json

# Load the JSON configuration file
with open("../config.json", "r") as file:
    config = json.load(file)

# consumer = KafkaConsumer('trades', bootstrap_servers='localhost:29092',
#                          value_deserializer=lambda x: json.loads(x.decode('utf-8')))
consumer = KafkaConsumer(
    'trades',
    bootstrap_servers=['localhost:29092'],
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='my-group',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)
# Initialize InfluxDB client
token = config["influxdb"]["token"]
org = config["influxdb"]["org"]
url = config["influxdb"]["url"]
client = InfluxDBClient(url=url, token=token, org=org)

write_api = client.write_api(write_options=SYNCHRONOUS)


def remove_duplicates(trades, seen):
    print('remove duplicate for ')
    unique_trades = []
    for trade in trades:
        trade_id = trade['id']
        if trade_id != seen:
            unique_trades.append(trade)
        else:
            break

    return unique_trades


def save_trade_data(trades, symbol, bucket=config["influxdb"]["bucket"]):
    print('here in save influx')
    points = []

    # Save trades
    for trade in trades:
        # print(trade)
        # print(type(trade['id']))
        # print(type(trade['side']))
        # print(type(trade['qty']))
        # print(type(trade['timestamp']))
        points.append(Point("trades") \
                      .tag("side", trade['side']) \
                      .tag("symbol", symbol) \
                      .field("id", trade['id']) \
                      .field("quantity", float(trade['qty'])) \
                      .field("price", float(trade['price'])) \
                      .time(trade['timestamp']))

    # # Save image (convert to base64 for storage)
    # _, buffer = cv2.imencode('.png', image)
    # encoded_image = base64.b64encode(buffer).decode('utf-8')
    # point = Point("trade_image") \
    #     .field("image", encoded_image) \
    #     .time(timestamp, WritePrecision.MS)
    a = write_api.write(bucket=bucket, org=org, record=points)
    print('type', type(a))
    print('this ia a:', a)
    print('Data Writed to influx for symbol:', symbol)


buffer_size = 10 * 10
trade_buffer = {i: [] for i in config["symbols"]}
lastTradeID = {i: 0 for i in config["symbols"]}

while True:
    try:
        for message in consumer:
            trades = message.value
            print('fetch trade from kafka broker')
            # print(trades)
            symbol = trades[0]
            trades = trades[1:]
            # print(trades)
            print(len(trades))
            print('trade_buffer %s:%d' % (symbol, len(trade_buffer[symbol])))
            if len(trade_buffer[symbol]) == 0 and lastTradeID[symbol] == 0:
                lastTradeID[symbol] = trades[0]['id']
                trade_buffer[symbol].extend(trades)
            else:
                unique_trades = remove_duplicates(trades[1:], lastTradeID[symbol])
                lastTradeID[symbol] = trades[0]['id']
                print('unique length for %s: %d ' % (symbol, len(unique_trades)))
                if len(unique_trades) > 0:
                    save_trade_data(unique_trades, symbol)
                trade_buffer[symbol].extend(unique_trades)
            print('trade_buffer size for %s : %d' % (symbol, len(trade_buffer[symbol])))

            if len(trade_buffer[symbol]) >= buffer_size:
                print('IN IF STATEMENT')
                trade_batch = trade_buffer[symbol][:buffer_size]
                trade_buffer[symbol] = trade_buffer[symbol][buffer_size:]
                print('trade_buffer size after %s : %d' % (symbol, len(trade_buffer[symbol])))

    except Exception as e:
        print(f'Error: {e}')
        time.sleep(5)  # Wait for 5 seconds before retrying

