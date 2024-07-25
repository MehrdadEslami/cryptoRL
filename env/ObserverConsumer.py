import time
import numpy as np
import json
from kafka import KafkaConsumer
from agents import DDPG_agent
from env.environment import TradingEnv

# Load the JSON configuration file
with open("../config.json", "r") as file:
    config = json.load(file)

consumer = KafkaConsumer(
    'imagesTopic',
    bootstrap_servers=['localhost:29092'],
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='my-group',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

buffer_size = config['buffer_size']

while True:
    try:
        for message in consumer:
            # image = message.value
            print('fetch trade from kafka broker')
            print(message.value)


    except Exception as e:
        print(f'Error: {e}')
        time.sleep(5)  # Wait for 5 seconds before retrying

# for message in consumer:
# trades = message.value
# print('fetch trade from kafka broker')
# print(trades)
# symbol = trades[0]
# trades = trades[1:]
# print(trades)
# print(len(trades))
# print('trade_buffer %s:%d' % (symbol, len(trade_buffer[symbol])))
# if len(trade_buffer[symbol]) == 0 and lastTradeID[symbol] == 0:
#     lastTradeID[symbol] = trades[0]['id']
#     trade_buffer[symbol].extend(trades)
# else:
#     unique_trades = remove_duplicates(trades[1:], lastTradeID[symbol])
#     lastTradeID[symbol] = trades[0]['id']
#     print('unique length for %s: %d '%(symbol,len(unique_trades)) )
#     if len(unique_trades) > 0:
#         save_trade_data(unique_trades, symbol)
#     trade_buffer[symbol].extend(unique_trades)
# print('trade_buffer size for %s : %d' % (symbol, len(trade_buffer[symbol])))
#
# if len(trade_buffer[symbol]) >= buffer_size:
#     print('IN IF STATEMENT')
#     trade_batch = trade_buffer[symbol][:buffer_size]
#     trade_buffer[symbol] = trade_buffer[symbol][buffer_size:]
#     print('trade_buffer size after %s : %d' % (symbol, len(trade_buffer[symbol])))
#     trade_image = trades_to_image(trade_batch, 10)

# _, buffer = cv2.imencode('.png', trade_image)
# encoded_image = base64.b64encode(buffer).decode('utf-8')

# Example usage
