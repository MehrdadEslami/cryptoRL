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



def trades_to_image(trades, image_size):
    """
    Convert trades data to an image with 3 channels.

    Args:
    trades (list): List of trade dictionaries with 'quantity', 'price', and 'side' keys.
    image_size (tuple): Desired image size (height, width).

    Returns:
    np.ndarray: Image representation of trades.
    """

    print("IN IMAGE CREATION....")
    # Initialize an empty image
    img = np.zeros((image_size[0], image_size[1], 2), dtype=np.float32)

    # Normalize trade data to fit into the image
    # max_quantity = max(trade['quantity'] for trade in trades)
    # max_price = max(trade['price'] for trade in trades)

    # Populate the image
    for i, trade in enumerate(trades):
        y = i // image_size[1]  # Row index
        x = i % image_size[1]  # Column index

        if y >= image_size[0]:
            break  # Stop if we exceed the image size

        if trade['side'] == 'buy':
            img[y, x, 0] = trade['qty']  # / max_quantity  # Normalize buy quantity
            img[y, x, 1] = 0  # No sell quantity for buy trades
        else:  # trade['side'] == 'sell'
            img[y, x, 0] = 0  # No buy quantity for sell trades
            img[y, x, 1] = trade['qty']  # / max_quantity  # Normalize sell quantity

        # img[y, x, 2] = trade['price']  # / max_price  # Normalize price

    # Scale the image to the range [0, 255]
    print("image shape is ", img.shape)
    img = (img * 255).astype(np.uint8)
    # show_image(img)
    return img


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
    write_api.write(bucket=bucket, org=org, record=points)


def trades_to_image(trades, image_size=10):
    buy_trades = [trade for trade in trades if trade['side'] == 'buy']
    sell_trades = [trade for trade in trades if trade['side'] == 'sell']

    buy_volume = np.array([trade['qty'] for trade in buy_trades])
    sell_volume = np.array([trade['qty'] for trade in sell_trades])

    if len(buy_volume) < image_size ** 2:
        buy_volume = np.pad(buy_volume, (0, image_size ** 2 - len(buy_volume)))
    if len(sell_volume) < image_size ** 2:
        sell_volume = np.pad(sell_volume, (0, image_size ** 2 - len(sell_volume)))

    buy_volume = buy_volume[:image_size ** 2].reshape(image_size, image_size)
    sell_volume = sell_volume[:image_size ** 2].reshape(image_size, image_size)

    # scaler = MinMaxScaler()
    # buy_volume = scaler.fit_transform(buy_volume)
    # sell_volume = scaler.fit_transform(sell_volume)

    image = np.stack((buy_volume, sell_volume), axis=-1)
    return image


def process_image(trade_image):
    global model
    trade_image_resized = np.expand_dims(trade_image, axis=0)
    trade_image_resized = preprocess_input(trade_image_resized)
    preds = model.predict(trade_image_resized)
    print('Predictions:', preds)


# Initialize the VGG16 model
# model = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 2))


buffer_size = 10 * 10
trade_buffer = {i: [] for i in config["symbols"]}
lastTradeID = {i: 0 for i in config["symbols"]}

# while True:
#         try:
#             # Poll for new messages
#             msg_pack = consumer.poll(timeout_ms=1000)
#
#             if msg_pack:
#                 for tp, messages in msg_pack.items():
#                     for message in messages:
#                         print(f"Received trade data: {message.value}")
#             else:
#                 print("No new messages, retrying after a delay...")
#                 time.sleep(5)  # Sleep for 5 seconds before retrying
#
#         except Exception as e:
#             print(f"Error while consuming messages: {e}")
#             time.sleep(10)  # Sleep for 10 seconds before retrying
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
                trade_image = trades_to_image(trade_batch, 10)

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
