import csv
import random

from influxdb_client import InfluxDBClient
import numpy as np
import pandas as pd
import json
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime as dt

# from agents.DQN_agent import DQNAgent
# from agents.MY_DQN_agent import MYDQNAgent

with open("config.json", "r") as file:
    config = json.load(file)

influxdb_config = config['influxdb']
client = InfluxDBClient(**influxdb_config)
query_api = client.query_api()
bucket_trades = influxdb_config['bucket']
bucket_ohlcv = influxdb_config['bucket_ohlcv']
org = influxdb_config['org']
url = influxdb_config['url']

times2 = []
times = []
price = []
count = []
count2 = []
price2 = []
times3 = []

def merge_count_price():
    i, j = 0, 0
    while i != len(count) or j != len(price):
        print('i:', i)
        if times[i] == times2[j]:
            count2.append(count[i])
            price2.append(price[j])
            times3.append(times2[j])
            i += 1
            j += 1
        elif times[i] < times2[j]:
            i += 1
        else:
            j += 1


class Query:

    def __init__(self, local_config):
        start = local_config['start']
        end = local_config['end']
        self.start_time = start.strftime('%Y-%m-%dT%H:%M:%SZ')
        self.stop_time = end.strftime('%Y-%m-%dT%H:%M:%SZ')
        self.symbol = local_config['symbol']
        self.window = local_config['window']

    def query_trades_count(self):
        query1 = f'''
                    from(bucket: "{bucket_trades}")
                      |> range(start: {self.start_time}, stop: {self.stop_time})
                      |> filter(fn: (r) => r["_measurement"] == "trades")
                      |> filter(fn: (r) => r["_field"] == "id") // Use the trade ID field for counting trades
                      |> filter(fn: (r) => r["symbol"] == "{self.symbol}")
                      |> filter(fn: (r) => r["side"] == "buy" or r["side"] == "sell")
                      |> unique(column: "_value")  // Fetch distinct trade IDs
                      |> aggregateWindow(every: {self.window}, fn: count, createEmpty: false) // Count trades every hour
                      |> yield(name: "count")
                    '''

        tables_count = query_api.query(query1)
        client.close()

        i = 0
        j = 0
        for table in tables_count:
            for record in table:
                t = pd.to_datetime(record['_time'])
                if t in times:
                    k = times.index(t)
                    count[k] += record['_value']
                else:
                    i += 1
                    times.append(t)
                    count.append(record['_value'])

        d = {'_time': times, 'count': count}
        df = pd.DataFrame(d)
        df.set_index('_time', inplace=True)
        # change temp
        # df._set_value(20, 'count', 0.9)

        # Plot the bar chart
        return df

    def query_trades_count_ohlcv(self):
        query1 = f'''
            from(bucket: "{bucket_trades}")
              |> range(start: {self.start_time}, stop: {self.stop_time})
              |> filter(fn: (r) => r["_measurement"] == "trades")
              |> filter(fn: (r) => r["_field"] == "id") // Use the trade ID field for counting trades
              |> filter(fn: (r) => r["symbol"] == "{self.symbol}")
              |> filter(fn: (r) => r["side"] == "buy" or r["side"] == "sell")
              |> unique(column: "_value")  // Fetch distinct trade IDs
              |> aggregateWindow(every: {self.window}, fn: count, createEmpty: false) // Count trades every hour
              |> yield(name: "count")
            '''
        print(query1)
        query2 = f'''
            from(bucket: "{bucket_ohlcv}")
              |> range(start: {self.start_time}, stop: {self.stop_time})
              |> filter(fn: (r) => r["_measurement"] == "ohlcvBucket")
              |> filter(fn: (r) => r["_field"] == "close") // Use the trade ID field for counting trades
              |> filter(fn: (r) => r["symbol"] == "{self.symbol}")
              |> aggregateWindow(every: {self.window}, fn: mean, createEmpty: false) // Count trades every hour
              |> yield(name: "mean")
            '''
        # print(query2)
        tables_count = query_api.query(query1)
        tables_price = query_api.query(query2)
        client.close()

        i = 0
        j = 0
        for table in tables_count:
            for record in table:
                t = pd.to_datetime(record['_time'])
                if t in times:
                    k = times.index(t)
                    count[k] += record['_value']
                else:
                    print(i)
                    i += 1
                    times.append(t)
                    count.append(record['_value'])

        for table in tables_price:
            for record in table:
                t = pd.to_datetime(record['_time'])
                times2.append(t)
                price.append(record['_value'])

        merge_count_price()
        c = (count2 - np.min(count)) / (np.max(count) - np.min(count))
        p = (price2 - np.min(price)) / (np.max(price) - np.min(price))
        for k in range(len(c)):
            c[k] = round(c[k], 2)
            p[k] = round(p[k], 2)

        d = {'_time': times3, 'count': c, 'price': p}
        df = pd.DataFrame(d)
        # Set the time as the index for plotting
        # df.set_index('_time', inplace=True)

        # change temp
        df._set_value(20, 'count', 0.9)

        # Plot the bar chart
        return df

    def plot_count_ohlcv(self, df):
        plt.figure(figsize=(12, 6))
        df['count'].plot(kind='bar', width=0.3, color='blue', label='Number of Trades')
        df['price'].plot(linewidth=1.0, color='red', label='price')
        plt.title('Number of Trades Done Per 4Hour and Price')
        plt.xlabel('Time ')
        plt.ylabel('Number of Trades, Price')
        # plt.annotate()
        plt.xticks(rotation=45)
        plt.tight_layout()
        print('Creating Plot.....')
        plt.savefig('plots/_count_price_24h.png')
        print('DONE')

    def plot_count(self, df, label, c, filename):

        plt.figure(figsize=(9, 3))
        plt.bar(range(len(df)), df['count'], width=0.8, color=c, label=label)

        plt.title("Number of Trades %s Per 24Hour" % label)
        plt.xlabel('Time ')
        plt.ylabel('Number of Trades')
        # plt.annotate()
        # plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        print('Creating Plot.....')
        plt.savefig(filename)
        print('DONE')

    def plot_from_csv(self, filename, x, y, xlabel, ylabel):
        # Read data from CSV
        data = pd.read_csv(filename)
        data['Reward'] *= 500
        # t = np.zeros(100)
        # for i in range(100):
        #     t[99 - i] = data.iloc[i]['Reward']
        # Set dark background
        plt.style.use('dark_background')

        # Create the plot
        plt.figure(figsize=(10, 6))

        plt.plot(data[x], data[y], label='loss value', marker='o')

        # Set labels
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title('Reward values for MY_DQN model')
        # Set grid
        plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
        plt.legend()
        # Show the plot
        print('Creating plot -----')
        plt.savefig('result/reward-MY4.png')
        print('done')

    def price_action(self, filename):
        query = f'''
                   from(bucket: "{bucket_ohlcv}")
                     |> range(start: {self.start_time}, stop: {self.stop_time})
                     |> filter(fn: (r) => r["_measurement"] == "ohlcvBucket")
                     |> filter(fn: (r) => r["_field"] == "close") // Use the trade ID field for counting trades
                     |> filter(fn: (r) => r["symbol"] == "{self.symbol}")
                     |> aggregateWindow(every: {self.window}, fn: mean, createEmpty: false) // Count trades every hour
                     |> yield(name: "mean")
                   '''
        # print(query2)
        tables_price = query_api.query(query)
        client.close()

        j = 0
        for table in tables_price:
            for record in table:
                t = pd.to_datetime(record['_time'])
                times.append(t)
                price.append(record['_value'])
        # Save logs to file
        print('# Save ohlcv to file')
        with open('result/article/test_close_%s.csv' % self.symbol, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['times', 'close'])
            for i in range(len(times)):
                writer.writerow((times[i], price[i]))

        plt.figure(figsize=(12, 6))
        plt.style.use('dark_background')

        # Customize the plot
        plt.title('Price Diagram with Buy/Sell Actions (Triangles) %s'%self.symbol)
        plt.xlabel('Time')
        plt.ylabel('Price')
        # plt.xticks(rotation=45)
        plt.plot(times, price, color='white', label='price')

        # read csv file
        data = pd.read_csv(filename)
        num = 1
        for action, action_time, action_price in zip(data['action'], data['action_time'], data['action_price']):
            print('action in for', action)
            print('action in for', action_time)
            if action_price == 0: continue
            if action == 0:
                plt.scatter(pd.to_datetime(action_time), action_price, color='red', marker='v', s=100, label=f'{num}')
            elif action == 1:
                plt.scatter(pd.to_datetime(action_time), action_price, color='red', marker='v', s=100, label=f'{num}')
            elif action == 3:
                plt.scatter(pd.to_datetime(action_time), action_price, color='green', marker='^', s=100, label=f'{num}')
            if action == 4:
                plt.scatter(pd.to_datetime(action_time), action_price, color='green', marker='^', s=100, label=f'{num}')
            elif action == 2:
                plt.scatter(pd.to_datetime(action_time), action_price, color='yellow', marker='*', s=100, label=f'{num}')
            # plt.annotate(str(num), xy=(action_time, action_price-100))
            num += 1
            # # Find the closest price point for the action_time
            # closest_time_idx = df['_time'].sub(action_time).abs().idxmin()
            # closest_price = df.loc[closest_time_idx, 'price']
        #

        # Show the plot


        # Ensure layout fits well
        plt.tight_layout()
        # Show the plot
        print('Creating plot ....')
        plt.savefig('result/article/test_my_dqn_%s_%s.png' % (config['slice_size'], self.symbol))
        print('save done')
        return j

    def price_action_plot_from_csv(self):
        # Read data from CSV
        filename_ohlcv = 'result/test_close_%s.csv' % self.symbol
        filename_action = 'result/test_action_%s_e.csv' % self.symbol

        data_ohlcv = pd.read_csv(filename_ohlcv)

        plt.style.use('dark_background')

        # Create the plot
        plt.figure(figsize=(10, 6))

        plt.plot(data_ohlcv['times'], data_ohlcv['close'], label='price %s'%self.symbol, color = 'white')

        # Set labels
        plt.xlabel('time')
        plt.ylabel('price')
        plt.title('Price Diagram with Sell/Buy actions (triangles) %s'%self.symbol)
        data = pd.read_csv(filename_action)
        num = 1
        for action, action_time, action_price in zip(data['action'], data['action_time'], data['action_price']):
            if action_price == 0: continue
            if action < 2:
                plt.scatter(action_time, action_price, color='red', marker='v', s=100, label=f'{num}')
            elif action > 2:
                plt.scatter(action_time, action_price, color='green', marker='^', s=100, label=f'{num}')
            elif action == 2:
                plt.scatter(action_time, action_price, color='yellow', marker='+', s=100,
                            label=f'{num}')
            # plt.annotate(str(num), xy=(action_time, action_price-100))
            num += 1
        # Show the plot
        print('Creating plot -----')
        plt.savefig('result/test_action_%s_e.png'%self.symbol)
        print('done')


# i, j = query_trades_count()
# agent = DQNAgent(config)
# agent.test()
#
# query = f'''
#             from(bucket: "tradeBucket")
#               |> range(start: {start_time}, stop: {stop_time})
#               |> filter(fn: (r) => r["_measurement"] == "trades")
#               |> filter(fn: (r) => r["_field"] == "quantity") // Use the trade ID field for counting trades
#               |> filter(fn: (r) => r["symbol"] == "BTCUSDT")
#               |> aggregateWindow(every: 4h, fn: sum, createEmpty: false) // Count trades every hour
#               |> yield(name: "sum")
#             '''
# tables_price = query_api.query(query)
# client.close()
#
# i = 0
# for table in tables_price:
#     for record in table:
#         t = pd.to_datetime(record['_time'])
#         times.append(t)
#         # index.append(i)
# #         i += 1
#         quantity.append(record['_value'])
# # timestamps = pd.date_range(start='2024-07-21', end='2024-08-21', freq='5T')
# d = {'_time': times, 'quantity': quantity}
# df2 = pd.DataFrame(d)
# df2['_time'] = pd.to_datetime(df2['_time'])
# # Set the time as the index for plotting
# # df.set_index('index', inplace=True)
#
# # print('\ntime', times)
# # print('price', price)
# print('time len', len(times))
# print('quantity len', len(quantity))
# print('df len', len(df2))


# Add random triangles at 12-hour intervals
# for action_time, action in zip(agent.test_action_time, agent.test_action):
#     # Find the closest price point for the action_time
#     closest_time_idx = df['_time'].sub(action_time).abs().idxmin()
#     closest_price = df.loc[closest_time_idx, 'price']
#
#     # Determine the color and orientation of the triangle based on the action
#     if action == 0:
#         color = 'red'
#         marker = 'v'  # Upward triangle
#         size = 200
#     elif action == 1:
#         color = 'blue'
#         marker = '+'  # Upward triangle
#         size = 50
#     elif action == 2:
#         color = 'green'
#         marker = '^'  # Downward triangle
#         size = 200
#     elif action == 3:
#         color = 'green'
#         marker = '^'  # Downward triangle
#         size = 100
#     elif action == 4:
#         color = 'green'
#         marker = '^'  # Downward triangle
#         size = 300
#     else:
#         continue  # Skip actions that are not 0, 1, 3, or 4
#
#     # Plot the triangle
#     plt.scatter(action_time, closest_price-1000, color=color, marker=marker, s=size, label=f'Action {action}')

# READ SCV FILE
# csv_file = f'dataset/BTCUSDT_ohlcv.csv'
# df = pd.read_csv(csv_file)
# #
# # # Ensure 'time' is in datetime format
# df['time'] = pd.to_datetime(df['time'])
# volume = []
# vtime  = []
# j = 1
# vsum = 0
# for i in df.iterrows():
#     vsum += i[1]['volume']
#     if j%8 == 0:
#         volume.append(vsum)
#         vtime.append(vtime)
#         vsum = 0
#     j += 1
# d = {'time': vtime, 'volume': volume}
# df_csv = pd.DataFrame(d)
# #
# # # Plot the price line
# # plt.figure(figsize=(20, 12), facecolor='#6857f5' )
# # # # df.sort_values(by=['_time'])
# # # ax = plt.axes()
# # # ax.set_facecolor('#6857f5')
# # # df2['quantity'].plot(kind='bar', label='quantity', color='white')
# #
# # # df['time'] = pd.to_datetime(df['time'])
# # plt.plot(vtime, volume, kind='bar', label='quantity', color='green')
# df_csv['volume'].plot(kind='bar', label='quantity', color='blue')
# # Customize the plot
# plt.title('Price Diagram with Buy/Sell Actions (Triangles)')
# plt.xlabel('Time')
# plt.ylabel('Price')
# # plt.legend(loc='best')
# plt.xticks(rotation=45)
# # plt.tight_layout()
#
# plt.savefig('plots/volume_quantity_csv.png')

local_config = {
    'start': dt.datetime.fromisoformat('2024-07-20T00:00:00'),
    'end': dt.datetime.fromisoformat('2024-09-12T00:00:00'),
    'symbol': config['symbol'],
    'window': '30m'
}
q = Query(local_config)
# df = q.query_trades_count()

# q.symbol = 'BTCUSDT'
# df = q.query_trades_count()
# q.plot_count(df, 'BTCUSDT', 'red', 'plots/_trades_count_BU_24h.png')

# q.plot_from_csv('result/training_logs_MY_dqn_4channel_16_2.csv', 'Episode', 'reward3', 'Epoch', 'Reward')
# agent = MYDQNAgent(config)
# print('After Creating Agent in Main')
# agent.test()
q.price_action('result/article/test_action_%s_%s.csv' % (config['slice_size'], q.symbol))
# q.price_action_plot_from_csv()