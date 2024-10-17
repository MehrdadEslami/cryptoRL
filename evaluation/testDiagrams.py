import csv
import datetime
import random

from influxdb_client import InfluxDBClient
import numpy as np
import pandas as pd
import json
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime as dt
from datetime import timedelta

# from agents.DQN_agent import DQNAgent
# from agents.MY_DQN_agent import MYDQNAgent
from agents.AutoEncoder_DDQN_agent import AutoEncoderDDQNAgent

with open("../config.json", "r") as file:
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
        # df.set_index('_time', inplace=True)
        print(df)
        print(df.shape)
        print(np.sum(df['count']))
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

    @staticmethod
    def plot_count(df, label, c, filename):

        plt.figure(figsize=(9, 3))
        day = []
        for i in range(len(df)):
            day.append(df.iloc[i]['_time'].day)
        plt.bar(day, df['count'], width=0.8, color=c, label=label)

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

    @staticmethod
    def plot_count_ohlcv(df):
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

    @staticmethod
    def plot_from_csv(filename, x, y, xlabel, ylabel):
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

    def price_action(self, action_filename, filename_date):
        query = f'''
                   from(bucket: "{bucket_ohlcv}")
                     |> range(start: {self.start_time}, stop: {self.stop_time})
                     |> filter(fn: (r) => r["_measurement"] == "ohlcvBucket")
                     |> filter(fn: (r) => r["_field"] == "close") // Use the trade ID field for counting trades
                     |> filter(fn: (r) => r["symbol"] == "{self.symbol}")
                     |> aggregateWindow(every: {self.window}, fn: mean, createEmpty: false) // Count trades every hour
                     |> yield(name: "mean")
                   '''
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
        with open('../result/AutoEncoder/test_close_%s.csv' % self.symbol, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['times', 'close'])
            for i in range(len(price)):
                writer.writerow((times[i], price[i]))

        plt.figure(figsize=(12, 6))
        plt.style.use('dark_background')
        # plt.style.use('fivethirtyeight')
        # Customize the plot
        plt.title('Price Diagram with Buy/Sell Actions (Triangles) %s'%self.symbol)
        plt.xlabel('Time')
        plt.ylabel('Price')
        # plt.xticks(rotation=45)
        plt.plot(times, price, color='white', label='price')

        # read csv file
        data = pd.read_csv(action_filename)
        num = 1
        for action, action_time, action_price in zip(data['action'], data['action_time'], data['action_price']):
            print('action in for', action)
            print('action in for', action_time)
            if action_price == 0:
                continue
            if action == 0:
                plt.scatter(pd.to_datetime(action_time), action_price, color='red', marker='v', s=100, label=f'{num}')
            elif action == 1:
                plt.scatter(pd.to_datetime(action_time), action_price, color='yellow', marker='*', s=100,
                            label=f'{num}')
                # plt.scatter(pd.to_datetime(action_time), action_price, color='red', marker='v', s=100, label=f'{num}')
            elif action == 3:
                plt.scatter(pd.to_datetime(action_time), action_price, color='green', marker='^', s=100, label=f'{num}')
            if action == 4:
                plt.scatter(pd.to_datetime(action_time), action_price, color='green', marker='^', s=100, label=f'{num}')
            elif action == 2:
                plt.scatter(pd.to_datetime(action_time), action_price, color='green', marker='^', s=100, label=f'{num}')

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

        plt.savefig('../result/AutoEncoder/test_my_dqn_%s_%s_%s.png' % (config['slice_size'], self.symbol, filename_date))
        print('save done')
        return j


class Evaluation:
    def __init__(self, local_config):
        self.symbol = local_config['symbol']
        self.USDT_balance = local_config['USDT_balance']
        self.ETH_balance = local_config['symbol_balance']
        self.action_df = pd.read_csv('result/article/test_action_%s_%s.csv' % (config['slice_size'], self.symbol))
        self.close_df = pd.read_csv('result/article/test_close_%s.csv' % self.symbol)

        # Convert action_time to datetime format
        self.action_df['action_time'] = pd.to_datetime(self.action_df['action_time'])
        self.close_df['times'] = pd.to_datetime(self.close_df['times'])
        self.minutes = local_config['minutes']
        self.hours = local_config['hours']
        # Store portfolio values and rewards
        self.portfolio_values = []
        self.rewards = []
        self.initial_portfolio_value = self.USDT_balance

    # Helper function to get the close price 2 hours after the action time
    def get_close_price_later(self, action_time):
        close_time = action_time + timedelta(minutes=self.minutes, hours=self.hours)
        nearest_close = self.close_df[self.close_df['times'] >= close_time]
        if nearest_close.empty:
            return self.close_df.iloc[-1]['close']
        else:
            return nearest_close.iloc[0]['close']


    def evaluate(self):
        # Loop through each action and calculate portfolio value, reward
        last_action = 'hold'
        for index, row in self.action_df.iterrows():
            action = row['action']
            action_time = row['action_time']
            action_price = row['action_price']

            # Get the close price 2 hours after the action
            close_price_after = self.get_close_price_later(action_time)

            # Calculate portfolio based on action space
            if action == 0 or action == 1:  # Sell 75% of ETH
                if last_action == 'buy' or last_action == 'hold':
                    self.USDT_balance += (self.ETH_balance * 1) * action_price
                    self.ETH_balance -= self.ETH_balance * 1
                    last_action = 'sell'
                elif last_action == 'sell':
                    last_action = 'hold'

            elif action == 3 or action == 4:  # Sell 25% of ETH
                if last_action == 'sell' or last_action == 'hold':
                    amount_to_buy = (self.USDT_balance * 1) / action_price
                    self.ETH_balance += amount_to_buy
                    self.USDT_balance -= self.USDT_balance * 1
                    last_action = 'buy'
                elif last_action == 'buy':
                    last_action = 'hold'
            else:
                last_action = 'hold'


            # Calculate current portfolio value
            current_portfolio_value = (self.ETH_balance * close_price_after) + self.USDT_balance
            self.portfolio_values.append(current_portfolio_value)

            # Calculate reward as the difference between current and previous portfolio value
            if index == 0:
                reward = current_portfolio_value - self.initial_portfolio_value
            else:
                reward = current_portfolio_value - self.portfolio_values[-2]
            self.rewards.append(reward)

    # Calculate Sharpe Ratio
    def sharp_ratio(self):
        returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(len(returns))
        return returns

    def Max_Drawdown(self):
        # Calculate Maximum Drawdown
        cumulative_returns = np.array(self.portfolio_values) / self.initial_portfolio_value
        running_max = np.maximum.accumulate(cumulative_returns)
        draw_down = (running_max - cumulative_returns) / running_max
        max_draw_down = np.max(draw_down)
        return max_draw_down

    def plot_reward(self):
        # Plot reward
        plt.plot(self.rewards)
        plt.title('Rewards after each action')
        plt.xlabel('Action number')
        plt.ylabel('Reward')
        plt.grid(True)
        print('Creating Plot')
        plt.savefig('result/article/reward_%s.png'%self.symbol)
        print('Plot Done')

        # Print key metrics
        SR = self.sharp_ratio()
        MDD = self.Max_Drawdown()
        print(f"Sharpe Ratio: {SR}")
        print(f"Maximum Drawdown: {MDD}")
        print(f"Max Reward: {np.max(self.rewards)}")
        print(f"Min Reward: {np.min(self.rewards)}")
        print(f"var Reward: {np.var(self.rewards)}")
        print(f"Average Reward: {np.mean(self.rewards)}")
        print(f"sum Reward: {np.sum(self.rewards)}")

    def plot_sharp_ratio(self):
        # Assuming you have a list of Sharpe ratios over time
        sharpe_ratios = self.sharp_ratio()

        # Plotting the Sharpe Ratio over time
        plt.plot(sharpe_ratios)
        plt.title('Sharpe Ratio Over Time')
        plt.xlabel('Trade/Interval')
        plt.ylabel('Sharpe Ratio')
        plt.savefig('result/article/sharoRatio_%s.png'%self.symbol)

        # Calculate mean Sharpe Ratio
        mean_sharpe_ratio = np.mean(sharpe_ratios)
        print(f"Mean Sharpe Ratio: {mean_sharpe_ratio:.2f}")

    def Base_model(self):
        # Extract start and end prices from the close_df
        start_price = self.close_df.iloc[0]['close']
        end_price = self.close_df.iloc[-1]['close']
        initial_balance = 1000

        buy_and_hold_profit = (end_price / start_price - 1) * initial_balance
        print(f"Buy and Hold Profit: {buy_and_hold_profit:.2f} USDT")
        sell_and_hold_profit = (1 - end_price / start_price) * initial_balance
        print(f"Sell and Hold Profit: {sell_and_hold_profit:.2f} USDT")

class_config = {
    'start': dt.datetime.fromisoformat('2024-07-10T00:00:00'),
    'end': dt.datetime.fromisoformat('2024-10-04T00:00:00'),
    'symbol': config['symbol'],
    'window': '1h',
    'USDT_balance': 1000,
    'symbol_balance': 0,
    'minutes': 15,
    'hours': 0
}
q = Query(class_config)
# df = q.query_trades_count()
# print(len(df))

# df = q.query_trades_count()
# q.plot_count(df, 'BTCUSDT', 'blue', 'result/article/trades_count_BU_24h.png')

# q.plot_from_csv('result/training_logs_MY_dqn_4channel_16_2.csv', 'Episode', 'reward3', 'Epoch', 'Reward')
# agent = MYDQNAgent(config)
agent = AutoEncoderDDQNAgent(config)
print('After Creating Agent in Main')
agent.test()
d = datetime.datetime.fromisoformat(agent.env.observer.last_trade_time)
filename_date = '%s-%s-%sT%s:%s' % (d.year, d.month, d.day, d.hour, d.minute)
# filename_date = '%s-%s-%sT%s:%s' % (2024, 10, 4, 14, 19)
q.price_action(action_filename='../result/AutoEncoder/test_action_%s_%s_%s.csv' % (config['slice_size'], q.symbol, filename_date),
               filename_date=filename_date)

#EVALUATING
# e = Evaluation(class_config)
# e.evaluate()
# e.plot_reward()
# e.plot_sharp_ratio()
# e.Base_model()