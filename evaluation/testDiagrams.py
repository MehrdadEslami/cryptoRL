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

def merge_count_price(times,times2,count, price):
    i, j = 0, 0
    times3 = []
    count2 = []
    price2 = []
    while i != len(count) and j != len(price):
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
    return count2, price2, times3

class Evaluation:

    def __init__(self, local_config, filename_date = None):
        start = local_config['start']
        end = local_config['end']
        self.start_time = start.strftime('%Y-%m-%dT%H:%M:%SZ')
        self.stop_time = end.strftime('%Y-%m-%dT%H:%M:%SZ')
        self.symbol = local_config['symbol']
        self.window = local_config['window']
        self.model_name = local_config['model_name']
        self.filename_date = filename_date

        self.symbol = local_config['symbol']
        self.first_balance = local_config['first_balance']
        self.second_balance = local_config['second_balance']

        if self.filename_date:
            self.action_df = pd.read_csv('../result/%s/test_action_%s_%s_%s.csv' % (
                self.model_name, config['slice_size'], self.symbol, self.filename_date))
            self.close_df = pd.read_csv('../result/%s/test_close_%s.csv' % (self.model_name, self.symbol))

        # Convert action_time to datetime format
        self.action_df['action_time'] = pd.to_datetime(self.action_df['action_time'])
        self.close_df['times'] = pd.to_datetime(self.close_df['times'])

        self.minutes = local_config['minutes']
        self.hours = local_config['hours']

        # Store portfolio values and rewards
        self.portfolio_values = []
        self.rewards = []
        self.initial_portfolio_value = self.first_balance

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
        print('sum',np.sum(df['count']))
        print('mean',np.mean(df['count']))
        print('max',np.max(df['count']))
        print('min',np.min(df['count']))

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
        times=[]
        count=[]
        times2 = []
        times = []
        price = []
        count = []
        for table in tables_count:
            for record in table:
                print(j)
                j += 1

                t = pd.to_datetime(record['_time'])
                if t in times:
                    k = times.index(t)
                    count[k] += record['_value']
                else:
                    print('i:',i)
                    i += 1
                    times.append(t)
                    count.append(record['_value'])
        times2 = []
        price = []
        for table in tables_price:
            for record in table:
                t = pd.to_datetime(record['_time'])
                times2.append(t)
                price.append(record['_value'])

        [count2, price2, times3] = merge_count_price(times,times2,count, price)
        c = (count2 - np.min(count)) / (np.max(count) - np.min(count))
        p = (price2 - np.min(price)) / (np.max(price) - np.min(price))
        # for k in range(len(c)):
        #     c[k] = round(c[k], 2)
        #     p[k] = round(p[k], 2)

        d = {'_time': times3, 'count': c, 'price': p}
        df = pd.DataFrame(d)
        # Set the time as the index for plotting
        # df.set_index('_time', inplace=True)

        # change temp
        # df._set_value(20, 'count', 0.9)

        # Plot the bar chart
        return df

    @staticmethod
    def plot_count(df, label, c, filename):

        plt.figure(figsize=(6, 3))
        day = []
        for i in range(len(df)):
            day.append(df.iloc[i]['_time'].day)
        plt.bar(day, df['count'], width=0.2, color=c, label=label)

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
        plt.figure(figsize=(8, 4))
        df['count'].plot(kind='bar', width=0.4, color='blue', label='Number of Trades')
        df['price'].plot(linewidth=2.0, color='red', label='price')
        plt.title('Number of ETH-USDT    Trades Done Per 4 hours and normilized Price')
        plt.xlabel('Time ')
        plt.ylabel('Number of Trades, Price')
        plt.xticks([])
        # plt.annotate()
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        print('Creating Plot.....')
        plt.savefig('../result/AutoEncoder/_count_price_4h_EU.png')
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

    def price_action(self):
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
        times = []
        price = []
        j = 0
        for table in tables_price:
            for record in table:
                t = pd.to_datetime(record['_time'])
                times.append(t)
                price.append(record['_value'])
        # Save logs to file
        print('# Save ohlcv to file')
        with open('../result/%s/test_close_%s.csv' % (self.model_name, self.symbol), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['times', 'close'])
            for i in range(len(price)):
                writer.writerow((times[i], price[i]))

        plt.figure(figsize=(12, 6))
        # plt.style.use('dark_background')
        # plt.style.use('Solarize_Light2')
        plt.style.use('Solarize_Light2')

        # Customize the plot
        plt.title('Price Diagram with Buy/Sell Actions %s (%s)'% (self.symbol, self.model_name) )
        plt.xlabel('Time')
        plt.ylabel('Price')
        # plt.xticks(rotation=45)
        plt.plot(times, price,  label='close price')
        # plt.plot(times[-240:], price[-240:], color='brown', label='testing')

        # read csv file
        data = self.action_df
        self.current_action = data.iloc[0]['action']
        num = 1
        allow_label = [True, True, True]
        for action, action_time, action_price in zip(data['action'], data['action_time'], data['action_price']):
            print('action in for', action)
            print('action in for', action_time)
            if action_price == 0:
                continue
            if self.current_action == action:
                action = 1

            if action == 0:
                if allow_label[0]:
                    plt.scatter(pd.to_datetime(action_time), action_price, color='red', marker='v', s=150, label='Short')
                    allow_label[0]=False
                else:
                    plt.scatter(pd.to_datetime(action_time), action_price, color='red', marker='v', s=150)
            elif action == 4:
                plt.scatter(pd.to_datetime(action_time), action_price, color='red', marker='v', s=150)
                # plt.scatter(pd.to_datetime(action_time), action_price, color='red', marker='v', s=100, label=f'{num}')
            elif action == 2:
                if allow_label[1]:
                    plt.scatter(pd.to_datetime(action_time), action_price, color='green', marker='^', s=150, label='Long')
                    allow_label[1] = False
                else:
                    plt.scatter(pd.to_datetime(action_time), action_price, color='green', marker='^', s=150)
            if action == 3:
                plt.scatter(pd.to_datetime(action_time), action_price, color='green', marker='^', s=150)
            elif action == 1:
                if allow_label[2]:
                    plt.scatter(pd.to_datetime(action_time), action_price, color='yellow', marker='*', s=150, label='Hold')
                    allow_label[2] = False
                else:
                    plt.scatter(pd.to_datetime(action_time), action_price, color='yellow', marker='*', s=150)

            self.current_action = action
            # plt.annotate(str(num), xy=(action_time, action_price-100))
            num += 1
            # # Find the closest price point for the action_time
            # closest_time_idx = df['_time'].sub(action_time).abs().idxmin()
            # closest_price = df.loc[closest_time_idx, 'price']
        #

        # Show the plot


        # Ensure layout fits well
        plt.tight_layout()
        plt.legend()
        # Show the plot
        print('Creating plot ....')

        plt.savefig('../result/%s/test_my_dqn_%s_%s_%s.png' % (self.model_name, config['slice_size'], self.symbol, filename_date))
        print('save done')
        return j

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
            action_reward = row['reward']

            # Get the close price 2 hours after the action
            close_price_after = self.get_close_price_later(action_time)

            # Calculate portfolio based on action space
            if action == 0 or action == 1:  # Sell 75% of ETH
                if last_action == 'buy' or last_action == 'hold':
                    self.first_balance += (self.second_balance * 1) * action_price
                    self.second_balance -= self.second_balance * 1
                    last_action = 'sell'
                elif last_action == 'sell':
                    last_action = 'hold'

            elif action == 3 or action == 4:  # Sell 25% of ETH
                if last_action == 'sell' or last_action == 'hold':
                    amount_to_buy = (self.first_balance * 1) / action_price
                    self.second_balance += amount_to_buy
                    self.first_balance -= self.first_balance * 1
                    last_action = 'buy'
                elif last_action == 'buy':
                    last_action = 'hold'
            else:
                last_action = 'hold'


            # Calculate current portfolio value
            current_portfolio_value = (self.second_balance * close_price_after) + self.first_balance
            self.portfolio_values.append(current_portfolio_value)

            # Calculate reward as the difference between current and previous portfolio value
            if index == 0:
                reward = current_portfolio_value - self.initial_portfolio_value
            else:
                reward = current_portfolio_value - self.portfolio_values[-2]
            self.rewards.append(reward)

    def evaluate_3action(self):
        # Loop through each action and calculate portfolio value, reward
        last_action = 'hold'
        for index, row in self.action_df.iterrows():
            action = row['action']
            action_time = row['action_time']
            action_price = row['action_price']
            self.rewards.append(row['reward'])

            # Get the close price 2 hours after the action
            close_price_after = self.get_close_price_later(action_time)

            if action == 0:  # Sell
                if last_action == 'buy' or last_action == 'hold':
                    self.first_balance += (self.second_balance * 1) * action_price
                    self.second_balance -= self.second_balance * 1
                    last_action = 'sell'
                elif last_action == 'sell':
                    last_action = 'hold'

            elif action == 2:  # buy
                if last_action == 'sell' or last_action == 'hold':
                    amount_to_buy = (self.first_balance * 1) / action_price
                    self.second_balance += amount_to_buy
                    self.first_balance -= self.first_balance * 1
                    last_action = 'buy'
                elif last_action == 'buy':
                    last_action = 'hold'
            else:
                last_action = 'hold'
            # Calculate current portfolio value
            current_portfolio_value = (self.second_balance * close_price_after) + self.first_balance
            if action != 1:
                self.portfolio_values.append(current_portfolio_value)

            # # Calculate reward as the difference between current and previous portfolio value
            # if index == 0:
            #     reward = current_portfolio_value - self.initial_portfolio_value
            # else:
            #     reward = current_portfolio_value - self.portfolio_values[-1]
            # self.rewards.append(reward)

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
        plt.figure(figsize=(8, 4))
        plt.style.use('Solarize_Light2')
        plt.plot(range(len(self.rewards)), self.rewards)
        plt.title('Rewards after each action (%s)' % self.model_name)
        plt.xlabel('Action number')
        plt.ylabel('Reward')
        plt.grid(True)
        print('Creating Plot')
        plt.savefig('../result/%s/reward_%s.png'%(self.model_name, self.symbol))
        print('Plot Done')

        # Print key metrics
        SR = self.sharp_ratio()
        MDD = self.Max_Drawdown()
        print(f"Sharpe Ratio:", np.mean(SR))
        print(f"Average Reward: {np.mean(self.rewards)}")
        print(f"sum Reward: {np.sum(self.rewards)}")
        print(f"Min Reward: {np.min(self.rewards)}")
        print(f"Max Reward: {np.max(self.rewards)}")
        print(f"var Reward: {np.var(self.rewards)}")
        print(f"Maximum Drawdown: {MDD}")


    def plot_sharp_ratio(self):
        # Assuming you have a list of Sharpe ratios over time
        sharpe_ratios = self.sharp_ratio()

        # Plotting the Sharpe Ratio over time
        plt.figure(figsize=(8, 4))
        plt.style.use('Solarize_Light2')
        plt.plot(range(len(sharpe_ratios)), sharpe_ratios)
        plt.title('Sharpe Ratio Over Time (%s)'%self.model_name)
        plt.xlabel('Trade/Interval')
        plt.ylabel('Sharpe Ratio')
        plt.savefig('../result/%s/sharoRatio_%s.png'%(self.model_name,self.symbol))

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

    def nearest_close_price(self, current_time, df_close):

        # Convert current_time to a pandas datetime object
        current_time = pd.to_datetime(current_time)

        # Ensure the 'time' column in df_close is in datetime format
        df_close['time'] = pd.to_datetime(df_close['time'])

        # set timedelta
        delta = td(minutes=30)

        # Filter for close times after the current time - 30 minutes
        future_close_times = df_close[df_close['time'] > current_time - delta]

        # Get the nearest close time
        if not future_close_times.empty:
            current_close_price = min(future_close_times.iloc[0]['close'], future_close_times.iloc[1]['close'])
            nearest_row = future_close_times.iloc[2]
            nearest_close_price = nearest_row['close']
            return current_close_price, nearest_close_price
        else:
            # If no future close time is found, return None or handle it
            return 56752.16, 56274.86

    def calculate_metrics(self):
        # Load the action and close price data
        df_close = self.close_df
        actions = self.action_df
        d = {'time': actions['action_time'], 'action': actions['action'], 'price': actions['action_price']}
        df = pd.DataFrame(d)

        # Sort the dataframe by time
        df = df.sort_values(by='time')

        # Initialize counts
        TP, FP, FN = 0, 0, 0

        # Iterate over the dataframe
        for i in range(len(df) - 1):
            current_action = df.iloc[i]['action']
            current_price = df.iloc[i]['price']
            next_price = df.iloc[i + 1]['price']
            # current_time = df.iloc[i]['time']
            # current_price, next_price = nearest_close_price(current_time, df_close)

            # Buy action: Expect future price to increase
            if current_action > 2:
                if next_price > current_price:
                    TP += 1  # True Positive: Successful buy
                else:
                    FP += 1  # False Positive: Failed buy

            # Sell action: Expect future price to decrease
            elif current_action < 2:
                if next_price < current_price:
                    TP += 1  # True Positive: Successful sell
                else:
                    FP += 1  # False Positive: Failed sell

            # Hold action: Missed opportunity if price changes drastically
            elif current_action == 2:
                if next_price > current_price * 1.02:
                    FN += 1  # False Negative: Missed opportunity to buy
                elif next_price * 1.02 < current_price:
                    FN += 1  # False Negative: Missed opportunity to sell
                else:
                    TP += 1

        return TP, FP, FN


class_config = {
    'start': dt.datetime.fromisoformat('2024-07-20T00:00:00'),
    'end': dt.datetime.fromisoformat('2024-11-01T00:00:00'),
    'symbol': config['symbol'],
    'model_name' : config['model_name'],
    'window': '1h',
    'first_balance': 1000,
    'second_balance': 0,
    'minutes': 15,
    'hours': 0
}

# agent = MYDQNAgent(config)
agent = AutoEncoderDDQNAgent(config)
print('After Creating Agent in Main')
filename_date = agent.test()

# filename_date = '%s:%s:%s-%s:%s' % (2024, 11, 2, 15, 57)
# filename_date = '%s-%s-%sT%s:%s' % (2024, 11, 2, 15, 57)
e = Evaluation(class_config, filename_date)

e.price_action()

# d = datetime.datetime.fromisoformat(agent.env.observer.last_trade_time)
# filename_date = '%s-%s-%sT%s:%s' % (d.year, d.month, d.day, d.hour, d.minute)


#EVALUATING
e.evaluate_3action()
e.plot_reward()
e.plot_sharp_ratio()
e.Base_model()


#Precision and RECALL

TP, FP, FN = e.calculate_metrics()
# df = calculate_metrics(action_file, close_file)

# Output the results
print(f"True Positives: {TP}")
print(f"False Positives: {FP}")
print(f"False Negatives: {FN}")

print('precision is', TP / (TP+FP))
print('recall is', TP / (TP+FN) )