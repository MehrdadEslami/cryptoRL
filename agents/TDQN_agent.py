import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.preprocessing import MinMaxScaler
from collections import deque
import random
import matplotlib.pyplot as plt


class TDQNAgent:
    def __init__(self, state_size, action_size):
        self.return_rate = None
        self.sharpe_ratio = None
        self.rewards = None
        self.max_drawdown = None
        self.state_size = state_size
        self.action_size = action_size
        self.USDT_balance = 1000
        self.ETH_balance = 0
        self.data = None
        self.scaled_data = None
        self.scaler = None
        self.portfolio_values = None
        self.sequences = None
        self.memory = deque(maxlen=2000)
        self.seq_length = 30

        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self.build_tdqn_model()
        self.target_model = self.build_tdqn_model()
        self.update_target_model()

    def fetch_data(self):
        self.data = yf.download('ETH-USD', start='201801-01', end='2024-07-01', interval='1d')
        self.data = self.data[['Close']]  # فقط قیمت‌های پایانی را نیاز داریم

    def preprocess_data(self):
        if self.data is None:
            self.fetch_data()
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaled_data = self.scaler.fit_transform(self.data)

    def create_sequences(self):
        if self.scaled_data is None:
            self.preprocess_data()
        for i in range(len(self.scaled_data) - self.seq_length):
            self.sequences.append(self.scaled_data[i: i + self.seq_length])
        print('seq len', self.sequences.shape)

    def build_tdqn_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, episodes=1):
        if self.sequences is None:
            self.create_sequences()
        for e in range(episodes):
            print('-----------------episode: %d----------'%e)
            state = self.sequences[0]  # یک دنباله زمانی اولیه برای شروع
            state = np.reshape(state, [1, self.seq_length])
            for time in range(len(self.sequences) - 1):  # 500 قدم معاملاتی
                print('-----------------time: %d, episode:%d----------' % (time,e))
                action = self.act(state)
                next_state = self.sequences[time + 1]
                next_state = np.reshape(next_state, [1, self.seq_length])
                price_change = (next_state[0][-1] - state[0][0]) / state[0][0] * 100
                if action == 0:
                    real_action = -1
                elif action == 2:
                    real_action = 1
                else:
                    real_action = 0
                reward = round(real_action * price_change, 3)  # محاسبه پاداش بر اساس استراتژی معاملاتی
                if reward == -0.0:
                    reward = 0
                done = time == 2200
                print('action:%d, reward:%f' % (action, reward))
                self.remember(state, action, reward, next_state, done)
                if time % 50 == 0:
                    # ذخیره مدل آموزش دیده
                    agent.save("/model_weights/tdqn_ethusdt.weights.h5")
                state = next_state
                if done:
                    self.update_target_model()
                    print(f"Episode: {e}/{episodes}, Score: {time}, Epsilon: {agent.epsilon}")
                    break
                if len(agent.memory) > 32:
                    agent.replay(32)

    def test_tdqn(self, test_data):
        self.USDT_balance = 1000  # مقدار اولیه USDT
        self.ETH_balance = 0  # مقدار اولیه ETH
        self.portfolio_values = []  # ذخیره ارزش سبد سهام در طول زمان

        for i in range(len(test_data) - self.seq_length):
            state = test_data[i: i + self.seq_length]
            state = np.reshape(state, [1, self.seq_length])

            action = self.act(state)

            # اگر خرید باشد
            if action == 0 and self.USDT_balance > 0:
                self.ETH_balance += self.USDT_balance / test_data[i + self.seq_length][0]  # خرید ETH به ازای USDT
                self.USDT_balance = 0
            # اگر فروش باشد
            elif action == 1 and self.ETH_balance > 0:
                self.USDT_balance += self.ETH_balance * test_data[i + self.seq_length][0]  # فروش ETH به USDT
                self.ETH_balance = 0

            # ارزش فعلی سبد سهام (USDT و ETH)
            portfolio_value = self.USDT_balance + self.ETH_balance * test_data[i + self.seq_length][0]
            self.portfolio_values.append(portfolio_value)

        return portfolio_values

    def calculate_max_drawdown(self):
        peak = self.portfolio_values[0]
        self.max_drawdown = 0
        for value in self.portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
        return self.max_drawdown * 100  # به درصد

    def calculate_rewards(self):
        self.rewards = np.diff(self.portfolio_values)  # تغییرات سبد سهام (پاداش)

    def plot_rewards_and_drawdown(self):
        # محاسبه پاداش‌ها
        rewards = self.calculate_rewards(self.portfolio_values)

        # رسم نمودار پاداش‌ها
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(rewards, color='green')
        plt.title('Rewards over Time')
        plt.xlabel('Time Steps')
        plt.ylabel('Rewards')
        plt.grid(True)

        # محاسبه حداکثر افت
        self.calculate_max_drawdown(portfolio_values)

        # رسم نمودار حداکثر افت
        plt.subplot(2, 1, 2)
        plt.plot(portfolio_values, color='blue')
        plt.title(f'Portfolio Value and Max Drawdown: {self.max_drawdown:.2f}%')
        plt.xlabel('Time Steps')
        plt.ylabel('Portfolio Value')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('result/article/TDQN_P_R.png')

    def calculate_sharpe_ratio(self):
        returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]  # بازده‌های روزانه
        mean_return = np.mean(returns)  # میانگین بازده روزانه
        std_return = np.std(returns)  # انحراف معیار بازده روزانه
        self.sharpe_ratio = mean_return / std_return * np.sqrt(252)  # 252 تعداد روزهای معاملاتی در یک سال

    def calculate_return(self):
        initial_value = 1000  # مقدار اولیه USDT
        final_value = self.portfolio_values[-1]  # مقدار نهایی سبد
        self.return_rate = (final_value - initial_value) / initial_value * 100  # بازده به درصد

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


agent = TDQNAgent(state_size=30, action_size=3)  # 3 اکشن: خرید، فروش، نگه داشتن
agent.train(episodes=100)

# فراخوانی تابع برای رسم نمودار
# plot_rewards_and_drawdown(portfolio_values)

# دریافت داده‌های تست (برای مثال داده‌های ETHUSDT از yfinance)
# test_data = fetch_data()

# شبیه‌سازی سبد سهام و محاسبه ارزش سبد
# portfolio_values = test_tdqn(agent, scaled_eth_data)

# محاسبه بازده نهایی
# return_rate = calculate_return(portfolio_values)
# print(f"Final Return: {return_rate:.2f}%")

# محاسبه نسبت شارپ
# sharpe_ratio = calculate_sharpe_ratio(portfolio_values)
# print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
