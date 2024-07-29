import json
from typing import List, Any

import gym
from gym import spaces
import numpy as np
from env.actions import BSH
from env.rewards import SimpleProfit
from env.observers import ObserverM




class TradingEnv(gym.Env):
    def __init__(self, trading_pair, config):
        super(TradingEnv, self).__init__()

        self.symbol = config['symbol']
        self.max_steps = int(config['max_steps'])
        self.trading_pair = trading_pair
        self.start_query_time = 0
        self.end_query_time = "now()"

        self.action_scheme = BSH(self.trading_pair)
        self.observer = ObserverM(config['influxdb'], self.symbol, buffer_size=int(config['buffer_size']))
        self.reward_scheme = SimpleProfit()

        self.action_space = self.action_scheme.action_space
        self.observation_space = self.observer.observation_space

        self.step_count = 0
        self.current_state = None
        self.current_state_mean_price = 0

    def step(self, action, usdt_balance, btc_balance):
        print('NOW IN ENVIRONMENT STEP: ', self.step_count)
        usdt_balance, btc_balance = self.action_scheme.perform(action, usdt_balance, btc_balance, self.current_state_mean_price)
        self.step_count += 1
        current_price = self.current_state_mean_price
        next_state, self.current_state_mean_price = self.observer.observe()
        if len(next_state) == 1:
            return [-1, -1, -1, True, usdt_balance, btc_balance]
        new_state_price = self.current_state_mean_price
        reward = self.reward_scheme.reward(action, current_price, new_state_price)
        self.current_state = next_state
        # Define when the episode is done
        done = False
        if usdt_balance < 0 or btc_balance < 0 or self.step_count >= self.max_steps:
            done = True
            print("Episode finished:")
            if usdt_balance <= 0:
                print("USDT balance fell to 0 or below.")
            if btc_balance <= 0:
                print("BTC balance fell to 0 or below.")
            if self.step_count >= self.max_steps:
                print("Reached maximum number of steps.")
        return [next_state, self.current_state_mean_price,  reward, done, usdt_balance, btc_balance]

    def calculate_max_q(self, next_state):
        next_next_state, next_next_mean_price = self.observer.observe()
        if len(next_next_state) == 1:
            return -1;
        buy = np.array(next_state[:, :, 0])
        sell = np.array(next_state[:, :, 1])
        price = np.array(next_state[:, :, 2])
        time = np.array(next_state[:, :, 3])
        print('price shape:', price.shape)
        profit = (buy * next_next_mean_price + sell * next_next_mean_price) - (buy * price + sell * price)
        return np.max(profit)

    def reset(self):
        print('NOW ENVIRONMENT IS RESESING >>>> ')
        self.step_count = 0
        self.observer.reset()
        self.current_state, self.current_state_mean_price = self.observer.observe()
        return self.current_state, self.current_state_mean_price

    def render(self, mode='human'):
        pass
