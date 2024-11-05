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

        self.action_scheme = BSH()
        self.observer = ObserverM(config)
        self.reward_scheme = SimpleProfit()

        self.action_space = self.action_scheme.action_space
        self.observation_space = self.observer.observation_space

        self.step_count = 0
        self.current_state = None
        self.next_state_price = 0
        print('ENVIRONMENT object Ceated')

    def step(self, action_index, usdt_balance, btc_balance):
        print('NOW IN ENVIRONMENT STEP: ', self.step_count)
        action = self.action_scheme.actions[action_index]
        usdt_balance, btc_balance = self.action_scheme.perform(action, usdt_balance, btc_balance, self.next_state_price)
        current_price = self.next_state_price
        next_state_i = self.observer.next_image_i
        next_state, self.next_state_price = self.observer.observe()
        # self.observer.next_image_i -= self.observer.slice_size

        new_state_price = self.next_state_price
        # reward = self.reward_scheme.reward(action, current_price, new_state_price)
        reward = self.reward_scheme.reward(usdt_balance, btc_balance, action, current_price, new_state_price)
        self.current_state = next_state
        self.step_count += 1
        # Define when the episode is done
        done = False
        if usdt_balance < 0 or btc_balance < 0 or len(self.current_state) == 1 or self.step_count >= self.max_steps:
            done = True
            print("Episode finished:")
            if usdt_balance <= 0:
                print("USDT balance fell to 0 or below.")
            if btc_balance <= 0:
                print("BTC balance fell to 0 or below.")
            if self.step_count >= self.max_steps:
                print("Reached maximum number of steps.")
            if len(self.current_state) == -1:
                print('THE current state is -1')
        return [next_state, self.next_state_price, reward, done, usdt_balance, btc_balance, next_state_i]

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
        self.current_state, self.next_state_price = self.observer.observe()
        return self.current_state, self.next_state_price

    def render(self, mode='human'):
        pass
