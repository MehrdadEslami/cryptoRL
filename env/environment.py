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
        [usdt_balance, btc_balance] = self.action_scheme.perform(self, action, usdt_balance, btc_balance, self.current_state_mean_price)
        self.step_count += 1
        current_price = self.current_state_mean_price
        next_state = self.observer.observe()
        new_state_price = self.current_state_mean_price
        reward = self.reward_scheme.reward(action, current_price, new_state_price)
        self.current_state = next_state
        done = False
        return [next_state, reward, done, usdt_balance, btc_balance]

    def reset(self):
        print('NOW ENVIRONMENT IS RESESING >>>> ')
        self.step_count = 0
        self.observer.reset()
        self.current_state, self.current_state_mean_price = self.observer.observe()
        return self.current_state, self.current_state_mean_price

    def render(self, mode='human'):
        pass
