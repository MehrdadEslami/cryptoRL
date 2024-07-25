import json
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

        self.current_step = 0

    def step(self, action):
        print('IN ENV STEP')
        state = 1
        reward = 1
        done = True
        # self.action_scheme.perform(self, action)
        # self.current_step += 1
        # done = True
        # reward = self.reward_scheme.reward(self)
        # state = self.observer.observe(self)

        return state, reward, done, {}

    def reset(self):
        self.current_step = 0
        self.balance = 10000
        self.position = 0
        self.observer.reset()
        return self.observer.observe(self)

    def render(self, mode='human'):
        pass
