import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# from agents.DDPG_agent import DDPGAgent
# from agents.DQN_agent import DQNAgent
# from agents.SAC_agent import SACAgent
# from agents.ActorCritic_agent import ActorCriticAgent
from agents.MY_DQN_agent import MYDQNAgent
# from agents.LSTM_DQN_agent import LSTMDQNAgent
from agents.AutoEncoder_DDQN_agent import AutoEncoderDDQNAgent

with open("config.json", "r") as file:
    config = json.load(file)


if __name__ == "__main__":
    # agent = ActorCriticAgent(config)
    # agent = DDPGAgent(config)
    # agent = DQNAgent(config)
    # agent = MYDQNAgent(config)
    # agent = LSTMDQNAgent(config)
    agent = AutoEncoderDDQNAgent(config)
    print('After Creating Agent in Main')
    # agent.load_weights()


    # Train
    agent.train(num_episodes=250)

