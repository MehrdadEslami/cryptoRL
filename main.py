import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from agents.DDPG_agent import DDPGAgent


with open("config.json", "r") as file:
    config = json.load(file)


def plot_training_logs(filename):
    episodes = []
    rewards = []
    profits = []

    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # skip header
        for row in reader:
            episodes.append(int(row[0]))
            rewards.append(float(row[1]))
            profits.append(float(row[2]))

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(episodes, rewards, label='Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward vs Episode')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(episodes, profits, label='Profit')
    plt.xlabel('Episode')
    plt.ylabel('Profit')
    plt.title('Profit vs Episode')
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    agent = DDPGAgent(config)
    print('After Creating Agent in Main')

    # Usage
    agent = DDPGAgent(config)
    agent.train(num_episodes=10)
    plot_training_logs('training_logs.csv')
