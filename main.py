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
# from agents.MY_DQN_agent import MYDQNAgent
# from agents.LSTM_DQN_agent import LSTMDQNAgent
from agents.AutoEncoder_DDQN_agent import AutoEncoderDDQNAgent

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

    # plt.show()
    plt.savefig('temp.png')
    print('done')

def plot_loss(loss_list):
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(loss_list)), loss_list, color='blue', label='Loss')
    plt.title('Loss Function of AutoEncoder')
    plt.xlabel('step ')
    plt.ylabel('Loss Value')
    # plt.annotate()
    plt.tight_layout()
    print('Creating Plot.....')
    plt.savefig('result/AutoEncoder/loss_exe_Q_16_5.png')
    print('DONE')



if __name__ == "__main__":
    # agent = ActorCriticAgent(config)
    # agent = DDPGAgent(config)
    # agent = DQNAgent(config)
    # agent = MYDQNAgent(config)
    # agent = LSTMDQNAgent(config)
    agent = AutoEncoderDDQNAgent(config)
    print('After Creating Agent in Main')
    filename = './model_weights/AutoEncoder/New_MY_EDDQN_32_512_2.h5'
    agent.load_weights(filename)


    # Train
    agent.train(num_episodes=20)
    # loss_list = agent.train_encoder_decoder()
    # agent.train(num_episodes=15)

    # plot_training_logs('result/temp.csv')
    # agent.test()

    #test
    # agent.load_weights()
    # agent.test()


#test code

# state, _ = agent.env.reset()
# action_probs = agent.actor_model.predict(np.expand_dims(state, axis=0))[0]
# action = np.random.choice(agent.env.action_space.n, p=action_probs)
# next_state, next_state_price, reward, done, agent.usdt_balance, agent.btc_balance, next_state_image_i = agent.env.step(
#                     action, agent.usdt_balance, agent.btc_balance)
# for i in range(16):
#     agent.memory.append((state, action, reward, next_state, done, next_state_image_i))
#
# import random
# minibatch = random.sample(agent.memory, agent.batch_size)
# state, action, reward, next_state, done, next_state_i = minibatch[9]
# action_index_, max_reward = agent.calculate_max_reward(next_state_i)