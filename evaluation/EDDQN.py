import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_from_csv(filename):
    # Read data from CSV
    data = pd.read_csv(filename)
    print('mean', np.mean(data['Q_Loss']))

    #     t[99 - i] = data.iloc[i]['Reward']
    # Set dark background
    plt.style.use('fivethirtyeight')


    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(range(len(data['Reward'])), data['Reward'], color='blue', label='Q value', marker='o')
    # plt.plot(range(len(data['Q_Loss'])), data['Q_Loss'], color='blue', label='Q value', marker='o')
    # plt.plot(range(len(data.iloc[660:]['ED_loss'])), data.iloc[660:]['ED_loss'], color='green', label='EncDec loss', marker='o')

    # Set labels
    plt.xlabel('iteration')
    plt.ylabel('reward Values ')
    plt.title('reward of Model')
    # Set grid
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    # Show the plot
    print('Creating plot -----')
    plt.savefig('../result/AutoEncoder/base_model_reward.png')
    print('done')


# plot_from_csv('../result/AutoEncoder/EDDQN_while_32_1.csv')
plot_from_csv('../result/AutoEncoder/EDDQN_32_episode_1.csv')
