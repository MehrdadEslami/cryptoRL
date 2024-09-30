import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_from_csv(filename):
    # Read data from CSV
    data = pd.read_csv(filename)


    #     t[99 - i] = data.iloc[i]['Reward']
    # Set dark background
    plt.style.use('fivethirtyeight')

    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(range(len(data['Q_loss'])), data['Q_loss'], color='blue', label='Q value', marker='o')
    # plt.plot(range(len(data['ED_Loss'])), data['ED_Loss'], color='green', label='EncDec loss', marker='o')

    # Set labels
    plt.xlabel('iteration')
    plt.ylabel('loss EnoderDecoder ')
    plt.title('Loss value of Q')
    # Set grid
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    # Show the plot
    print('Creating plot -----')
    plt.savefig('../result/AutoEncoder/plots/QLoss.png')
    print('done')


plot_from_csv('../result/AutoEncoder/EDDQN_while_32_1.csv')
# plot_from_csv('../result/AutoEncoder/EDDQN_32_episode_1.csv')