import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_from_csv(filename, measure_label):
    # Read data from CSV
    data = pd.read_csv(filename)
    print('mean loss', np.mean(data['Q_Loss']))
    print('var loss', np.var(data['Q_Loss']))
    print('mean Reward', np.mean(data['Reward']))
    print('var Reward', np.var(data['Reward']))
    print('mean ED_Loss', np.mean(data['ED_Loss']))
    data1 = []
    for i in range(len(data)):
        print(i)
        print(data.iloc[-i]['Q_Loss'])
        data1.append(data.iloc[-i]['Q_Loss'])
    #     t[99 - i] = data.iloc[i]['Reward']
    # Set dark background
    plt.style.use('fivethirtyeight')
    # plt.style.use('Solarize_Light2')

    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(range(len(data[measure_label])), data[measure_label], color='blue', label=measure_label, marker='o')
    # plt.plot(range(len(data['Q_Loss'])), data['Q_Loss'], color='blue', label='Q value', marker='o')
    # plt.plot(range(len(data1)), data1, label='Q Loss', marker='o')
    # plt.plot(range(len(data['ED_loss'])), data['ED_loss'], color='green', label='EncDec loss', marker='o')

    # Set labels
    plt.xlabel('iteration')
    plt.ylabel(measure_label)
    plt.title('%s (%s)' % (measure_label, model_name))
    # Set grid
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    # Show the plot
    print('Creating plot -----')
    plt.savefig('../result/%s/%s.png' % (model_name, measure_label))
    print('done')

model_name = 'Vanilla ED DQN'
filename = '../result/%s/temp_episode.csv'%model_name
# filename = '../result/%s/EDDQN_32_episode_4.csv'%model_name
plot_from_csv(filename=filename, measure_label='Reward')
plot_from_csv(filename=filename, measure_label='Q_Loss')
plot_from_csv(filename=filename, measure_label='ED_Loss')
# plot_from_csv(filename=filename, measure_label='Target_Loss')
