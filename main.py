import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from agents.DDPG_agent import DDPGAgent


with open("config.json", "r") as file:
    config = json.load(file)

if __name__ == "__main__":
    agent = DDPGAgent(config)
    print('After Creating Agent in Main')

    # Example data
    time_vector = np.arange(10)  # Example time vector
    price_vector = np.random.randn(10).cumsum() + 100  # Example price vector
    action_vector = [0, 0.25, 0, -0.5, 1, -0.25, 0, 0.5, -1, 0]  # Example action vector
    action_time_vector = np.arange(10)  # Example action time vector

    # Create figure and axes
    fig, ax = plt.subplots()

    # Dark background
    plt.style.use('dark_background')

    # Plot price vs time
    ax.plot(time_vector, price_vector, label='Price')

    # Highlight actions
    for i in range(len(action_vector)):
        if action_vector[i] > 0:
            ax.scatter(action_time_vector[i], price_vector[i], color='green', marker='^', s=100)  # Buy action
        elif action_vector[i] < 0:
            ax.scatter(action_time_vector[i], price_vector[i], color='red', marker='v', s=100)  # Sell action

    # Labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.set_title('Price over Time with Buy/Sell Actions')
    ax.legend()

    # Show plot
    plt.show()
    # agent = DDPGAgent(env.observation_space.shape, env.action_space.n)

    # episodes = 1000
    # for episode in range(episodes):
    #     state = env.reset()
    #     done = False
    #     while not done:
    #         action = agent.act(state)
    #         next_state, reward, done, _ = env.step(action)
    #         agent.remember(state, action, reward, next_state, done)
    #         state = next_state
    #
    #     agent.replay()
