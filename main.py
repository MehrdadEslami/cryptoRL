import json
from agents.DDPG_agent import DDPGAgent
from env.environment import TradingEnv


with open("config.json", "r") as file:
    config = json.load(file)

if __name__ == "__main__":
    agent = DDPGAgent(config)
    print('After Creating Agent in Main')

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
