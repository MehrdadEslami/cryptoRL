import csv
import os
import tensorflow as tf
import numpy as np
from keras.applications import VGG16
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Dropout, ZeroPadding2D, Convolution2D, MaxPooling2D, Input, concatenate
from keras.optimizers import Adam
import random
from env.environment import TradingEnv

class DQNAgent:
    def __init__(self, config):
        self.env = TradingEnv(trading_pair='BTC/USD', config=config)
        self.observation_shape = self.env.observation_space.shape
        self.buffer_size = int(config['buffer_size']) #THE IMAGE SIZE IS buffer_size*buffer_size
        self.batch_size = int(config['batch_size'])   # THE MINIBATH FOR TRAIN ON BATCH
        self.model_path = config['model_weights_path']
        self.memory = []
        self.gamma = 0.8
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01
        self.learning_rate = 0.01
        self.optimizer = Adam(lr=self.learning_rate)
        self.id = 3620398178
        self.env.agent_id = self.id

        self.usdt_balance = 1000
        self.btc_balance = 0
        # Q-network
        self.model = self._build_q_network()

        # Target network
        self.target_model = self._build_q_network()

        # Synchronize target model weights with the Q-network
        self.update_target_network()

    def _build_modified_vgg16(self):
        vgg = VGG16(weights='imagenet', include_top=False)
        vgg_input_shape = (self.buffer_size, self.buffer_size, 3)

        new_input = Input(shape=vgg_input_shape)
        x = ZeroPadding2D((1, 1))(new_input)
        x = Convolution2D(64, (3, 3), activation='relu', name='block1_conv1_3ch')(x)
        for layer in vgg.layers[2:]:
            x = layer(x)
        model = Model(new_input, x)
        return model

    def _build_q_network(self):
        vgg = self._build_modified_vgg16()

        x = vgg.output
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        output = Dense(self.env.action_scheme.action_n, activation='linear')(x)

        model = Model(inputs=vgg.input, outputs=output)
        model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')
        return model

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = np.array(state)
        state = np.expand_dims(state, axis=0)
        print('in act function with state shape', state.shape)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.env.action_scheme.action_n)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        states = []
        # actions = np.array([m[1] for m in minibatch])
        # rewards = np.array([m[2] for m in minibatch])
        # next_states = np.array([m[3] for m in minibatch])
        # dones = np.array([m[4] for m in minibatch])
        target_q_values = []
        for state, action, reward, next_state, done in minibatch:
            states.append(state)
            if done:
                target_q_values.append(reward)
            if not done:
                next_q_values = self.target_model.predict(np.expand_dims(next_state, axis=0))[0]
                target_q_values.append(reward + self.gamma * np.amax(next_q_values))


            # target_f = self.model.predict(np.expand_dims(state, axis=0))
            # target_f[0][action] = target
        target_q_values = np.array(target_q_values)
        states = np.array(states)
        print('states shape', states.shape)
        print('target shape', target_q_values.shape)
        self.model.train_on_batch(states, target_q_values)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, num_episodes=1):
        rewards_log = []
        profits_log = []
        for episode in range(num_episodes):
            print('---------------------------START FOR ITERATION:%d-----------------' %(episode))

            self.usdt_balance = 1000
            self.btc_balance = 0
            state, _ = self.env.reset()
            episode_reward = 0
            episode_profit = 0
            self.state_reward = []
            self.state_action = []
            self.action_time = []
            self.memory = []
            done = False

            while not done:
                print('*******************************start while')
                action = self.act(state)
                print('action is', action)

                next_state, next_state_price, reward, done, self.usdt_balance, self.btc_balance = self.env.step(
                    action, self.usdt_balance,
                    self.btc_balance)
                if done:
                    break
                print('reward after one step in Environment is ', reward)
                self.memory.append((state, action, reward, next_state, done))
                state = next_state
                self.replay()
                self.state_action.append(action)
                self.state_reward.append(reward)
                # self.action_time.append(next_state[0, 0, 3])
                episode_reward += reward
                episode_profit = (self.btc_balance * next_state_price + self.usdt_balance) - 1000
            rewards_log.append(episode_reward)
            profits_log.append(episode_profit)
            self.update_target_network()
            print(f"Episode: {episode + 1}, Reward: {episode_reward}, Profit: {episode_profit}")

        # Save the model weights at the end of training
        print('Save the model weights at the end of training')
        self.save_weights()

        # Save logs to file
        print('# Save logs to file')
        with open('training_logs_dqn_64_1.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Episode', 'Reward', 'Profit'])
            for i in range(num_episodes):
                writer.writerow([i + 1, rewards_log[i], profits_log[i]])

        # with open('training_state_logs_dqn_64_1.csv', 'w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(['action', 'Reward', 'action_time'])
        #     for i in range(len(self.state_action)):
        #         # writer.writerow([self.state_action[i], self.state_reward[i], self.action_time[i]])
        #         writer.writerow([self.state_action[i], self.state_reward[i]])

    def save_weights(self):
        self.model.save_weights(os.path.join(self.model_path, 'dqn_model_64_1.h5'))

    def load_weights(self):
        if os.path.exists(os.path.join(self.model_path, 'dqn_model_64_1.h5')):
            self.model.load_weights(os.path.join(self.model_path, 'dqn_model_64_1.h5'))
            self.target_model.load_weights(os.path.join(self.model_path, 'dqn_model_64_1.h5'))
            print("Loaded model weights.")

