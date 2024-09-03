import csv
import os
import numpy as np
from keras.applications import VGG16
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Dropout, ZeroPadding2D, Convolution2D, MaxPooling2D, Input, LSTM, Reshape
from keras.optimizers import Adam
import random
from env.environment import TradingEnv
import datetime as dt

class LSTMDQNAgent:
    def __init__(self, config):
        self.env = TradingEnv(trading_pair='BTC/USD', config=config)
        self.observation_shape = self.env.observation_space.shape
        self.buffer_size = int(config['buffer_size']) #THE IMAGE SIZE IS buffer_size*buffer_size
        self.batch_size = int(config['batch_size'])   # THE MINIBATH FOR TRAIN ON BATCH
        self.model_path = config['model_weights_path']
        self.memory = []
        self.gamma = 0.7
        self.beta = 0.3
        self.epsilon = 0.5
        self.epsilon_decay = 0.9
        self.epsilon_min = 0.1
        self.learning_rate = 0.01
        self.optimizer = Adam(lr=self.learning_rate)
        self.id = 3620398178
        self.env.agent_id = self.id

        self.usdt_balance = 1000
        self.btc_balance = 0
        self.test_action = []
        self.test_action_time = []
        # Q-network
        self.model = self._build_q_network()
        # Target network
        self.target_model = self._build_q_network()

        # Synchronize target model weights with the Q-network
        # self.load_weights()
        self.update_target_network()

    def _build_modified_vgg16(self):
        vgg = VGG16(weights='imagenet', include_top=False)
        vgg_input_shape = (self.buffer_size, self.buffer_size, 4)

        new_input = Input(shape=vgg_input_shape)
        x = ZeroPadding2D((1, 1))(new_input)
        x = Convolution2D(64, (3, 3), activation='relu', name='block1_conv1_4ch')(x)
        for layer in vgg.layers[2:]:
            if isinstance(layer, MaxPooling2D) and x.shape[1] < 2:
                # Skip pooling layers when input spatial dimensions are too small
                continue
            x = layer(x)
        model = Model(new_input, x)
        return model

    def _build_q_network(self):
        vgg = self._build_modified_vgg16()

        x = vgg.output
        print(x.shape)
        x = Flatten()(x)
        print('len x', x.shape)
        timesteps = self.env.observer.slice_size
        features = x.shape[-1] // timesteps

        # Reshape to 3D input for LSTM
        x = Reshape((timesteps, features))(x)
        print("Shape after Reshape:", x.shape)
        x = LSTM(128, return_sequences=True)(x)
        x = LSTM(128, return_sequences=True)(x)
        x = LSTM(128)(x)

        # Continue with Dense layers
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
            print('in e greedy random ')
            return random.randrange(self.env.action_scheme.action_n)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def calculate_max_reward(self, next_state_i):
        best_trade_index = None
        if next_state_i + self.buffer_size ** 2 < len(self.env.observer.trades):
            last_trade = self.env.observer.trades.iloc[next_state_i + self.buffer_size ** 2]
        else:
            last_trade = self.env.observer.trades.iloc[-1]

        trade_time = last_trade['time']
        if trade_time.minute < 15:
            start = trade_time - dt.timedelta(minutes=30)
        else:
            start = trade_time
        end = start + dt.timedelta(minutes=30)
        # change format to Flux suitable format
        start = start.strftime('%Y-%m-%dT%H:%M:%SZ')
        end = end.strftime('%Y-%m-%dT%H:%M:%SZ')
        ohlcv = self.env.observer.query_ohlcv_by_time(start, end)

        #find Best next action Reward MaxR`
        trades = self.env.observer.trades[next_state_i - self.buffer_size ** 2:next_state_i]
        index = next_state_i - self.buffer_size ** 2
        max_reward = {'index': index, 'quantity': 0, 'price': 0, 'side': 0, 'reward': 0}
        for j, trade in trades.iterrows():
            reward = round(trade['quantity']*( (ohlcv['close'] - trade['price'])/trade['price']*100), 3)
            if reward < 0 and trade['side'] == 'sell':
                reward *= -1
            if reward > max_reward['reward']:
                max_reward['index'] = index
                max_reward['quantity'] = trade['quantity']
                max_reward['price'] = trade['price']
                max_reward['side'] = trade['side']
                max_reward['reward'] = round(reward, 5)
            index += 1
        # print('max reward in current state', max_reward)
        return max_reward['reward']

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        states = []
        target_q_values = []
        for state, action, reward, next_state, done, next_state_i in minibatch:
            states.append(state)
            if done:
                target_q_values.append(reward)
            if not done:
                max_reward_next_state = self.calculate_max_reward(
                    next_state_i)  # Assuming next_states[i] represents trade rewards

                next_q_values = self.target_model.predict(np.expand_dims(next_state, axis=0))[0]
                target_q_values.append(
                    reward + self.gamma * (self.beta * max_reward_next_state + (1 - self.beta) * np.max(next_q_values)))
                print('action:%d, reward: %f, max_reward:%f' % (action, reward, max_reward_next_state))
        target_q_values = np.array(target_q_values)
        states = np.array(states)
        print('states shape', states.shape)
        print('target shape', target_q_values.shape)
        loss = self.model.train_on_batch(states, target_q_values)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss

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
            self.loss = []
            self.memory = []
            done = False

            while not done:
                print('*******************************start while iteration:%d' % (episode))
                action = self.act(state)
                print('action is', action)

                next_state, next_state_price, reward, done, self.usdt_balance, self.btc_balance, next_state_i = self.env.step(
                    action, self.usdt_balance, self.btc_balance)
                if done:
                    break
                print('reward after one step in Environment is ', reward)
                self.memory.append((state, action, reward, next_state, done, next_state_i))
                state = next_state
                loss = self.replay()
                self.loss.append(loss)
                self.state_action.append(action)
                self.state_reward.append(reward)
                self.action_time.append(next_state[0, 0, 3])
                episode_reward += reward
                episode_profit += (self.btc_balance * next_state_price + self.usdt_balance) - 1000
                if self.env.step_count % 30 == 0:
                    self.update_target_network()
                    # Save the model weights at the end of training
                    print('Save the model weights at the end of training')
                    self.save_weights()
            rewards_log.append(episode_reward / self.env.step_count)
            profits_log.append(episode_profit / self.env.step_count)
            print(f"Episode: {episode + 1}, Reward: {episode_reward}, Profit: {episode_profit}")

            #save each episode all while loop reward and ...
            with open('result/training_state_logs_lstm_dqn_32_1.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['episode', 'action', 'Reward', 'action_time', 'loss'])
                for i in range(len(self.state_action)):
                    # writer.writerow([self.state_action[i], self.state_reward[i], self.action_time[i]])
                    writer.writerow([episode, self.state_action[i], self.state_reward[i], self.action_time[i], self.loss[i]])

        # Save logs to file
        print('# Save logs to file')
        with open('result/training_logs_lstm_dqn_4channel_32_1.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Episode', 'Reward', 'Profit'])
            for i in range(num_episodes):
                writer.writerow([i + 1, rewards_log[i], profits_log[i]])

    def test(self):
        self.usdt_balance = 1000
        self.btc_balance = 0
        state, _ = self.env.reset()
        episode_reward = 0

        done = False
        while not done:
            print('*******************************start while')
            action = self.act(state)
            print('action is', action)
            self.test_action.append(action)
            if self.env.observer.next_image_i < len(self.env.observer.trades):
                self.test_action_time.append(self.env.observer.trades['time'].iloc[self.env.observer.next_image_i])
            next_state, next_state_price, reward, done, self.usdt_balance, self.btc_balance = self.env.step(
                action, self.usdt_balance,
                self.btc_balance)

            if done:
                break
            print('reward after one step in Environment is ', reward)

            episode_reward += reward

            print(f"Reward: {episode_reward}")
        print('test action len:',len(self.test_action))
        print('test action time len:',len(self.test_action_time))

    def save_weights(self):
        self.model.save_weights(os.path.join(self.model_path, 'lstm_dqn_4channel_32_1.h5'))

    def load_weights(self):
        if os.path.exists(os.path.join(self.model_path, 'lstm_dqn_4channel_32_1.h5')):
            self.model.load_weights(os.path.join(self.model_path, 'lstm_dqn_4channel_32_1.h5'))
            self.target_model.load_weights(os.path.join(self.model_path, 'lstm_dqn_4channel_32_1.h5'))
            print("Loaded model weights.")

