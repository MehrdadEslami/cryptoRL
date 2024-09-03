import csv
import json
import os
import tensorflow as tf
import numpy as np
from keras.applications import VGG16
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Dropout, ZeroPadding2D, Convolution2D, MaxPooling2D, Input, concatenate
from keras.optimizers import Adam
import random
from env.environment import TradingEnv
import datetime as dt

with open("config.json", "r") as file:
    config = json.load(file)

class DDPGAgent:
    def __init__(self, config):
        self.target_q_values = None
        self.env = TradingEnv(trading_pair='BTC/USD', config=config)
        self.observation_shape = self.env.observation_space.shape
        self.buffer_size = int(config['buffer_size'])
        self.model_path = config['model_weights_path']
        self.memory = []
        self.gamma = 0.9
        self.beta = 0.3
        self.epsilon = 0.2
        # self.epsilon_decay = 0.9
        # self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.batch_size = int(config['batch_size'])
        self.optimizer = Adam(lr=self.learning_rate)
        self.id = 3620398178
        self.env.agent_id = self.id

        self.usdt_balance = 1000
        self.btc_balance = 0

        self.actor_model = self._build_actor_network()
        self.critic_model = self._build_critic_network()
        self.target_actor_model = self._build_actor_network()
        self.target_critic_model = self._build_critic_network()

        # Sync target models
        # self.update_target_networks(tau=1.0)

        # Load weights if they exist
        self.load_weights()

    def _build_modified_vgg16(self):
        vgg = VGG16(weights=None, include_top=False)
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

    def _build_actor_network(self):
        vgg = self._build_modified_vgg16()

        x = vgg.output
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        output = Dense(self.env.action_scheme.action_n, activation='softmax')(x)

        model = Model(inputs=vgg.input, outputs=output)
        model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')
        return model

    def _build_critic_network(self):
        state_input = Input(shape=(self.buffer_size, self.buffer_size, 4))
        action_input = Input(shape=(self.env.action_scheme.action_n,))

        vgg = self._build_modified_vgg16()
        state_features = vgg(state_input)
        state_features = Flatten()(state_features)

        merged = concatenate([state_features, action_input])
        x = Dense(512, activation='relu')(merged)
        x = Dense(256, activation='relu')(x)
        # x = Dropout(0.5)(x)
        output = Dense(1, activation='linear')(x)

        model = Model(inputs=[state_input, action_input], outputs=output)
        model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')
        return model

    def update_target_networks(self, tau=0.1):
        new_weights = []
        for target, source in zip(self.target_actor_model.weights, self.actor_model.weights):
            new_weights.append(target * (1 - tau) + source * tau)
        self.target_actor_model.set_weights(new_weights)

        new_weights = []
        for target, source in zip(self.target_critic_model.weights, self.critic_model.weights):
            new_weights.append(target * (1 - tau) + source * tau)
        self.target_critic_model.set_weights(new_weights)

    def save_weights(self):
        self.actor_model.save_weights(os.path.join(self.model_path, 'ddpg_actor_model_16_1.h5'))
        self.critic_model.save_weights(os.path.join(self.model_path, 'ddpg_critic_model_16_1.h5'))

    def load_weights(self):
        if os.path.exists(os.path.join(self.model_path, 'ddpg_actor_model_16_1.h5')):
            self.actor_model.load_weights(os.path.join(self.model_path, 'ddpg_actor_model_16_1.h5'))
            self.target_actor_model.load_weights(os.path.join(self.model_path, 'ddpg_actor_model_16_1.h5'))
            print("Loaded actor model weights.")
        if os.path.exists(os.path.join(self.model_path, 'ddpg_critic_model_16_1.h5')):
            self.critic_model.load_weights(os.path.join(self.model_path, 'ddpg_critic_model_16_1.h5'))
            self.target_critic_model.load_weights(os.path.join(self.model_path, 'ddpg_critic_model_16_1.h5'))
            print("Loaded critic model weights.")

    def calculate_max_action_reward(self, next_state_i):
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

        # find Best next action Reward MaxR`
        trades = self.env.observer.trades[next_state_i - self.buffer_size ** 2:next_state_i]
        index = next_state_i - self.buffer_size ** 2
        max_reward = {'index': index, 'quantity': 0, 'price': 0, 'side': 0, 'reward': 0}
        for j, trade in trades.iterrows():
            reward = trade['quantity'] * (ohlcv['close'] - trade['price'])

            if reward < 0 and trade['side'] == 'sell':
                reward *= -1
            if reward > max_reward['reward']:
                max_reward['index'] = index
                max_reward['quantity'] = trade['quantity']
                max_reward['price'] = trade['price']
                max_reward['side'] = trade['side']
                max_reward['reward'] = round(reward, 5)
            index += 1
        return max_reward['reward']

    def update_net(self):
        print('Here is update_net')
        if len(self.memory) < self.batch_size:
            return
        print('IN UPDATE SELECTING MINIBATCH')
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([m[0] for m in minibatch])
        actions = np.array([m[1] for m in minibatch])
        rewards = np.array([m[2] for m in minibatch])
        next_states = np.array([m[3] for m in minibatch])
        dones = np.array([m[4] for m in minibatch])
        next_states_i = np.array([m[5] for m in minibatch])

        target_q_values = []
        for i in range(self.batch_size):
            if dones[i]:
                target_q_values.append(rewards[i])
            else:
                reward = rewards[i]
                next_state_i = next_states_i[i]
                max_reward_next_state = self.calculate_max_action_reward(next_state_i)
                next_state = np.expand_dims(next_states[i], axis=0)
                target_action = self.target_actor_model.predict(next_state)[0]
                # print('target action shape', target_action.shape)
                # print('target action', target_action)
                target_q_value = self.target_critic_model.predict([next_state, np.expand_dims(target_action, axis=0)])[
                    0]
                # target_q_values.append(rewards[i] + self.gamma * target_q_value)
                # Hybrid update: mix immediate reward max with long-term reward estimate
                target_q_values.append(
                    rewards[i] + self.gamma * (self.beta * max_reward_next_state + (1 - self.beta) * target_q_value))

                print('r+gamma*target q value is : %f + %f*%f = %f'%(rewards[i], self.gamma, target_q_value, rewards[i] + self.gamma * target_q_value))

        # print('IN UPDATE: \nstates shape', states.shape)
        # print('actions shape', actions.shape)
        # print('target_q_value shape', len(target_q_values))
        self.target_q_values = np.array(target_q_values)
        # print('target_q_values', self.target_q_values)
        self.critic_model.train_on_batch([states, actions], self.target_q_values)

        with tf.GradientTape() as tape:
            actions_pred = self.actor_model(states)
            print('action_pre', actions_pred.shape)
            critic_value = self.critic_model([states, actions_pred])
            actor_loss = -tf.reduce_mean(critic_value)

        actor_grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.optimizer.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))

        self.update_target_networks()

    def train(self, num_episodes=10):
        rewards_log = []
        profits_log = []

        for episode in range(num_episodes):
            print('---------------------------START FOR ITERATION:%d-----------------' %(episode))
            self.usdt_balance = 1000
            self.btc_balance = 1
            state, _ = self.env.reset()
            episode_reward = 0
            episode_profit = 0
            self.state_reward = []
            self.state_action = []
            self.action_time = []
            self.memory=[]
            done = False
            while_index = 1
            while not done:
                print('************************start while:%d   episode: %d ***********************' % (while_index, episode))
                # epsilon Greedy
                if np.random.rand() > self.epsilon:
                    action_probs = self.actor_model.predict(np.expand_dims(state, axis=0))[0]
                else:
                    action_probs = np.zeros(self.env.action_scheme.action_n)
                    action_probs[self.env.action_space.sample()] = 1

                # action[0] = round(action[0], 1)
                next_state, next_state_price, reward, done, self.usdt_balance, self.btc_balance, next_state_i = self.env.step(
                    np.argmax(action_probs), self.usdt_balance, self.btc_balance)
                print('in train while: action_probs',action_probs)
                if done == True:
                    print('FINISH WHILE LOOP')
                    break
                print('reward after one step in Environment is ', reward)
                self.memory.append((state, action_probs, reward, next_state, done, next_state_i))
                state = next_state
                self.update_net()
                self.state_action.append(action_probs)
                self.state_reward.append(reward)
                self.action_time.append(next_state[0, 0, 3])
                episode_reward += reward
                episode_profit += ((self.btc_balance * next_state_price + self.usdt_balance) - 1000)
                while_index += 1
            rewards_log.append(episode_reward//self.env.step_count)
            profits_log.append(episode_profit/self.env.step_count)

            print(f"Episode: {episode + 1}, Reward: {episode_reward}, Profit: {episode_profit}")

            if self.env.step_count % 30 == 0:
                # Save the model weights at the end of training
                print('Save the model weights at the end of training')
                self.save_weights()

        with open('training_state_logs_ddpg_16_1.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Episode', 'while_index', 'action', 'Reward', 'action_time'])
            for i in range(len(self.state_action)):
                writer.writerow([episode, i, self.state_action[i], self.state_reward[i], self.action_time[i]])

        # Save logs to file
        print('# Save logs to file')
        with open('result/training_logs_ddpg_16_1.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Episode', 'Reward', 'Profit'])
            for i in range(num_episodes):
                writer.writerow([i + 1, rewards_log[i], profits_log[i]])


    def test(self):
        self.usdt_balance = 1000
        self.btc_balance = 0
        state, _ = self.env.reset()
        episode_reward = []
        episode_profit = []
        done = False
        while_index = 0
        while not done:
            print(
                '************************start while:%d ***********************' % while_index)

            action_probs = self.actor_model.predict(np.expand_dims(state, axis=0))[0]

            # action[0] = round(action[0], 1)
            next_state, next_state_price, reward, done, self.usdt_balance, self.btc_balance, next_state_i = self.env.step(
                np.argmax(action_probs), self.usdt_balance, self.btc_balance)
            print('in train while: action_probs', action_probs)
            if done == True:
                print('FINISH WHILE LOOP')
                break
            print('reward after one step in Environment is ', reward)
            state = next_state
            episode_reward.append(reward)
            episode_profit.append((self.btc_balance * next_state_price + self.usdt_balance) - 1000)
            while_index += 1

        print(f"Episode: {while_index}, Reward: {episode_reward}, Profit: {episode_profit}")
        print('# Save logs to file')
        with open('result/test_DDPG_4channel_16.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Episode', 'Reward', 'Profit'])
            for i in range(while_index):
                writer.writerow([i, episode_reward[i], episode_profit[i]])
