import csv
import math
import os
import pandas as pd

import numpy as np
from keras.applications import VGG16
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Dropout, ZeroPadding2D, Convolution2D, MaxPooling2D, Input, concatenate, \
    Reshape
from keras.optimizers import Adam
from tensorflow.keras import backend as K
import tensorflow as tf
import random
from env.environment import TradingEnv
import datetime as dt


class AutoEncoderDDQNAgent:
    def __init__(self, config):

        self.env = TradingEnv(trading_pair='BTC/USD', config=config)
        self.observation_shape = self.env.observation_space.shape
        self.buffer_size = int(config['buffer_size'])
        self.batch_size = int(config['batch_size'])
        self.encoded_size = int(config['encoded_size'])
        self.ESRI = int(config['ESRI'])  # Encoded state Reshape Index
        self.model_path = config['model_weights_path']

        self.q_values = None
        self.target_values = None
        self.encoded_state = None
        self.max_reward_next_state = None
        self.mean_reward_states = None
        self.max_mean_reward = 0
        self.mean_max_reward = 0

        self.current_action = None
        self.memory = []
        self.gamma = 0.9
        self.lambda_ = 0.9
        self.epsilon = 0.9
        self.epsilon_decay = 0.02
        self.epsilon_min = 0.01
        self.learning_rate = 0.01
        self.optimizer = Adam(lr=self.learning_rate)
        self.id = 3620398178
        self.env.agent_id = self.id

        self.usdt_balance = 1000
        self.btc_balance = 0
        self.Qloss = []
        self.Recloss = []
        self.best_loss = float('inf')

        # Q-network and encoder-decoder network
        self.encoder_decoder = self._build_encoder_decoder()
        # self.q_network_pretrain = self._build_q_network()
        self.q_network = self.build_q_network()
        self.model = self.build_model()
        # Target network
        self.target_q_network = self.build_q_network()
        self.load_weights(os.path.join(self.model_path, 'AutoEncoder/New_MY_EDDQN_32_512_3.h5'))

    def _build_encoder_decoder(self):
        input_shape = (self.buffer_size, self.buffer_size, 4)
        inputs = Input(shape=input_shape)

        # Encoder
        input_layer = Flatten()(inputs)
        hidden_encoder_1 = Dense(256, activation='relu')(input_layer)
        encoded = Dense(self.encoded_size, activation='relu')(hidden_encoder_1)  # Encoded features

        # Decoder
        latent_inputs = Input(shape=(self.encoded_size,))
        hidden_decoder_1 = Dense(256, activation='relu')(latent_inputs)
        output_layer = Dense(self.buffer_size * self.buffer_size * 4, activation='sigmoid')(hidden_decoder_1)
        decoded = Reshape((self.buffer_size, self.buffer_size, 4))(output_layer)

        # Encoder model (connects inputs to encoded output)
        # Ensure the depth is 4 for Conv2D layers
        self.encoder = Model(inputs=inputs, outputs=encoded)

        # Decoder model (from latent space to original space)
        self.decoder = Model(inputs=latent_inputs, outputs=decoded)

        # Full Encoder-Decoder model (combines both)
        model = Model(inputs=inputs, outputs=self.decoder(self.encoder(inputs)))
        model.compile(optimizer=Adam(lr=self.learning_rate), loss=self.custom_loss)
        return model

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

    def build_q_network(self):
        # Freeze layers in the pretrained Q-network except the last few layers
        # for layer in self.q_network_pretrain.layers[1:-4]:
        #     layer.trainable = False
        vgg = self._build_modified_vgg16()

        # Create a new input layer for the encoded input from the encoder
        encoded_input = Input(shape=self.encoder.output.shape[1:])  # Exclude batch size in the shape
        # Transform the encoded output to match the required input shape of the Q-network
        a, b, c = int(self.ESRI), int(self.ESRI / 2), int(self.ESRI / 4)
        transformed_encoded_output = Dense(a * b * c, activation='relu')(encoded_input)
        reshaped_encoded_output = Reshape((a, b, c))(transformed_encoded_output)
        x = reshaped_encoded_output
        # Pass through the layers of the pretrained Q-network
        for layer in vgg.layers[1:]:
            if isinstance(layer, MaxPooling2D) and x.shape[1] < 4:
                # Skip pooling layers when input spatial dimensions are too small
                continue
            if isinstance(layer, Flatten):
                break
            x = layer(x)

        # Continue building the Q-network
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        output = Dense(self.env.action_scheme.action_n, activation='linear')(x)

        # Build the Q-network model
        model = Model(inputs=encoded_input, outputs=output)
        model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')

        return model

    def build_model(self):
        encoded_output = self.encoder.output
        print('encoded_output shape:', encoded_output.shape)

        # Connect the encoder and Q-network
        q_values = self.q_network(encoded_output)

        # Build the combined model
        combined_model = Model(inputs=self.encoder.input, outputs=q_values)

        # Compile the model with Adam optimizer and MSE loss for Q-learning
        combined_model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')
        print(combined_model.summary())
        return combined_model

    def custom_loss(self, y_true, y_pred):
        print('in custome loss')
        reconstruction_loss = K.mean(K.square(y_pred - y_true), axis=-1)

        # target = self.max_reward_next_state + self.gamma * np.max(self.target_values, axis=1)
        # target = self.max_mean_reward
        target = self.mean_max_reward
        # target = (self.gamma * (self.beta * self.max_reward_next_state + (1 - self.beta) * np.max(self.target_values, axis=1)))-np.max(self.q_values, axis=1)
        # print('target:', target.shape)
        # print('target:', target)
        q_value_loss = K.mean(K.square(target - np.max(self.q_values, axis=1)))
        # temp = tf.convert_to_tensor(q_value_loss)
        return (1 - self.lambda_) * reconstruction_loss + self.lambda_ * q_value_loss

    def custom_mse_loss(self, y_true, y_pred):
        return K.mean(K.square(y_pred - y_true), axis=-1)

    def simulated_annealing(self):
        # Small random change to the parameters
        new_gamma = self.gamma + random.uniform(-0.01, 0.01)
        new_lambda = self.lambda_ + random.uniform(-0.01, 0.01)

        # Ensure new values stay within valid ranges
        new_gamma = max(0, min(1, new_gamma))
        new_lambda = max(0, new_lambda)

        return new_gamma, new_lambda

    def acceptance_probability(self, old_loss, new_loss, temperature):
        # Accept if new loss is lower, or accept probabilistically if worse
        if new_loss < old_loss:
            return 1.0
        return math.exp((old_loss - new_loss) / temperature)

    def update_target_network(self):
        self.target_q_network.set_weights(self.q_network.get_weights())

    def decision(self, state):
        # Encode the state and fetch to q_network to return q_values

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.env.action_scheme.action_n)

        state = np.array(state)
        state = np.expand_dims(state, axis=0)
        print('state shape befor model', state.shape)
        q_values = self.model.predict(state)
        print('q_values shape after model predict', q_values.shape)
        action = np.argmax(q_values[0])
        return action
        if self.current_action == action:
            print('in if action: 1')
            return np.array(1)
        print('out if action', action)
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def calculate_max_mean_reward(self, next_state_i):
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

        # find Best next action Reward MaxR`
        trades = self.env.observer.trades[next_state_i - self.buffer_size ** 2:next_state_i]
        index = next_state_i - self.buffer_size ** 2
        max_reward = {'index': index, 'quantity': 0, 'price': 0, 'side': 0, 'reward': 0}
        sum_reward = 0
        for j, trade in trades.iterrows():
            quantity_norm = (trade['quantity'] - self.env.observer.min_all_qty) \
                            / (self.env.observer.max_all_qty - self.env.observer.min_all_qty)
            reward = quantity_norm * ((ohlcv['close'] - trade['price']) / trade['price'] * 100)
            if reward < 0 and trade['side'] == 'sell':
                reward *= -1
            sum_reward += reward
            if reward > max_reward['reward']:
                max_reward['index'] = index
                max_reward['quantity'] = trade['quantity']
                max_reward['price'] = trade['price']
                max_reward['side'] = trade['side']
                max_reward['reward'] = round(reward, 5)
            index += 1
        # print('max reward in current state',max_reward)
        # return sum_reward / (self.buffer_size * self.buffer_size)  # max_reward['reward']
        return max_reward['reward'], sum_reward/(self.buffer_size * self.buffer_size)

    def train_encoder_decoder(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        states_list = []
        next_states_list = []
        for state, action, reward, next_state, done, next_state_i in minibatch:
            states_list.append(state)
            next_states_list.append(next_state)

        states_batch = np.array(states_list)
        next_states_batch = np.array(next_states_list)
        encoded_output = self.encoder(states_batch)  # This processes y_true through the encoder
        self.q_values = self.q_network(encoded_output)
        next_encoded_output = self.encoder(next_states_batch)
        self.target_values = self.target_q_network(encoded_output)
        print('GO to train encoder decoder train on batch')
        ED_loss = self.encoder_decoder.train_on_batch(states_batch, states_batch)

        return ED_loss

    def train_Q_Network(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        states_list = []
        rewards_list = []
        actions_list = []
        next_states_list = []
        target_q_values = []
        for state, action, reward, next_state, done, next_state_i in minibatch:
            states_list.append(state)
            next_states_list.append(next_state)
            rewards_list.append(reward)
            actions_list.append(action)

        state_batch = np.array(states_list)
        next_state_batch = np.array(next_states_list)
        encoded_output = self.encoder(state_batch)  # This processes y_true through the encoder
        self.q_values = self.q_network(encoded_output)
        next_encoded_output = self.encoder(next_state_batch)
        self.target_values = self.target_q_network(next_encoded_output)
        print('GO to train Q_network')
        # target_q_values = rewards_list + self.gamma * (self.beta * self.max_reward_next_state + (1 - self.beta) * np.max(self.target_values))
        target_q_values = rewards_list + np.multiply(self.gamma, [float(self.target_values[w][actions_list[w]]) for w in range(self.batch_size)])

        target_q_values = np.array(target_q_values)
        print('target_q_values shape', target_q_values.shape)
        print('encoded_output shape', encoded_output.shape)
        loss = self.q_network.train_on_batch(encoded_output, target_q_values)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss

    def train_Target_Q_Network(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        states_list = []
        rewards_list = []
        for state, action, reward, next_state, done, next_state_i in minibatch:
            states_list.append(state)
            rewards_list.append(reward)

        state_batch = np.array(states_list)
        encoded_output = self.encoder(state_batch)  # This processes y_true through the encoder
        self.target_values = self.target_q_network(encoded_output)
        print('GO to train Target_Q_network')
        target_q_values = np.multiply(1-self.gamma, rewards_list) + self.gamma * self.mean_max_reward

        target_q_values = np.array(target_q_values)
        loss = self.q_network.train_on_batch(encoded_output, target_q_values)

        return loss

    def train(self, num_episodes=1, T_start=100, T_end=1, anneal_rate=0.9):
        rewards_log = []
        q_loss_log = []
        ed_loss_log = []
        best_gamma = self.gamma
        best_lambda_ = self.lambda_
        temperature = T_start
        for episode in range(num_episodes):
            print('---------------------------START FOR ITERATION:%d-----------------' % (episode))

            self.usdt_balance = 1000
            self.btc_balance = 0
            state, _ = self.env.reset()
            episode_reward = 0
            ED_loss = []
            Q_loss = []
            states_reward = []
            self.memory = []
            done = False


            while not done:
                print('*******************************start while iteration:%d' % (episode))
                action = self.decision(state)  # Select action based on encoded state
                self.current_action = action
                next_state, next_state_price, reward, done, self.usdt_balance, self.btc_balance, next_state_i = self.env.step(
                    action, self.usdt_balance, self.btc_balance)
                if done or int(next_state_price) == 0:
                    break

                episode_reward += reward
                states_reward.append(reward)
                self.memory.append((state, action, reward, next_state, done, next_state_i))
                state = next_state
                self.max_reward_next_state, self.mean_reward_states = self.calculate_max_mean_reward(next_state_i)
                self.mean_max_reward = self.lambda_ * self.max_reward_next_state + (1 - self.lambda_) * self.mean_max_reward
                if self.max_mean_reward < self.mean_reward_states:
                    self.max_mean_reward = self.mean_reward_states

                # Train encoder-decoder and qNetwork

                ed_loss = self.train_encoder_decoder()
                edq_loss = self.train_Q_Network()
                # ed_loss = edq_loss
                if ed_loss != None:
                    ED_loss.append(ed_loss)
                if edq_loss != None:
                    Q_loss.append(edq_loss)

                if self.env.step_count % 10 == 0:
                    self.train_Target_Q_Network()
                    self.save_weights()

                print(
                    f"Episode: {episode}, Action: {action}, Reward: {reward}, Q_loss: {edq_loss}, Re_loss: {ed_loss}")

            # Simulated annealing step
            new_gamma, new_lambda = self.simulated_annealing()
            self.gamma, self.lambda_ = new_gamma, new_lambda

            # Evaluate new parameter set
            new_loss = np.mean(Q_loss) + np.mean(ED_loss)
            prob = self.acceptance_probability(self.best_loss, new_loss, temperature)

            if random.random() < prob:
                best_gamma = new_gamma
                best_lambda = new_lambda
                self.best_loss = new_loss
            else:
                # Revert to best values if the new set is not accepted
                self.gamma, self.lambda_ = best_gamma, best_lambda

            # Decrease temperature
            temperature = max(T_end, temperature * anneal_rate)


            q_loss_log.append(np.mean(Q_loss))
            ed_loss_log.append(np.mean(ED_loss))
            rewards_log.append(episode_reward)
            print(
                f"Episode: {episode + 1}, Reward: {episode_reward}, Q_loss: {edq_loss}, ED_loss: {ed_loss}, Best Loss: {self.best_loss}")

            print(f"Best parameters - Gamma: {best_gamma}, Lambda: {best_lambda}")

            filename = 'result/AutoEncoder/EDDQN_while_32_1.csv'
            file_exit = False
            if os.path.exists(filename):
                file_exit = True
            with open(filename, 'a', newline='') as file:
                writer = csv.writer(file)
                if not file_exit:
                    writer.writerow(['episode', 'Reward', 'Q_loss', 'ED_Loss'])
                for i in range(len(Q_loss)):
                    writer.writerow([episode, states_reward[i],
                                     Q_loss[i], ED_loss[i]])


        # Save logs to file
        print('# Save logs to file')
        file_exit = False
        if os.path.exists('result/AutoEncoder/EDDQN_32_episode_1.csv'):
            file_exit = True
        with open('result/AutoEncoder/EDDQN_32_episode_1.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            if not file_exit:
                writer.writerow(['Episode', 'Reward', 'Q_Loss', 'ED_loss'])
            for i in range(num_episodes):
                writer.writerow([i + 1, rewards_log[i], q_loss_log[i], ed_loss_log[i]])

    def test(self):
        self.usdt_balance = 1000
        self.btc_balance = 0
        state, _ = self.env.reset()
        episode_reward = 0
        self.memory = []
        done = False

        while not done:
            print('*******************************start while')

            action = self.decision(state)

            print('action is', action)
            if self.env.observer.next_image_i < len(self.env.observer.trades):
                action_time = self.env.observer.trades['time'].iloc[self.env.observer.next_image_i]

            next_state, next_state_price, reward, done, self.usdt_balance, self.btc_balance, _ = self.env.step(
                action, self.usdt_balance,
                self.btc_balance)
            self.memory.append((action, action_time, next_state_price))
            print('action is', action)
            print('action time is', action_time)
            if done:
                break
            state = next_state
            print('reward after one step in Environment is ', reward)

            episode_reward += reward

            print(f"Reward: {episode_reward}")

        # Save logs to file
        print('# Save logs to file')
        d = dt.datetime.fromisoformat(self.env.observer.last_trade_time)
        filename_date = '%s-%s-%sT%s:%s' % (d.year, d.month, d.day, d.hour, d.minute)
        with open('../result/AutoEncoder/test_action_%s_%s_%s.csv' % (self.env.observer.slice_size, self.env.symbol, filename_date), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['i', 'action', 'action_time', 'action_price'])
            for i in range(len(self.memory)):
                writer.writerow((i, self.memory[i][0], self.memory[i][1], self.memory[i][2]))

    # Remaining methods for training, testing, saving/loading weights remain unchanged.
    def save_weights(self):
        # self.encoder_decoder.save_weights(os.path.join(self.model_path, 'AutoEncoder/EncoderDec_32_512_1.h5'))
        # self.q_network.save_weights(os.path.join(self.model_path, 'AutoEncoder/QNetwork_32_512_1.h5'))
        self.target_q_network.save_weights(os.path.join(self.model_path, 'AutoEncoder/TargetQNetwork_32_512_3.h5'))
        self.model.save_weights(os.path.join(self.model_path, 'AutoEncoder/New_MY_EDDQN_32_512_3.h5'))

    def load_weights(self, filename):
        if os.path.exists(filename):
            print('FOUND')
            # self.encoder_decoder.load_weights(os.path.join(self.model_path, 'AutoEncoder/EncoderDec_32_512_2.h5'))
            # self.q_network.load_weights(os.path.join(self.model_path, 'AutoEncoder/QNetwork_32_512_1.h5'))
            self.target_q_network.load_weights(os.path.join(self.model_path, 'AutoEncoder/TargetQNetwork_32_512_3.h5'))
            self.model.load_weights(os.path.join(self.model_path, 'AutoEncoder/New_MY_EDDQN_32_512_3.h5'))
            # self.model.load_weights(filename)
        else:
            print('NOT FOUND')
