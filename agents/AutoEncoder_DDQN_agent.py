import csv
import os
import pandas as pd

import numpy as np
from keras.applications import VGG16
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Dropout, ZeroPadding2D, Convolution2D, MaxPooling2D, Input, concatenate, \
    Reshape
from keras.optimizers import Adam
from tensorflow.keras import backend as K
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

        self.current_action = None
        self.memory = []
        self.gamma = 0.7
        self.beta = 0.2
        self.epsilon = 0.9
        self.epsilon_decay = 0.02
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.optimizer = Adam(lr=self.learning_rate)
        self.id = 3620398178
        self.env.agent_id = self.id

        self.usdt_balance = 1000
        self.btc_balance = 0
        self.Qloss = []
        self.Recloss = []

        # Q-network and encoder-decoder network
        self.encoder_decoder = self._build_encoder_decoder()
        # self.q_network_pretrain = self._build_q_network()
        self.load_weights()
        self.q_network = self.build_q_network()
        self.model = self.build_model()
        # Target network
        self.target_q_network = self.build_q_network()

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

    # def _build_q_network(self):
    #     vgg = self._build_modified_vgg16()
    #
    #     x = vgg.output
    #     x = Flatten()(x)
    #     x = Dense(512, activation='relu')(x)
    #     x = Dense(256, activation='relu')(x)
    #     output = Dense(5, activation='linear')(x)
    #
    #     model = Model(inputs=vgg.input, outputs=output)
    #     model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')
    #     return model

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

        # target = (self.gamma * (self.beta * self.max_reward_next_state + (1 - self.beta) * np.max(self.target_values, axis=1)))-np.max(self.q_values, axis=1)
        # print('target:', target.shape)
        # print('target:', target)
        # q_value_loss = K.mean(K.square(target - self.q_values))
        return reconstruction_loss #+ float(np.min(target))  #self.gamma * (target - np.max(self.q_values, axis=1))

    def custom_mse_loss(self, y_true, y_pred):
        return K.mean(K.square(y_pred - y_true), axis=-1)

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
        if self.current_action == action:
            print('in if action: 1')
            return np.array(1)
        print('out if action', action)
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

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

        # find Best next action Reward MaxR`
        trades = self.env.observer.trades[next_state_i - self.buffer_size ** 2:next_state_i]
        index = next_state_i - self.buffer_size ** 2
        max_reward = {'index': index, 'quantity': 0, 'price': 0, 'side': 0, 'reward': 0}
        for j, trade in trades.iterrows():
            reward = trade['quantity'] * ((ohlcv['close'] - trade['price']) / trade['price'] * 100)
            if reward < 0 and trade['side'] == 'sell':
                reward *= -1
            if reward > max_reward['reward']:
                max_reward['index'] = index
                max_reward['quantity'] = trade['quantity']
                max_reward['price'] = trade['price']
                max_reward['side'] = trade['side']
                max_reward['reward'] = round(reward, 5)
            index += 1
        # print('max reward in current state',max_reward)
        return max_reward['reward']

    def train_encoder_decoder(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        states_list = []
        for state, action, reward, next_state, done, next_state_i in minibatch:
            states_list.append(state)

        state_batch = np.array(states_list)
        encoded_output = self.encoder(state_batch)  # This processes y_true through the encoder
        self.q_values = self.q_network(encoded_output)
        self.target_values = self.target_q_network(encoded_output)
        print('GO to train encoder decoder train on batch')
        ED_loss = self.encoder_decoder.train_on_batch(state_batch, state_batch)

        return ED_loss

    def train_Q_Network(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        states_list = []
        rewards_list = []
        encoded_states = []
        target_q_values = []
        for state, action, reward, next_state, done, next_state_i in minibatch:
            states_list.append(state)
            rewards_list.append(reward)

        state_batch = np.array(states_list)
        encoded_output = self.encoder(state_batch)  # This processes y_true through the encoder
        self.q_values = self.q_network(encoded_output)
        self.target_values = self.target_q_network(encoded_output)
        print('GO to train Q_network')
        target_q_values = rewards_list + self.gamma * (self.beta * self.max_reward_next_state + (1 - self.beta) * np.max(self.target_values))

        target_q_values = np.array(target_q_values)
        print('target_q_values shape', target_q_values.shape)
        print('encoded_output shape', encoded_output.shape)
        loss = self.q_network.train_on_batch(encoded_output, target_q_values)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss

    def train(self, num_episodes=1):
        rewards_log = []
        q_loss_log = []
        ed_loss_log = []

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
                self.max_reward_next_state = self.calculate_max_reward(next_state_i)
                # Train encoder-decoder and qNetwork

                # ed_loss = self.train_encoder_decoder()
                edq_loss = self.train_Q_Network()
                ed_loss = edq_loss
                if ed_loss != None:
                    ED_loss.append(ed_loss)
                if edq_loss != None:
                    Q_loss.append(edq_loss)

                if self.env.step_count % 30 == 0:
                    self.update_target_network()
                    self.save_weights()

                print(
                    f"Episode: {episode}, Action: {action}, Reward: {reward}, Q_loss: {edq_loss}, Re_loss: {ed_loss}")

            q_loss_log.append(np.mean(Q_loss))
            ed_loss_log.append(np.mean(ED_loss))
            rewards_log.append(episode_reward / self.env.step_count)
            print(f"Episode: {episode + 1}, Reward: {episode_reward}, {edq_loss}, ED_loss: {edq_loss}")

            filename = 'result/AutoEncoder/EDDQN_while_32_1.csv'
            with open(filename, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['episode', 'Reward', 'Q_loss', 'ED_Loss'])
                for i in range(len(Q_loss)):
                    writer.writerow([episode, states_reward[i],
                                     Q_loss[i], ED_loss[i]])


        # Save logs to file
        print('# Save logs to file')
        with open('result/AutoEncoder/EDDQN_32_episode_1.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Episode', 'Reward', 'Q_Loss', 'ED_loss'])
            for i in range(num_episodes):
                writer.writerow([i + 1, np.mean(rewards_log), np.mean(q_loss_log), np.mean(ed_loss_log)])

    # Remaining methods for training, testing, saving/loading weights remain unchanged.
    def save_weights(self):
        self.encoder_decoder.save_weights(os.path.join(self.model_path, 'AutoEncoder/EncoderDec_32_512_1.h5'))
        self.q_network.save_weights(os.path.join(self.model_path, 'AutoEncoder/QNetwork_32_512_1.h5'))
        self.model.save_weights(os.path.join(self.model_path, 'AutoEncoder/EDDQN_32_512_1.h5'))
    def load_weights(self):
        if os.path.exists(os.path.join(self.model_path, 'AutoEncoder/encoderDecoder_Q_16_1.h5')):
            print('FOUND')
            self.encoder_decoder.load_weights(os.path.join(self.model_path, 'AutoEncoder/EncoderDec_32_512_1.h5'))
            # filename = '~/workspace/CryptoAnalysis/model_weights/AutoEncoder/EDDQN_32_512_1.h5'
            # self.model.load_weights(filename)
        else:
            print('NOT FOUND')
