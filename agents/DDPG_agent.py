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


class DDPGAgent:
    def __init__(self, config):
        self.env = TradingEnv(trading_pair='BTC/USD', config=config)
        self.n_actions = 1#self.env.action_space.n
        self.observation_shape = self.env.observation_space.shape
        self.buffer_size = int(config['buffer_size'])
        self.model_path = config['model_weights_path']
        self.memory = []
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.batch_size = 10
        self.optimizer = Adam(lr=self.learning_rate)
        self.id = 3620398178
        self.env.agent_id = self.id

        self.usdt_balance = 1000
        self.btc_balance = 0

        self.actor_model = self._build_actor_network()
        self.critic_model = self._build_critic_network()

        # Load weights if they exist
        # self.load_weights()

    def VGG_16(self, weights_path=None):
        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering="th"))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering="th"))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering="th"))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering="th"))


        return model

    def _build_modified_vgg16(self):
        vgg = VGG16(weights='imagenet', include_top=False)
        vgg_input_shape = (self.buffer_size, self.buffer_size, 4)

        new_input = Input(shape=vgg_input_shape)
        x = ZeroPadding2D((1, 1))(new_input)
        x = Convolution2D(64, (3, 3), activation='relu', name='block1_conv1_4ch')(x)
        for layer in vgg.layers[2:]:
            x = layer(x)
        model = Model(new_input, x)
        return model

    def _build_actor_network(self):
        vgg = self._build_modified_vgg16()
        # vgg = self.VGG_16()

        x = vgg.output
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        output = Dense(self.env.action_scheme.action_n, activation='tanh')(x)

        model = Model(inputs=vgg.input, outputs=output)
        model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')
        return model

    def _build_critic_network(self):
        state_input = Input(shape=(self.buffer_size, self.buffer_size, 4))
        action_input = Input(shape=(self.n_actions,))

        vgg = self._build_modified_vgg16()
        state_features = vgg(state_input)
        state_features = Flatten()(state_features)

        merged = concatenate([state_features, action_input])
        x = Dense(256, activation='relu')(merged)
        x = Dropout(0.5)(x)
        output = Dense(1, activation='linear')(x)

        model = Model(inputs=[state_input, action_input], outputs=output)
        model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')
        return model

    def save_weights(self):
        self.actor_model.save_weights(os.path.join(self.model_path, 'actor_model_128.h5'))
        self.critic_model.save_weights(os.path.join(self.model_path, 'critic_model_128.h5'))

    def load_weights(self):
        if os.path.exists(os.path.join(self.model_path, 'actor_model.h5')):
            self.actor_model.load_weights(os.path.join(self.model_path, 'actor_model.h5'))
            print("Loaded actor model weights.")
        if os.path.exists(os.path.join(self.model_path, 'critic_model.h5')):
            self.critic_model.load_weights(os.path.join(self.model_path, 'critic_model.h5'))
            print("Loaded critic model weights.")

    def update_net(self):
        print('Here is update_net')
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([m[0] for m in minibatch])
        actions = np.array([m[1] for m in minibatch])
        rewards = np.array([m[2] for m in minibatch])
        next_states = np.array([m[3] for m in minibatch])
        dones = np.array([m[4] for m in minibatch])

        target_q_values = []
        for i in range(self.batch_size):
            if dones[i]:
                target_q_values.append(rewards[i])
            else:
                next_state = next_states[i]
                max_q_value = self.env.calculate_max_q(next_state)
                target_q_values.append(rewards[i] + self.gamma * max_q_value)

        target_q_values = np.array(target_q_values)
        self.critic_model.train_on_batch([states, actions], target_q_values)

        with tf.GradientTape() as tape:
            actions_pred = self.actor_model(states)
            critic_value = self.critic_model([states, actions_pred])
            actor_loss = -tf.reduce_mean(critic_value)

        actor_grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.optimizer.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def predict(self, state):
        state = np.expand_dims(state, axis=0)
        action_probs = self.actor_model.predict(state)
        return action_probs[0]

    def train(self, num_episodes=10):
        rewards_log = []
        profits_log = []

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_profit = 0
            self.state_reward = []
            self.state_action = []
            self.action_time = []
            done = False

            while not done:
                action = self.predict(state) if np.random.rand() > self.epsilon else self.env.action_space.sample()
                next_state, next_state_price, reward, done, self.usdt_balance, self.btc_balance = self.env.step(action[0], self.usdt_balance,
                                                                                              self.btc_balance)
                if len(next_state) == 1:
                    break
                print('reward after one step in Environment is ', reward)
                self.memory.append((state, action, reward, next_state, done))
                state = next_state
                self.update_net()
                self.state_action.append(action)
                self.state_reward.append(reward)
                self.action_time.append( next_state[0, 0, 3] )
                episode_reward += reward
                episode_profit = self.usdt_balance - (self.btc_balance * next_state_price + self.usdt_balance)
                print('------------------------------------------------------------------')
            rewards_log.append(episode_reward)
            profits_log.append(episode_profit)

            print(f"Episode: {episode + 1}, Reward: {episode_reward}, Profit: {episode_profit}")

        # Save the model weights at the end of training
        print('Save the model weights at the end of training')
        self.save_weights()

        # Save logs to file
        print('# Save logs to file')
        with open('training_logs.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Episode', 'Reward', 'Profit'])
            for i in range(num_episodes):
                writer.writerow([i + 1, rewards_log[i], profits_log[i]])

        with open('training_state_logs.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['action', 'Reward', 'action_time'])
            for i in range(len(self.state_action)):
                writer.writerow([self.state_action[i], self.state_reward[i], self.action_time[i]])