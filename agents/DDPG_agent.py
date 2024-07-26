import json

import tensorflow as tf
import numpy as np
from keras.applications import VGG16
from keras.layers import Flatten
from keras.optimizers import Adam
from nltk import Model
from tensorflow.python.keras.layers import Dense
from env.environment import TradingEnv

with open("../config.json", "r") as file:
    config = json.load(file)


class DDPGAgent:
    def __init__(self, config):
        self.env = TradingEnv(trading_pair='BTC/USD', config=config)
        self.n_actions = self.env.action_space.n
        self.observation_shape = self.env.observation_space.shape
        self.buffer_size = int(config['buffer_size'])
        self.memory = []
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.batch_size = 64
        # self.model = self.build_model()
        self.optimizer = Adam(lr=self.learning_rate)
        self.id = 3620398178
        self.env.agent_id = self.id

        self.usdt_balance = 1000
        self.btc_balance = 0

    def build_model(self):
        vgg16 = VGG16(include_top=False, input_shape=(self.buffer_size, self.buffer_size, 3))
        x = Flatten()(vgg16.output)
        x = Dense(512, activation='relu')(x)
        actor_output = Dense(self.n_actions, activation='softmax')(x)
        critic_output = Dense(1)(x)
        model = Model(inputs=vgg16.input, outputs=[actor_output, critic_output])
        model.compile(optimizer=Adam(lr=self.learning_rate), loss=['categorical_crossentropy', 'mse'])
        return model

    def predict(self, state):
        print('in predict')
        state = np.expand_dims(state, axis=0)
        action_probs, value = self.model.predict(state)
        return action_probs[0], value[0]

    def train(self):
        print('STAET TRAIN LOOP:', self.env.step_count)
        state, state_price = self.env.reset()
        done = False
        while not done:
            # action_probs, _ = self.predict(state)
            # action = np.argmax(action_probs) if np.random.rand() > self.epsilon else self.env.action_space.sample()
            action = 0.25
            [ next_state, \
              reward, \
              done, \
              self.usdt_balance,
              self.btc_balance ] = self.env.step(action, self.usdt_balance, self.btc_balance)
            self.memory.append((state, action, reward, next_state, done))
            state = next_state
            print('-----------------------------------------')
            # self.replay()

    def update_model_parameters(self, states, actions, advantages, returns):
        with tf.GradientTape() as tape:
            action_probs, values = self.model(states, training=True)
            action_log_probs = tf.math.log(action_probs + 1e-10)
            selected_action_log_probs = tf.reduce_sum(action_log_probs * actions, axis=1)
            actor_loss = -tf.reduce_mean(selected_action_log_probs * advantages)
            critic_loss = tf.reduce_mean(tf.square(returns - values))
            loss = actor_loss + critic_loss

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([m[0] for m in minibatch])
        actions = np.array([m[1] for m in minibatch])
        rewards = np.array([m[2] for m in minibatch])
        next_states = np.array([m[3] for m in minibatch])
        dones = np.array([m[4] for m in minibatch])

        action_probs, values = self.model.predict(states)
        next_action_probs, next_values = self.model.predict(next_states)

        returns = rewards + self.gamma * next_values * (1 - dones)
        advantages = returns - values

        actions_one_hot = tf.keras.utils.to_categorical(actions, num_classes=self.n_actions)
        self.update_model_parameters(states, actions_one_hot, advantages, returns)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
