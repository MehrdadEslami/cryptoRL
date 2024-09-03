import csv
import os
import random
from keras.applications import VGG16
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Dropout, ZeroPadding2D, Convolution2D, MaxPooling2D, Input, concatenate
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import layers
from env.environment import TradingEnv
import numpy as np


class SACAgent:
    def __init__(self, config):
        self.env = TradingEnv(trading_pair='BTC/USD', config=config)
        self.gamma = 0.99
        self.alpha = 0.2  # Entropy coefficient
        self.learning_rate = 3e-4
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.batch_size = int(config['batch_size'])
        self.buffer_size = int(config['buffer_size'])

        self.actor_model = self.build_actor()
        self.critic1_model = self.build_critic()
        self.critic2_model = self.build_critic()
        self.target_critic1_model = self.build_critic()
        self.target_critic2_model = self.build_critic()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.memory = []

    def save_weights(self):
        self.actor_model.save_weights(os.path.join(self.model_path, 'SAC_actor_32_1.h5'))
        self.critic1_model.save_weights(os.path.join(self.model_path, 'SAC_critic_32_1.h5'))

    def load_weights(self):
        if os.path.exists(os.path.join(self.model_path, 'actor_model_32_3.h5')):
            self.actor_model.load_weights(os.path.join(self.model_path, 'actor_model_32_3.h5'))
            self.target_actor_model.load_weights(os.path.join(self.model_path, 'actor_model_32_3.h5'))
            print("Loaded actor model weights.")
        if os.path.exists(os.path.join(self.model_path, 'critic_model_32_3.h5')):
            self.critic_model.load_weights(os.path.join(self.model_path, 'critic_model_32_3.h5'))
            self.target_critic_model.load_weights(os.path.join(self.model_path, 'critic_model_32_3.h5'))
            print("Loaded critic model weights.")

    def _build_modified_vgg16(self):
        vgg = VGG16(weights='imagenet', include_top=False)

        state_input = layers.Input(shape=self.env.observation_space.shape)
        x = ZeroPadding2D((1, 1))(state_input)
        x = Convolution2D(64, (3, 3), activation='relu', name='block1_conv1_3ch')(x)
        for layer in vgg.layers[2:]:
            if isinstance(layer, MaxPooling2D) and x.shape[1] < 2:
                # Skip pooling layers when input spatial dimensions are too small
                continue
            x = layer(x)
        model = Model(state_input, x)
        return model

    def build_actor(self):
        vgg = VGG16(include_top=False)
        state_input = layers.Input(shape=self.env.observation_space.shape)

        x = ZeroPadding2D((1, 1))(state_input)
        x = Convolution2D(64, (3, 3), activation='relu', name='block1_conv1_3ch')(x)
        for layer in vgg.layers[2:]:
            x = layer(x)

        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        mean = layers.Dense(self.env.action_space.n, activation="linear")(x)
        log_std = layers.Dense(self.env.action_space.n, activation="linear")(x)

        model = tf.keras.models.Model(inputs=state_input, outputs=[mean, log_std])
        model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')
        return model

    def build_critic(self):
        state_input = layers.Input(shape=self.env.observation_space.shape)
        action_input = layers.Input(shape=(self.env.action_space.n,))

        vgg = self._build_modified_vgg16()
        state_features = vgg(state_input)
        state_features = Flatten()(state_features)

        merged = concatenate([state_features, action_input])
        x = Dense(128, activation='relu')(merged)
        x = Dense(64, activation='relu')(x)
        # x = Dropout(0.5)(x)
        output = Dense(1, activation='linear')(x)

        model = Model(inputs=[state_input, action_input], outputs=output)
        model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')
        return model

    def act(self, state):
        state = np.array(state)
        state = np.expand_dims(state, axis=0)
        print('in act function with state shape', state.shape)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.env.action_scheme.action_n)
        action_probs = self.actor_model.predict(state)
        return np.argmax(action_probs[0])

    def update(self):

        if len(self.memory) < self.batch_size:
            return
        print('Here is update_net')
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = tf.convert_to_tensor(states)
        next_states = tf.convert_to_tensor(next_states)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.convert_to_tensor(rewards)
        dones = tf.convert_to_tensor(dones)

        # Critic loss
        print('IN UPDATE SELECTING MINIBATCH')
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
                next_state = np.expand_dims(next_states[i], axis=0)
                target_action = self.target_actor_model.predict(next_state)[0]
                target_q_value = self.target_critic_model.predict([next_state, np.expand_dims(target_action, axis=0)])[
                    0]
                target_q_values.append(rewards[i] + self.gamma * target_q_value)
                print('r+gamma*target q value is : %f + %f*%f = %f' % (
                rewards[i], self.gamma, target_q_value, rewards[i] + self.gamma * target_q_value))

        # target_q1_value = self.target_critic1_model([next_states, actions])
        target_q1_value = self.target_critic1_model([next_states, np.expand_dims(actions, axis=0)])[
            0]
        target_q2_value = self.target_critic2_model([next_states, actions])
        min_q_value = tf.minimum(target_q1_value, target_q2_value)

        target_value = rewards + self.gamma * min_q_value * (1 - dones)
        q1_value = self.critic1_model([states, actions])
        q2_value = self.critic2_model([states, actions])

        critic1_loss = tf.reduce_mean(tf.square(q1_value - target_value))
        critic2_loss = tf.reduce_mean(tf.square(q2_value - target_value))

        # Actor loss
        mean, log_std = self.actor_model(states)
        std = tf.exp(log_std)
        new_action = mean + std * tf.random.normal(shape=mean.shape)
        log_probs = -0.5 * ((new_action - mean) / (std + 1e-6)) ** 2 - log_std

        q1_value = self.critic1_model([states, new_action])
        q2_value = self.critic2_model([states, new_action])
        min_q_value = tf.minimum(q1_value, q2_value)

        actor_loss = tf.reduce_mean(self.alpha * log_probs - min_q_value)

        # Apply gradients
        critic1_grads = tf.GradientTape().gradient(critic1_loss, self.critic1_model.trainable_variables)
        critic2_grads = tf.GradientTape().gradient(critic2_loss, self.critic2_model.trainable_variables)
        self.optimizer.apply_gradients(zip(critic1_grads, self.critic1_model.trainable_variables))
        self.optimizer.apply_gradients(zip(critic2_grads, self.critic2_model.trainable_variables))

        actor_grads = tf.GradientTape().gradient(actor_loss, self.actor_model.trainable_variables)
        self.optimizer.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))

    def train(self, num_episodes=1):
        rewards_log = []
        profits_log = []
        for episode in range(num_episodes):
            print('-----------------------------START FOR ITERATION:%d-------------------' %(episode))

            self.usdt_balance = 1000
            self.btc_balance = 0
            state, _ = self.env.reset()
            episode_reward = 0
            episode_profit = 0
            self.memory = []
            done = False

            while not done:
                print('***start while***')
                action = self.act(state)
                print('action is', action)

                next_state, next_state_price, reward, done, self.usdt_balance, self.btc_balance,_ = self.env.step(
                    action, self.usdt_balance,
                    self.btc_balance)
                if done:
                    break
                print('reward after one step in Environment is ', reward)
                self.memory.append((state, action, reward, next_state, done))
                state = next_state
                self.update()
                episode_reward += reward
                episode_profit = (self.btc_balance * next_state_price + self.usdt_balance) - 1000
            rewards_log.append(episode_reward)
            profits_log.append(episode_profit)
            print(f"Episode: {episode + 1}, Reward: {episode_reward}, Profit: {episode_profit}")

        # Save the model weights at the end of training
        print('Save the model weights at the end of training')
        self.save_weights()

        # Save logs to file
        print('# Save logs to file')
        with open('training_logs_SAC_32_1.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Episode', 'Reward', 'Profit'])
            for i in range(num_episodes):
                writer.writerow([i + 1, rewards_log[i], profits_log[i]])
