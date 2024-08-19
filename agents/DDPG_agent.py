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
        self.observation_shape = self.env.observation_space.shape
        self.buffer_size = int(config['buffer_size'])
        self.model_path = config['model_weights_path']
        self.memory = []
        self.gamma = 0.8
        self.epsilon = 1.0
        self.epsilon_decay = 0.9
        self.epsilon_min = 0.01
        self.learning_rate = 0.01
        self.batch_size = 6
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
        self.update_target_networks(tau=1.0)

        # Load weights if they exist
        self.load_weights()

    def _build_modified_vgg16(self):
        vgg = VGG16(weights='imagenet', include_top=False)
        vgg_input_shape = (self.buffer_size, self.buffer_size, 4)

        new_input = Input(shape=vgg_input_shape)
        # x = ZeroPadding2D((1, 1))(new_input)
        x = Convolution2D(64, (3, 3), activation='relu', name='block1_conv1_4ch')(x)
        for layer in vgg.layers[2:]:
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
        self.actor_model.save_weights(os.path.join(self.model_path, 'actor_model_32_3.h5'))
        self.critic_model.save_weights(os.path.join(self.model_path, 'critic_model_32_3.h5'))

    def load_weights(self):
        if os.path.exists(os.path.join(self.model_path, 'actor_model_32_3.h5')):
            self.actor_model.load_weights(os.path.join(self.model_path, 'actor_model_32_3.h5'))
            self.target_actor_model.load_weights(os.path.join(self.model_path, 'actor_model_32_3.h5'))
            print("Loaded actor model weights.")
        if os.path.exists(os.path.join(self.model_path, 'critic_model_32_3.h5')):
            self.critic_model.load_weights(os.path.join(self.model_path, 'critic_model_32_3.h5'))
            self.target_critic_model.load_weights(os.path.join(self.model_path, 'critic_model_32_3.h5'))
            print("Loaded critic model weights.")

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
                print('r+gamma*target q value is : %f + %f*%f = %f'%(rewards[i], self.gamma, target_q_value, rewards[i] + self.gamma * target_q_value))

        print('IN UPDATE: \nstates shape',states.shape)
        print('actions shape', actions.shape)
        print('target_q_value shape', len(target_q_values))
        self.target_q_values = np.array(target_q_values)
        print('target_q_values', self.target_q_values)
        self.critic_model.train_on_batch([states, actions], self.target_q_values)

        with tf.GradientTape() as tape:
            actions_pred = self.actor_model(states)
            print('action_pre', actions_pred.shape)
            critic_value = self.critic_model([states, actions_pred])
            actor_loss = -tf.reduce_mean(critic_value)

        actor_grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.optimizer.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))

        self.update_target_networks()

    def predict(self, state):
        # state = np.expand_dims(state, axis=0)
        print('HELLO IN PREDICT *********WERWER*****************')
        state = np.array(state[0])
        state = np.expand_dims(state, axis=0)
        action_probs = self.actor_model.predict(state)
        return action_probs

    def train(self, num_episodes=10):
        rewards_log = []
        profits_log = []

        for episode in range(num_episodes):
            print('---------------------------START FOR ITERATION:%d-----------------' %(episode))

            state, _ = self.env.reset()
            episode_reward = 0
            episode_profit = 0
            self.state_reward = []
            self.state_action = []
            self.action_time = []
            done = False

            while not done:
                print('*******************************start while')
                # epsilon Greedy
                if np.random.rand() > self.epsilon:
                    action_probs = self.actor_model.predict(state)[0]
                else:
                    action_probs = np.zeros(self.env.action_scheme.action_n)
                    action_probs[self.env.action_space.sample()] = 1

                # action[0] = round(action[0], 1)
                next_state, next_state_price, reward, done, self.usdt_balance, self.btc_balance = self.env.step(
                    np.argmax(action_probs), self.usdt_balance,
                    self.btc_balance)
                print('in train while: action_probs',action_probs)
                if done == True:
                    print('FINISH WHILE LOOP')
                    break
                print('reward after one step in Environment is ', reward)
                self.memory.append((state, action_probs, reward, next_state, done))
                state = next_state
                self.update_net()
                self.state_action.append(action_probs)
                self.state_reward.append(reward)
                self.action_time.append(next_state[0, 0, 3])
                episode_reward += reward
                episode_profit = self.usdt_balance - (self.btc_balance * next_state_price + self.usdt_balance)
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