from keras.layers import Dense, Flatten, Input, concatenate, ZeroPadding2D, Convolution2D, MaxPooling2D
from keras.applications import VGG16
from keras.models import Model
from keras.optimizers import Adam
from env.environment import TradingEnv
import tensorflow as tf
import datetime as dt
import numpy as np
import random
import os


class ActorCriticAgent:
    def __init__(self, config):

        self.env = TradingEnv(trading_pair='BTC/USD', config=config)
        self.observation_shape = self.env.observation_space.shape
        self.buffer_size = int(config['buffer_size'])  # THE IMAGE SIZE IS buffer_size*buffer_size
        self.batch_size = int(config['batch_size'])  # THE MINIBATH FOR TRAIN ON BATCH
        self.model_path = config['model_weights_path']
        self.memory = []
        self.gamma = 0.7
        self.beta = 0.3
        self.epsilon = 0.2
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.optimizer = Adam(lr=self.learning_rate)

        self.usdt_balance = 1000
        self.btc_balance = 0
        # Build and load pretrained actor and critic networks
        self.actor_model = self._build_actor_network()
        self.target_actor_model = self._build_actor_network()
        self.critic_model = self._build_critic_network()
        self.target_critic_model = self._build_critic_network()

        # Load weights from the pretrained model
        # self.actor_model.load_weights(os.path.join(self.model_path, 'dqn_4channel_16_1.h5'), by_name=True)
        # self.critic_model.load_weights(os.path.join(self.model_path, 'dqn_4channel_16_1.h5'), by_name=True)
        # self.target_critic_model.load_weights(os.path.join(self.model_path, 'dqn_4channel_16_1.h5'), by_name=True)
        #
        # # Freeze all layers in VGG16 except fully connected layers
        # for layer in self.actor_model.layers[:-3]:  # Exclude the last 3 layers (fully connected)
        #     layer.trainable = False
        # for layer in self.critic_model.layers[:-3]:
        #     layer.trainable = False
        # for layer in self.target_critic_model.layers[:-3]:
        #     layer.trainable = False

    def _build_modified_vgg16(self):
        vgg = VGG16(weights='imagenet', include_top=False)
        vgg_input_shape = (self.buffer_size, self.buffer_size, 4)

        new_input = Input(shape=vgg_input_shape)
        x = ZeroPadding2D((1, 1))(new_input)
        x = Convolution2D(64, (3, 3), activation='relu')(x)
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
        output = Dense(self.env.action_space.n, activation='softmax')(x)  # Use softmax for probabilities

        model = Model(inputs=vgg.input, outputs=output)
        model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')
        return model

    def _build_critic_network(self):
        state_input = Input(shape=(self.buffer_size, self.buffer_size, 4))
        action_input = Input(shape=(self.env.action_space.n,))

        vgg = self._build_modified_vgg16()
        state_features = vgg(state_input)
        print('state_feature shape', state_features.shape)
        state_features = Flatten()(state_features)

        # Concatenate state and action
        x = concatenate([state_features, action_input])
        print('x shape after Concatenate', x.shape)

        x = Dense(512, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        output = Dense(1, activation='linear')(x)  # Output a single Q-value

        model = Model(inputs=[state_input, action_input], outputs=output)
        model.compile(optimizer=Adam(lr=0.001), loss='mse')
        return model

    def calculate_max_action_reward(self, next_state_i):
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
        # if the price does not change much do nothing
        # norm_quantity = (max_reward['quantity'] - np.min(trades['quantity'])) / (
        #             np.max(trades['quantity']) - np.min(trades['quantity']))
        # if 0.3 < norm_quantity < 0.50 and max_reward['side'] == 'buy':
        #     action_index = 3
        # elif 0.3 < norm_quantity < 0.50 and max_reward['side'] == 'sell':
        #     action_index = 1
        # elif norm_quantity > 0.50 and max_reward['side'] == 'buy':
        #     action_index = 4
        # elif norm_quantity > 0.50 and max_reward['side'] == 'sell':
        #     action_index = 0
        # else:
        #     action_index = 2
        #
        # # create action prob vector
        # n = self.env.action_scheme.action_n
        # action_prob = np.zeros(n)
        # for l in range(len(action_prob)):
        #     if l != action_index:
        #         action_prob[l] = (1/n)-(1/n*.6)
        # action_prob[action_index] = 1 - np.sum(action_prob)
        # if np.abs((ohlcv['close'] - trade['price'])/ohlcv['close']*100) < 0.5:
        #     return 0
            # action_prob[2] += action_prob[action_index]/2
            # action_prob[action_index] = 1 + action_prob[action_index] - np.sum(action_prob)

        # return action_prob, max_reward['reward']
        return max_reward['reward']

    def update_networks(self):
        print('Here is update_net')
        if len(self.memory) < self.batch_size:
            return

        print('IN UPDATE SELECTING MINIBATCH')
        minibatch = random.sample(self.memory, self.batch_size)

        states = []
        actions = []
        target_q_values = []

        for state, action, reward, next_state, done, next_state_i in minibatch:
            states.append(state)
            actions.append(action)
            if done:
                # No future reward if done
                target_q_values.append(reward)
            else:
                # Compute max immediate reward (max R_t) from current trades
                # target_action, max_reward_next_state = self.calculate_max_action_reward(next_state_i)  # Assuming next_states[i] represents trade rewards
                max_reward_next_state = self.calculate_max_action_reward(next_state_i)  # Assuming next_states[i] represents trade rewards

                # Get the predicted action from the target actor network for the next state
                next_state = np.expand_dims(next_state, axis=0)
                target_action = self.target_actor_model.predict(next_state)[0]

                # Critic's estimate of the future Q-value
                target_q_value = self.target_critic_model.predict([next_state, np.expand_dims(target_action, axis=1)])[0]

                # Hybrid update: mix immediate reward max with long-term reward estimate
                target_q_values.append(
                    reward + self.gamma * (self.beta * max_reward_next_state + (1 - self.beta) * target_q_value))

                # print(
                #     'r + gamma * (beta * max_reward + (1 - beta) * target_q_value) = %f + %f * (%f * %f + %f * %f) = %f' %
                #     (reward, self.gamma, self.beta, max_reward_next_state, 1 - self.beta, target_q_value,
                #      reward + self.gamma * (self.beta * max_reward_next_state + (1 - self.beta) * target_q_value)))

        self.target_q_values = np.array(target_q_values)
        print('target_q_values', self.target_q_values)

        # Critic update with hybrid target Q-values
        self.critic_model.train_on_batch([states, actions], self.target_q_values)

        # Actor update via policy gradient
        with tf.GradientTape() as tape:
            actions_pred = self.actor_model(states)
            print('action_pred', actions_pred.shape)
            critic_value = self.critic_model([states, actions_pred])
            actor_loss = -tf.reduce_mean(critic_value)

        # Compute and apply gradients
        actor_grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.optimizer.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))

        # Update target networks
        self.update_target_networks()

    def train(self, num_episodes=1):
        rewards_log = []
        profits_log = []
        for episode in range(num_episodes):
            print('---------------------------START FOR ITERATION:%d-----------------' % (episode))

            self.usdt_balance = 1000
            self.btc_balance = 0
            state, _ = self.env.reset()
            episode_reward = 0
            episode_profit = 0
            episode_reward = 0
            self.memory = []
            done = False

            while not done:
                print('*******************************start while')

                # Predict action probabilities from the actor network
                action_probs = self.actor_model.predict(np.expand_dims(state, axis=0))[0]
                action = np.random.choice(self.env.action_space.n, p=action_probs)

                # Take the action in the environment
                next_state, next_state_price, reward, done, self.usdt_balance, self.btc_balance, next_state_image_i = self.env.step(
                    action, self.usdt_balance, self.btc_balance)

                # Store the transition (state, action, reward, next_state, done) in memory
                self.memory.append((state, action, reward, next_state, done, next_state_image_i))

                # Update networks at each step
                # self.update_networks()

                if done:
                    break

                print('reward after one step in Environment is ', reward)
                self.memory.append((state, action, reward, next_state, done, next_state_image_i))
                self.update_networks()
                if self.env.step_count % 30 == 0:
                    # Save the model weights at the end of training
                    print('Save the model weights at the end of training')
                    self.save_weights()
                state = next_state
                episode_reward += reward
                episode_profit += ((self.btc_balance * next_state_price + self.usdt_balance) - 1000)
            rewards_log.append(episode_reward/self.env.step_count)
            profits_log.append(episode_profit/self.env.step_count)

            print(f"Episode: {episode + 1}, Reward: {episode_reward}, Profit: {episode_profit}")



        # Save logs to file
        print('# Save logs to file')
        with open('training_logs_MY_dqn_4channel_16_1.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Episode', 'Reward', 'Profit'])
            for i in range(num_episodes):
                writer.writerow([i + 1, rewards_log[i], profits_log[i]])

            print(f"Episode {episode + 1}, Reward: {episode_reward}")
