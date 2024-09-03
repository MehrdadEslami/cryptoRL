import tensorflow as tf
from tensorflow.keras import layers
from env.environment import TradingEnv


class PPO3Agent:
    def __init__(self, config):
        self.env = TradingEnv(trading_pair='BTC/USD', config=config)
        self.buffer_size = int(config['buffer_size'])
        self.gamma = 0.99
        self.epsilon = 0.2  # Clipping epsilon for PPO
        self.learning_rate = 3e-4
        self.batch_size = 64

        self.actor_model = self.build_actor()
        self.critic_model = self.build_critic()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.memory = []

    def build_actor(self):
        state_input = layers.Input(shape=self.env.observation_space.shape)
        x = layers.Dense(64, activation="relu")(state_input)
        x = layers.Dense(64, activation="relu")(x)
        output = layers.Dense(self.env.action_space.n, activation="softmax")(x)

        model = tf.keras.models.Model(inputs=state_input, outputs=output)
        return model

    def build_critic(self):
        state_input = layers.Input(shape=self.env.observation_space.shape)
        x = layers.Dense(64, activation="relu")(state_input)
        x = layers.Dense(64, activation="relu")(x)
        output = layers.Dense(1)(x)

        model = tf.keras.models.Model(inputs=state_input, outputs=output)
        return model

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = tf.convert_to_tensor(states)
        next_states = tf.convert_to_tensor(next_states)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.convert_to_tensor(rewards)
        dones = tf.convert_to_tensor(dones)

        with tf.GradientTape() as tape:
            advantages = rewards + self.gamma * self.critic_model(next_states) * (1 - dones) - self.critic_model(states)
            action_probs = self.actor_model(states)
            entropy = -tf.reduce_mean(action_probs * tf.math.log(action_probs + 1e-8))
            log_probs = tf.math.log(tf.reduce_sum(action_probs * actions, axis=1) + 1e-8)

            old_log_probs = tf.math.log(tf.reduce_sum(self.actor_model(states) * actions, axis=1) + 1e-8)
            ratio = tf.exp(log_probs - old_log_probs)
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon)
            actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages) + 0.01 * entropy)

        actor_grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.optimizer.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))
