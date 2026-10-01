import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class DQN:
    def __init__(self, input_dim=32, hidden1=64, hidden2=128, output_dim=5, lr=0.001, epsilon=0.7, gamma=0.9):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.epsilon = epsilon
        self.gamma = gamma
        self.replay_buffer = ReplayBuffer(capacity=100000)

        self.model = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(hidden1, activation='tanh', kernel_initializer=tf.keras.initializers.RandomUniform(minval=0.0, maxval=0.1)),
            layers.Dense(hidden2, activation='tanh', kernel_initializer=tf.keras.initializers.RandomUniform(minval=0.0, maxval=0.1)),
            layers.Dense(output_dim, kernel_initializer=tf.keras.initializers.RandomUniform(minval=0.0, maxval=0.1))
        ])

        self.optimizer = optimizers.Adam(learning_rate=lr)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    def choose_action(self, state):
        state = np.expand_dims(state, axis=0).astype(np.float32)
        q_values = self.model(state)
        return np.argmax(q_values.numpy()[0])

    def train(self, state, action, target):
        state = np.expand_dims(state, axis=0).astype(np.float32)
        with tf.GradientTape() as tape:
            q_values = self.model(state)
            q_value = q_values[0, action]
            # Ensure target and q_value are tensors with shape (1,) for compatibility
            target_tensor = tf.convert_to_tensor([target], dtype=tf.float32)
            q_value_tensor = tf.convert_to_tensor([q_value], dtype=tf.float32)
            loss = self.loss_fn(target_tensor, q_value_tensor)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def train_batch(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = states.astype(np.float32)
        next_states = next_states.astype(np.float32)

        q_next = self.model.predict(next_states, verbose=0)
        targets = rewards + (1 - dones) * self.gamma * np.max(q_next, axis=1)

        for i in range(batch_size):
            self.train(states[i], actions[i], targets[i])

    def update(self, state, action, reward, next_state, done, batch_size=32):
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.train_batch(batch_size)

def preprocess_input(data):
    return (np.array(data, dtype=np.float32) - 128.0) / 128.0
