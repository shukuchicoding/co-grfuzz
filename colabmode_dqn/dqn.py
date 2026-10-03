import math
import random
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

import dqn_core


class ReplayBuffer:
    def __init__(self, capacity=0):
        self.buffer = deque(maxlen=max(int(capacity), 1))
        self.enabled = int(capacity) > 0

    def push(self, state, action, reward, next_state, done):
        if not self.enabled:
            return
        self.buffer.append(
            (
                np.asarray(state, dtype=np.float32).copy(),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32).copy(),
                bool(done),
            )
        )

    def sample(self, batch_size):
        if not self.enabled or len(self.buffer) < int(batch_size):
            return None
        batch = random.sample(self.buffer, int(batch_size))
        state, action, reward, next_state, done = map(np.asarray, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer) if self.enabled else 0


class DQN:
    """DQN policy for the Deep Reinforcement Fuzzing baseline.

    Replay is disabled by default because the original study reported no gain
    from replay memory in the configuration used for subsequent measurements.
    Epsilon follows a monotonic high-to-low schedule and is configurable.
    """

    def __init__(
        self,
        input_dim=32,
        hidden1=64,
        hidden2=128,
        output_dim=5,
        lr=0.001,
        epsilon=0.7,
        epsilon_min=0.1,
        epsilon_decay=0.999,
        gamma=0.9,
        replay_capacity=0,
    ):
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        if self.input_dim < 1 or self.output_dim < 1:
            raise ValueError("input_dim and output_dim must be >= 1")

        self.epsilon = min(max(float(epsilon), 0.0), 1.0)
        self.epsilon_min = min(max(float(epsilon_min), 0.0), 1.0)
        self.epsilon_decay = min(max(float(epsilon_decay), 0.0), 1.0)
        self.gamma = min(max(float(gamma), 0.0), 1.0)
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)

        initializer = tf.keras.initializers.RandomUniform(minval=0.0, maxval=0.1)
        self.model = models.Sequential(
            [
                layers.Input(shape=(self.input_dim,)),
                layers.Dense(hidden1, activation="tanh", kernel_initializer=initializer),
                layers.Dense(hidden2, activation="tanh", kernel_initializer=initializer),
                layers.Dense(self.output_dim, kernel_initializer=initializer),
            ]
        )
        self.optimizer = optimizers.Adam(learning_rate=float(lr))

    def _checked_state(self, state):
        arr = np.asarray(state, dtype=np.float32).reshape(-1)
        if arr.size != self.input_dim:
            fixed = np.zeros(self.input_dim, dtype=np.float32)
            copy_len = min(arr.size, self.input_dim)
            if copy_len:
                fixed[:copy_len] = arr[:copy_len]
            arr = fixed
        if not np.all(np.isfinite(arr)):
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
        return arr.astype(np.float32, copy=False)

    def choose_action(self, state):
        state = self._checked_state(state)
        q_values = self.model(np.expand_dims(state, axis=0), training=False).numpy()[0]
        q_values = np.nan_to_num(q_values, nan=-np.inf, posinf=np.finfo(np.float32).max, neginf=-np.inf)
        if not np.any(np.isfinite(q_values)):
            return 0
        return int(np.argmax(q_values))

    def select_action(self, state):
        greedy = self.choose_action(state)
        action = dqn_core.epsilon_greedy(greedy, self.output_dim, self.epsilon)
        self.epsilon = dqn_core.decay_epsilon(self.epsilon, self.epsilon_min, self.epsilon_decay)
        return int(action)

    def _train_one(self, state, action, target):
        state = self._checked_state(state)
        action = int(action)
        if not 0 <= action < self.output_dim:
            return None
        target = float(target)
        if not math.isfinite(target):
            return None

        state_tensor = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        target_tensor = tf.convert_to_tensor(target, dtype=tf.float32)
        with tf.GradientTape() as tape:
            q_values = self.model(state_tensor, training=True)
            q_value = q_values[0, action]
            loss = tf.square(target_tensor - q_value)
        grads = tape.gradient(loss, self.model.trainable_variables)
        pairs = [(g, v) for g, v in zip(grads, self.model.trainable_variables) if g is not None]
        if pairs:
            self.optimizer.apply_gradients(pairs)
        return float(loss.numpy())

    def train_batch(self, batch_size=32):
        sampled = self.replay_buffer.sample(batch_size)
        if sampled is None:
            return
        states, actions, rewards, next_states, dones = sampled
        next_states = np.asarray([self._checked_state(s) for s in next_states], dtype=np.float32)
        q_next = self.model(next_states, training=False).numpy()
        q_next = np.nan_to_num(q_next, nan=0.0, posinf=0.0, neginf=0.0)
        targets = rewards.astype(np.float32) + (1.0 - dones.astype(np.float32)) * self.gamma * np.max(q_next, axis=1)
        for state, action, target in zip(states, actions, targets):
            self._train_one(state, int(action), float(target))

    def update(self, state, action, reward, next_state, done=False, batch_size=32):
        state = self._checked_state(state)
        next_state = self._checked_state(next_state)
        reward = float(reward) if math.isfinite(float(reward)) else 0.0
        done = bool(done)

        if self.replay_buffer.enabled:
            self.replay_buffer.push(state, action, reward, next_state, done)
            self.train_batch(batch_size)
            return

        q_next = self.model(next_state[None, :], training=False).numpy()[0]
        q_next = np.nan_to_num(q_next, nan=0.0, posinf=0.0, neginf=0.0)
        target = reward if done else reward + self.gamma * float(np.max(q_next))
        self._train_one(state, action, target)


def preprocess_input(data, width=32):
    """Backward-compatible state preprocessing wrapper."""
    return dqn_core.make_state(data, width=width)
