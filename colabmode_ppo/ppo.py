import math

import numpy as np
import tensorflow as tf

import ppo_core


pg_model = None
optimizer_policy_net = None
memory = None
max_input_size = None
intermediate_layers_size = None
learning_rate = None
clip_param = None
temperature = None
buffer_length = None
activation_function = None
num_layers = None


def init(
    _max_input_size,
    _intermediate_layers_size,
    _learning_rate,
    _clip_param,
    _temperature,
    _n_layers,
    _buffer_length,
    _activation_function,
    _num_layers,
):
    global pg_model, optimizer_policy_net, memory
    global max_input_size, intermediate_layers_size, learning_rate
    global clip_param, temperature, buffer_length, activation_function, num_layers

    max_input_size = int(_max_input_size)
    intermediate_layers_size = int(_intermediate_layers_size)
    learning_rate = float(_learning_rate)
    clip_param = float(_clip_param)
    temperature = float(_temperature)
    buffer_length = int(_buffer_length)
    activation_function = _activation_function
    num_layers = int(_num_layers)

    if max_input_size < 1:
        raise ValueError("max_input_size must be >= 1")
    if intermediate_layers_size < 1 or num_layers < 0:
        raise ValueError("invalid policy-network dimensions")
    if buffer_length < 1:
        raise ValueError("buffer_length must be >= 1")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")
    if not math.isfinite(clip_param) or clip_param < 0:
        raise ValueError("clip_param must be >= 0")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be >= 0")

    memory = Memory(buffer_length)
    pg_model = PolicyGradientModel(max_input_size, max_input_size, num_layers)
    optimizer_policy_net = tf.keras.optimizers.Adam(learning_rate)


class Memory:
    def __init__(self, size):
        self.size = max(int(size), 1)
        self.clear()

    def push(self, state, action, probability, reward, valid_length=None):
        state = bytes(state or b"")
        if not state:
            state = b"\x00"
        valid_length = len(state) if valid_length is None else int(valid_length)
        valid_length = min(max(valid_length, 1), len(state), int(max_input_size))
        action = min(max(int(action), 0), valid_length - 1)
        self.states.append(state)
        self.actions.append(action)
        self.probabilities.append(ppo_core.clamp_old_probability(probability))
        reward = float(reward)
        self.rewards.append(reward if math.isfinite(reward) else 0.0)
        self.valid_lengths.append(valid_length)
        self.count += 1

    def clear(self):
        self.states = []
        self.actions = []
        self.probabilities = []
        self.rewards = []
        self.valid_lengths = []
        self.count = 0

    def is_full(self):
        return self.count >= self.size


class PolicyGradientModel(tf.keras.Model):
    def __init__(self, num_inputs, num_outputs, layers_count):
        super().__init__()
        network_layers = [tf.keras.layers.Input(shape=(num_inputs,))]
        for _ in range(layers_count):
            network_layers.append(
                tf.keras.layers.Dense(
                    intermediate_layers_size,
                    activation=activation_function,
                    kernel_initializer=tf.keras.initializers.Constant(value=0.5),
                )
            )
        network_layers.append(
            tf.keras.layers.Dense(
                num_outputs,
                activation=tf.nn.log_softmax,
                kernel_initializer=tf.keras.initializers.Constant(value=0.5),
            )
        )
        self.NN = tf.keras.models.Sequential(network_layers)

    def call(self, inputs):
        return self.NN(inputs)


def _state_vector(data):
    data = bytes(data or b"")[:max_input_size]
    data = data.ljust(max_input_size, b"\x00")
    return np.frombuffer(data, dtype=np.uint8).astype(np.float32)


def _valid_log_policy(state, valid_length, training=False):
    valid_length = min(max(int(valid_length), 1), max_input_size)
    x = tf.convert_to_tensor(_state_vector(state)[None, :], dtype=tf.float32)
    full_log_policy = pg_model(x, training=training)[0]
    valid = full_log_policy[:valid_length]
    # The network is normalized over max_input_size outputs. Re-normalize over
    # only positions that actually exist in the current seed.
    return valid - tf.reduce_logsumexp(valid)


def get_heatmap(input, n_actions):
    if pg_model is None:
        raise RuntimeError("ppo.init() must be called before get_heatmap()")
    n_actions = min(max(int(n_actions), 1), len(input or b""), max_input_size)
    if n_actions < 1:
        raise ValueError("input must contain at least one byte")
    valid_log_probs = _valid_log_policy(input, n_actions, training=False).numpy()
    probs = ppo_core.sanitize_probabilities(np.exp(valid_log_probs))
    entropy_terms = np.where(probs > 0, probs * np.log(probs), 0.0)
    entropy = float(-np.sum(entropy_terms))
    return probs, entropy


def pick_action(input, n_actions):
    n_actions = min(max(int(n_actions), 1), len(input or b""), max_input_size)
    probs, entropy = get_heatmap(input, n_actions)
    action = int(np.random.choice(n_actions, p=probs))
    probability = ppo_core.clamp_old_probability(probs[action])
    return action, probability, entropy


def get_loss():
    if memory is None or memory.count == 0:
        return tf.convert_to_tensor(0.0, dtype=tf.float32)

    log_probs = []
    old_log_probs = []
    advantages = []
    entropies = []

    for i in range(memory.count):
        valid_log_policy = _valid_log_policy(memory.states[i], memory.valid_lengths[i], training=True)
        action = min(max(int(memory.actions[i]), 0), memory.valid_lengths[i] - 1)
        log_prob = valid_log_policy[action]
        probs = tf.exp(valid_log_policy)
        entropy = -tf.reduce_sum(probs * valid_log_policy)

        log_probs.append(log_prob)
        old_log_probs.append(math.log(ppo_core.clamp_old_probability(memory.probabilities[i])))
        advantages.append(float(memory.rewards[i]))
        entropies.append(entropy)

    log_probs = tf.stack(log_probs)
    old_log_probs = tf.convert_to_tensor(old_log_probs, dtype=tf.float32)
    advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
    entropies = tf.stack(entropies)

    ratio = tf.exp(tf.clip_by_value(log_probs - old_log_probs, -20.0, 20.0))
    surr1 = ratio * advantages
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
    policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
    loss = policy_loss - temperature * tf.reduce_mean(entropies)
    return tf.where(tf.math.is_finite(loss), loss, tf.zeros_like(loss))


def train_one_batch():
    if memory is None or memory.count == 0:
        return 0.0
    with tf.GradientTape() as tape:
        loss = get_loss()
    gradients = tape.gradient(loss, pg_model.trainable_variables)
    pairs = []
    for gradient, variable in zip(gradients, pg_model.trainable_variables):
        if gradient is None:
            continue
        gradient = tf.where(tf.math.is_finite(gradient), gradient, tf.zeros_like(gradient))
        pairs.append((gradient, variable))
    if pairs:
        optimizer_policy_net.apply_gradients(pairs)
    return float(loss.numpy())


def add_experience(state, action, probability, reward, valid_length=None):
    memory.push(state, action, probability, reward, valid_length=valid_length)
    if memory.is_full():
        train_one_batch()
        memory.clear()


def finished_callback():
    tf.keras.backend.clear_session()
