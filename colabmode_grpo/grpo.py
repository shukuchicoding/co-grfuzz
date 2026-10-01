import tensorflow as tf
import numpy as np

class PolicyGradientModel(tf.keras.Model):
    def __init__(self, num_inputs, num_outputs, num_layers):
        super(PolicyGradientModel, self).__init__()
        layers = [tf.keras.layers.Input(shape=(num_inputs,))]
        for _ in range(num_layers):
            layers += [tf.keras.layers.Dense(intermediate_layers_size, activation=activation_function, kernel_initializer=tf.keras.initializers.GlorotUniform())]
        layers += [tf.keras.layers.Dense(num_outputs, activation=tf.nn.log_softmax, kernel_initializer=tf.keras.initializers.Constant(value=0.5))]
        self.NN = tf.keras.models.Sequential(layers)
    
    def call(self, inputs):
        output = self.NN(inputs)
        return output

def init(_max_input_size, _num_layers, _intermediate_layers_size, _learning_rate, _activation_function, _group_size, _clip_param, _temperature):
    global max_input_size
    global num_layers
    global intermediate_layers_size
    global learning_rate
    global activation_function
    global group_size
    global clip_param
    global temperature

    global pg_model
    global policy_net_optimizer
    global group_policy

    max_input_size = _max_input_size
    num_layers = _num_layers
    intermediate_layers_size = _intermediate_layers_size
    learning_rate = _learning_rate
    activation_function = _activation_function
    group_size = _group_size
    clip_param = _clip_param
    temperature = _temperature

    pg_model = PolicyGradientModel(max_input_size, max_input_size, num_layers)
    policy_net_optimizer = tf.keras.optimizers.Adam(learning_rate)
    group_policy = np.full(group_size, 1/group_size, dtype=np.float32)

def get_policy(input, n_actions):
    x = np.frombuffer(input.ljust(max_input_size, b'\x00'), dtype=np.uint8)
    probs = np.squeeze(np.exp(pg_model(np.atleast_2d(x))))
    final_probs = probs[:n_actions]
    sum = np.sum(final_probs)
    return final_probs / sum if sum != 0 else np.asarray([1]*n_actions) / n_actions

def sampling(input, n_actions):
    probs = get_policy(input, n_actions)
    group_actions = np.argsort(probs)[-group_size:][::-1]
    return group_actions, probs[group_actions]

def compute_group_advantages(rewards):
    rewards = tf.cast(rewards, dtype=tf.float32)
    group_mean = tf.reduce_mean(rewards)
    group_std = tf.math.reduce_std(rewards) + 1e-8
    return (rewards - group_mean) / group_std

def get_loss(state, rewards):
    global group_policy
    advantages = compute_group_advantages(rewards)
    out = pg_model(tf.convert_to_tensor(np.atleast_2d(np.frombuffer(state.ljust(max_input_size, b'\x00'), dtype=np.uint8))))
    log_policy = out[0]
    entropy = -tf.reduce_sum(log_policy*tf.exp(log_policy), axis=-1)
    log_group_probs, _ = tf.math.top_k(out[0], k=group_size)
    ratio = tf.exp(log_group_probs - group_policy)
    # group_policy = log_group_probs
    group_policy = tf.stop_gradient(log_group_probs)
    surr1 = ratio * advantages
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
    pol_surr = -tf.reduce_mean(tf.minimum(surr1, surr2))
    return pol_surr - temperature * entropy

def train_on_group(state, rewards):
    with tf.GradientTape() as tape:
        loss = get_loss(state, rewards)
    gradients = tape.gradient(loss, pg_model.trainable_variables)
    policy_net_optimizer.apply_gradients(zip(gradients, pg_model.trainable_variables))

def finished_callback():
    tf.keras.backend.clear_session()
    return