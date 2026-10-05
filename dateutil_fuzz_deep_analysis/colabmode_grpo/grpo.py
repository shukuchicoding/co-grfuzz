import numpy as np
import tensorflow as tf

import grpo_core


class PolicyGradientModel(tf.keras.Model):
    def __init__(self, num_inputs, num_outputs, num_layers):
        super().__init__()
        layers = [tf.keras.layers.Input(shape=(num_inputs,))]
        for _ in range(num_layers):
            layers.append(
                tf.keras.layers.Dense(
                    intermediate_layers_size,
                    activation=activation_function,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(),
                )
            )
        # The network emits log-probabilities over the fixed maximum input size.
        # _valid_log_policy() renormalizes them over the actual seed length so
        # padded positions can never be sampled.
        layers.append(
            tf.keras.layers.Dense(
                num_outputs,
                activation=tf.nn.log_softmax,
                kernel_initializer=tf.keras.initializers.Constant(value=0.5),
            )
        )
        self.NN = tf.keras.Sequential(layers)

    def call(self, inputs):
        return self.NN(inputs)


def init(
    _max_input_size,
    _num_layers,
    _intermediate_layers_size,
    _learning_rate,
    _activation_function,
    _group_size,
    _clip_param,
    _temperature,
):
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

    if int(_max_input_size) < 1:
        raise ValueError("max_input_size must be >= 1")
    if int(_group_size) < 1:
        raise ValueError("group_size must be >= 1")
    if int(_num_layers) < 0:
        raise ValueError("num_layers must be >= 0")
    if int(_intermediate_layers_size) < 1:
        raise ValueError("intermediate_layers_size must be >= 1")
    if float(_learning_rate) <= 0:
        raise ValueError("learning_rate must be > 0")
    if not 0.0 <= float(_clip_param) < 1.0:
        raise ValueError("clip_param must satisfy 0 <= clip_param < 1")
    if float(_temperature) < 0:
        raise ValueError("temperature must be >= 0")

    max_input_size = int(_max_input_size)
    num_layers = int(_num_layers)
    intermediate_layers_size = int(_intermediate_layers_size)
    learning_rate = float(_learning_rate)
    activation_function = _activation_function
    group_size = int(_group_size)
    clip_param = float(_clip_param)
    temperature = float(_temperature)

    pg_model = PolicyGradientModel(max_input_size, max_input_size, num_layers)
    policy_net_optimizer = tf.keras.optimizers.Adam(learning_rate)


def _state_array(state):
    raw = bytes(state[:max_input_size])
    raw = raw.ljust(max_input_size, b"\x00")
    return np.frombuffer(raw, dtype=np.uint8).astype(np.float32)


def _valid_log_policy(state):
    """Return log probabilities normalized only over real byte positions."""
    valid_len = min(len(state), max_input_size)
    if valid_len <= 0:
        # Callers treat an empty tensor as "no valid action". This avoids a
        # reduce_logsumexp over an empty set and keeps len==0 from crashing.
        return tf.zeros((0,), dtype=tf.float32)

    x = tf.convert_to_tensor(np.atleast_2d(_state_array(state)), dtype=tf.float32)
    full_log_policy = pg_model(x)[0]
    valid_logits = full_log_policy[:valid_len]
    return valid_logits - tf.reduce_logsumexp(valid_logits)


def get_policy(state):
    log_policy = _valid_log_policy(state)
    if int(log_policy.shape[0]) == 0:
        return np.asarray([], dtype=np.float64)
    return grpo_core.probabilities_from_log_probs(log_policy.numpy())


def sampling(state, k=None, rng=None):
    """Sample a GRPO group from the old policy over valid byte positions.

    Unlike the previous top-k implementation, this draws independent actions
    with replacement from pi_old(position | seed), matching GRPO's group sampling
    semantics. Therefore a short seed can still produce group_size=64 samples,
    and duplicate positions are valid by design.
    """
    if k is None:
        k = group_size
    k = int(k)
    if k <= 0 or len(state) <= 0:
        return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.float32)

    log_policy = _valid_log_policy(state)
    if int(log_policy.shape[0]) == 0:
        return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.float32)

    return grpo_core.sample_actions_from_log_probs(log_policy.numpy(), k, rng=rng)


def log_probs_for_actions(state, actions):
    """Return current-policy log-probabilities for valid action indices."""
    actions = np.asarray(actions, dtype=np.int32).reshape(-1)
    if actions.size == 0 or len(state) <= 0:
        return np.asarray([], dtype=np.float32)

    log_policy = _valid_log_policy(state)
    valid_len = int(log_policy.shape[0])
    if valid_len <= 0:
        return np.asarray([], dtype=np.float32)
    if np.any(actions < 0) or np.any(actions >= valid_len):
        raise ValueError("action index is outside the valid seed length")
    return tf.gather(log_policy, actions).numpy().astype(np.float32)


def compute_group_advantages(rewards):
    # Rewards do not require gradients, so use the tested NumPy implementation
    # and convert the stable result back to a TensorFlow tensor.
    return tf.convert_to_tensor(grpo_core.group_advantages(rewards), dtype=tf.float32)


def get_loss(state, actions, old_log_probs, rewards):
    actions = tf.convert_to_tensor(actions, dtype=tf.int32)
    old_log_probs = tf.stop_gradient(tf.convert_to_tensor(old_log_probs, dtype=tf.float32))
    advantages = compute_group_advantages(rewards)

    log_policy = _valid_log_policy(state)
    if int(log_policy.shape[0]) == 0:
        return tf.constant(0.0, dtype=tf.float32)

    selected_log_probs = tf.gather(log_policy, actions)
    ratio = tf.exp(selected_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
    policy_loss = -tf.reduce_mean(tf.minimum(unclipped, clipped))

    probs = tf.exp(log_policy)
    entropy = -tf.reduce_sum(probs * log_policy)
    loss = policy_loss - temperature * entropy
    # Avoid propagating NaN/Inf into model weights in the unlikely event of a
    # numerical failure. A zero loss means "skip this malformed update".
    return tf.where(tf.math.is_finite(loss), loss, tf.constant(0.0, dtype=tf.float32))


def train_on_group(state, actions, old_log_probs, rewards):
    actions = np.asarray(actions, dtype=np.int32).reshape(-1)
    old_log_probs = np.asarray(old_log_probs, dtype=np.float32).reshape(-1)
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)

    if not (len(actions) == len(old_log_probs) == len(rewards)):
        raise ValueError("actions, old_log_probs, and rewards must have equal lengths")
    if len(actions) == 0 or len(state) == 0:
        return None

    valid_len = min(len(state), max_input_size)
    if np.any(actions < 0) or np.any(actions >= valid_len):
        raise ValueError("training action index is outside the valid seed length")

    # NaN/Inf rewards are sanitized by group_advantages(), but old log-probs are
    # part of the importance ratio and must be valid. Skip rather than corrupting
    # the policy if an unexpected numerical anomaly is observed.
    if not np.all(np.isfinite(old_log_probs)):
        return None

    current_log_policy = _valid_log_policy(state).numpy()
    if current_log_policy.size == 0 or not np.all(np.isfinite(current_log_policy)):
        return None

    with tf.GradientTape() as tape:
        loss = get_loss(state, actions, old_log_probs, rewards)
    gradients = tape.gradient(loss, pg_model.trainable_variables)
    pairs = []
    for gradient, variable in zip(gradients, pg_model.trainable_variables):
        if gradient is not None and bool(tf.reduce_all(tf.math.is_finite(gradient)).numpy()):
            pairs.append((gradient, variable))
    if pairs:
        policy_net_optimizer.apply_gradients(pairs)
    return float(loss.numpy())


def finished_callback():
    tf.keras.backend.clear_session()
