"""TensorFlow-independent helpers for the DQN fuzzing baseline."""

import math
import random

import numpy as np


def make_state(data, width=32):
    """Return a fixed-width normalized float32 state in [-1, 1).

    Inputs shorter than ``width`` are zero-padded at byte level.  Empty inputs
    are accepted defensively and become an all-zero byte vector before
    normalization.  ``width`` must be positive.
    """
    width = int(width)
    if width < 1:
        raise ValueError("state width must be >= 1")
    raw = bytes(data or b"")[:width]
    raw = raw + (b"\x00" * (width - len(raw)))
    arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
    return (arr - 128.0) / 128.0


def coverage_reward(input_edges):
    """Coverage-only reward used by the cleaned DQN baseline.

    Deep Reinforcement Fuzzing evaluates coverage of the current execution when
    using the coverage reward with an empty history.  In our Python adaptation,
    line-transition edges are the coverage units.
    """
    try:
        value = float(len(set(input_edges)))
    except TypeError:
        value = 0.0
    return value if math.isfinite(value) and value >= 0.0 else 0.0


def epsilon_greedy(greedy_action, n_actions, epsilon, rng=None):
    """Choose an action with epsilon-greedy exploration."""
    n_actions = int(n_actions)
    if n_actions < 1:
        raise ValueError("n_actions must be >= 1")
    epsilon = float(epsilon)
    if not math.isfinite(epsilon):
        epsilon = 0.0
    epsilon = min(max(epsilon, 0.0), 1.0)
    rng = rng or random
    if rng.random() < epsilon:
        return int(rng.randrange(n_actions))
    greedy_action = int(greedy_action)
    if not 0 <= greedy_action < n_actions:
        return int(rng.randrange(n_actions))
    return greedy_action


def decay_epsilon(current, minimum=0.1, decay=0.999):
    """Monotonically decay epsilon while preserving a configured floor."""
    current = float(current)
    minimum = float(minimum)
    decay = float(decay)
    if not math.isfinite(current):
        current = 1.0
    if not math.isfinite(minimum):
        minimum = 0.0
    if not math.isfinite(decay):
        decay = 1.0
    minimum = min(max(minimum, 0.0), 1.0)
    decay = min(max(decay, 0.0), 1.0)
    current = min(max(current, minimum), 1.0)
    return max(minimum, current * decay)
