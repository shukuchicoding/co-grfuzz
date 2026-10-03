"""TensorFlow-independent helpers for the RainFuzz/PPO baseline."""

import math

import numpy as np


def sanitize_probabilities(values):
    """Return a finite normalized probability vector, with uniform fallback."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr[arr < 0.0] = 0.0
    total = float(arr.sum())
    if not math.isfinite(total) or total <= 0.0:
        return np.full(arr.size, 1.0 / arr.size, dtype=np.float64)
    arr /= total
    # Re-normalize after finite arithmetic to keep np.random.choice happy.
    arr /= float(arr.sum())
    return arr


def rainfuzz_r1(original_counts, mutant_counts_list):
    """Compute RainFuzz reward R1.

    For each mutant coverage vector, sum only positive edge-hit-count increases
    relative to the original seed, then average those sums over mutants.
    Missing edges are treated as hit count zero.  Malformed/negative counters are
    clamped defensively instead of propagating NaN/negative rewards.
    """
    original_counts = dict(original_counts or {})
    mutants = list(mutant_counts_list or [])
    if not mutants:
        return 0.0

    def safe_count(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(value) or value < 0:
            return 0.0
        return value

    diffs = []
    for counts in mutants:
        counts = dict(counts or {})
        diff = 0.0
        for edge, value in counts.items():
            inc = safe_count(value) - safe_count(original_counts.get(edge, 0))
            if inc > 0.0:
                diff += inc
        diffs.append(diff)

    reward = float(np.mean(diffs)) if diffs else 0.0
    return reward if math.isfinite(reward) and reward >= 0.0 else 0.0


def clamp_old_probability(probability, eps=1e-12):
    try:
        probability = float(probability)
    except (TypeError, ValueError):
        probability = eps
    if not math.isfinite(probability):
        probability = eps
    return min(max(probability, eps), 1.0)
