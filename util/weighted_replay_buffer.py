import numpy as np
import math
from collections import deque, namedtuple

# -------------------------
# Replay buffer with weights
# -------------------------
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done', 'weight', 'seq_inp'])

class WeightedReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size, seed=42):
        # --- Deterministic RNG when seed is provided ---
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random  # original nondeterministic behavior

        weights = np.array([t.weight for t in self.buffer], dtype=np.float64)

        if weights.sum() <= 0:
            probs = np.ones(len(weights)) / len(weights)
        else:
            probs = weights / weights.sum()

        idx = rng.choice(
            len(self.buffer),
            size=min(batch_size, len(self.buffer)),
            replace=False,
            p=probs
        )

        batch = [self.buffer[i] for i in idx]
        return batch

    def upweight_recent(self, window=200, multiplier=2.0):
        n = len(self.buffer)
        for i in range(max(0, n-window), n):
            t = self.buffer[i]
            new_w = t.weight * multiplier
            self.buffer[i] = Transition(
                t.state, t.action, t.reward, t.next_state,
                t.done, new_w, t.seq_inp
            )
