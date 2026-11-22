"""
Gymnasium import helper.

On a normal install (pip install -r requirements.txt) this simply re-exports the
REAL gymnasium classes. Only if gymnasium is not installed do we fall back to a
tiny stand-in with the same interface, so the code can still be imported.
Check `USING_REAL_GYMNASIUM` to know which one you are getting.
"""
from __future__ import annotations

import numpy as np

try:  # the normal case
    import gymnasium as _gym
    from gymnasium import spaces as _spaces

    Env = _gym.Env
    Box = _spaces.Box
    Discrete = _spaces.Discrete
    USING_REAL_GYMNASIUM = True
except ImportError:  # minimal fallback (same method names as gymnasium)
    USING_REAL_GYMNASIUM = False

    class Discrete:  # type: ignore[no-redef]
        def __init__(self, n):
            self.n = int(n)

        def contains(self, x):
            try:
                return int(x) == x and 0 <= int(x) < self.n
            except (TypeError, ValueError):
                return False

        def sample(self):
            return int(np.random.randint(self.n))

    class Box:  # type: ignore[no-redef]
        def __init__(self, low, high, shape, dtype=np.float32):
            self.shape = tuple(shape)
            self.dtype = dtype
            self.low = np.full(self.shape, low, dtype=dtype)
            self.high = np.full(self.shape, high, dtype=dtype)

        def contains(self, x):
            x = np.asarray(x)
            return x.shape == self.shape and bool(np.all(x >= self.low) and np.all(x <= self.high))

        def sample(self):
            return np.random.uniform(self.low, self.high).astype(self.dtype)

    class Env:  # type: ignore[no-redef]
        metadata: dict = {}
        np_random = np.random.default_rng()

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                self.np_random = np.random.default_rng(seed)
