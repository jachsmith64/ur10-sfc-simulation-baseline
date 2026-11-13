"""Causal slope at independent measurement times, in seconds."""

from collections import deque
from math import isfinite


WINDOWS = {"B0": 2, "L3": 3, "L5": 5}


class ShortWindow:
    def __init__(self, mode, gap_s):
        self.samples = deque(maxlen=WINDOWS[mode])
        if not isfinite(gap_s) or gap_s <= 0:
            raise ValueError("gap_s must be positive and finite")
        self.gap_s = gap_s
        self.reset()

    def reset(self):
        self.samples.clear()
        self.slope = self.span_s = 0.0

    def push(self, sample_s, position):
        if not (isfinite(sample_s) and isfinite(position)):
            return self.slope
        if self.samples:
            last_s = self.samples[-1][0]
            if sample_s <= last_s:
                return self.slope
            if sample_s - last_s > self.gap_s:
                self.reset()
        self.samples.append((sample_s, position))
        if len(self.samples) == 1:
            return self.slope
        newest_s, newest_z = self.samples[-1]
        if len(self.samples) < self.samples.maxlen or self.samples.maxlen == 2:
            # Use the two most recent points while building a full window.
            prev_s, prev_z = self.samples[-2]
            self.span_s = newest_s - prev_s
            self.slope = (newest_z - prev_z) / self.span_s
            return self.slope
        self.span_s = newest_s - self.samples[0][0]
        times = [t - newest_s for t, _ in self.samples]
        values = [z - newest_z for _, z in self.samples]
        mean_t = sum(times) / len(times)
        mean_z = sum(values) / len(values)
        covariance = sum((t - mean_t) * (z - mean_z) for t, z in zip(times, values))
        variance = sum((t - mean_t) ** 2 for t in times)
        self.slope = covariance / variance
        return self.slope
