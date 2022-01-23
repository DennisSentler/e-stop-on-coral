import time
import numpy as np


class InferenceClock:
    def __init__(self, name="default"):
        self.name = name
        self._start_time = 0.0
        self.times_ms = []

    def start(self):
        self._start_time = time.perf_counter()

    def stop(self):
        if (self._start_time != 0.0):
            inference_time = (time.perf_counter() - self._start_time) * 1000
            self.times_ms += [inference_time]

    def calculate_avg_IPS(self):
        sum_seconds = np.sum(self.times_ms)/1000
        return len(self.times_ms) / sum_seconds

    def reset(self):
        self._start_time = 0.0
        self.times_ms = []

    def report(self):
        return (f'=== Clock report for "{self.name}" ===\r\n'
                f'Average IPS = {self.calculate_avg_IPS():.2f}\r\n'
                f'Standard deviation = {np.std(self.times_ms):.2f}\r\n'
                f'Minimum processing time = {np.min(self.times_ms):.2f}ms\r\n'
                f'Maximum processing time = {np.max(self.times_ms):.2f}ms\r\n'
                 '=== report end ===')
