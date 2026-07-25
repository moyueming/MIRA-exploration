from typing import Sequence

import numpy as np


class RunningWeightAverage:
    def __init__(self, start_fraction: float):
        self.start_fraction = min(max(float(start_fraction), 0.0), 1.0)
        self._weights = None
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    @property
    def active(self) -> bool:
        return self._count > 0

    def update(self, weights: Sequence[np.ndarray], progress: float) -> bool:
        if min(max(float(progress), 0.0), 1.0) < self.start_fraction:
            return False

        incoming = [np.asarray(weight) for weight in weights]
        if self._weights is None:
            self._weights = [weight.copy() for weight in incoming]
            self._count = 1
            return True

        self._validate(incoming)
        next_count = self._count + 1
        for average, weight in zip(self._weights, incoming):
            average += (weight - average) / next_count
        self._count = next_count
        return True

    def formal_weights(self, online_weights: Sequence[np.ndarray]):
        source = self._weights if self._weights is not None else online_weights
        return [np.asarray(weight).copy() for weight in source]

    def _validate(self, weights: Sequence[np.ndarray]) -> None:
        if len(weights) != len(self._weights):
            raise ValueError(
                f"weight count mismatch: expected {len(self._weights)}, got {len(weights)}"
            )
        for index, (average, weight) in enumerate(zip(self._weights, weights)):
            if average.shape != weight.shape:
                raise ValueError(
                    f"weight {index} shape mismatch: expected {average.shape}, got {weight.shape}"
                )
