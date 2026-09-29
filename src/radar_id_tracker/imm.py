"""Two-model IMM filter: low-maneuver CV and constant-acceleration CA."""

from __future__ import annotations

from dataclasses import dataclass
from math import log, pi

import numpy as np
from numpy.typing import NDArray

from .types import Detection

FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class _Model:
    mean: FloatArray
    covariance: FloatArray
    kind: str


def _symmetrize(covariance: FloatArray) -> FloatArray:
    return (covariance + covariance.T) * 0.5


def _update(
    mean: FloatArray,
    covariance: FloatArray,
    observation: FloatArray,
    estimate: FloatArray,
    jacobian: FloatArray,
    noise: FloatArray,
) -> tuple[FloatArray, FloatArray, float]:
    residual = observation - estimate
    innovation_cov = jacobian @ covariance @ jacobian.T + noise
    solved = np.linalg.solve(innovation_cov, residual)
    gain = np.linalg.solve(innovation_cov, jacobian @ covariance).T
    updated_mean = mean + gain @ residual
    identity = np.eye(mean.size)
    residual_map = identity - gain @ jacobian
    updated_cov = _symmetrize(
        residual_map @ covariance @ residual_map.T + gain @ noise @ gain.T
    )
    sign, logdet = np.linalg.slogdet(innovation_cov)
    if sign <= 0:
        raise ArithmeticError("innovation covariance is not positive definite")
    log_likelihood = -0.5 * (
        float(residual @ solved) + logdet + observation.size * log(2.0 * pi)
    )
    return updated_mean, updated_cov, log_likelihood


def _radial_geometry(mean: FloatArray, sensor_xy: tuple[float, float]) -> tuple[float, FloatArray] | None:
    dx = mean[0] - sensor_xy[0]
    dy = mean[1] - sensor_xy[1]
    squared_range = dx * dx + dy * dy
    if squared_range < 1e-6:
        return None
    distance = squared_range**0.5
    prediction = (dx * mean[2] + dy * mean[3]) / distance
    jacobian = np.zeros(6, dtype=np.float64)
    jacobian[0] = mean[2] / distance - prediction * dx / squared_range
    jacobian[1] = mean[3] / distance - prediction * dy / squared_range
    jacobian[2] = dx / distance
    jacobian[3] = dy / distance
    return float(prediction), jacobian


class IMMFilter:
    """State order: x, y, vx, vy, ax, ay.

    Both models share this state so their means and covariances can be mixed.
    The CV model resets acceleration on prediction; CA retains it.
    """

    def __init__(
        self,
        detection: Detection,
        sensor_xy: tuple[float, float],
        position_sigma: float,
        doppler_sigma: float,
    ) -> None:
        self.position_sigma = position_sigma
        self.doppler_sigma = doppler_sigma
        dx = detection.x - sensor_xy[0]
        dy = detection.y - sensor_xy[1]
        distance = (dx * dx + dy * dy) ** 0.5
        initial_vx = initial_vy = 0.0
        if detection.radial_velocity is not None and distance > 1e-3:
            initial_vx = detection.radial_velocity * dx / distance
            initial_vy = detection.radial_velocity * dy / distance
        initial_mean = np.array(
            [detection.x, detection.y, initial_vx, initial_vy, 0.0, 0.0],
            dtype=np.float64,
        )
        initial_cov = np.diag(
            [position_sigma**2, position_sigma**2, 25.0, 25.0, 100.0, 100.0]
        ).astype(np.float64)
        self.models = [
            _Model(initial_mean.copy(), initial_cov.copy(), "cv"),
            _Model(initial_mean.copy(), initial_cov.copy(), "ca"),
        ]
        self.probabilities = np.array([0.8, 0.2], dtype=np.float64)
        self.transition = np.array([[0.97, 0.03], [0.08, 0.92]], dtype=np.float64)

    def mean_covariance(self) -> tuple[FloatArray, FloatArray]:
        mean = sum(
            (weight * model.mean for weight, model in zip(self.probabilities, self.models)),
            start=np.zeros(6, dtype=np.float64),
        )
        covariance = np.zeros((6, 6), dtype=np.float64)
        for weight, model in zip(self.probabilities, self.models):
            delta = model.mean - mean
            covariance += weight * (model.covariance + np.outer(delta, delta))
        return mean, _symmetrize(covariance)

    def predict(self, dt: float) -> None:
        if not 0.0 < dt <= 10.0:
            raise ValueError("dt must be within (0, 10] seconds")
        prior = self.probabilities @ self.transition
        mixed: list[tuple[FloatArray, FloatArray]] = []
        for destination in range(2):
            weights = self.probabilities * self.transition[:, destination] / prior[destination]
            mean = sum(
                (weights[i] * self.models[i].mean for i in range(2)),
                start=np.zeros(6, dtype=np.float64),
            )
            covariance = np.zeros((6, 6), dtype=np.float64)
            for i, model in enumerate(self.models):
                delta = model.mean - mean
                covariance += weights[i] * (model.covariance + np.outer(delta, delta))
            mixed.append((mean, covariance))

        for model, (mean, covariance) in zip(self.models, mixed):
            f = np.eye(6, dtype=np.float64)
            f[0, 2] = f[1, 3] = dt
            q = np.zeros((6, 6), dtype=np.float64)
            if model.kind == "cv":
                f[4, 4] = f[5, 5] = 0.0
                acceleration_variance = 2.0**2
                for position, velocity, acceleration in ((0, 2, 4), (1, 3, 5)):
                    q[position, position] = acceleration_variance * dt**4 / 4
                    q[position, velocity] = q[velocity, position] = acceleration_variance * dt**3 / 2
                    q[velocity, velocity] = acceleration_variance * dt**2
                    q[acceleration, acceleration] = 4.0
            else:
                f[0, 4] = f[1, 5] = dt**2 / 2
                f[2, 4] = f[3, 5] = dt
                jerk_spectral_density = 8.0**2
                block = jerk_spectral_density * np.array(
                    [
                        [dt**5 / 20, dt**4 / 8, dt**3 / 6],
                        [dt**4 / 8, dt**3 / 3, dt**2 / 2],
                        [dt**3 / 6, dt**2 / 2, dt],
                    ],
                    dtype=np.float64,
                )
                for indices in ((0, 2, 4), (1, 3, 5)):
                    q[np.ix_(indices, indices)] = block
            model.mean = f @ mean
            model.covariance = _symmetrize(f @ covariance @ f.T + q)
        self.probabilities = prior

    def update(self, detection: Detection, sensor_xy: tuple[float, float]) -> None:
        position_h = np.zeros((2, 6), dtype=np.float64)
        position_h[0, 0] = position_h[1, 1] = 1.0
        position_noise = np.eye(2, dtype=np.float64) * self.position_sigma**2
        log_likelihoods = np.zeros(2, dtype=np.float64)

        for index, model in enumerate(self.models):
            model.mean, model.covariance, position_loglike = _update(
                model.mean,
                model.covariance,
                np.array([detection.x, detection.y], dtype=np.float64),
                position_h @ model.mean,
                position_h,
                position_noise,
            )
            log_likelihoods[index] = position_loglike
            if detection.radial_velocity is not None:
                radial = _radial_geometry(model.mean, sensor_xy)
                if radial is not None:
                    prediction, jacobian = radial
                    model.mean, model.covariance, radial_loglike = _update(
                        model.mean,
                        model.covariance,
                        np.array([detection.radial_velocity], dtype=np.float64),
                        np.array([prediction], dtype=np.float64),
                        jacobian.reshape(1, 6),
                        np.array([[self.doppler_sigma**2]], dtype=np.float64),
                    )
                    log_likelihoods[index] += radial_loglike

        log_weights = np.log(np.maximum(self.probabilities, 1e-12)) + log_likelihoods
        weights = np.exp(log_weights - np.max(log_weights))
        self.probabilities = weights / weights.sum()

    def position_distance_squared(self, detection: Detection) -> float:
        mean, covariance = self.mean_covariance()
        residual = np.array([detection.x, detection.y], dtype=np.float64) - mean[:2]
        innovation_cov = covariance[:2, :2] + np.eye(2) * self.position_sigma**2
        return float(residual @ np.linalg.solve(innovation_cov, residual))

    def radial_residual(self, detection: Detection, sensor_xy: tuple[float, float]) -> float | None:
        if detection.radial_velocity is None:
            return None
        mean, _ = self.mean_covariance()
        radial = _radial_geometry(mean, sensor_xy)
        if radial is None:
            return None
        return abs(detection.radial_velocity - radial[0])
