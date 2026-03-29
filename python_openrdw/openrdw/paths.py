from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from .geometry import Vector2


@dataclass(frozen=True)
class SamplingDistribution:
    kind: str
    min_value: float
    max_value: float
    alternation: str = "none"
    mu: float = 0.0
    sigma: float = 0.0


@dataclass(frozen=True)
class PathSeed:
    waypoint_count: int
    distance_distribution: SamplingDistribution
    angle_distribution: SamplingDistribution

    @staticmethod
    def ninety_turn() -> "PathSeed":
        return PathSeed(
            waypoint_count=40,
            distance_distribution=SamplingDistribution("uniform", 2, 8),
            angle_distribution=SamplingDistribution("uniform", 90, 90, "random"),
        )

    @staticmethod
    def random_turn() -> "PathSeed":
        return PathSeed(
            waypoint_count=50,
            distance_distribution=SamplingDistribution("uniform", 2, 8),
            angle_distribution=SamplingDistribution("uniform", -180, 180),
        )

    @staticmethod
    def straight_line() -> "PathSeed":
        return PathSeed(
            waypoint_count=10,
            distance_distribution=SamplingDistribution("uniform", 20, 20),
            angle_distribution=SamplingDistribution("uniform", 0, 0),
        )

    @staticmethod
    def sawtooth() -> "PathSeed":
        return PathSeed(
            waypoint_count=40,
            distance_distribution=SamplingDistribution("uniform", 5, 5),
            angle_distribution=SamplingDistribution("uniform", 140, 140, "constant"),
        )


def _sample_distribution(distribution: SamplingDistribution, rng: random.Random) -> float:
    if distribution.kind == "uniform":
        value = rng.uniform(distribution.min_value, distribution.max_value)
    elif distribution.kind == "normal":
        value = rng.gauss(distribution.mu, distribution.sigma)
        value = min(distribution.max_value, max(distribution.min_value, value))
    else:
        raise ValueError(f"Unsupported distribution kind: {distribution.kind}")
    if distribution.alternation == "random" and rng.random() < 0.5:
        value = -value
    return value


def generate_initial_path_by_seed(
    seed: PathSeed,
    target_distance: float,
    rng: random.Random | None = None,
) -> list[Vector2]:
    rng = rng or random.Random(3041)
    position = Vector2(0.0, 0.0)
    forward = Vector2(0.0, 1.0)
    points = [position]
    alternator = 1
    total_distance = 0.0

    while total_distance < target_distance:
        distance = _sample_distribution(seed.distance_distribution, rng)
        if distance + total_distance >= target_distance:
            distance = target_distance - total_distance
        angle = _sample_distribution(seed.angle_distribution, rng)
        if seed.angle_distribution.alternation == "constant":
            angle *= alternator
        next_position = position + forward * distance
        points.append(next_position)
        position = next_position
        forward = forward.rotate(angle).normalized()
        total_distance += distance
        alternator *= -1
        if distance <= 0:
            break
    return points


def generate_circle_path(
    radius: float,
    waypoint_count: int,
    figure_eight: bool = False,
) -> list[Vector2]:
    center = Vector2(radius, 0.0)
    start_vec = Vector2(-radius, 0.0)
    points = [Vector2(0.0, 0.0)]
    step_angle = 360.0 / waypoint_count
    position = points[0]
    for index in range(waypoint_count):
        vec = start_vec.rotate(-step_angle * (index + 1))
        next_position = center + vec
        points.append(next_position)
        position = next_position
    if figure_eight:
        center = Vector2(-radius, 0.0)
        start_vec = Vector2(radius, 0.0)
        for index in range(waypoint_count):
            vec = start_vec.rotate(step_angle * (index + 1))
            next_position = center + vec
            points.append(next_position)
            position = next_position
    return points


def load_waypoints_from_file(path: str | Path, first_waypoint_is_start_point: bool = False) -> list[Vector2]:
    raw_lines = Path(path).read_text(encoding="utf-8").splitlines()
    waypoints: list[Vector2] = []
    first_point = Vector2(0.0, 0.0)
    for index, raw_line in enumerate(raw_lines):
        line = raw_line.strip()
        if not line:
            continue
        parts = [part for part in line.replace(",", " ").split() if part]
        if len(parts) != 2:
            raise ValueError(f"Invalid waypoint line: {raw_line!r}")
        x = float(parts[0])
        y = float(parts[1])
        if index == 0:
            first_point = Vector2(x, y)
        point = Vector2(x, y)
        if first_waypoint_is_start_point:
            point = point - first_point
        waypoints.append(point)
    return waypoints


def load_sampling_intervals_from_file(path: str | Path) -> list[float]:
    values: list[float] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line:
            values.append(float(line))
    return values
