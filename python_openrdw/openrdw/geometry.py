from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


EPSILON = 1e-6


@dataclass(frozen=True)
class Vector2:
    x: float
    y: float

    def __add__(self, other: "Vector2") -> "Vector2":
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector2") -> "Vector2":
        return Vector2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vector2":
        return Vector2(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> "Vector2":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Vector2":
        return Vector2(self.x / scalar, self.y / scalar)

    @property
    def magnitude(self) -> float:
        return math.hypot(self.x, self.y)

    def normalized(self) -> "Vector2":
        mag = self.magnitude
        if mag <= EPSILON:
            return Vector2(0.0, 0.0)
        return self / mag

    def dot(self, other: "Vector2") -> float:
        return self.x * other.x + self.y * other.y

    def cross(self, other: "Vector2") -> float:
        return self.x * other.y - self.y * other.x

    def angle_to(self, other: "Vector2") -> float:
        denom = max(self.magnitude * other.magnitude, EPSILON)
        cos_value = max(-1.0, min(1.0, self.dot(other) / denom))
        return math.degrees(math.acos(cos_value))

    def rotate(self, degrees: float) -> "Vector2":
        radians = math.radians(-degrees)
        cos_theta = math.cos(radians)
        sin_theta = math.sin(radians)
        return Vector2(
            self.x * cos_theta - self.y * sin_theta,
            self.x * sin_theta + self.y * cos_theta,
        )


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def heading_to_vector(heading_deg: float) -> Vector2:
    radians = math.radians(heading_deg)
    return Vector2(math.sin(radians), math.cos(radians))


def vector_to_heading(direction: Vector2) -> float:
    if direction.magnitude <= EPSILON:
        return 0.0
    return math.degrees(math.atan2(direction.x, direction.y))


def normalize_heading(heading_deg: float) -> float:
    heading = heading_deg % 360.0
    if heading < 0:
        heading += 360.0
    return heading


def signed_angle(from_vec: Vector2, to_vec: Vector2) -> float:
    if from_vec.magnitude <= EPSILON or to_vec.magnitude <= EPSILON:
        return 0.0
    angle = from_vec.angle_to(to_vec)
    cross = from_vec.cross(to_vec)
    if abs(cross) <= EPSILON and from_vec.dot(to_vec) < 0:
        return 180.0
    return math.copysign(angle, cross)


def polygon_centroid(points: Iterable[Vector2]) -> Vector2:
    pts = list(points)
    if not pts:
        return Vector2(0.0, 0.0)
    return Vector2(
        sum(point.x for point in pts) / len(pts),
        sum(point.y for point in pts) / len(pts),
    )


def point_in_polygon(point: Vector2, polygon: list[Vector2]) -> bool:
    if not polygon:
        return False
    inside = False
    j = len(polygon) - 1
    for i, pi in enumerate(polygon):
        pj = polygon[j]
        intersects = ((pi.y > point.y) != (pj.y > point.y)) and (
            point.x
            < (pj.x - pi.x) * (point.y - pi.y) / max(pj.y - pi.y, EPSILON) + pi.x
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def distance_point_to_segment(point: Vector2, a: Vector2, b: Vector2) -> float:
    ab = b - a
    if ab.magnitude <= EPSILON:
        return (point - a).magnitude
    t = clamp((point - a).dot(ab) / max(ab.dot(ab), EPSILON), 0.0, 1.0)
    projection = a + ab * t
    return (point - projection).magnitude


def closest_point_on_segment(point: Vector2, a: Vector2, b: Vector2) -> Vector2:
    ab = b - a
    if ab.magnitude <= EPSILON:
        return a
    t = clamp((point - a).dot(ab) / max(ab.dot(ab), EPSILON), 0.0, 1.0)
    return a + ab * t


def nearest_point_on_polygon(point: Vector2, polygon: list[Vector2]) -> Vector2:
    if not polygon:
        return point
    best_point = polygon[0]
    best_distance = float("inf")
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        candidate = closest_point_on_segment(point, start, end)
        distance = (point - candidate).magnitude
        if distance < best_distance:
            best_distance = distance
            best_point = candidate
    return best_point


def nearest_distance_and_point(point: Vector2, polygons: list[list[Vector2]]) -> tuple[float, Vector2]:
    best_distance = float("inf")
    best_point = point
    for polygon in polygons:
        if len(polygon) < 2:
            continue
        candidate = nearest_point_on_polygon(point, polygon)
        distance = (point - candidate).magnitude
        if distance < best_distance:
            best_distance = distance
            best_point = candidate
    return best_distance, best_point


def nearest_distance_to_polygons(point: Vector2, polygons: list[list[Vector2]]) -> float:
    best, _ = nearest_distance_and_point(point, polygons)
    return best
