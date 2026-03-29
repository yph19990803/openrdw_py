from __future__ import annotations

import math

from .geometry import EPSILON, Vector2, signed_angle


def ray_segment_intersection(origin: Vector2, direction: Vector2, a: Vector2, b: Vector2) -> tuple[float, Vector2] | None:
    segment = b - a
    denom = direction.cross(segment)
    if abs(denom) <= EPSILON:
        return None
    diff = a - origin
    t = diff.cross(segment) / denom
    u = diff.cross(direction) / denom
    if t >= 0.0 and 0.0 <= u <= 1.0:
        hit = origin + direction * t
        return t, hit
    return None


def compute_visibility_polygon(origin: Vector2, polygons: list[list[Vector2]]) -> list[Vector2]:
    vertices: list[Vector2] = []
    for polygon in polygons:
        vertices.extend(polygon)
    if not vertices:
        return []

    angles: list[float] = []
    for vertex in vertices:
        base = math.atan2(vertex.y - origin.y, vertex.x - origin.x)
        angles.extend([base - 1e-5, base, base + 1e-5])

    hits: list[tuple[float, Vector2]] = []
    for angle in angles:
        direction = Vector2(math.cos(angle), math.sin(angle))
        best_distance = float("inf")
        best_point: Vector2 | None = None
        for polygon in polygons:
            if len(polygon) < 2:
                continue
            for index, start in enumerate(polygon):
                end = polygon[(index + 1) % len(polygon)]
                result = ray_segment_intersection(origin, direction, start, end)
                if result is None:
                    continue
                distance, point = result
                if distance < best_distance:
                    best_distance = distance
                    best_point = point
        if best_point is not None:
            hits.append((angle, best_point))

    hits.sort(key=lambda item: item[0])
    deduped: list[Vector2] = []
    for _, point in hits:
        if not deduped or (point - deduped[-1]).magnitude > 1e-4:
            deduped.append(point)
    return deduped


def compute_slice_bisectors(origin: Vector2, polygon: list[Vector2]) -> list[Vector2]:
    bisectors: list[Vector2] = []
    if not polygon:
        return bisectors
    weights: list[float] = []
    for index, point in enumerate(polygon):
        next_point = polygon[(index + 1) % len(polygon)]
        vec1 = (point - origin).normalized()
        vec2 = (next_point - origin).normalized()
        bisector = (vec1 + vec2).normalized()
        weight = abs((point - origin).cross(next_point - origin))
        bisectors.append(bisector * weight)
        weights.append(weight)
    total = sum(weights)
    if total <= EPSILON:
        return bisectors
    return [bisector / total for bisector in bisectors]


def active_slice_index(direction: Vector2, bisectors: list[Vector2]) -> int:
    best_index = 0
    best_angle = float("inf")
    for index, bisector in enumerate(bisectors):
        angle = abs(direction.angle_to(bisector))
        if angle < best_angle:
            best_angle = angle
            best_index = index
    return best_index


def most_similar_slice_weight(target_weight: float, bisectors: list[Vector2]) -> Vector2:
    if not bisectors:
        return Vector2(0.0, 0.0)
    best = bisectors[0]
    best_score = abs(target_weight - bisectors[0].magnitude)
    for bisector in bisectors[1:]:
        score = abs(target_weight - bisector.magnitude)
        if score < best_score:
            best = bisector
            best_score = score
    return best.normalized()


def priority_from_force(force: Vector2, heading: Vector2, a1: float = 1.0, a2: float = 1.0) -> float:
    if force.magnitude <= EPSILON:
        return 0.0
    return -(a1 * force.magnitude + a2 * abs(signed_angle(force, heading)))
