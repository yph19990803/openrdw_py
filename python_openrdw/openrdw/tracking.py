from __future__ import annotations

import math
from pathlib import Path

from .geometry import Vector2
from .models import InitialConfiguration


TARGET_AREA = 400.0


def _rotate(point: Vector2, degrees: float) -> Vector2:
    return point.rotate(degrees)


def generate_polygon_tracking_space_points(side_count: int, radius: float = 5.0) -> list[Vector2]:
    points: list[Vector2] = []
    sampled_rotation = 360.0 / side_count
    if side_count % 2 == 1:
        start = Vector2(0.0, radius)
    else:
        start = Vector2(
            radius * math.sin(math.radians(sampled_rotation / 2.0)),
            radius * math.cos(math.radians(sampled_rotation / 2.0)),
        )
    for index in range(side_count):
        points.append(_rotate(start, -sampled_rotation * index))
    return points


def generate_rectangle_tracking_space(
    width: float,
    height: float,
    obstacle_type: int = 0,
) -> tuple[list[Vector2], list[list[Vector2]], list[InitialConfiguration]]:
    tracking = [
        Vector2(width / 2.0, height / 2.0),
        Vector2(-width / 2.0, height / 2.0),
        Vector2(-width / 2.0, -height / 2.0),
        Vector2(width / 2.0, -height / 2.0),
    ]
    obstacles: list[list[Vector2]] = []
    initial_configs: list[InitialConfiguration] = []
    for x in (-width / 2.0 + 1.0, width / 2.0 - 1.0):
        for y in (-height / 2.0 + 1.0, height / 2.0 - 1.0):
            point = Vector2(x, y)
            initial_configs.append(InitialConfiguration(point, (point * -1.0).normalized()))

    if obstacle_type == 1:
        half_side = 0.5
        obstacles.append(
            [
                Vector2(-half_side, half_side),
                Vector2(-half_side, -half_side),
                Vector2(half_side, -half_side),
                Vector2(half_side, half_side),
            ]
        )
    elif obstacle_type == 2:
        half_side = 0.25
        offset = 1.5
        for ix in (-1, 1):
            for iy in (-1, 1):
                center_x = ix * offset
                center_y = iy * offset
                obstacles.append(
                    [
                        Vector2(center_x - half_side, center_y + half_side),
                        Vector2(center_x - half_side, center_y - half_side),
                        Vector2(center_x + half_side, center_y - half_side),
                        Vector2(center_x + half_side, center_y + half_side),
                    ]
                )
    return tracking, obstacles, initial_configs


def generate_square_tracking_space(
    side_length: float,
    obstacle_type: int = 0,
) -> tuple[list[Vector2], list[list[Vector2]], list[InitialConfiguration]]:
    return generate_rectangle_tracking_space(side_length, side_length, obstacle_type)


def generate_triangle_tracking_space(
    obstacle_type: int = 0,
    radius: float | None = None,
) -> tuple[list[Vector2], list[list[Vector2]], list[InitialConfiguration]]:
    radius = radius or math.sqrt(4 * TARGET_AREA / (3 * math.sqrt(3)))
    tracking = generate_polygon_tracking_space_points(3, radius)
    obstacles: list[list[Vector2]] = []

    initial_configs: list[InitialConfiguration] = []
    for point in [Vector2(0.0, radius - 5.0), Vector2(0.0, radius - 8.0)]:
        for index in range(3):
            new_point = _rotate(point, 120 * index)
            initial_configs.append(InitialConfiguration(new_point, (new_point * -1.0).normalized()))

    if obstacle_type == 1:
        h = math.sqrt(3)
        obstacles.append([Vector2(3.0, h), Vector2(-3.0, h), Vector2(0.0, -2.0 * h)])
    elif obstacle_type == 2:
        side_width = radius * math.sqrt(3)
        segment = side_width / 6.0
        height = segment * math.sqrt(3)
        rect_width = 0.5
        for index in range(4):
            center = Vector2(-index * segment / 2.0, height)
            for offset in range(index + 1):
                x = center.x + offset * segment
                y = center.y
                obstacles.append(
                    [
                        Vector2(x + rect_width / 2.0, y + rect_width / 2.0),
                        Vector2(x - rect_width / 2.0, y + rect_width / 2.0),
                        Vector2(x - rect_width / 2.0, y - rect_width / 2.0),
                        Vector2(x + rect_width / 2.0, y - rect_width / 2.0),
                    ]
                )
            height -= segment * math.sqrt(3) / 2.0
        initial_configs = []
        for point in [Vector2(0.0, height / 2.0), Vector2(-segment, height / 2.0), Vector2(segment, height / 2.0)]:
            for index in range(3):
                new_point = _rotate(point, 120 * index)
                initial_configs.append(InitialConfiguration(new_point, (new_point * -1.0).normalized()))
    return tracking, obstacles, initial_configs


def generate_trapezoid_tracking_space_points(w1: float = 10.0, w2: float = 6.0, height: float = 10.0) -> list[Vector2]:
    return [
        Vector2(-height / 2.0, (w1 + w2) / 4.0),
        Vector2(-height / 2.0, (w1 + w2) / 4.0 - w1),
        Vector2(height / 2.0, (w1 + w2) / 4.0 - w2),
        Vector2(height / 2.0, (w1 + w2) / 4.0),
    ]


def generate_trapezoid_tracking_space(
    obstacle_type: int = 0,
    w1: float | None = None,
    w2: float | None = None,
    height: float | None = None,
) -> tuple[list[Vector2], list[list[Vector2]], list[InitialConfiguration]]:
    if w1 is None or w2 is None or height is None:
        w2 = math.sqrt(TARGET_AREA / 3.0)
        w1 = 2.0 * w2
        height = w1
    tracking = generate_trapezoid_tracking_space_points(w1, w2, height)
    obstacles: list[list[Vector2]] = []
    initial_configs = [
        InitialConfiguration(Vector2(-height / 2.0 + 2.0, 5.0), Vector2(1.0, 0.0)),
        InitialConfiguration(Vector2(-height / 2.0 + 2.0, -5.0), Vector2(1.0, 0.0)),
        InitialConfiguration(Vector2(height / 2.0 - 2.0, 7.0), Vector2(0.0, -1.0)),
        InitialConfiguration(Vector2(height / 2.0 - 2.0, -2.0), Vector2(0.0, 1.0)),
    ]
    half_side = 2.0
    if obstacle_type == 1:
        obstacles.append(
            [
                Vector2(-height / 2.0, 1.0),
                Vector2(-height / 2.0, -2.0),
                Vector2(-height / 2.0 + 10.0, -2.0),
                Vector2(-height / 2.0 + 10.0, 1.0),
            ]
        )
    elif obstacle_type == 2:
        obstacles.append([Vector2(1.0, -1.0), Vector2(-1.0, -1.0), Vector2(-1.0, -3.0), Vector2(1.0, -3.0)])
        obstacles.append(
            [
                Vector2(-height / 4.0 + half_side, (w1 + w2) / 8.0 + half_side),
                Vector2(-height / 4.0 - half_side, (w1 + w2) / 8.0 + half_side),
                Vector2(-height / 4.0 - half_side, (w1 + w2) / 8.0 - half_side),
                Vector2(-height / 4.0 + half_side, (w1 + w2) / 8.0 - half_side),
            ]
        )
        obstacles.append(
            [
                Vector2(height / 4.0 + half_side, (w1 + w2) / 8.0 + half_side),
                Vector2(height / 4.0 - half_side, (w1 + w2) / 8.0 + half_side),
                Vector2(height / 4.0 - half_side, (w1 + w2) / 8.0 - half_side),
                Vector2(height / 4.0 + half_side, (w1 + w2) / 8.0 - half_side),
            ]
        )
        initial_configs = [
            InitialConfiguration(Vector2(-height / 2.0 + 2.0, 0.0), Vector2(1.0, 0.0)),
            InitialConfiguration(Vector2(height / 2.0 - 2.0, -1.0), Vector2(-1.0, 0.0)),
            InitialConfiguration(Vector2(-height / 2.0 + 2.0, -6.0), Vector2(-1.0, 0.0)),
            InitialConfiguration(Vector2(-height / 2.0 + 2.0, -3.0), Vector2(1.0, 0.0)),
        ]
    return tracking, obstacles, initial_configs


def generate_cross_tracking_space_points(w: float = 5.0, h: float = 10.0) -> list[Vector2]:
    points: list[Vector2] = []
    base = [Vector2(w / 2.0, w / 2.0), Vector2(w / 2.0, h + w / 2.0), Vector2(-w / 2.0, h + w / 2.0)]
    for index in range(4):
        for point in base:
            points.append(_rotate(point, -90 * index))
    return points


def generate_cross_tracking_space(
    obstacle_type: int = 0,
    w: float | None = None,
    h: float | None = None,
) -> tuple[list[Vector2], list[list[Vector2]], list[InitialConfiguration]]:
    if w is None or h is None:
        k = 2.0
        w = math.sqrt(TARGET_AREA / (4.0 * k + 1.0))
        h = k * w
    tracking = generate_cross_tracking_space_points(w, h)
    obstacles: list[list[Vector2]] = []
    base = Vector2(0.0, w / 2.0 + h - 2.0)
    initial_configs = [InitialConfiguration(_rotate(base, 90 * index), (_rotate(base, 90 * index) * -1.0).normalized()) for index in range(4)]
    half_side = 0.25
    half_side2 = 0.75
    width = 4.0
    if obstacle_type == 1:
        obstacles.append(
            [
                Vector2(-half_side, w / 2.0 + h - width),
                Vector2(-half_side, -w / 2.0 - h + width),
                Vector2(half_side, -w / 2.0 - h + width),
                Vector2(half_side, w / 2.0 + h - width),
            ]
        )
        initial_configs = [
            InitialConfiguration(Vector2(w / 2.0 + h - 2.0, 0.0), Vector2(-(w / 2.0 + h - 1.0), 0.0).normalized()),
            InitialConfiguration(Vector2(w / 2.0 + h - 5.0, 0.0), Vector2(-(w / 2.0 + h - 3.0), 0.0).normalized()),
            InitialConfiguration(Vector2(-w / 2.0 - h + 2.0, 0.0), Vector2(-(-w / 2.0 - h + 1.0), 0.0).normalized()),
            InitialConfiguration(Vector2(-w / 2.0 - h + 5.0, 0.0), Vector2(-(-w / 2.0 - h + 3.0), 0.0).normalized()),
        ]
    elif obstacle_type == 2:
        obstacle_positions = [Vector2(w / 2.0 - half_side2, w / 2.0 + half_side2), Vector2(-w / 2.0 + half_side2, w / 2.0 + half_side2)]
        for index in range(4):
            for obstacle_position in obstacle_positions:
                point = _rotate(obstacle_position, index * 90)
                obstacles.append(
                    [
                        Vector2(point.x + half_side2, point.y + half_side2),
                        Vector2(point.x - half_side2, point.y + half_side2),
                        Vector2(point.x - half_side2, point.y - half_side2),
                        Vector2(point.x + half_side2, point.y - half_side2),
                    ]
                )
    return tracking, obstacles, initial_configs


def generate_l_shape_tracking_space_points(w1: float = 5.0, w2: float = 5.0) -> list[Vector2]:
    return [
        Vector2(w1 / 2.0, w1 / 2.0),
        Vector2(w1 / 2.0, w1 / 2.0 + w2),
        Vector2(-w1 / 2.0, w1 / 2.0 + w2),
        Vector2(-w1 / 2.0, -w1 / 2.0),
        Vector2(w1 / 2.0 + w2, -w1 / 2.0),
        Vector2(w1 / 2.0 + w2, w1 / 2.0),
    ]


def generate_l_shape_tracking_space(
    obstacle_type: int = 0,
    w1: float | None = None,
    w2: float | None = None,
) -> tuple[list[Vector2], list[list[Vector2]], list[InitialConfiguration]]:
    if w1 is None or w2 is None:
        k = 1.0
        w1 = math.sqrt(TARGET_AREA / (2.0 * k + 1.0))
        w2 = k * w1
    tracking = generate_l_shape_tracking_space_points(w1, w2)
    obstacles: list[list[Vector2]] = []
    half_side = 1.0
    dist = 2.0
    initial_configs = [
        InitialConfiguration(Vector2(w1 / 2.0 + w2 - dist, half_side + dist), Vector2(-1.0, 0.0)),
        InitialConfiguration(Vector2(w1 / 2.0 + w2 - dist, -half_side - dist), Vector2(-1.0, 0.0)),
        InitialConfiguration(Vector2(-half_side - dist, w1 / 2.0 + w2 - dist), Vector2(0.0, -1.0)),
        InitialConfiguration(Vector2(half_side + dist, w1 / 2.0 + w2 - dist), Vector2(0.0, -1.0)),
    ]
    if obstacle_type == 1:
        obstacles.append([Vector2(2.0, 2.0), Vector2(-2.0, 2.0), Vector2(-2.0, -2.0), Vector2(2.0, -2.0)])
    elif obstacle_type == 2:
        obstacles.append(
            [
                Vector2(half_side, w1 / 2.0 + 1.0),
                Vector2(half_side, w1 / 2.0 + w2 - 3.0),
                Vector2(-half_side, w1 / 2.0 + w2 - 3.0),
                Vector2(-half_side, w1 / 2.0 + 1.0),
            ]
        )
        obstacles.append(
            [
                Vector2(w1 / 2.0 + 1.0, half_side),
                Vector2(w1 / 2.0 + 1.0, -half_side),
                Vector2(w1 / 2.0 + w2 - 3.0, -half_side),
                Vector2(w1 / 2.0 + w2 - 3.0, half_side),
            ]
        )
    return tracking, obstacles, initial_configs


def generate_t_shape_tracking_space_points(w1: float = 4.0, w2: float = 2.0, w3: float = 8.0) -> list[Vector2]:
    return [
        Vector2(w1 / 2.0 + w2, w1 / 2.0),
        Vector2(-w1 / 2.0 - w2, w1 / 2.0),
        Vector2(-w1 / 2.0 - w2, -w1 / 2.0),
        Vector2(-w1 / 2.0, -w1 / 2.0),
        Vector2(-w1 / 2.0, -w1 / 2.0 - w3),
        Vector2(w1 / 2.0, -w1 / 2.0 - w3),
        Vector2(w1 / 2.0, -w1 / 2.0),
        Vector2(w1 / 2.0 + w2, -w1 / 2.0),
    ]


def generate_t_shape_tracking_space(
    obstacle_type: int = 0,
    w1: float | None = None,
    w2: float | None = None,
    w3: float | None = None,
) -> tuple[list[Vector2], list[list[Vector2]], list[InitialConfiguration]]:
    if w1 is None or w2 is None or w3 is None:
        k = 3.0 / 4.0
        c = 3.0 / 2.0
        w1 = math.sqrt(TARGET_AREA / (1.0 + 2.0 * k + c))
        w2 = k * w1
        w3 = c * w1
    tracking = generate_t_shape_tracking_space_points(w1, w2, w3)
    obstacles: list[list[Vector2]] = []
    half_side = 1.0
    initial_configs = [
        InitialConfiguration(Vector2(w1 / 2.0 + w2 - 3.0, 0.0), Vector2(-1.0, 0.0)),
        InitialConfiguration(Vector2(-w1 / 2.0 - w2 + 3.0, 0.0), Vector2(1.0, 0.0)),
        InitialConfiguration(Vector2(0.0, -w1 / 2.0 - w3 + 3.0), Vector2(0.0, 1.0)),
        InitialConfiguration(Vector2(0.0, -w1 / 2.0 - w3 + 6.0), Vector2(0.0, 1.0)),
    ]
    if obstacle_type == 1:
        obstacles.append([Vector2(2.0, 2.0), Vector2(-2.0, 2.0), Vector2(-2.0, -2.0), Vector2(2.0, -2.0)])
    elif obstacle_type == 2:
        for center in [Vector2(0.0, -w1 / 2.0 - w3 + 4.0), Vector2(-w1 / 2.0 - w2 + 4.0, 0.0), Vector2(w1 / 2.0 + w2 - 4.0, 0.0)]:
            obstacles.append(
                [
                    Vector2(center.x + half_side, center.y + half_side),
                    Vector2(center.x - half_side, center.y + half_side),
                    Vector2(center.x - half_side, center.y - half_side),
                    Vector2(center.x + half_side, center.y - half_side),
                ]
            )
        initial_configs = [
            InitialConfiguration(Vector2(w1 / 2.0 + w2 - 8.0, 0.0), Vector2(-1.0, 0.0)),
            InitialConfiguration(Vector2(-w1 / 2.0 - w2 + 8.0, 0.0), Vector2(1.0, 0.0)),
            InitialConfiguration(Vector2(0.0, -w1 / 2.0 - w3 + 7.0), Vector2(0.0, 1.0)),
            InitialConfiguration(Vector2(0.0, -w1 / 2.0 - w3 + 10.0), Vector2(0.0, 1.0)),
        ]
    return tracking, obstacles, initial_configs


def load_tracking_space_from_file(path: str | Path) -> tuple[list[Vector2], list[list[Vector2]]]:
    content = Path(path).read_text(encoding="utf-8").splitlines()
    tracking_space: list[Vector2] = []
    obstacles: list[list[Vector2]] = []
    current_polygon: list[Vector2] = []
    parsing_obstacles = False
    for raw_line in content:
        line = raw_line.strip()
        if not line:
            if parsing_obstacles and len(current_polygon) > 2:
                obstacles.append(current_polygon)
            current_polygon = []
            parsing_obstacles = True
            continue
        split = [part.strip() for part in line.split(",")]
        if len(split) != 2:
            raise ValueError(f"Invalid tracking-space line: {raw_line!r}")
        point = Vector2(float(split[0]), float(split[1]))
        if not parsing_obstacles:
            tracking_space.append(point)
        else:
            current_polygon.append(point)
    if parsing_obstacles and len(current_polygon) > 2:
        obstacles.append(current_polygon)
    return tracking_space, obstacles
