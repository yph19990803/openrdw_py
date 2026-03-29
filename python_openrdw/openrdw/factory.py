from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

from .geometry import Vector2, signed_angle, vector_to_heading
from .models import AgentState, Environment, GainsConfig, InitialConfiguration, Pose2D
from .paths import (
    PathSeed,
    generate_circle_path,
    generate_initial_path_by_seed,
    load_sampling_intervals_from_file,
    load_waypoints_from_file,
)
from .redirectors import (
    DeepLearningRedirector,
    DynamicApfRedirector,
    MessingerApfRedirector,
    NullRedirector,
    PassiveHapticApfRedirector,
    S2CRedirector,
    S2ORedirector,
    ThomasApfRedirector,
    VisPolyRedirector,
    ZigZagRedirector,
)
from .resetters import ApfResetter, NullResetter, TwoOneTurnResetter
from .scheduler import MultiAgentScheduler, ScheduledAgent
from .tracking import (
    generate_cross_tracking_space,
    generate_l_shape_tracking_space,
    generate_rectangle_tracking_space,
    generate_square_tracking_space,
    generate_t_shape_tracking_space,
    generate_trapezoid_tracking_space,
    generate_triangle_tracking_space,
    load_tracking_space_from_file,
)


REDIRECTOR_OPTIONS = {
    "none": NullRedirector,
    "s2c": S2CRedirector,
    "s2o": S2ORedirector,
    "zigzag": ZigZagRedirector,
    "thomas_apf": ThomasApfRedirector,
    "messinger_apf": MessingerApfRedirector,
    "dynamic_apf": DynamicApfRedirector,
    "deep_learning": DeepLearningRedirector,
    "passive_haptic_apf": PassiveHapticApfRedirector,
    "vispoly": VisPolyRedirector,
}

RESETTER_OPTIONS = {
    "none": NullResetter,
    "two_one_turn": TwoOneTurnResetter,
    "apf": ApfResetter,
}

PATH_OPTIONS = {
    "ninety_turn": PathSeed.ninety_turn,
    "random_turn": PathSeed.random_turn,
    "straight_line": PathSeed.straight_line,
    "sawtooth": PathSeed.sawtooth,
    "circle": None,
    "figure_eight": None,
    "file_path": None,
    "real_user_path": None,
}

TRACKING_SPACE_OPTIONS = {
    "rectangle": generate_rectangle_tracking_space,
    "square": generate_square_tracking_space,
    "triangle": generate_triangle_tracking_space,
    "trapezoid": generate_trapezoid_tracking_space,
    "cross": generate_cross_tracking_space,
    "l_shape": generate_l_shape_tracking_space,
    "t_shape": generate_t_shape_tracking_space,
}


@dataclass(frozen=True)
class SimulationConfig:
    redirector: str = "s2c"
    resetter: str = "two_one_turn"
    path_mode: str = "random_turn"
    tracking_space_shape: str = "rectangle"
    movement_controller: str = "autopilot"
    tracking_space_file: str | None = None
    waypoints_file: str | None = None
    sampling_intervals_file: str | None = None
    first_waypoint_is_start_point: bool = True
    align_to_initial_forward: bool = True
    obstacle_type: int = 0
    physical_width: float = 5.0
    physical_height: float = 5.0
    virtual_width: float = 20.0
    virtual_height: float = 20.0
    physical_obstacle_count: int = 0
    virtual_obstacle_count: int = 0
    physical_obstacle_specs: tuple[dict, ...] | None = None
    agent_count: int = 1
    total_path_length: float = 30.0
    time_step: float = 1.0 / 60.0
    translation_speed: float = 1.0
    rotation_speed: float = 90.0
    max_trans_gain: float = 0.26
    min_trans_gain: float = -0.14
    max_rot_gain: float = 0.49
    min_rot_gain: float = -0.20
    curvature_radius: float = 7.5
    body_collider_diameter: float = 0.1
    physical_space_buffer: float = 0.4
    obstacle_buffer: float = 0.4
    reset_trigger_buffer: float | None = None
    physical_targets: tuple[Vector2, ...] | None = None
    physical_target_forwards: tuple[Vector2, ...] | None = None
    sampling_frequency: float = 10.0
    use_custom_sampling_frequency: bool = False
    draw_real_trail: bool = True
    draw_virtual_trail: bool = True
    trail_visual_time: float = -1.0
    virtual_world_visible: bool = True
    tracking_space_visible: bool = True
    buffer_visible: bool = True
    seed: int = 3041


def build_redirector(name: str):
    try:
        return REDIRECTOR_OPTIONS[name.lower()]()
    except KeyError as exc:
        raise ValueError(f"Unsupported redirector: {name}") from exc


def build_resetter(name: str):
    try:
        return RESETTER_OPTIONS[name.lower()]()
    except KeyError as exc:
        raise ValueError(f"Unsupported resetter: {name}") from exc


def build_waypoints(config: SimulationConfig) -> tuple[list[Vector2], list[float] | None]:
    rng = random.Random(config.seed)
    if config.path_mode == "circle":
        radius = config.total_path_length / 2.0 / 3.141592653589793
        return generate_circle_path(radius=radius, waypoint_count=20), None
    if config.path_mode == "figure_eight":
        radius = config.total_path_length / 2.0 / 3.141592653589793 / 2.0
        return generate_circle_path(radius=radius, waypoint_count=20, figure_eight=True), None
    if config.path_mode == "file_path":
        if not config.waypoints_file:
            raise ValueError("file_path mode requires waypoints_file")
        return load_waypoints_from_file(config.waypoints_file, config.first_waypoint_is_start_point), None
    if config.path_mode == "real_user_path":
        if not config.waypoints_file or not config.sampling_intervals_file:
            raise ValueError("real_user_path mode requires both waypoints_file and sampling_intervals_file")
        return (
            load_waypoints_from_file(config.waypoints_file, config.first_waypoint_is_start_point),
            load_sampling_intervals_from_file(config.sampling_intervals_file),
        )
    try:
        seed_factory = PATH_OPTIONS[config.path_mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported path mode: {config.path_mode}") from exc
    if seed_factory is None:
        raise ValueError(f"Path mode {config.path_mode} is not configured correctly")
    return generate_initial_path_by_seed(seed_factory(), target_distance=config.total_path_length, rng=rng), None


def align_waypoints_to_initial_configuration(
    waypoints: list[Vector2],
    initial_position: Vector2,
    initial_forward: Vector2,
    first_waypoint_is_start_point: bool = True,
    align_to_initial_forward: bool = True,
) -> list[Vector2]:
    if not waypoints:
        return []
    delta_position = initial_position - Vector2(0.0, 0.0)
    aligned = [point + delta_position for point in waypoints]
    if not align_to_initial_forward:
        return aligned
    if len(aligned) < 2:
        return aligned
    if first_waypoint_is_start_point:
        virtual_dir = aligned[1] - initial_position
    else:
        virtual_dir = aligned[0] - initial_position
    rotation_angle = -signed_angle(virtual_dir, initial_forward)
    return [initial_position + (point - initial_position).rotate(rotation_angle) for point in aligned]


def generate_random_rect_obstacles(
    count: int,
    width: float,
    height: float,
    rng: random.Random,
) -> list[list[Vector2]]:
    obstacles: list[list[Vector2]] = []
    if count <= 0:
        return obstacles
    max_half_w = max(0.5, width / 10.0)
    max_half_h = max(0.5, height / 10.0)
    for _ in range(count):
        half_w = rng.uniform(0.5, max_half_w)
        half_h = rng.uniform(0.5, max_half_h)
        center_x = rng.uniform(-width / 2.0 + half_w + 0.5, width / 2.0 - half_w - 0.5)
        center_y = rng.uniform(-height / 2.0 + half_h + 0.5, height / 2.0 - half_h - 0.5)
        if abs(center_x) < 1.5 and abs(center_y) < 1.5:
            center_x += 2.0
            center_y += 2.0
        obstacles.append(
            [
                Vector2(center_x - half_w, center_y - half_h),
                Vector2(center_x - half_w, center_y + half_h),
                Vector2(center_x + half_w, center_y + half_h),
                Vector2(center_x + half_w, center_y - half_h),
            ]
        )
    return obstacles


def build_custom_obstacle_polygon(spec: dict) -> list[Vector2]:
    shape = str(spec.get("shape", "square")).lower()
    center = Vector2(float(spec.get("x", 0.0)), float(spec.get("y", 0.0)))
    if shape == "square":
        size = max(float(spec.get("size", 1.0)), 0.01)
        half = size / 2.0
        return [
            Vector2(center.x - half, center.y - half),
            Vector2(center.x - half, center.y + half),
            Vector2(center.x + half, center.y + half),
            Vector2(center.x + half, center.y - half),
        ]
    if shape == "rectangle":
        width = max(float(spec.get("width", 1.0)), 0.01)
        height = max(float(spec.get("height", 1.0)), 0.01)
        half_w = width / 2.0
        half_h = height / 2.0
        return [
            Vector2(center.x - half_w, center.y - half_h),
            Vector2(center.x - half_w, center.y + half_h),
            Vector2(center.x + half_w, center.y + half_h),
            Vector2(center.x + half_w, center.y - half_h),
        ]
    if shape == "triangle":
        width = max(float(spec.get("width", 1.0)), 0.01)
        height = max(float(spec.get("height", 1.0)), 0.01)
        return [
            Vector2(center.x, center.y + height / 2.0),
            Vector2(center.x - width / 2.0, center.y - height / 2.0),
            Vector2(center.x + width / 2.0, center.y - height / 2.0),
        ]
    if shape == "circle":
        radius = max(float(spec.get("radius", 0.5)), 0.01)
        segments = 24
        return [
            Vector2(
                center.x + radius * math.cos(2.0 * math.pi * idx / segments),
                center.y + radius * math.sin(2.0 * math.pi * idx / segments),
            )
            for idx in range(segments)
        ]
    raise ValueError(f"Unsupported obstacle shape: {shape}")


def _build_tracking_space(config: SimulationConfig) -> tuple[list[Vector2], list[list[Vector2]], list]:
    if config.tracking_space_shape == "file_path":
        if not config.tracking_space_file:
            raise ValueError("tracking_space_shape=file_path requires tracking_space_file")
        tracking, obstacles = load_tracking_space_from_file(config.tracking_space_file)
        default_forward = (Vector2(0.0, 0.0) - tracking[0]).normalized() if tracking else Vector2(0.0, 1.0)
        initials = [InitialConfiguration(tracking[0] if tracking else Vector2(0.0, 0.0), default_forward)]
        return tracking, obstacles, initials
    if config.tracking_space_shape == "rectangle":
        return generate_rectangle_tracking_space(config.physical_width, config.physical_height, config.obstacle_type)
    if config.tracking_space_shape == "square":
        return generate_square_tracking_space(config.physical_width, config.obstacle_type)
    generator = TRACKING_SPACE_OPTIONS[config.tracking_space_shape]
    return generator(config.obstacle_type)


def build_environment(config: SimulationConfig) -> Environment:
    tracking, shape_obstacles, _ = _build_tracking_space(config)
    rng = random.Random(config.seed)
    if config.physical_obstacle_specs is not None:
        physical_obstacles = [build_custom_obstacle_polygon(spec) for spec in config.physical_obstacle_specs]
    else:
        physical_obstacles = list(shape_obstacles)
        if config.physical_obstacle_count > len(physical_obstacles):
            xs = [point.x for point in tracking]
            ys = [point.y for point in tracking]
            physical_obstacles.extend(
                generate_random_rect_obstacles(
                    config.physical_obstacle_count - len(physical_obstacles),
                    max(xs) - min(xs),
                    max(ys) - min(ys),
                    rng,
                )
            )
        elif config.physical_obstacle_count < len(physical_obstacles):
            physical_obstacles = physical_obstacles[: config.physical_obstacle_count]

    virtual_boundary = [
        Vector2(config.virtual_width / 2.0, config.virtual_height / 2.0),
        Vector2(-config.virtual_width / 2.0, config.virtual_height / 2.0),
        Vector2(-config.virtual_width / 2.0, -config.virtual_height / 2.0),
        Vector2(config.virtual_width / 2.0, -config.virtual_height / 2.0),
    ]
    virtual_obstacles = [virtual_boundary]
    virtual_obstacles.extend(
        generate_random_rect_obstacles(
            config.virtual_obstacle_count,
            config.virtual_width,
            config.virtual_height,
            random.Random(config.seed + 101),
        )
    )
    physical_targets = list(config.physical_targets) if config.physical_targets is not None else [
        Vector2(0.0, 0.0),
        Vector2(1.0, 0.0),
        Vector2(0.0, 10.0),
        Vector2(100.0, 0.0),
    ]
    physical_target_forwards = list(config.physical_target_forwards) if config.physical_target_forwards is not None else [
        Vector2(0.0, 1.0),
        Vector2(1.0, 1.0).normalized(),
        Vector2(1.0, 0.0),
        Vector2(0.0, 1.0),
    ]
    return Environment(
        tracking_space=tracking,
        obstacles=physical_obstacles,
        virtual_obstacles=virtual_obstacles,
        physical_targets=physical_targets,
        physical_target_forwards=physical_target_forwards,
        shape=config.tracking_space_shape,
    )


def build_gains(config: SimulationConfig) -> GainsConfig:
    return GainsConfig(
        max_trans_gain=config.max_trans_gain,
        min_trans_gain=config.min_trans_gain,
        max_rot_gain=config.max_rot_gain,
        min_rot_gain=config.min_rot_gain,
        curvature_radius=config.curvature_radius,
        body_collider_diameter=config.body_collider_diameter,
        physical_space_buffer=config.physical_space_buffer,
        obstacle_buffer=config.obstacle_buffer,
        reset_trigger_buffer=config.reset_trigger_buffer,
        time_step=config.time_step,
        translation_speed=config.translation_speed,
        rotation_speed=config.rotation_speed,
    )


def build_scheduler(config: SimulationConfig) -> MultiAgentScheduler:
    environment = build_environment(config)
    gains = build_gains(config)
    base_waypoints, sampling_intervals = build_waypoints(config)
    _, _, initial_configs = _build_tracking_space(config)
    agents: list[ScheduledAgent] = []
    for agent_index in range(config.agent_count):
        initial = initial_configs[agent_index % len(initial_configs)]
        heading = vector_to_heading(initial.forward)
        pose = Pose2D(initial.position, heading)
        state = AgentState(
            virtual_pose=pose,
            physical_pose=pose,
            agent_index=agent_index,
            if_just_end_reset=True,
            active_waypoint=base_waypoints[0] if base_waypoints else None,
            final_waypoint=base_waypoints[-1] if base_waypoints else None,
        )
        aligned_waypoints = align_waypoints_to_initial_configuration(
            base_waypoints,
            initial.position,
            initial.forward,
            first_waypoint_is_start_point=config.first_waypoint_is_start_point,
            align_to_initial_forward=config.align_to_initial_forward,
        )
        if aligned_waypoints:
            state.active_waypoint = aligned_waypoints[0]
            state.final_waypoint = aligned_waypoints[-1]
        agents.append(
            ScheduledAgent(
                agent_id=str(agent_index),
                state=state,
                environment=environment,
                gains=gains,
                redirector=build_redirector(config.redirector),
                resetter=build_resetter(config.resetter),
                waypoints=aligned_waypoints,
                sampling_intervals=sampling_intervals,
            )
        )
    return MultiAgentScheduler(agents)
