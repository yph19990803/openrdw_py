from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .geometry import Vector2, heading_to_vector, polygon_centroid


def _heading_delta_deg(current: float, previous: float) -> float:
    return ((current - previous + 540.0) % 360.0) - 180.0


@dataclass
class Pose2D:
    position: Vector2
    heading_deg: float

    @property
    def forward(self) -> Vector2:
        return heading_to_vector(self.heading_deg)


@dataclass
class GainsConfig:
    max_trans_gain: float = 0.26
    min_trans_gain: float = -0.14
    max_rot_gain: float = 0.49
    min_rot_gain: float = -0.20
    curvature_radius: float = 7.5
    body_collider_diameter: float = 0.1
    physical_space_buffer: float = 0.4
    obstacle_buffer: float = 0.4
    reset_trigger_buffer: float | None = None
    translation_speed: float = 1.0
    rotation_speed: float = 90.0
    time_step: float = 1.0 / 60.0

    def __post_init__(self) -> None:
        # Backward compatibility for older configs and command files that only
        # specify a single reset trigger buffer.
        if self.reset_trigger_buffer is not None:
            self.physical_space_buffer = self.reset_trigger_buffer
            self.obstacle_buffer = self.reset_trigger_buffer


@dataclass
class InitialConfiguration:
    position: Vector2
    forward: Vector2


@dataclass
class Environment:
    tracking_space: list[Vector2]
    obstacles: list[list[Vector2]] = field(default_factory=list)
    virtual_obstacles: list[list[Vector2]] = field(default_factory=list)
    physical_targets: list[Vector2] = field(default_factory=list)
    physical_target_forwards: list[Vector2] = field(default_factory=list)
    shape: str = "rectangle"

    @property
    def center(self) -> Vector2:
        return polygon_centroid(self.tracking_space)

    @property
    def all_polygons(self) -> list[list[Vector2]]:
        return [self.tracking_space, *self.obstacles]

    @property
    def all_virtual_polygons(self) -> list[list[Vector2]]:
        if self.virtual_obstacles:
            return self.virtual_obstacles
        return [self.tracking_space, *self.obstacles]


@dataclass
class RedirectCommand:
    translation: Vector2 = field(default_factory=lambda: Vector2(0.0, 0.0))
    rotation_deg: float = 0.0
    curvature_deg: float = 0.0
    priority: float = 0.0
    translation_gain: float = 0.0
    rotation_gain: float = 0.0
    curvature_gain: float = 0.0
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResetCommand:
    plane_rotation_deg: float = 0.0
    user_rotation_deg: float = 0.0
    finished: bool = False
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    virtual_pose: Pose2D
    physical_pose: Pose2D
    root_pose: Pose2D | None = None
    tracking_space_pose: Pose2D | None = None
    tracking_space_local_position: Vector2 = field(default_factory=lambda: Vector2(0.0, 0.0))
    tracking_space_local_heading_deg: float = 0.0
    agent_index: int = 0
    prev_virtual_pose: Pose2D | None = None
    prev_physical_pose: Pose2D | None = None
    prev_root_pose: Pose2D | None = None
    observed_virtual_pose: Pose2D | None = None
    observed_physical_pose: Pose2D | None = None
    observed_root_pose: Pose2D | None = None
    observed_tracking_space_pose: Pose2D | None = None
    current_waypoint: int = 0
    in_reset: bool = False
    mission_complete: bool = False
    if_just_end_reset: bool = False
    redirection_time: float = 0.0
    walk_distance: float = 0.0
    priority: float = 0.0
    total_force: Vector2 = field(default_factory=lambda: Vector2(0.0, 0.0))
    active_waypoint: Vector2 | None = None
    final_waypoint: Vector2 | None = None
    base_delta_translation: Vector2 = field(default_factory=lambda: Vector2(0.0, 0.0))
    physical_delta_translation: Vector2 = field(default_factory=lambda: Vector2(0.0, 0.0))
    base_delta_rotation_deg: float = 0.0
    sampling_time_accumulator: float = 0.0
    waypoint_time_accumulator: float = 0.0
    zigzag_target_index: int = 1
    manual_translation: Vector2 = field(default_factory=lambda: Vector2(0.0, 0.0))
    manual_rotation_deg: float = 0.0
    same_pos_time: float = 0.0
    last_command: RedirectCommand = field(default_factory=RedirectCommand)
    last_reset_command: ResetCommand = field(default_factory=ResetCommand)

    @property
    def virtual_position(self) -> Vector2:
        return self.virtual_pose.position

    @property
    def physical_position(self) -> Vector2:
        return self.physical_pose.position

    @property
    def virtual_forward(self) -> Vector2:
        return self.virtual_pose.forward

    @property
    def physical_forward(self) -> Vector2:
        return self.physical_pose.forward

    @property
    def root_forward(self) -> Vector2:
        return self.root_pose.forward if self.root_pose is not None else self.virtual_pose.forward

    @property
    def curr_pos(self) -> Vector2:
        if self.observed_virtual_pose is not None:
            return self.observed_virtual_pose.position
        return self.virtual_pose.position

    @property
    def curr_pos_real(self) -> Vector2:
        if self.observed_physical_pose is not None:
            return self.observed_physical_pose.position
        return self.physical_pose.position

    @property
    def prev_pos(self) -> Vector2:
        if self.prev_virtual_pose is None:
            return self.virtual_pose.position
        return self.prev_virtual_pose.position

    @property
    def prev_pos_real(self) -> Vector2:
        if self.prev_physical_pose is None:
            return self.physical_pose.position
        return self.prev_physical_pose.position

    @property
    def curr_dir(self) -> Vector2:
        if self.observed_virtual_pose is not None:
            return self.observed_virtual_pose.forward
        return self.virtual_pose.forward

    @property
    def curr_dir_real(self) -> Vector2:
        if self.observed_physical_pose is not None:
            return self.observed_physical_pose.forward
        return self.physical_pose.forward

    @property
    def prev_dir(self) -> Vector2:
        if self.prev_virtual_pose is None:
            return self.virtual_pose.forward
        return self.prev_virtual_pose.forward

    @property
    def prev_dir_real(self) -> Vector2:
        if self.prev_physical_pose is None:
            return self.physical_pose.forward
        return self.prev_physical_pose.forward

    @property
    def delta_virtual_translation(self) -> Vector2:
        if self.prev_virtual_pose is None:
            return self.base_delta_translation
        return self.virtual_pose.position - self.prev_virtual_pose.position

    @property
    def delta_virtual_rotation_deg(self) -> float:
        if self.prev_virtual_pose is None:
            return self.base_delta_rotation_deg
        return _heading_delta_deg(self.virtual_pose.heading_deg, self.prev_virtual_pose.heading_deg)

    @property
    def delta_physical_translation(self) -> Vector2:
        if self.prev_physical_pose is None:
            return Vector2(0.0, 0.0)
        return self.physical_pose.position - self.prev_physical_pose.position

    @property
    def delta_physical_rotation_deg(self) -> float:
        if self.prev_physical_pose is None:
            return 0.0
        return _heading_delta_deg(self.physical_pose.heading_deg, self.prev_physical_pose.heading_deg)

    @property
    def delta_pos(self) -> Vector2:
        return self.base_delta_translation

    @property
    def delta_dir(self) -> float:
        return self.base_delta_rotation_deg


@dataclass
class ExperimentSetup:
    environment: Environment
    waypoints: list[Vector2]
    initial_configuration: InitialConfiguration
    redirector: str
    resetter: str
    sampling_intervals: list[float] | None = None


@dataclass
class StepTrace:
    step_index: int
    simulation_time_s: float
    virtual_x: float
    virtual_y: float
    virtual_heading_deg: float
    physical_x: float
    physical_y: float
    physical_heading_deg: float
    in_reset: bool
    translation_injection_x: float
    translation_injection_y: float
    rotation_injection_deg: float
    curvature_injection_deg: float
    translation_gain: float = 0.0
    rotation_gain: float = 0.0
    curvature_gain: float = 0.0
    total_force_x: float = 0.0
    total_force_y: float = 0.0
    priority: float = 0.0
    observed_virtual_x: float | None = None
    observed_virtual_y: float | None = None
    observed_virtual_heading_deg: float | None = None
    observed_physical_x: float | None = None
    observed_physical_y: float | None = None
    observed_physical_heading_deg: float | None = None
    observed_tracking_x: float | None = None
    observed_tracking_y: float | None = None
    observed_tracking_heading_deg: float | None = None
