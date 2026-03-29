from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .geometry import Vector2, vector_to_heading
from .models import AgentState, Environment, GainsConfig, ResetCommand


def _same_agent(lhs: AgentState, rhs: AgentState) -> bool:
    lhs_index = getattr(lhs, "agent_index", None)
    rhs_index = getattr(rhs, "agent_index", None)
    if lhs_index is not None and rhs_index is not None:
        return lhs_index == rhs_index
    return lhs is rhs


class Resetter(Protocol):
    def is_reset_required(self, state: AgentState, environment: Environment, gains: GainsConfig, other_agents: list[AgentState]) -> bool:
        ...

    def begin(self, state: AgentState, environment: Environment, gains: GainsConfig) -> None:
        ...

    def simulated_walker_update(self, state: AgentState, environment: Environment, gains: GainsConfig) -> float:
        ...

    def inject_resetting(self, state: AgentState, environment: Environment, gains: GainsConfig, delta_rotation_deg: float) -> ResetCommand:
        ...

    def step(self, state: AgentState, environment: Environment, gains: GainsConfig) -> ResetCommand:
        ...


class NullResetter:
    def is_reset_required(self, state: AgentState, environment: Environment, gains: GainsConfig, other_agents: list[AgentState]) -> bool:
        return False

    def begin(self, state: AgentState, environment: Environment, gains: GainsConfig) -> None:
        return None

    def simulated_walker_update(self, state: AgentState, environment: Environment, gains: GainsConfig) -> float:
        return 0.0

    def inject_resetting(self, state: AgentState, environment: Environment, gains: GainsConfig, delta_rotation_deg: float) -> ResetCommand:
        return ResetCommand(finished=True)

    def step(self, state: AgentState, environment: Environment, gains: GainsConfig) -> ResetCommand:
        return ResetCommand(finished=True)


def collision_happens(state: AgentState, environment: Environment, gains: GainsConfig, other_agents: list[AgentState]) -> bool:
    real_pos = state.curr_pos_real
    real_dir = state.curr_dir_real

    if _collide_with_polygons(real_pos, real_dir, [environment.tracking_space], gains.physical_space_buffer):
        return True
    if _collide_with_polygons(real_pos, real_dir, environment.obstacles, gains.obstacle_buffer):
        return True

    for other in other_agents:
        if _same_agent(other, state):
            continue
        if collide_with_point(real_pos, real_dir, other.curr_pos_real, gains.physical_space_buffer):
            return True
    return False


def _collide_with_polygons(real_pos: Vector2, real_dir: Vector2, polygons: list[list[Vector2]], buffer_size: float) -> bool:
    for polygon in polygons:
        for index, start in enumerate(polygon):
            end = polygon[(index + 1) % len(polygon)]
            if collide_with_point(real_pos, real_dir, start, buffer_size):
                return True
            edge = end - start
            if edge.magnitude <= 1e-6:
                continue
            if abs(edge.cross(real_pos - start)) / edge.magnitude <= buffer_size:
                if edge.dot(real_pos - start) >= 0 and (start - end).dot(real_pos - end) >= 0:
                    if abs(edge.cross(real_dir)) > 1e-3 and (1 if edge.cross(real_dir) > 0 else -1) != (1 if edge.cross(real_pos - start) > 0 else -1):
                        return True
    return False


def collide_with_point(real_pos: Vector2, real_dir: Vector2, obstacle: Vector2, buffer_size: float) -> bool:
    direction = obstacle - real_pos
    if direction.magnitude <= buffer_size:
        return direction.angle_to(real_dir) < 89.0
    return False


@dataclass
class TwoOneTurnResetter:
    overall_injected_rotation_deg: float = 0.0
    remaining_plane_rotation_deg: float = 0.0
    remaining_user_rotation_deg: float = 0.0

    def is_reset_required(self, state: AgentState, environment: Environment, gains: GainsConfig, other_agents: list[AgentState]) -> bool:
        return collision_happens(state, environment, gains, other_agents)

    def begin(self, state: AgentState, environment: Environment, gains: GainsConfig) -> None:
        self.overall_injected_rotation_deg = 0.0
        self.remaining_plane_rotation_deg = 180.0
        self.remaining_user_rotation_deg = 180.0

    def simulated_walker_update(self, state: AgentState, environment: Environment, gains: GainsConfig) -> float:
        rotate_amount = min(gains.rotation_speed * gains.time_step, self.remaining_user_rotation_deg)
        self.remaining_user_rotation_deg -= rotate_amount
        return rotate_amount

    def inject_resetting(self, state: AgentState, environment: Environment, gains: GainsConfig, delta_rotation_deg: float) -> ResetCommand:
        if abs(self.overall_injected_rotation_deg) >= 180.0:
            return ResetCommand(finished=True)

        remaining_rotation = 180.0 - self.overall_injected_rotation_deg if delta_rotation_deg > 0 else -180.0 - self.overall_injected_rotation_deg
        if abs(remaining_rotation) < abs(delta_rotation_deg) or self.remaining_user_rotation_deg <= 1e-6:
            plane_rotation = remaining_rotation
            self.overall_injected_rotation_deg += remaining_rotation
            self.remaining_plane_rotation_deg = 0.0
            finished = True
        else:
            plane_rotation = delta_rotation_deg
            self.overall_injected_rotation_deg += delta_rotation_deg
            self.remaining_plane_rotation_deg = max(0.0, 180.0 - abs(self.overall_injected_rotation_deg))
            finished = False
        return ResetCommand(
            plane_rotation_deg=plane_rotation,
            user_rotation_deg=delta_rotation_deg,
            finished=finished,
        )

    def step(self, state: AgentState, environment: Environment, gains: GainsConfig) -> ResetCommand:
        user_rotation = self.simulated_walker_update(state, environment, gains)
        return self.inject_resetting(state, environment, gains, user_rotation)


@dataclass
class ApfResetter:
    required_rotate_steer_angle: float = 0.0
    required_rotate_angle: float = 0.0
    rotate_dir: float = 1.0
    speed_ratio: float = 1.0

    def is_reset_required(self, state: AgentState, environment: Environment, gains: GainsConfig, other_agents: list[AgentState]) -> bool:
        return collision_happens(state, environment, gains, other_agents)

    def begin(self, state: AgentState, environment: Environment, gains: GainsConfig) -> None:
        total_force = state.total_force
        if total_force.magnitude <= 1e-6:
            total_force = (environment.center - state.curr_pos_real).normalized()
        current_heading = state.physical_pose.heading_deg % 360.0
        target_heading = vector_to_heading(total_force) % 360.0
        shortest_heading_delta = ((target_heading - current_heading + 540.0) % 360.0) - 180.0
        self.rotate_dir = -1.0 if shortest_heading_delta > 0 else 1.0
        self.required_rotate_steer_angle = abs(shortest_heading_delta)
        self.required_rotate_angle = 360.0 - self.required_rotate_steer_angle if self.required_rotate_steer_angle > 1e-6 else 0.0
        self.speed_ratio = 1.0 if self.required_rotate_angle == 0 else self.required_rotate_steer_angle / self.required_rotate_angle

    def simulated_walker_update(self, state: AgentState, environment: Environment, gains: GainsConfig) -> float:
        rotate_amount = min(gains.rotation_speed * gains.time_step, self.required_rotate_angle)
        self.required_rotate_angle -= rotate_amount
        return rotate_amount * self.rotate_dir

    def inject_resetting(self, state: AgentState, environment: Environment, gains: GainsConfig, delta_rotation_deg: float) -> ResetCommand:
        steer_rotation = self.speed_ratio * delta_rotation_deg
        if abs(self.required_rotate_steer_angle) <= abs(steer_rotation) or self.required_rotate_angle <= 1e-6:
            plane_rotation = self.required_rotate_steer_angle
            self.required_rotate_steer_angle = 0.0
            finished = True
        else:
            plane_rotation = steer_rotation
            self.required_rotate_steer_angle = max(0.0, self.required_rotate_steer_angle - abs(steer_rotation))
            finished = False
        return ResetCommand(
            plane_rotation_deg=plane_rotation,
            user_rotation_deg=delta_rotation_deg,
            finished=finished,
        )

    def step(self, state: AgentState, environment: Environment, gains: GainsConfig) -> ResetCommand:
        user_rotation = self.simulated_walker_update(state, environment, gains)
        return self.inject_resetting(state, environment, gains, user_rotation)
