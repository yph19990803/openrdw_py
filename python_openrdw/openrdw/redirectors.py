from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .geometry import (
    Vector2,
    clamp,
    nearest_distance_to_polygons,
    nearest_point_on_polygon,
    signed_angle,
    vector_to_heading,
)
from .models import AgentState, Environment, GainsConfig, RedirectCommand
from .visibility import (
    active_slice_index,
    compute_slice_bisectors,
    compute_visibility_polygon,
    most_similar_slice_weight,
    priority_from_force,
)

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional dependency
    ort = None


class Redirector(Protocol):
    def inject(self, state: AgentState, environment: Environment, gains: GainsConfig, other_agents: list[AgentState]) -> RedirectCommand:
        ...


def _sign_from_angle(angle_deg: float) -> int:
    if angle_deg > 0:
        return 1
    if angle_deg < 0:
        return -1
    return 0


def _desired_steering_direction(current_forward: Vector2, desired_forward: Vector2) -> float:
    return -1.0 * _sign_from_angle(signed_angle(current_forward, desired_forward))


def _physical_to_virtual_point(state: AgentState, point: Vector2) -> Vector2:
    local = point - state.curr_pos_real
    tracking_pose = state.observed_tracking_space_pose or state.tracking_space_pose
    delta_heading = tracking_pose.heading_deg if tracking_pose is not None else (
        state.virtual_pose.heading_deg - state.physical_pose.heading_deg
    )
    return state.curr_pos + local.rotate(delta_heading)


def _translation_gain_from_negative_gradient(force: Vector2, current_forward: Vector2, gains: GainsConfig) -> float:
    if force.dot(current_forward) < 0:
        return -gains.min_trans_gain
    return 0.0


def _agent_index(state: AgentState, other_agents: list[AgentState]) -> int:
    if getattr(state, "agent_index", None) is not None:
        return state.agent_index
    for index, other in enumerate(other_agents):
        if other is state:
            return index
    return 0


def _same_agent(lhs: AgentState, rhs: AgentState) -> bool:
    lhs_index = getattr(lhs, "agent_index", None)
    rhs_index = getattr(rhs, "agent_index", None)
    if lhs_index is not None and rhs_index is not None:
        return lhs_index == rhs_index
    return lhs is rhs


def _apply_negative_gradient(
    state: AgentState,
    gains: GainsConfig,
    negative_gradient: Vector2,
    curvature_gain_cap_deg_per_sec: float = 15.0,
    rotation_gain_cap_deg_per_sec: float = 30.0,
) -> RedirectCommand:
    if negative_gradient.magnitude <= 1e-6:
        return RedirectCommand(debug={"mode": "zero_force"})

    negative_gradient = negative_gradient.normalized()
    state.total_force = negative_gradient
    current_forward = state.curr_dir_real
    desired_steering_direction = _desired_steering_direction(current_forward, negative_gradient)
    translation_gain = _translation_gain_from_negative_gradient(negative_gradient, current_forward, gains)
    dt = gains.time_step
    max_rotation_from_curvature = curvature_gain_cap_deg_per_sec * dt
    max_rotation_from_rotation = rotation_gain_cap_deg_per_sec * dt
    rotation_from_curvature = math.degrees(state.delta_pos.magnitude / max(gains.curvature_radius, 1e-6))
    curvature = desired_steering_direction * min(rotation_from_curvature, max_rotation_from_curvature)

    delta_dir = state.delta_dir
    if delta_dir * desired_steering_direction < 0:
        rotation = desired_steering_direction * min(abs(delta_dir * gains.min_rot_gain), max_rotation_from_rotation)
        rotation_gain = gains.min_rot_gain
    else:
        rotation = desired_steering_direction * min(abs(delta_dir * gains.max_rot_gain), max_rotation_from_rotation)
        rotation_gain = gains.max_rot_gain

    translation = state.delta_pos * translation_gain
    curvature_gain = 0.0
    if state.delta_pos.magnitude > 1e-6:
        curvature_gain = math.radians(curvature) / state.delta_pos.magnitude

    if abs(rotation) > abs(curvature):
        curvature = 0.0
        curvature_gain = 0.0
    else:
        rotation = 0.0
        rotation_gain = 0.0

    return RedirectCommand(
        translation=translation,
        rotation_deg=rotation,
        curvature_deg=curvature,
        translation_gain=translation_gain,
        rotation_gain=rotation_gain,
        curvature_gain=curvature_gain,
        debug={
            "force_x": negative_gradient.x,
            "force_y": negative_gradient.y,
        },
    )


def _nearest_positions_for_thomas(state: AgentState, environment: Environment, other_agents: list[AgentState]) -> list[Vector2]:
    nearest_positions: list[Vector2] = []
    curr_pos_real = state.curr_pos_real
    for index, point in enumerate(environment.tracking_space):
        next_point = environment.tracking_space[(index + 1) % len(environment.tracking_space)]
        nearest_positions.append(nearest_point_on_polygon(curr_pos_real, [point, next_point]))
    for obstacle in environment.obstacles:
        nearest_positions.append(nearest_point_on_polygon(curr_pos_real, obstacle))
    for other in other_agents:
        if _same_agent(other, state):
            continue
        nearest_positions.append(other.curr_pos_real)
    return nearest_positions


def _nearest_point_on_segment(point: Vector2, start: Vector2, end: Vector2) -> Vector2:
    return nearest_point_on_polygon(point, [start, end])


class NullRedirector:
    def inject(self, state: AgentState, environment: Environment, gains: GainsConfig, other_agents: list[AgentState]) -> RedirectCommand:
        return RedirectCommand()

    def get_priority(self, state: AgentState, environment: Environment, gains: GainsConfig, other_agents: list[AgentState]) -> float:
        return state.priority


@dataclass
class SteerToRedirector:
    movement_threshold: float = 0.2
    rotation_threshold: float = 1.5
    curvature_gain_cap_deg_per_sec: float = 15.0
    rotation_gain_cap_deg_per_sec: float = 30.0
    distance_threshold_for_dampening: float = 1.25
    bearing_threshold_for_dampening: float = 45.0
    smoothing_factor: float = 0.125
    last_rotation_applied: float = 0.0

    def pick_target(self, state: AgentState, environment: Environment, gains: GainsConfig) -> Vector2:
        raise NotImplementedError

    def inject(self, state: AgentState, environment: Environment, gains: GainsConfig, other_agents: list[AgentState]) -> RedirectCommand:
        target = self.pick_target(state, environment, gains)
        desired_facing_direction = target - state.curr_pos
        desired_steering_direction = _desired_steering_direction(state.curr_dir, desired_facing_direction)
        dt = gains.time_step
        rotation_from_curvature = 0.0
        speed = state.delta_pos.magnitude / max(dt, 1e-6)
        if speed > self.movement_threshold:
            rotation_from_curvature = math.degrees(state.delta_pos.magnitude / max(gains.curvature_radius, 1e-6))
            rotation_from_curvature = min(rotation_from_curvature, self.curvature_gain_cap_deg_per_sec * dt)

        rotation_from_rotation = 0.0
        delta_dir = state.delta_dir
        if abs(delta_dir) / max(dt, 1e-6) >= self.rotation_threshold:
            if delta_dir * desired_steering_direction < 0:
                rotation_from_rotation = min(abs(delta_dir * gains.min_rot_gain), self.rotation_gain_cap_deg_per_sec * dt)
                rotation_gain = gains.min_rot_gain
            else:
                rotation_from_rotation = min(abs(delta_dir * gains.max_rot_gain), self.rotation_gain_cap_deg_per_sec * dt)
                rotation_gain = gains.max_rot_gain
        else:
            rotation_gain = 0.0

        rotation_proposed = desired_steering_direction * max(rotation_from_rotation, rotation_from_curvature)
        curvature_gain_used = rotation_from_curvature > rotation_from_rotation
        if abs(rotation_proposed) <= 1e-6:
            return RedirectCommand(debug={"target_x": target.x, "target_y": target.y})

        bearing_to_target = state.curr_dir.angle_to(desired_facing_direction)
        if bearing_to_target <= self.bearing_threshold_for_dampening:
            rotation_proposed *= math.sin(math.radians(90.0 * bearing_to_target / self.bearing_threshold_for_dampening))
        if desired_facing_direction.magnitude <= self.distance_threshold_for_dampening:
            rotation_proposed *= desired_facing_direction.magnitude / self.distance_threshold_for_dampening

        final_rotation = (1.0 - self.smoothing_factor) * self.last_rotation_applied + self.smoothing_factor * rotation_proposed
        self.last_rotation_applied = final_rotation
        if curvature_gain_used:
            curvature_gain = 0.0
            if state.delta_pos.magnitude > 1e-6:
                curvature_gain = math.radians(final_rotation) / state.delta_pos.magnitude
            return RedirectCommand(
                curvature_deg=final_rotation,
                curvature_gain=curvature_gain,
                debug={"target_x": target.x, "target_y": target.y, "mode": "curvature"},
            )
        return RedirectCommand(
            rotation_deg=final_rotation,
            rotation_gain=rotation_gain,
            debug={"target_x": target.x, "target_y": target.y, "mode": "rotation"},
        )


class S2CRedirector(SteerToRedirector):
    bearing_threshold_deg: float = 160.0
    temp_target_distance: float = 4.0

    def pick_target(self, state: AgentState, environment: Environment, gains: GainsConfig) -> Vector2:
        user_to_center = environment.center - state.curr_pos
        bearing_to_center = user_to_center.angle_to(state.curr_dir)
        direction_to_center = _sign_from_angle(signed_angle(state.curr_dir, user_to_center))
        if bearing_to_center >= self.bearing_threshold_deg:
            return state.curr_pos + state.curr_dir.rotate(direction_to_center * 90.0) * self.temp_target_distance
        return environment.center


class S2ORedirector(SteerToRedirector):
    target_generation_angle_deg: float = 60.0

    def pick_target(self, state: AgentState, environment: Environment, gains: GainsConfig) -> Vector2:
        xs = [point.x for point in environment.tracking_space]
        ys = [point.y for point in environment.tracking_space]
        tracking_size_x = max(xs) - min(xs)
        tracking_size_y = max(ys) - min(ys)
        target_radius = 5.0
        if tracking_size_x <= 10 or tracking_size_y <= 10:
            target_radius = min(tracking_size_x, tracking_size_y) / 5.0
        center = environment.center
        user_to_center = center - state.curr_pos
        if user_to_center.magnitude < target_radius:
            alpha = self.target_generation_angle_deg
        else:
            alpha = math.degrees(math.acos(clamp(target_radius / max(user_to_center.magnitude, 1e-6), -1.0, 1.0)))
        dir1 = (user_to_center * -1.0).normalized().rotate(alpha)
        dir2 = (user_to_center * -1.0).normalized().rotate(-alpha)
        target1 = center + dir1 * target_radius
        target2 = center + dir2 * target_radius
        angle1 = state.curr_dir.angle_to(target1 - state.curr_pos)
        angle2 = state.curr_dir.angle_to(target2 - state.curr_pos)
        return target1 if angle1 <= angle2 else target2


class ThomasApfRedirector:
    def inject(self, state: AgentState, environment: Environment, gains: GainsConfig, other_agents: list[AgentState]) -> RedirectCommand:
        curr_pos_real = state.curr_pos_real
        repulsive_force = 0.0
        negative_gradient = Vector2(0.0, 0.0)
        for obstacle_position in _nearest_positions_for_thomas(state, environment, other_agents):
            delta = curr_pos_real - obstacle_position
            distance = max(delta.magnitude, 1e-6)
            repulsive_force += 1.0 / distance
            negative_gradient += delta / (distance ** 3)
        command = _apply_negative_gradient(state, gains, negative_gradient)
        command.debug["mode"] = "thomas_apf"
        command.debug["repulsive_force"] = repulsive_force
        return command

class PassiveHapticApfRedirector:
    alignment_state: bool = False

    def inject(self, state: AgentState, environment: Environment, gains: GainsConfig, other_agents: list[AgentState]) -> RedirectCommand:
        target_index = _agent_index(state, other_agents)
        if environment.physical_targets and target_index < len(environment.physical_targets):
            physical_target = environment.physical_targets[target_index]
        else:
            physical_target = environment.center
        self._update_alignment_state(state, environment, gains, physical_target, target_index)

        curr_pos_real = state.curr_pos_real
        attractive_negative_gradient = (physical_target - curr_pos_real) * 2.0
        obstacle_negative_gradient = Vector2(0.0, 0.0)
        for obstacle_position in _nearest_positions_for_thomas(state, environment, other_agents):
            delta = curr_pos_real - obstacle_position
            distance = max(delta.magnitude, 1e-6)
            obstacle_negative_gradient += delta / (distance ** 3)
        negative_gradient = (attractive_negative_gradient + obstacle_negative_gradient).normalized()
        command = _apply_negative_gradient(state, gains, negative_gradient)
        command.debug["mode"] = "passive_haptic_apf"
        command.debug["target_x"] = physical_target.x
        command.debug["target_y"] = physical_target.y
        command.debug["alignment_state"] = self.alignment_state
        return command

    def _update_alignment_state(
        self,
        state: AgentState,
        environment: Environment,
        gains: GainsConfig,
        physical_target: Vector2,
        target_index: int,
    ) -> None:
        if self.alignment_state:
            return
        virtual_obj_pos = state.final_waypoint
        if virtual_obj_pos is None:
            return
        curr_pos_real = state.curr_pos_real
        curr_dir_real = state.curr_dir_real
        physical_obj_forward = (
            environment.physical_target_forwards[target_index]
            if target_index < len(environment.physical_target_forwards)
            else Vector2(0.0, 1.0)
        )
        dv = (virtual_obj_pos - state.curr_pos).magnitude
        dp = (physical_target - curr_pos_real).magnitude
        gt = gains.min_trans_gain + 1.0
        g_t = gains.max_trans_gain + 1.0
        phi_p = curr_dir_real.angle_to(physical_target - curr_pos_real) * math.pi / 180.0 if dp > 1e-6 else 0.0
        curvature_limit = 0.0
        if dp > 1e-6 and gains.curvature_radius > 1e-6:
            ratio = (dp / gains.curvature_radius) / 2.0
            curvature_limit = math.asin(ratio) if -1.0 <= ratio <= 1.0 else float("nan")
        if gt * dp < dv < g_t * dp and phi_p < curvature_limit:
            self.alignment_state = True


class MessingerApfRedirector:
    target_seg_length: float = 1.0
    c_const: float = 0.00897
    lambda_const: float = 2.656
    gamma_const: float = 3.091
    curvature_gain_cap_deg_per_sec: float = 15.0
    rotation_gain_cap_deg_per_sec: float = 30.0
    max_steering_rate: float = 15.0
    base_rate: float = 1.5

    def inject(self, state: AgentState, environment: Environment, gains: GainsConfig, other_agents: list[AgentState]) -> RedirectCommand:
        total_force = self.get_total_force(state, environment, other_agents).normalized()
        return self._inject_from_force(state, environment, gains, total_force, mode="messinger_apf")

    def _inject_from_force(
        self,
        state: AgentState,
        environment: Environment,
        gains: GainsConfig,
        total_force: Vector2,
        *,
        mode: str,
    ) -> RedirectCommand:
        state.total_force = total_force
        if total_force.magnitude <= 1e-6:
            return RedirectCommand(debug={"mode": f"{mode}_zero", "force_x": 0.0, "force_y": 0.0})
        desired_steering_direction = _desired_steering_direction(state.curr_dir_real, total_force)
        dt = gains.time_step
        max_curvature = self.curvature_gain_cap_deg_per_sec * dt
        max_rotation = self.rotation_gain_cap_deg_per_sec * dt
        speed = state.delta_pos.magnitude / max(dt, 1e-6)
        moving_rate = 360.0 * speed / (2.0 * math.pi * max(gains.curvature_radius, 1e-6))
        distance_to_obstacle = nearest_distance_to_polygons(state.curr_pos_real, environment.all_polygons)
        if distance_to_obstacle < gains.curvature_radius:
            t = 1.0 - distance_to_obstacle / max(gains.curvature_radius, 1e-6)
            moving_rate = (1.0 - t) * moving_rate + t * self.max_steering_rate

        curvature = desired_steering_direction * min(moving_rate * dt, max_curvature)
        delta_dir = state.delta_dir
        if delta_dir * desired_steering_direction < 0:
            rotation = desired_steering_direction * max(self.base_rate * dt, min(abs(delta_dir * gains.min_rot_gain), max_rotation))
            rotation_gain = gains.min_rot_gain
        else:
            rotation = desired_steering_direction * max(self.base_rate * dt, min(abs(delta_dir * gains.max_rot_gain), max_rotation))
            rotation_gain = gains.max_rot_gain

        curvature_gain = 0.0
        if state.delta_pos.magnitude > 1e-6:
            curvature_gain = math.radians(curvature) / state.delta_pos.magnitude

        if abs(rotation) > abs(curvature):
            curvature = 0.0
            curvature_gain = 0.0
        else:
            rotation = 0.0
            rotation_gain = 0.0

        return RedirectCommand(
            rotation_deg=rotation,
            curvature_deg=curvature,
            rotation_gain=rotation_gain,
            curvature_gain=curvature_gain,
            debug={"mode": mode, "force_x": total_force.x, "force_y": total_force.y},
        )

    def get_total_force(self, state: AgentState, environment: Environment, other_agents: list[AgentState]) -> Vector2:
        wall_force = Vector2(0.0, 0.0)
        for polygon in [environment.tracking_space, *environment.obstacles]:
            if len(polygon) < 2:
                continue
            reverse_edges = polygon is not environment.tracking_space
            for index, start in enumerate(polygon):
                end = polygon[(index + 1) % len(polygon)]
                if reverse_edges:
                    wall_force += self.get_wall_force(state, end, start)
                else:
                    wall_force += self.get_wall_force(state, start, end)
        user_force = Vector2(0.0, 0.0)
        for other in other_agents:
            if _same_agent(other, state):
                continue
            user_force += self.get_user_force(state, other)
        return wall_force + user_force

    def get_wall_force(self, state: AgentState, start: Vector2, end: Vector2) -> Vector2:
        segment = end - start
        length = segment.magnitude
        if length <= 1e-6:
            return Vector2(0.0, 0.0)
        segment_count = int(length / self.target_seg_length)
        if segment_count * self.target_seg_length != length:
            segment_count += 1
        segment_length = length / max(segment_count, 1)
        unit = segment.normalized()
        total = Vector2(0.0, 0.0)
        for index in range(1, segment_count + 1):
            seg_start = start + unit * ((index - 1) * segment_length)
            seg_end = start + unit * (index * segment_length)
            center = (seg_start + seg_end) * 0.5
            delta = state.curr_pos_real - center
            if delta.magnitude <= 1e-6:
                continue
            normal = (seg_end - seg_start).rotate(-90.0).normalized()
            if normal.dot(delta.normalized()) > 0:
                total += delta.normalized() * (self.c_const * (seg_end - seg_start).magnitude / (delta.magnitude ** self.lambda_const))
        return total

    def get_user_force(self, state: AgentState, other: AgentState) -> Vector2:
        other_position = other.curr_pos_real
        other_direction = other.curr_dir_real
        current_position = state.curr_pos_real
        current_direction = state.curr_dir_real
        theta1 = (other_position - current_position).angle_to(current_direction)
        theta2 = (current_position - other_position).angle_to(other_direction)
        k = clamp((math.cos(math.radians(theta1)) + math.cos(math.radians(theta2))) / 2.0, 0.0, 1.0)
        delta = current_position - other_position
        if delta.magnitude <= 1e-6:
            return Vector2(0.0, 0.0)
        return k * delta.normalized() / (delta.magnitude ** self.gamma_const)


class DynamicApfRedirector(MessingerApfRedirector):
    avatar_force_weight: float = 0.25
    steering_target_weight_boundary: float = 2.0
    steering_target_weight_center: float = 1.0
    priority_weight_force: float = 1.0
    priority_weight_angle: float = 0.02

    def inject(self, state: AgentState, environment: Environment, gains: GainsConfig, other_agents: list[AgentState]) -> RedirectCommand:
        force_t = self.get_total_force(state, environment, other_agents)
        gravitation = self.get_gravitational_dir(state, environment, other_agents) * force_t.magnitude
        total_force = (force_t + gravitation).normalized()
        command = self._inject_from_force(state, environment, gains, total_force, mode="dynamic_apf")
        state.total_force = force_t.normalized() if force_t.magnitude > 1e-6 else force_t
        command.debug["mode"] = "dynamic_apf"
        command.debug["force_x"] = total_force.x
        command.debug["force_y"] = total_force.y
        command.debug["reset_force_x"] = state.total_force.x
        command.debug["reset_force_y"] = state.total_force.y
        return command

    def get_total_force(self, state: AgentState, environment: Environment, other_agents: list[AgentState]) -> Vector2:
        total = super().get_total_force(state, environment, other_agents)
        for other in other_agents:
            if _same_agent(other, state):
                continue
            avatar_pos = other.curr_pos_real + other.curr_dir_real.normalized()
            current_pos = state.curr_pos_real
            theta1 = (avatar_pos - current_pos).angle_to(state.curr_dir_real)
            theta2 = (current_pos - avatar_pos).angle_to(other.curr_dir_real)
            k = clamp((math.cos(math.radians(theta1)) + math.cos(math.radians(theta2))) / 2.0, 0.0, 1.0)
            delta = current_pos - avatar_pos
            if delta.magnitude > 1e-6:
                total += self.avatar_force_weight * k * delta.normalized() / (delta.magnitude ** self.gamma_const)
        return total

    def get_gravitational_dir(self, state: AgentState, environment: Environment, other_agents: list[AgentState]) -> Vector2:
        if len(environment.tracking_space) < 4:
            return (environment.center - state.curr_pos_real).normalized()

        current_pos = state.curr_pos_real
        current_dir = state.curr_dir_real
        foot_point1, foot_point2 = self._lock_potential_area(current_pos, current_dir, environment.tracking_space)
        primary_target = self._search_primary_target(current_pos, foot_point1, foot_point2, other_agents, state)
        selection_bounds = self._lock_selection_area(primary_target, environment.tracking_space, other_agents, state)
        steering_target = self._search_steering_target(primary_target, selection_bounds, environment.tracking_space, current_pos)
        return (steering_target - current_pos).normalized()

    def get_priority(self, state: AgentState, environment: Environment, gains: GainsConfig, other_agents: list[AgentState]) -> float:
        force = MessingerApfRedirector.get_total_force(self, state, environment, other_agents)
        if force.magnitude <= 1e-6:
            return 0.0
        return -(self.priority_weight_force * force.magnitude + self.priority_weight_angle * force.angle_to(state.curr_dir_real))

    def _lock_potential_area(self, current_pos: Vector2, current_dir: Vector2, tracking_space: list[Vector2]) -> tuple[Vector2, Vector2]:
        foot_point1 = tracking_space[0]
        foot_point2 = tracking_space[1]
        count = len(tracking_space)
        for index in range(count):
            foot_point1 = _nearest_point_on_segment(current_pos, tracking_space[index], tracking_space[(index + 1) % count])
            foot_point2 = _nearest_point_on_segment(current_pos, tracking_space[(index + 1) % count], tracking_space[(index + 2) % count])
            if (foot_point1 - current_pos).dot(current_dir) >= 0 and (foot_point2 - current_pos).dot(current_dir) >= 0:
                break
        return foot_point1, foot_point2

    def _search_primary_target(
        self,
        current_pos: Vector2,
        foot_point1: Vector2,
        foot_point2: Vector2,
        other_agents: list[AgentState],
        state: AgentState,
    ) -> Vector2:
        x_dir = (foot_point1 - current_pos).normalized()
        y_dir = (foot_point2 - current_pos).normalized()
        primary_target = current_pos
        max_sum = 0.0
        for i in range(max(0, int((foot_point1 - current_pos).magnitude))):
            for j in range(max(0, int((foot_point2 - current_pos).magnitude))):
                target = current_pos + x_dir * (i + 0.5) + y_dir * (j + 0.5)
                total = 0.0
                for other in other_agents:
                    if _same_agent(other, state):
                        continue
                    total += (other.curr_pos_real - target).magnitude
                if total > max_sum:
                    primary_target = target
                    max_sum = total
        return primary_target

    def _lock_selection_area(
        self,
        primary_target: Vector2,
        tracking_space: list[Vector2],
        other_agents: list[AgentState],
        state: AgentState,
    ) -> tuple[float, float, float, float]:
        select_points = list(tracking_space)
        for other in other_agents:
            if _same_agent(other, state):
                continue
            select_points.append(other.curr_pos_real)
        select_points.sort(key=lambda point: point.x)

        max_area = 0.0
        right = float("inf")
        left = float("-inf")
        up = float("inf")
        down = float("-inf")
        for left_index in range(len(select_points) - 1):
            for right_index in range(left_index + 1, len(select_points)):
                left_x = select_points[left_index].x - primary_target.x
                right_x = select_points[right_index].x - primary_target.x
                if left_x <= 0 <= right_x:
                    up_y = float("inf")
                    down_y = float("-inf")
                    found_up = False
                    found_down = False
                    for index in range(left_index, right_index + 1):
                        displacement = select_points[index].y - primary_target.y
                        if displacement >= 0 and displacement < up_y:
                            up_y = displacement
                            found_up = True
                        if displacement < 0 and displacement > down_y:
                            down_y = displacement
                            found_down = True
                    area = (up_y - down_y) * (right_x - left_x) if found_up and found_down else -1.0
                    if found_up and found_down and area >= max_area:
                        max_area = area
                        right = right_x
                        left = left_x
                        up = up_y
                        down = down_y
        return left, right, down, up

    def _search_steering_target(
        self,
        primary_target: Vector2,
        bounds: tuple[float, float, float, float],
        tracking_space: list[Vector2],
        current_pos: Vector2,
    ) -> Vector2:
        left, right, down, up = bounds
        if not all(math.isfinite(value) for value in bounds):
            return primary_target
        steering_target = primary_target
        x_dir = (tracking_space[1] - tracking_space[2]).normalized()
        y_dir = (tracking_space[2] - tracking_space[3]).normalized()
        start_point = primary_target + x_dir * left + y_dir * down
        max_sum = float("-inf")
        width = max(0, int(right - left))
        height = max(0, int(up - down))
        for i in range(width):
            for j in range(height):
                target = start_point + x_dir * (i + 0.5) + y_dir * (j + 0.5)
                d1 = nearest_distance_to_polygons(target, [tracking_space])
                center = start_point + x_dir * ((right - left) / 2.0) + y_dir * ((up - down) / 2.0)
                d2 = (target - center).magnitude
                total = 2.0 * d1 - 1.0 * d2
                if total > max_sum:
                    steering_target = target
                    max_sum = total
        if (steering_target - current_pos).magnitude <= 1e-6:
            return primary_target
        return steering_target


@dataclass
class DeepLearningRedirector:
    model_directory: Path = Path("/Users/yph/Desktop/My_project/Unity/OpenRdw_vbdi/Assets/OpenRDW/Resources")
    wait_time: int = 20
    states: list[Vector2] = field(default_factory=list)
    state_vectors: list[tuple[float, float, float]] = field(default_factory=list)
    call_count: int = 0
    action_mean: list[float] | None = None
    session: object | None = None
    output_name: str | None = None

    def inject(self, state: AgentState, environment: Environment, gains: GainsConfig, other_agents: list[AgentState]) -> RedirectCommand:
        if ort is None:
            return RedirectCommand(debug={"mode": "deep_learning_unavailable", "error": "onnxruntime not installed"})
        if self.session is None:
            try:
                self._load_model(environment)
            except Exception as exc:
                return RedirectCommand(debug={"mode": "deep_learning_unavailable", "error": str(exc)})
        if not self.state_vectors or state.if_just_end_reset:
            self.state_vectors = []
            for _ in range(10):
                self._add_state(state, environment)
        self.call_count += 1
        if (self.call_count - 1) % self.wait_time == 0:
            self._add_state(state, environment)
            input_data = [value for triple in self.state_vectors[-10:] for value in triple]
            import numpy as np

            outputs = self.session.run(None, {self.session.get_inputs()[0].name: np.array([input_data], dtype=np.float32)})
            action_tensor = None
            output_names = [output.name for output in self.session.get_outputs()]
            if "24" in output_names:
                preferred = outputs[output_names.index("24")].reshape(-1)
                if preferred.shape[0] >= 3:
                    action_tensor = preferred
            if action_tensor is None:
                for output in outputs:
                    flat = output.reshape(-1)
                    if flat.shape[0] >= 3:
                        action_tensor = flat
                        break
            if action_tensor is None:
                raise RuntimeError("DeepLearning ONNX model did not return a 3-value action tensor")
            self.action_mean = [float(action_tensor[0]), float(action_tensor[1]), float(action_tensor[2])]
        if self.action_mean is None:
            return RedirectCommand(debug={"mode": "deep_learning_warmup"})

        g_t = self._convert(-1.0, 1.0, gains.min_trans_gain, gains.max_trans_gain, self.action_mean[0])
        g_r = self._convert(-1.0, 1.0, gains.min_rot_gain, gains.max_rot_gain, self.action_mean[1])
        g_c = self._convert(-1.0, 1.0, -1.0 / gains.curvature_radius, 1.0 / gains.curvature_radius, self.action_mean[2])
        translation = state.delta_pos * g_t
        rotation = g_r * state.delta_dir
        curvature = g_c * state.delta_pos.magnitude * math.degrees(1.0)
        return RedirectCommand(
            translation=translation,
            rotation_deg=rotation,
            curvature_deg=curvature,
            translation_gain=g_t,
            rotation_gain=g_r,
            curvature_gain=g_c,
            debug={"mode": "deep_learning"},
        )

    def _load_model(self, environment: Environment) -> None:
        xs = [point.x for point in environment.tracking_space]
        ys = [point.y for point in environment.tracking_space]
        if environment.shape == "square":
            box_width = max(xs) - min(xs)
        else:
            box_width = ((max(xs) - min(xs)) + (max(ys) - min(ys))) / 2.0
        target_width = min([10, 20, 30, 40, 50], key=lambda candidate: abs(candidate - box_width))
        model_path = self.model_directory / f"SRLNet_{target_width}.onnx"
        available = list(ort.get_available_providers())
        preferred_order = [
            "CUDAExecutionProvider",
            "CoreMLExecutionProvider",
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ]
        providers = [provider for provider in preferred_order if provider in available] or available
        self.session = ort.InferenceSession(model_path.as_posix(), providers=providers)

    def _add_state(self, state: AgentState, environment: Environment) -> None:
        xs = [point.x for point in environment.tracking_space]
        ys = [point.y for point in environment.tracking_space]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        pos = state.curr_pos_real
        direction = state.curr_dir_real
        x_value = self._convert(min_x, max_x, 0.0, 1.0, pos.x)
        y_value = self._convert(min_y, max_y, 0.0, 1.0, pos.y)
        z_value = self._convert(-180.0, 180.0, 0.0, 1.0, signed_angle(Vector2(1.0, 0.0), direction))
        self.state_vectors.append((x_value, y_value, z_value))

    def _convert(self, l1: float, r1: float, l2: float, r2: float, value: float) -> float:
        if abs(r1 - l1) <= 1e-6:
            return l2
        return (value - l1) / (r1 - l1) * (r2 - l2) + l2


@dataclass
class ZigZagRedirector:
    real_target0_default: Vector2 = Vector2(0.0, 0.0)
    real_target1_default: Vector2 = Vector2(3.0, 3.0)
    waypoint_update_distance: float = 0.4
    slow_down_velocity_threshold: float = 0.25
    heading_to_target0: bool = False
    last_waypoint_index: int = 1

    def inject(self, state: AgentState, environment: Environment, gains: GainsConfig, other_agents: list[AgentState]) -> RedirectCommand:
        real_target0 = environment.physical_targets[0] if len(environment.physical_targets) >= 1 else self.real_target0_default
        real_target1 = environment.physical_targets[1] if len(environment.physical_targets) >= 2 else self.real_target1_default
        if state.current_waypoint != self.last_waypoint_index:
            self.heading_to_target0 = not self.heading_to_target0
            self.last_waypoint_index = state.current_waypoint

        virtual_target_position = state.active_waypoint or (state.curr_pos + state.curr_dir * 1.0)

        real_target_physical = real_target0 if self.heading_to_target0 else real_target1
        real_target_virtual = _physical_to_virtual_point(state, real_target_physical)
        angle_to_real_target = signed_angle(state.curr_dir, real_target_virtual - state.curr_pos)
        angle_to_virtual_target = signed_angle(state.curr_dir, virtual_target_position - state.curr_pos)
        distance_to_real_target = (real_target_physical - state.curr_pos_real).magnitude
        user_to_virtual_target = virtual_target_position - state.curr_pos
        user_to_real_target = real_target_virtual - state.curr_pos
        required_angle_injection = signed_angle(user_to_real_target, user_to_virtual_target)

        minimum_real_translation_remaining = user_to_virtual_target.magnitude / (1.0 + gains.max_trans_gain)
        minimum_real_rotation_remaining = angle_to_virtual_target
        expected_rotation_from_rotation_gain = math.copysign(
            min(abs(required_angle_injection), abs(minimum_real_rotation_remaining * gains.min_rot_gain)),
            required_angle_injection if abs(required_angle_injection) > 1e-6 else 1.0,
        )
        remaining_rotation_for_curvature = required_angle_injection - expected_rotation_from_rotation_gain
        expected_rotation_from_curvature = math.copysign(
            min(
                minimum_real_translation_remaining * math.degrees(1.0 / max(gains.curvature_radius, 1e-6)),
                abs(2.0 * remaining_rotation_for_curvature),
            ),
            required_angle_injection if abs(required_angle_injection) > 1e-6 else 1.0,
        )
        required_translation_injection = (real_target_virtual - virtual_target_position).magnitude

        if distance_to_real_target < 0.1:
            g_c = 0.0
            g_r = 0.0
            g_t = 0.0
        else:
            g_c = expected_rotation_from_curvature / max(minimum_real_translation_remaining, 1e-6)
            g_r = 0.0 if abs(angle_to_real_target) < math.radians(1.0) else expected_rotation_from_rotation_gain / max(abs(minimum_real_rotation_remaining), 1e-6)
            g_t = required_translation_injection / max(distance_to_real_target, 1e-6)

        if state.delta_pos.magnitude > 1e-6:
            g_t = math.cos(math.radians(signed_angle(state.delta_pos, virtual_target_position - real_target_virtual))) * abs(g_t)
        g_r *= _sign_from_angle(state.delta_dir)
        g_t = clamp(g_t, gains.min_trans_gain, gains.max_trans_gain)
        g_r = clamp(g_r, gains.min_rot_gain, gains.max_rot_gain)

        translation = state.delta_pos * g_t
        rotation = g_r * state.delta_dir
        curvature = g_c * state.delta_pos.magnitude
        return RedirectCommand(
            translation=translation,
            rotation_deg=rotation,
            curvature_deg=curvature,
            translation_gain=g_t,
            rotation_gain=g_r,
            curvature_gain=g_c,
            debug={
                "mode": "zigzag",
                "real_target_x": real_target_physical.x,
                "real_target_y": real_target_physical.y,
            },
        )


class VisPolyRedirector:
    def inject(self, state: AgentState, environment: Environment, gains: GainsConfig, other_agents: list[AgentState]) -> RedirectCommand:
        physical_poly = compute_visibility_polygon(state.curr_pos_real, [*environment.obstacles, environment.tracking_space])
        virtual_poly = compute_visibility_polygon(state.curr_pos, environment.all_virtual_polygons)
        if len(physical_poly) < 2 or len(virtual_poly) < 2:
            return RedirectCommand(debug={"mode": "vispoly_empty"})
        physical_bisectors = compute_slice_bisectors(state.curr_pos_real, physical_poly)
        virtual_bisectors = compute_slice_bisectors(state.curr_pos, virtual_poly)
        active_index = active_slice_index(state.curr_dir, virtual_bisectors)
        virtual_slice_weight = virtual_bisectors[active_index].magnitude
        negative_gradient = most_similar_slice_weight(virtual_slice_weight, physical_bisectors)
        command = _apply_negative_gradient(state, gains, negative_gradient)
        command.debug["mode"] = "vispoly"
        command.debug["negative_gradient_x"] = negative_gradient.x
        command.debug["negative_gradient_y"] = negative_gradient.y
        command.debug["virtual_slice_weight"] = virtual_slice_weight
        return command
