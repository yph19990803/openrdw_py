from __future__ import annotations

from dataclasses import dataclass, field

from .geometry import Vector2, clamp, normalize_heading, signed_angle, vector_to_heading
from .models import AgentState, Environment, GainsConfig, Pose2D, RedirectCommand, ResetCommand, StepTrace
from .redirectors import Redirector
from .resetters import Resetter


@dataclass
class OpenRDWSimulator:
    environment: Environment
    gains: GainsConfig
    redirector: Redirector
    resetter: Resetter
    waypoints: list[Vector2]
    state: AgentState
    sampling_intervals: list[float] | None = None
    distance_to_waypoint_threshold: float = 0.3
    trace: list[StepTrace] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._ensure_root_and_tracking_pose()
        self._sync_physical_from_root_tracking()
        self._capture_previous_poses()

    def run(self, steps: int, other_agents: list[AgentState] | None = None) -> list[StepTrace]:
        other_agents = other_agents or [self.state]
        for step_index in range(steps):
            self.step(step_index, other_agents)
        return self.trace

    def step(self, step_index: int, other_agents: list[AgentState], manual_input: dict[str, bool] | None = None) -> StepTrace:
        self.prepare_frame()
        self.advance_movement(other_agents, manual_input=manual_input)
        self.apply_redirection_phase(other_agents)
        return self.finalize_frame(step_index)

    def prepare_frame(self) -> None:
        if not self.state.in_reset:
            self.state.redirection_time += self.gains.time_step
        self._capture_previous_poses()
        self._ensure_root_and_tracking_pose()
        self._sync_physical_from_root_tracking()
        self._clear_observed_poses()
        self._clear_frame_deltas()

    def advance_movement(self, other_agents: list[AgentState], manual_input: dict[str, bool] | None = None) -> None:
        physical_position_before_motion = self.state.physical_pose.position
        if self.state.in_reset:
            self._advance_reset_user_rotation()
            self._sync_physical_from_root_tracking()
        else:
            collision_before_move = self.resetter.is_reset_required(self.state, self.environment, self.gains, other_agents)
            if collision_before_move:
                self._stop_base_motion()
            elif manual_input:
                self._advance_manual_motion(manual_input)
            else:
                self._advance_base_motion()
            self._sync_physical_from_root_tracking()
        self.state.physical_delta_translation = self.state.physical_pose.position - physical_position_before_motion
        self._capture_observed_poses()
        self._update_same_pos_time()

    def apply_redirection_phase(self, other_agents: list[AgentState]) -> None:
        if self.resetter.is_reset_required(self.state, self.environment, self.gains, other_agents) and not self.state.in_reset and not self.state.if_just_end_reset:
            self.resetter.begin(self.state, self.environment, self.gains)
            self.state.in_reset = True

        if self.state.in_reset:
            reset_command = self.resetter.inject_resetting(
                self.state,
                self.environment,
                self.gains,
                self.state.base_delta_rotation_deg,
            )
            self._apply_reset_plane_rotation(reset_command)
            self._sync_physical_from_root_tracking()
            self.state.last_reset_command = reset_command
            self.state.last_command = RedirectCommand()
            if reset_command.finished:
                self.state.in_reset = False
                self.state.if_just_end_reset = True
        else:
            command = self.redirector.inject(self.state, self.environment, self.gains, other_agents)
            self._apply_redirect(command)
            self._sync_physical_from_root_tracking()
            self.state.last_command = command
            self.state.last_reset_command = ResetCommand()
            self.state.if_just_end_reset = False

    def finalize_frame(self, step_index: int) -> StepTrace:
        observed_virtual = self.state.observed_virtual_pose or self.state.virtual_pose
        observed_physical = self.state.observed_physical_pose or self.state.physical_pose
        observed_tracking = self.state.observed_tracking_space_pose or self.state.tracking_space_pose
        trace = StepTrace(
            step_index=step_index,
            simulation_time_s=(step_index + 1) * self.gains.time_step,
            virtual_x=self.state.virtual_pose.position.x,
            virtual_y=self.state.virtual_pose.position.y,
            virtual_heading_deg=self.state.virtual_pose.heading_deg,
            physical_x=self.state.physical_pose.position.x,
            physical_y=self.state.physical_pose.position.y,
            physical_heading_deg=self.state.physical_pose.heading_deg,
            in_reset=self.state.in_reset,
            translation_injection_x=self.state.last_command.translation.x,
            translation_injection_y=self.state.last_command.translation.y,
            rotation_injection_deg=self.state.last_command.rotation_deg,
            curvature_injection_deg=self.state.last_command.curvature_deg,
            translation_gain=self.state.last_command.translation_gain,
            rotation_gain=self.state.last_command.rotation_gain,
            curvature_gain=self.state.last_command.curvature_gain,
            total_force_x=self.state.total_force.x,
            total_force_y=self.state.total_force.y,
            priority=self.state.priority,
            observed_virtual_x=observed_virtual.position.x,
            observed_virtual_y=observed_virtual.position.y,
            observed_virtual_heading_deg=observed_virtual.heading_deg,
            observed_physical_x=observed_physical.position.x,
            observed_physical_y=observed_physical.position.y,
            observed_physical_heading_deg=observed_physical.heading_deg,
            observed_tracking_x=observed_tracking.position.x if observed_tracking is not None else None,
            observed_tracking_y=observed_tracking.position.y if observed_tracking is not None else None,
            observed_tracking_heading_deg=observed_tracking.heading_deg if observed_tracking is not None else None,
        )
        self.trace.append(trace)
        return trace

    def _update_same_pos_time(self) -> None:
        if self.state.mission_complete:
            self.state.same_pos_time = 0.0
            return
        if self.state.delta_pos.magnitude <= 1e-9:
            self.state.same_pos_time += self.gains.time_step
        else:
            self.state.same_pos_time = 0.0

    def _clear_frame_deltas(self) -> None:
        self.state.base_delta_translation = Vector2(0.0, 0.0)
        self.state.physical_delta_translation = Vector2(0.0, 0.0)
        self.state.base_delta_rotation_deg = 0.0
        self.state.manual_translation = Vector2(0.0, 0.0)
        self.state.manual_rotation_deg = 0.0

    def _stop_base_motion(self) -> None:
        self.state.base_delta_translation = Vector2(0.0, 0.0)
        self.state.physical_delta_translation = Vector2(0.0, 0.0)
        self.state.base_delta_rotation_deg = 0.0

    def _advance_reset_user_rotation(self) -> None:
        user_rotation = self.resetter.simulated_walker_update(self.state, self.environment, self.gains)
        self.state.virtual_pose = Pose2D(
            self.state.virtual_pose.position,
            normalize_heading(self.state.virtual_pose.heading_deg + user_rotation),
        )
        self.state.base_delta_translation = Vector2(0.0, 0.0)
        self.state.physical_delta_translation = Vector2(0.0, 0.0)
        self.state.base_delta_rotation_deg = user_rotation

    def _advance_manual_motion(self, manual_input: dict[str, bool]) -> None:
        forward = self.state.virtual_forward.normalized()
        right = Vector2(forward.y, -forward.x).normalized()
        movement = Vector2(0.0, 0.0)
        if manual_input.get("w"):
            movement += forward
        if manual_input.get("s"):
            movement -= forward
        if manual_input.get("d"):
            movement += right
        if manual_input.get("a"):
            movement -= right
        if movement.magnitude > 1e-6:
            movement = movement.normalized() * (self.gains.translation_speed * self.gains.time_step)
        rotation = 0.0
        if manual_input.get("left"):
            rotation -= self.gains.rotation_speed * self.gains.time_step
        if manual_input.get("right"):
            rotation += self.gains.rotation_speed * self.gains.time_step
        self.state.virtual_pose = Pose2D(
            self.state.virtual_pose.position + movement,
            normalize_heading(self.state.virtual_pose.heading_deg + rotation),
        )
        self.state.base_delta_translation = movement
        self.state.base_delta_rotation_deg = rotation
        self.state.manual_translation = movement
        self.state.manual_rotation_deg = rotation
        self.state.walk_distance += movement.magnitude

    def _advance_base_motion(self) -> None:
        if self.state.mission_complete:
            self.state.base_delta_translation = Vector2(0.0, 0.0)
            self.state.physical_delta_translation = Vector2(0.0, 0.0)
            self.state.base_delta_rotation_deg = 0.0
            self.state.active_waypoint = None
            return
        if not self.waypoints:
            self.state.base_delta_translation = Vector2(0.0, 0.0)
            self.state.physical_delta_translation = Vector2(0.0, 0.0)
            self.state.base_delta_rotation_deg = 0.0
            return
        waypoints = self._resolved_waypoints()
        if self.sampling_intervals:
            self._advance_real_user_path()
            return
        while not self.state.mission_complete:
            waypoint = waypoints[min(self.state.current_waypoint, len(waypoints) - 1)]
            if (self.state.virtual_pose.position - waypoint).magnitude >= self.distance_to_waypoint_threshold:
                break
            self._advance_to_next_waypoint()
        if self.state.mission_complete:
            self.state.base_delta_translation = Vector2(0.0, 0.0)
            self.state.physical_delta_translation = Vector2(0.0, 0.0)
            self.state.base_delta_rotation_deg = 0.0
            self.state.active_waypoint = None
            return
        waypoint = waypoints[min(self.state.current_waypoint, len(waypoints) - 1)]
        self.state.active_waypoint = waypoint
        self._turn_and_walk_to_waypoint(waypoint)
        if (self.state.virtual_pose.position - waypoint).magnitude < self.distance_to_waypoint_threshold:
            self._advance_to_next_waypoint()

    def _turn_and_walk_to_waypoint(self, waypoint: Vector2) -> None:
        user_to_target = waypoint - self.state.virtual_position
        rotation_to_target = signed_angle(self.state.virtual_forward, user_to_target)
        if user_to_target.magnitude > 1e-4:
            heading_step = -clamp(rotation_to_target, -self.gains.rotation_speed * self.gains.time_step, self.gains.rotation_speed * self.gains.time_step)
        else:
            heading_step = 0.0

        new_virtual_heading = self.state.virtual_pose.heading_deg + heading_step
        new_virtual_forward = Vector2(0.0, 1.0).rotate(new_virtual_heading)
        remaining_vector = waypoint - self.state.virtual_position
        remaining_angle = signed_angle(new_virtual_forward, remaining_vector)

        move_distance = 0.0
        if abs(remaining_angle) < 1.0:
            move_distance = min(self.gains.translation_speed * self.gains.time_step, remaining_vector.magnitude)

        virtual_translation = new_virtual_forward.normalized() * move_distance
        self.state.virtual_pose = Pose2D(self.state.virtual_pose.position + virtual_translation, normalize_heading(new_virtual_heading))
        self.state.base_delta_translation = virtual_translation
        self.state.base_delta_rotation_deg = heading_step
        self.state.walk_distance += virtual_translation.magnitude

    def _advance_real_user_path(self) -> None:
        waypoints = self._resolved_waypoints()
        if len(waypoints) < 2:
            self.state.base_delta_translation = Vector2(0.0, 0.0)
            self.state.physical_delta_translation = Vector2(0.0, 0.0)
            self.state.base_delta_rotation_deg = 0.0
            return
        current_index = min(self.state.current_waypoint, len(waypoints) - 1)
        interval = self._sampling_interval_for_index(current_index)
        elapsed_in_segment = max(self.state.redirection_time - self.state.waypoint_time_accumulator, 0.0)
        while current_index < len(waypoints) - 1 and elapsed_in_segment > interval:
            self.state.waypoint_time_accumulator += interval
            self.state.current_waypoint += 1
            current_index = min(self.state.current_waypoint, len(waypoints) - 1)
            interval = self._sampling_interval_for_index(current_index)
            elapsed_in_segment = max(self.state.redirection_time - self.state.waypoint_time_accumulator, 0.0)
        prev_index = max(current_index - 1, 0)
        start = waypoints[prev_index]
        end = waypoints[current_index]
        if interval <= 1e-6:
            alpha = 1.0
        else:
            alpha = clamp(elapsed_in_segment / interval, 0.0, 1.0)
        next_virtual_position = start + (end - start) * alpha
        delta_virtual = next_virtual_position - self.state.virtual_position
        target_direction = (end - start).normalized() if (end - start).magnitude > 1e-6 else self.state.virtual_forward
        next_heading = vector_to_heading(target_direction)
        heading_step = -signed_angle(self.state.virtual_forward, target_direction)
        self.state.virtual_pose = Pose2D(next_virtual_position, normalize_heading(next_heading))
        self.state.base_delta_translation = delta_virtual
        self.state.base_delta_rotation_deg = heading_step
        self.state.walk_distance += delta_virtual.magnitude
        if current_index >= len(waypoints) - 1 and (next_virtual_position - end).magnitude <= self.distance_to_waypoint_threshold:
            self.state.mission_complete = True
            self.state.active_waypoint = None
        else:
            self.state.active_waypoint = self._current_waypoint_target()

    def _advance_base_turn_only(self) -> None:
        if self.state.mission_complete or not self.waypoints:
            self.state.base_delta_translation = Vector2(0.0, 0.0)
            self.state.physical_delta_translation = Vector2(0.0, 0.0)
            self.state.base_delta_rotation_deg = 0.0
            return
        waypoints = self._resolved_waypoints()
        waypoint = waypoints[min(self.state.current_waypoint, len(waypoints) - 1)]
        self.state.active_waypoint = waypoint
        user_to_target = waypoint - self.state.virtual_position
        rotation_to_target = signed_angle(self.state.virtual_forward, user_to_target)
        if user_to_target.magnitude > 1e-4:
            heading_step = -clamp(rotation_to_target, -self.gains.rotation_speed * self.gains.time_step, self.gains.rotation_speed * self.gains.time_step)
        else:
            heading_step = 0.0
        self.state.virtual_pose = Pose2D(
            self.state.virtual_pose.position,
            normalize_heading(self.state.virtual_pose.heading_deg + heading_step),
        )
        self.state.base_delta_translation = Vector2(0.0, 0.0)
        self.state.base_delta_rotation_deg = heading_step

    def _sampling_interval_for_index(self, waypoint_index: int) -> float:
        if not self.sampling_intervals or waypoint_index == 0:
            return 0.0
        return self.sampling_intervals[waypoint_index]

    def _apply_redirect(self, command: RedirectCommand) -> None:
        self._ensure_root_and_tracking_pose()
        self.state.virtual_pose = Pose2D(
            position=self.state.virtual_pose.position + command.translation,
            heading_deg=normalize_heading(self.state.virtual_pose.heading_deg + command.rotation_deg + command.curvature_deg),
        )
        current_root = self.state.root_pose
        new_root_heading = normalize_heading(current_root.heading_deg + command.rotation_deg + command.curvature_deg)
        new_root_position = self._solve_root_position(
            self.state.virtual_pose.position,
            self.state.physical_pose.position,
            new_root_heading,
        )
        self.state.root_pose = Pose2D(new_root_position, new_root_heading)
        self._sync_tracking_space_from_root()
        self.state.priority = command.priority

    def _apply_reset(self, command: ResetCommand) -> None:
        self.state.virtual_pose = Pose2D(
            position=self.state.virtual_pose.position,
            heading_deg=normalize_heading(self.state.virtual_pose.heading_deg + command.user_rotation_deg),
        )
        self._sync_physical_from_root_tracking()
        self._apply_reset_plane_rotation(command)
        self._sync_physical_from_root_tracking()

    def _apply_reset_plane_rotation(self, command) -> None:
        self._ensure_root_and_tracking_pose()
        self.state.virtual_pose = Pose2D(
            position=self.state.virtual_pose.position,
            heading_deg=normalize_heading(self.state.virtual_pose.heading_deg + command.plane_rotation_deg),
        )
        current_root = self.state.root_pose
        new_root_heading = normalize_heading(current_root.heading_deg + command.plane_rotation_deg)
        new_root_position = self._solve_root_position(
            self.state.virtual_pose.position,
            self.state.physical_pose.position,
            new_root_heading,
        )
        self.state.root_pose = Pose2D(new_root_position, new_root_heading)
        self._sync_tracking_space_from_root()

    def _advance_to_next_waypoint(self) -> None:
        if self.state.current_waypoint >= len(self.waypoints) - 1:
            self.state.mission_complete = True
            self.state.active_waypoint = None
            return
        self.state.current_waypoint += 1
        self.state.active_waypoint = self._current_waypoint_target()

    def _current_waypoint_target(self) -> Vector2:
        waypoints = self._resolved_waypoints()
        target_index = min(self.state.current_waypoint, len(waypoints) - 1)
        return waypoints[target_index]

    def _resolved_waypoints(self) -> list[Vector2]:
        if not self.waypoints:
            return []
        resolved = [self.waypoints[0]]
        for waypoint in self.waypoints[1:]:
            resolved.append(self._clamp_waypoint_to_virtual_rectangle(waypoint, resolved[-1]))
        return resolved

    def _clamp_waypoint_to_virtual_rectangle(self, position: Vector2, previous_position: Vector2) -> Vector2:
        if not self.environment.all_virtual_polygons:
            return position
        boundary = self.environment.all_virtual_polygons[0]
        if len(boundary) < 4:
            return position

        min_x = min(point.x for point in boundary)
        max_x = max(point.x for point in boundary)
        min_y = min(point.y for point in boundary)
        max_y = max(point.y for point in boundary)

        current_previous = previous_position
        current_position = position
        for _ in range(16):
            if min_x <= current_position.x <= max_x and min_y <= current_position.y <= max_y:
                return current_position

            direction = current_position - current_previous
            if direction.magnitude <= 1e-6:
                return Vector2(clamp(current_position.x, min_x, max_x), clamp(current_position.y, min_y, max_y))

            if current_position.x < min_x and abs(direction.x) > 1e-6:
                t = (min_x - current_previous.x) / direction.x
                intersection = current_previous + direction * t
                remaining_distance = direction.magnitude * (1.0 - t)
                reflected_direction = Vector2(-direction.x, direction.y).normalized()
                current_previous = intersection
                current_position = intersection + reflected_direction * remaining_distance
                continue
            if current_position.x > max_x and abs(direction.x) > 1e-6:
                t = (max_x - current_previous.x) / direction.x
                intersection = current_previous + direction * t
                remaining_distance = direction.magnitude * (1.0 - t)
                reflected_direction = Vector2(-direction.x, direction.y).normalized()
                current_previous = intersection
                current_position = intersection + reflected_direction * remaining_distance
                continue
            if current_position.y < min_y and abs(direction.y) > 1e-6:
                t = (min_y - current_previous.y) / direction.y
                intersection = current_previous + direction * t
                remaining_distance = direction.magnitude * (1.0 - t)
                reflected_direction = Vector2(direction.x, -direction.y).normalized()
                current_previous = intersection
                current_position = intersection + reflected_direction * remaining_distance
                continue
            if current_position.y > max_y and abs(direction.y) > 1e-6:
                t = (max_y - current_previous.y) / direction.y
                intersection = current_previous + direction * t
                remaining_distance = direction.magnitude * (1.0 - t)
                reflected_direction = Vector2(direction.x, -direction.y).normalized()
                current_previous = intersection
                current_position = intersection + reflected_direction * remaining_distance
                continue
            break

        return Vector2(clamp(current_position.x, min_x, max_x), clamp(current_position.y, min_y, max_y))

    def _capture_previous_poses(self) -> None:
        self.state.prev_virtual_pose = Pose2D(self.state.virtual_pose.position, self.state.virtual_pose.heading_deg)
        self.state.prev_physical_pose = Pose2D(self.state.physical_pose.position, self.state.physical_pose.heading_deg)
        if self.state.root_pose is not None:
            self.state.prev_root_pose = Pose2D(self.state.root_pose.position, self.state.root_pose.heading_deg)

    def _capture_observed_poses(self) -> None:
        self.state.observed_virtual_pose = Pose2D(self.state.virtual_pose.position, self.state.virtual_pose.heading_deg)
        self.state.observed_physical_pose = Pose2D(self.state.physical_pose.position, self.state.physical_pose.heading_deg)
        if self.state.root_pose is not None:
            self.state.observed_root_pose = Pose2D(self.state.root_pose.position, self.state.root_pose.heading_deg)
        if self.state.tracking_space_pose is not None:
            self.state.observed_tracking_space_pose = Pose2D(
                self.state.tracking_space_pose.position,
                self.state.tracking_space_pose.heading_deg,
            )

    def _clear_observed_poses(self) -> None:
        self.state.observed_virtual_pose = None
        self.state.observed_physical_pose = None
        self.state.observed_root_pose = None
        self.state.observed_tracking_space_pose = None

    def _ensure_root_and_tracking_pose(self) -> None:
        if self.state.root_pose is None:
            root_heading = normalize_heading(self.state.virtual_pose.heading_deg - self.state.physical_pose.heading_deg)
            root_position = self._solve_root_position(
                self.state.virtual_pose.position,
                self.state.physical_pose.position,
                root_heading,
            )
            self.state.root_pose = Pose2D(root_position, root_heading)
        self._sync_tracking_space_from_root()

    def _sync_tracking_space_from_root(self) -> None:
        root = self.state.root_pose
        tracking_world_position = root.position + self.state.tracking_space_local_position.rotate(root.heading_deg)
        tracking_world_heading = normalize_heading(root.heading_deg + self.state.tracking_space_local_heading_deg)
        self.state.tracking_space_pose = Pose2D(tracking_world_position, tracking_world_heading)

    def _sync_physical_from_root_tracking(self) -> None:
        self._ensure_root_and_tracking_pose()
        root = self.state.root_pose
        tracking = self.state.tracking_space_pose
        local_position = (self.state.virtual_pose.position - tracking.position).rotate(-tracking.heading_deg)
        local_heading = normalize_heading(self.state.virtual_pose.heading_deg - root.heading_deg)
        self.state.physical_pose = Pose2D(local_position, local_heading)

    def _solve_root_position(
        self,
        virtual_position: Vector2,
        physical_local_position: Vector2,
        root_heading_deg: float,
    ) -> Vector2:
        tracking_heading_deg = normalize_heading(root_heading_deg + self.state.tracking_space_local_heading_deg)
        tracking_origin = self.state.tracking_space_local_position.rotate(root_heading_deg)
        physical_offset = physical_local_position.rotate(tracking_heading_deg)
        return virtual_position - tracking_origin - physical_offset
