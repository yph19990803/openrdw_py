from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .geometry import Vector2, nearest_distance_to_polygons, normalize_heading
from .models import Environment, StepTrace


MAX_RESET_COUNT = 1000


@dataclass(frozen=True)
class AvatarSummary:
    values: dict[str, str]
    one_dimensional_samples: dict[str, list[float]]
    two_dimensional_samples: dict[str, list[Vector2]]


@dataclass(frozen=True)
class TrialSummary:
    end_state: int
    avatars: list[AvatarSummary]

    def end_state_to_string(self) -> str:
        if self.end_state == -1:
            return "Invalid"
        if self.end_state == 0:
            return "Normal"
        if self.end_state == 1:
            return "Manually"
        return "Undefined"


def _average(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _average_abs(values: Iterable[float]) -> float:
    values = list(values)
    return sum(abs(value) for value in values) / len(values) if values else 0.0


def _average_vec(values: Iterable[Vector2]) -> Vector2:
    values = list(values)
    if not values:
        return Vector2(0.0, 0.0)
    total = Vector2(0.0, 0.0)
    for value in values:
        total += value
    return total / len(values)


def _trace_positions(trace: list[StepTrace], kind: str) -> list[Vector2]:
    if kind == "virtual":
        return [
            Vector2(
                getattr(row, "observed_virtual_x", None) if getattr(row, "observed_virtual_x", None) is not None else row.virtual_x,
                getattr(row, "observed_virtual_y", None) if getattr(row, "observed_virtual_y", None) is not None else row.virtual_y,
            )
            for row in trace
        ]
    return [
        Vector2(
            getattr(row, "observed_physical_x", None) if getattr(row, "observed_physical_x", None) is not None else row.physical_x,
            getattr(row, "observed_physical_y", None) if getattr(row, "observed_physical_y", None) is not None else row.physical_y,
        )
        for row in trace
    ]


def _observed_virtual_delta(trace: list[StepTrace], index: int) -> Vector2:
    if index <= 0:
        return Vector2(0.0, 0.0)
    current = trace[index]
    previous = trace[index - 1]
    current_x = getattr(current, "observed_virtual_x", None)
    current_y = getattr(current, "observed_virtual_y", None)
    previous_x = getattr(previous, "observed_virtual_x", None)
    previous_y = getattr(previous, "observed_virtual_y", None)
    if None not in (current_x, current_y, previous_x, previous_y):
        return Vector2(current_x - previous_x, current_y - previous_y)
    return Vector2(current.virtual_x - previous.virtual_x, current.virtual_y - previous.virtual_y)


def _distance_deltas(points: list[Vector2]) -> list[float]:
    if len(points) < 2:
        return []
    return [(curr - prev).magnitude for prev, curr in zip(points, points[1:])]


def _reset_segments(trace: list[StepTrace], step_dt: float, virtual_positions: list[Vector2]) -> tuple[int, list[float], list[float]]:
    reset_count = 0
    distances_between_resets: list[float] = []
    times_between_resets: list[float] = []
    accumulated_distance = 0.0
    accumulated_time = 0.0
    for index, row in enumerate(trace):
        if index > 0:
            accumulated_distance += (virtual_positions[index] - virtual_positions[index - 1]).magnitude
            accumulated_time += step_dt
        prev_in_reset = trace[index - 1].in_reset if index > 0 else False
        if row.in_reset and not prev_in_reset:
            reset_count += 1
            distances_between_resets.append(accumulated_distance)
            times_between_resets.append(accumulated_time)
            accumulated_distance = 0.0
            accumulated_time = 0.0
    distances_between_resets.append(accumulated_distance)
    times_between_resets.append(accumulated_time)
    return reset_count, distances_between_resets, times_between_resets


@dataclass
class _SampleBuffers:
    user_real_positions: list[Vector2] = field(default_factory=list)
    user_virtual_positions: list[Vector2] = field(default_factory=list)
    translation_gains: list[float] = field(default_factory=list)
    injected_translations: list[float] = field(default_factory=list)
    rotation_gains: list[float] = field(default_factory=list)
    injected_rotations_from_rotation_gain: list[float] = field(default_factory=list)
    curvature_gains: list[float] = field(default_factory=list)
    injected_rotations_from_curvature_gain: list[float] = field(default_factory=list)
    injected_rotations: list[float] = field(default_factory=list)
    distances_to_boundary: list[float] = field(default_factory=list)
    distances_to_center: list[float] = field(default_factory=list)


@dataclass
class _SampleSeries:
    user_real_positions: list[Vector2] = field(default_factory=list)
    user_virtual_positions: list[Vector2] = field(default_factory=list)
    translation_gains: list[float] = field(default_factory=list)
    injected_translations: list[float] = field(default_factory=list)
    rotation_gains: list[float] = field(default_factory=list)
    injected_rotations_from_rotation_gain: list[float] = field(default_factory=list)
    curvature_gains: list[float] = field(default_factory=list)
    injected_rotations_from_curvature_gain: list[float] = field(default_factory=list)
    injected_rotations: list[float] = field(default_factory=list)
    distances_to_boundary: list[float] = field(default_factory=list)
    distances_to_center: list[float] = field(default_factory=list)
    sampling_intervals: list[float] = field(default_factory=list)


@dataclass
class _UnityLikeStats:
    reset_count: int
    virtual_distances_between_resets: list[float]
    time_elapsed_between_resets: list[float]
    sum_injected_translation: float
    sum_injected_rotation_from_rotation_gain: float
    sum_injected_rotation_from_curvature_gain: float
    sum_virtual_distance_travelled: float
    sum_real_distance_travelled: float
    min_translation_gain: float
    max_translation_gain: float
    min_rotation_gain: float
    max_rotation_gain: float
    min_curvature_gain: float
    max_curvature_gain: float
    samples: _SampleSeries


def _finalize_float_sample(samples: list[float], buffer: list[float]) -> None:
    samples.append(sum(buffer) / len(buffer) if buffer else 0.0)
    buffer.clear()


def _finalize_vec_sample(samples: list[Vector2], buffer: list[Vector2]) -> None:
    if buffer:
        samples.append(_average_vec(buffer))
    else:
        samples.append(Vector2(0.0, 0.0))
    buffer.clear()


def _collect_unity_like_stats(
    *,
    trace: list[StepTrace],
    environment: Environment,
    time_step: float,
    sampling_frequency: float,
    use_custom_sampling_frequency: bool,
) -> _UnityLikeStats:
    positions_virtual = _trace_positions(trace, "virtual")
    positions_physical = _trace_positions(trace, "physical")

    buffers = _SampleBuffers()
    samples = _SampleSeries()

    reset_count = 0
    virtual_distances_between_resets: list[float] = []
    time_elapsed_between_resets: list[float] = []
    sum_injected_translation = 0.0
    sum_injected_rotation_from_rotation_gain = 0.0
    sum_injected_rotation_from_curvature_gain = 0.0
    sum_virtual_distance_travelled = 0.0
    sum_real_distance_travelled = 0.0
    min_translation_gain = float("inf")
    max_translation_gain = float("-inf")
    min_rotation_gain = float("inf")
    max_rotation_gain = float("-inf")
    min_curvature_gain = float("inf")
    max_curvature_gain = float("-inf")
    virtual_distance_since_last_reset = 0.0
    time_of_last_reset = 0.0
    last_sampling_time = 0.0
    last_position_sampling_time = 0.0

    for index, row in enumerate(trace):
        current_time = (index + 1) * time_step
        prev_in_reset = trace[index - 1].in_reset if index > 0 else False
        entered_reset = row.in_reset and not prev_in_reset

        observed_delta = _observed_virtual_delta(trace, index)
        real_distance = observed_delta.magnitude
        translation_injection = Vector2(row.translation_injection_x, row.translation_injection_y).magnitude
        rotation_injection = abs(row.rotation_injection_deg)
        curvature_injection = abs(row.curvature_injection_deg)

        sum_real_distance_travelled += real_distance
        sum_virtual_distance_travelled += real_distance
        virtual_distance_since_last_reset += real_distance

        if translation_injection > 0.0:
            sum_injected_translation += translation_injection
            sum_virtual_distance_travelled += (1.0 if row.translation_gain >= 0.0 else -1.0) * translation_injection
            virtual_distance_since_last_reset += (1.0 if row.translation_gain >= 0.0 else -1.0) * translation_injection
            min_translation_gain = min(min_translation_gain, row.translation_gain)
            max_translation_gain = max(max_translation_gain, row.translation_gain)

        if rotation_injection > 0.0:
            sum_injected_rotation_from_rotation_gain += rotation_injection
            min_rotation_gain = min(min_rotation_gain, row.rotation_gain)
            max_rotation_gain = max(max_rotation_gain, row.rotation_gain)

        if curvature_injection > 0.0:
            sum_injected_rotation_from_curvature_gain += curvature_injection
            min_curvature_gain = min(min_curvature_gain, row.curvature_gain)
            max_curvature_gain = max(max_curvature_gain, row.curvature_gain)

        if translation_injection > 0.0:
            buffers.translation_gains.append(row.translation_gain * time_step)
            buffers.injected_translations.append(translation_injection * time_step)
        if rotation_injection > 0.0:
            buffers.rotation_gains.append(row.rotation_gain * time_step)
            buffers.injected_rotations_from_rotation_gain.append(rotation_injection * time_step)
            buffers.injected_rotations.append(rotation_injection * time_step)
        if curvature_injection > 0.0:
            buffers.curvature_gains.append(row.curvature_gain * time_step)
            buffers.injected_rotations_from_curvature_gain.append(curvature_injection * time_step)
            buffers.injected_rotations.append(curvature_injection * time_step)

        should_sample_position = (not use_custom_sampling_frequency) or (
            current_time - last_position_sampling_time > (1.0 / max(sampling_frequency, 1e-9))
        )
        if should_sample_position and not row.in_reset:
            buffers.user_real_positions.append(positions_physical[index])
            buffers.user_virtual_positions.append(positions_virtual[index])
            buffers.distances_to_boundary.append(nearest_distance_to_polygons(positions_physical[index], environment.all_polygons))
            buffers.distances_to_center.append(positions_physical[index].magnitude)
        if use_custom_sampling_frequency and should_sample_position:
            last_position_sampling_time = current_time

        if entered_reset:
            reset_count += 1
            virtual_distances_between_resets.append(virtual_distance_since_last_reset)
            time_elapsed_between_resets.append(current_time - time_of_last_reset)
            virtual_distance_since_last_reset = 0.0
            time_of_last_reset = current_time

        if current_time - last_sampling_time > (1.0 / max(sampling_frequency, 1e-9)):
            _finalize_vec_sample(samples.user_real_positions, buffers.user_real_positions)
            _finalize_vec_sample(samples.user_virtual_positions, buffers.user_virtual_positions)
            _finalize_float_sample(samples.translation_gains, buffers.translation_gains)
            _finalize_float_sample(samples.injected_translations, buffers.injected_translations)
            _finalize_float_sample(samples.rotation_gains, buffers.rotation_gains)
            _finalize_float_sample(samples.injected_rotations_from_rotation_gain, buffers.injected_rotations_from_rotation_gain)
            _finalize_float_sample(samples.curvature_gains, buffers.curvature_gains)
            _finalize_float_sample(samples.injected_rotations_from_curvature_gain, buffers.injected_rotations_from_curvature_gain)
            _finalize_float_sample(samples.injected_rotations, buffers.injected_rotations)
            _finalize_float_sample(samples.distances_to_boundary, buffers.distances_to_boundary)
            _finalize_float_sample(samples.distances_to_center, buffers.distances_to_center)
            samples.sampling_intervals.append(current_time - last_sampling_time)
            last_sampling_time = current_time

    experiment_duration = len(trace) * time_step
    virtual_distances_between_resets.append(virtual_distance_since_last_reset)
    time_elapsed_between_resets.append(experiment_duration - time_of_last_reset)

    return _UnityLikeStats(
        reset_count=reset_count,
        virtual_distances_between_resets=virtual_distances_between_resets,
        time_elapsed_between_resets=time_elapsed_between_resets,
        sum_injected_translation=sum_injected_translation,
        sum_injected_rotation_from_rotation_gain=sum_injected_rotation_from_rotation_gain,
        sum_injected_rotation_from_curvature_gain=sum_injected_rotation_from_curvature_gain,
        sum_virtual_distance_travelled=sum_virtual_distance_travelled,
        sum_real_distance_travelled=sum_real_distance_travelled,
        min_translation_gain=min_translation_gain,
        max_translation_gain=max_translation_gain,
        min_rotation_gain=min_rotation_gain,
        max_rotation_gain=max_rotation_gain,
        min_curvature_gain=min_curvature_gain,
        max_curvature_gain=max_curvature_gain,
        samples=samples,
    )


def summarize_agent_trace(
    *,
    trace: list[StepTrace],
    environment: Environment,
    waypoints: list[Vector2],
    descriptor: dict[str, str],
    time_step: float,
    sampling_frequency: float = 10.0,
    use_custom_sampling_frequency: bool = False,
    execute_duration: float | None = None,
    passive_target: Vector2 | None = None,
    passive_target_forward: Vector2 | None = None,
) -> AvatarSummary:
    physical_positions = _trace_positions(trace, "physical")
    translation_gains = [row.translation_gain for row in trace]
    rotation_gains = [row.rotation_gain for row in trace]
    curvature_gains = [row.curvature_gain for row in trace]
    unity_stats = _collect_unity_like_stats(
        trace=trace,
        environment=environment,
        time_step=time_step,
        sampling_frequency=sampling_frequency,
        use_custom_sampling_frequency=use_custom_sampling_frequency,
    )
    virtual_way_distance = sum((end - start).magnitude for start, end in zip(waypoints, waypoints[1:]))
    experiment_duration = len(trace) * time_step
    average_sampling_interval = _average(unity_stats.samples.sampling_intervals)

    position_error = 0.0
    angle_error = 0.0
    if passive_target is not None and physical_positions:
        position_error = (passive_target - physical_positions[-1]).magnitude
    if passive_target_forward is not None and trace:
        final_heading = (
            getattr(trace[-1], "observed_physical_heading_deg", None)
            if getattr(trace[-1], "observed_physical_heading_deg", None) is not None
            else trace[-1].physical_heading_deg
        )
        target_heading = normalize_heading(passive_target_forward.heading_deg if hasattr(passive_target_forward, "heading_deg") else 0.0)
        target_forward_heading = passive_target_forward
        if isinstance(target_forward_heading, Vector2) and target_forward_heading.magnitude > 1e-6:
            from .geometry import vector_to_heading

            target_heading = vector_to_heading(target_forward_heading)
        angle_error = abs(normalize_heading(final_heading - target_heading))
        if angle_error > 180.0:
            angle_error = 360.0 - angle_error

    values = dict(descriptor)
    values["reset_count"] = str(unity_stats.reset_count)
    values["virtual_way_distance"] = str(virtual_way_distance)
    values["virtual_distance_between_resets_average"] = str(_average(unity_stats.virtual_distances_between_resets))
    values["time_elapsed_between_resets_average"] = str(_average(unity_stats.time_elapsed_between_resets))
    values["sum_injected_translation(IN METERS)"] = str(unity_stats.sum_injected_translation)
    values["sum_injected_rotation_g_r(IN DEGREES)"] = str(unity_stats.sum_injected_rotation_from_rotation_gain)
    values["sum_injected_rotation_g_c(IN DEGREES)"] = str(unity_stats.sum_injected_rotation_from_curvature_gain)
    values["sum_real_distance_travelled(IN METERS)"] = str(unity_stats.sum_real_distance_travelled)
    values["sum_virtual_distance_travelled(IN METERS)"] = str(unity_stats.sum_virtual_distance_travelled)
    values["min_g_t"] = str(unity_stats.min_translation_gain) if unity_stats.min_translation_gain != float("inf") else "N/A"
    values["max_g_t"] = str(unity_stats.max_translation_gain) if unity_stats.max_translation_gain != float("-inf") else "N/A"
    values["min_g_r"] = str(unity_stats.min_rotation_gain) if unity_stats.min_rotation_gain != float("inf") else "N/A"
    values["max_g_r"] = str(unity_stats.max_rotation_gain) if unity_stats.max_rotation_gain != float("-inf") else "N/A"
    values["min_g_c"] = str(unity_stats.min_curvature_gain) if unity_stats.min_curvature_gain != float("inf") else "N/A"
    values["max_g_c"] = str(unity_stats.max_curvature_gain) if unity_stats.max_curvature_gain != float("-inf") else "N/A"
    values["g_t_average"] = str(_average_abs(unity_stats.samples.translation_gains))
    values["injected_translation_average"] = str(_average(unity_stats.samples.injected_translations))
    values["g_r_average"] = str(_average_abs(unity_stats.samples.rotation_gains))
    values["injected_rotation_from_rotation_gain_average"] = str(_average(unity_stats.samples.injected_rotations_from_rotation_gain))
    values["g_c_average"] = str(_average_abs(unity_stats.samples.curvature_gains))
    values["injected_rotation_from_curvature_gain_average"] = str(_average(unity_stats.samples.injected_rotations_from_curvature_gain))
    values["injected_rotation_average"] = str(_average(unity_stats.samples.injected_rotations))
    values["real_position_average"] = str(_average_vec(unity_stats.samples.user_real_positions))
    values["virtual_position_average"] = str(_average_vec(unity_stats.samples.user_virtual_positions))
    values["distance_to_boundary_average"] = str(_average(unity_stats.samples.distances_to_boundary))
    values["distance_to_center_average"] = str(_average(unity_stats.samples.distances_to_center))
    values["experiment_duration"] = str(experiment_duration)
    values["execute_duration"] = str(execute_duration if execute_duration is not None else experiment_duration)
    values["average_sampling_interval"] = str(average_sampling_interval)
    if passive_target is not None:
        values["positionError"] = str(position_error)
        values["angleError"] = str(angle_error)

    one_dimensional_samples = {
        "distances_to_boundary": unity_stats.samples.distances_to_boundary,
        "distances_to_center": unity_stats.samples.distances_to_center,
        "g_t": unity_stats.samples.translation_gains,
        "injected_translations": unity_stats.samples.injected_translations,
        "g_r": unity_stats.samples.rotation_gains,
        "injected_rotations_from_rotation_gain": unity_stats.samples.injected_rotations_from_rotation_gain,
        "g_c": unity_stats.samples.curvature_gains,
        "injected_rotations_from_curvature_gain": unity_stats.samples.injected_rotations_from_curvature_gain,
        "injected_rotations": unity_stats.samples.injected_rotations,
        "virtual_distances_between_resets": unity_stats.virtual_distances_between_resets,
        "time_elapsed_between_resets": unity_stats.time_elapsed_between_resets,
        "sampling_intervals": unity_stats.samples.sampling_intervals,
    }
    two_dimensional_samples = {
        "user_real_positions": unity_stats.samples.user_real_positions,
        "user_virtual_positions": unity_stats.samples.user_virtual_positions,
    }
    return AvatarSummary(values, one_dimensional_samples, two_dimensional_samples)


def export_summary_results_scsv(
    path: str | Path,
    trial_summaries: list[TrialSummary],
    experiment_start_time: str,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        handle.write("sep=;\n")
        for trial_id, trial_summary in enumerate(trial_summaries):
            handle.write(f"TrialId = {trial_id};EndState = {trial_summary.end_state_to_string()}\n")
            if trial_summary.avatars:
                headers = list(trial_summary.avatars[0].values.keys())
                handle.write("experiment_start_time;")
                handle.write(";".join(headers))
                handle.write(";\n")
                for avatar in trial_summary.avatars:
                    handle.write(experiment_start_time + ";")
                    handle.write(";".join(avatar.values.get(header, "") for header in headers))
                    handle.write(";\n")
            if trial_id < len(trial_summaries) - 1:
                handle.write("\n")


def export_sampled_metrics(
    output_dir: str | Path,
    trial_name: str,
    avatar_summaries: list[AvatarSummary],
) -> None:
    root = Path(output_dir) / trial_name
    for avatar_index, avatar_summary in enumerate(avatar_summaries):
        avatar_dir = root / f"userId_{avatar_index}"
        avatar_dir.mkdir(parents=True, exist_ok=True)
        for name, values in avatar_summary.one_dimensional_samples.items():
            with (avatar_dir / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                for value in values:
                    writer.writerow([value])
        for name, values in avatar_summary.two_dimensional_samples.items():
            with (avatar_dir / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                for value in values:
                    writer.writerow([value.x, value.y])
