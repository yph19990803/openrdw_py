from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass, replace
from pathlib import Path

from .factory import (
    SimulationConfig,
    align_waypoints_to_initial_configuration,
    build_environment,
    build_gains,
    build_redirector,
    build_resetter,
    build_waypoints,
)
from .exporters import export_real_path_graph_png, export_trace_csv
from .geometry import Vector2, vector_to_heading
from .models import AgentState, InitialConfiguration, Pose2D
from .scheduler import MultiAgentScheduler, ScheduledAgent
from .stats import MAX_RESET_COUNT, TrialSummary, export_sampled_metrics, export_summary_results_scsv, summarize_agent_trace
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


PATH_NAME_MAP = {
    "90turn": "ninety_turn",
    "randomturn": "random_turn",
    "straightline": "straight_line",
    "sawtooth": "sawtooth",
    "circle": "circle",
    "figureeight": "figure_eight",
    "filepath": "file_path",
    "realuserpath": "real_user_path",
}

TRACKING_NAME_MAP = {
    "rectangle": "rectangle",
    "trapezoid": "trapezoid",
    "triangle": "triangle",
    "cross": "cross",
    "l_shape": "l_shape",
    "t_shape": "t_shape",
    "square": "square",
    "filepath": "file_path",
}

REDIRECTOR_NAME_MAP = {
    "null": "none",
    "s2c": "s2c",
    "s2o": "s2o",
    "zigzag": "zigzag",
    "thomasapf": "thomas_apf",
    "messingerapf": "messinger_apf",
    "dynamicapf": "dynamic_apf",
    "deeplearning": "deep_learning",
    "passivehapticapf": "passive_haptic_apf",
    "vispoly": "vispoly",
}

RESETTER_NAME_MAP = {
    "null": "none",
    "twooneturn": "two_one_turn",
    "apf": "apf",
}


@dataclass(frozen=True)
class AvatarExperimentSpec:
    redirector: str = "none"
    resetter: str = "none"
    path_mode: str = "straight_line"
    waypoints_file: str | None = None
    sampling_intervals_file: str | None = None
    initial_configuration: InitialConfiguration | None = None


@dataclass(frozen=True)
class TrialSpec:
    avatars: tuple[AvatarExperimentSpec, ...]
    config: SimulationConfig


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _decode_initial_configuration(value: str) -> InitialConfiguration:
    split = value.split(",")
    return InitialConfiguration(
        Vector2(float(split[0]), float(split[1])),
        Vector2(float(split[2]), float(split[3])).normalized(),
    )


def _rotate_waypoints(waypoints: list[Vector2], angle_deg: float) -> list[Vector2]:
    return [point.rotate(angle_deg) for point in waypoints]


def parse_command_file(path: str | Path) -> list[TrialSpec]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    trial_specs: list[TrialSpec] = []
    avatar = AvatarExperimentSpec()
    avatars: list[AvatarExperimentSpec] = []
    current_config = SimulationConfig(
        redirector="none",
        resetter="none",
        path_mode="straight_line",
        tracking_space_shape="rectangle",
        total_path_length=400.0,
    )
    first_avatar = True

    for line in lines:
        if not line.strip():
            continue
        split = line.split(",") if "," in line else line.split(" ")
        keyword = split[0].strip().lower()
        value = split[2].strip()
        if keyword == "newuser":
            if first_avatar:
                first_avatar = False
            else:
                avatars.append(avatar)
                avatar = replace(avatar, initial_configuration=None)
            continue
        if keyword == "redirector":
            avatar = replace(avatar, redirector=REDIRECTOR_NAME_MAP.get(value.lower(), "none"))
            continue
        if keyword == "resetter":
            avatar = replace(avatar, resetter=RESETTER_NAME_MAP.get(value.lower(), "none"))
            continue
        if keyword == "pathseedchoice":
            avatar = replace(avatar, path_mode=PATH_NAME_MAP.get(value.lower(), "straight_line"))
            continue
        if keyword == "waypointsfilepath":
            avatar = replace(avatar, waypoints_file=value)
            continue
        if keyword == "samplingintervalsfilepath":
            avatar = replace(avatar, sampling_intervals_file=value)
            continue
        if keyword == "trackingspacechoice":
            current_config = replace(current_config, tracking_space_shape=TRACKING_NAME_MAP.get(value.lower(), "rectangle"))
            continue
        if keyword == "obstacletype":
            current_config = replace(current_config, obstacle_type=int(value))
            continue
        if keyword == "squarewidth":
            current_config = replace(current_config, physical_width=float(value), physical_height=float(value))
            continue
        if keyword == "trackingspacefilepath":
            current_config = replace(current_config, tracking_space_file=value)
            continue
        if keyword == "initialconfiguration":
            avatar = replace(avatar, initial_configuration=_decode_initial_configuration(value))
            continue
        if keyword == "max_trans_gain":
            current_config = replace(current_config, max_trans_gain=float(value))
            continue
        if keyword == "min_trans_gain":
            current_config = replace(current_config, min_trans_gain=float(value))
            continue
        if keyword == "max_rot_gain":
            current_config = replace(current_config, max_rot_gain=float(value))
            continue
        if keyword == "min_rot_gain":
            current_config = replace(current_config, min_rot_gain=float(value))
            continue
        if keyword == "curvature_radius":
            current_config = replace(current_config, curvature_radius=float(value))
            continue
        if keyword == "reset_trigger_buffer":
            current_config = replace(
                current_config,
                reset_trigger_buffer=float(value),
                physical_space_buffer=float(value),
                obstacle_buffer=float(value),
            )
            continue
        if keyword == "samplingfrequency":
            current_config = replace(current_config, sampling_frequency=float(value))
            continue
        if keyword == "usecustomsamplingfrequency":
            current_config = replace(current_config, use_custom_sampling_frequency=_parse_bool(value))
            continue
        if keyword == "generatedpathlength":
            current_config = replace(current_config, total_path_length=float(value))
            continue
        if keyword == "firstwaypointisstartpoint":
            current_config = replace(current_config, first_waypoint_is_start_point=_parse_bool(value))
            continue
        if keyword == "aligntoinitialforward":
            current_config = replace(current_config, align_to_initial_forward=_parse_bool(value))
            continue
        if keyword == "translationspeed":
            current_config = replace(current_config, translation_speed=float(value))
            continue
        if keyword == "rotationspeed":
            current_config = replace(current_config, rotation_speed=float(value))
            continue
        if keyword == "drawrealtrail":
            current_config = replace(current_config, draw_real_trail=_parse_bool(value))
            continue
        if keyword == "drawvirtualtrail":
            current_config = replace(current_config, draw_virtual_trail=_parse_bool(value))
            continue
        if keyword == "trailvisualtime":
            current_config = replace(current_config, trail_visual_time=float(value))
            continue
        if keyword == "virtualworldvisible":
            current_config = replace(current_config, virtual_world_visible=_parse_bool(value))
            continue
        if keyword == "trackingspacevisible":
            current_config = replace(current_config, tracking_space_visible=_parse_bool(value))
            continue
        if keyword == "buffervisible":
            current_config = replace(current_config, buffer_visible=_parse_bool(value))
            continue
        if keyword == "end":
            avatars.append(avatar)
            trial_specs.append(TrialSpec(tuple(avatars), current_config))
            avatar = AvatarExperimentSpec()
            avatars = []
            current_config = replace(current_config)
            first_avatar = True
            continue
        raise ValueError(f"Invalid command line: {line}")
    return trial_specs


def _tracking_layout(config: SimulationConfig, avatar_count: int) -> tuple[list[Vector2], list[list[Vector2]], list[InitialConfiguration]]:
    if config.tracking_space_shape == "rectangle":
        return generate_rectangle_tracking_space(config.physical_width, config.physical_height, config.obstacle_type)
    if config.tracking_space_shape == "square":
        return generate_square_tracking_space(config.physical_width, config.obstacle_type)
    if config.tracking_space_shape == "triangle":
        return generate_triangle_tracking_space(config.obstacle_type)
    if config.tracking_space_shape == "trapezoid":
        return generate_trapezoid_tracking_space(config.obstacle_type)
    if config.tracking_space_shape == "cross":
        return generate_cross_tracking_space(config.obstacle_type)
    if config.tracking_space_shape == "l_shape":
        return generate_l_shape_tracking_space(config.obstacle_type)
    if config.tracking_space_shape == "t_shape":
        return generate_t_shape_tracking_space(config.obstacle_type)
    if config.tracking_space_shape == "file_path":
        tracking, obstacles = load_tracking_space_from_file(config.tracking_space_file or "")
        initials = [InitialConfiguration(tracking[0], (Vector2(0.0, 0.0) - tracking[0]).normalized())]
        return tracking, obstacles, initials
    raise ValueError(f"Unsupported tracking space shape: {config.tracking_space_shape}")


def _procedural_waypoint_cache(trial: TrialSpec, seed: int) -> dict[str, list[Vector2]]:
    cache: dict[str, list[Vector2]] = {}
    rng = random.Random(seed)
    for avatar in trial.avatars:
        if avatar.path_mode in {"file_path", "real_user_path"}:
            continue
        if avatar.path_mode in cache:
            continue
        waypoints, _ = build_waypoints(replace(trial.config, path_mode=avatar.path_mode, seed=seed))
        cache[avatar.path_mode] = _rotate_waypoints(waypoints, rng.uniform(0.0, 360.0))
    return cache


def build_scheduler_for_trial(trial: TrialSpec, seed: int | None = None) -> tuple[MultiAgentScheduler, list[dict[str, str]]]:
    trial_seed = trial.config.seed if seed is None else seed
    tracking_space, obstacle_polygons, initial_configurations = _tracking_layout(trial.config, len(trial.avatars))
    physical_obstacle_count = len(obstacle_polygons)
    environment = build_environment(
        replace(
            trial.config,
            seed=trial_seed,
            physical_obstacle_count=physical_obstacle_count,
        )
    )
    environment = replace_environment(environment, tracking_space, obstacle_polygons)
    gains = build_gains(trial.config)
    procedural_cache = _procedural_waypoint_cache(trial, trial_seed)
    agents: list[ScheduledAgent] = []
    descriptors: list[dict[str, str]] = []
    for index, avatar in enumerate(trial.avatars):
        initial = avatar.initial_configuration or initial_configurations[index % len(initial_configurations)]
        if avatar.path_mode in {"file_path", "real_user_path"}:
            base_waypoints, sampling_intervals = build_waypoints(
                replace(
                    trial.config,
                    path_mode=avatar.path_mode,
                    waypoints_file=avatar.waypoints_file,
                    sampling_intervals_file=avatar.sampling_intervals_file,
                    seed=trial_seed,
                )
            )
        else:
            base_waypoints = procedural_cache[avatar.path_mode]
            sampling_intervals = None
        waypoints = align_waypoints_to_initial_configuration(
            base_waypoints,
            initial.position,
            initial.forward,
            first_waypoint_is_start_point=trial.config.first_waypoint_is_start_point,
            align_to_initial_forward=trial.config.align_to_initial_forward,
        )
        heading = vector_to_heading(initial.forward)
        pose = Pose2D(initial.position, heading)
        state = AgentState(
            virtual_pose=pose,
            physical_pose=pose,
            agent_index=index,
            if_just_end_reset=True,
            active_waypoint=waypoints[0] if waypoints else None,
            final_waypoint=waypoints[-1] if waypoints else None,
        )
        agents.append(
            ScheduledAgent(
                agent_id=str(index),
                state=state,
                environment=environment,
                gains=gains,
                redirector=build_redirector(avatar.redirector),
                resetter=build_resetter(avatar.resetter),
                waypoints=waypoints,
                sampling_intervals=sampling_intervals,
            )
        )
        descriptors.append(
            {
                "trackingSpace": trial.config.tracking_space_shape,
                "squareWidth": str(trial.config.physical_width) if trial.config.tracking_space_shape == "square" else "",
                "obstacleType": str(trial.config.obstacle_type),
                "pathSeedChoice": avatar.path_mode,
                "redirector": avatar.redirector,
                "resetter": avatar.resetter,
            }
        )
    return MultiAgentScheduler(agents), descriptors


def replace_environment(environment, tracking_space: list[Vector2], obstacles: list[list[Vector2]]):
    environment.tracking_space = tracking_space
    environment.obstacles = obstacles
    return environment


def run_trial(
    trial: TrialSpec,
    *,
    seed: int | None = None,
    max_steps: int = 30000,
    sampling_frequency: float = 10.0,
) -> tuple[MultiAgentScheduler, TrialSummary]:
    wall_clock_start = time.perf_counter()
    scheduler, descriptors = build_scheduler_for_trial(trial, seed=seed)
    end_state = 0
    for step_index in range(max_steps):
        scheduler.step(step_index)
        all_complete = True
        for index, agent in enumerate(scheduler.agents):
            if not agent.state.mission_complete:
                all_complete = False
            reset_count = sum(1 for idx, row in enumerate(agent.trace) if row.in_reset and (idx == 0 or not agent.trace[idx - 1].in_reset))
            if reset_count > MAX_RESET_COUNT or agent.state.same_pos_time > 50.0:
                end_state = -1
                all_complete = True
                break
        if all_complete:
            break
    avatar_summaries = []
    execute_duration = time.perf_counter() - wall_clock_start
    for index, agent in enumerate(scheduler.agents):
        passive_target = agent.environment.physical_targets[index] if index < len(agent.environment.physical_targets) else None
        passive_target_forward = (
            agent.environment.physical_target_forwards[index] if index < len(agent.environment.physical_target_forwards) else None
        )
        active_sampling_frequency = trial.config.sampling_frequency if trial.config.sampling_frequency > 0 else sampling_frequency
        avatar_summaries.append(
            summarize_agent_trace(
                trace=agent.trace,
                environment=agent.environment,
                waypoints=agent.waypoints,
                descriptor=descriptors[index],
                time_step=agent.gains.time_step,
                sampling_frequency=active_sampling_frequency,
                use_custom_sampling_frequency=trial.config.use_custom_sampling_frequency,
                execute_duration=execute_duration,
                passive_target=passive_target,
                passive_target_forward=passive_target_forward,
            )
        )
    return scheduler, TrialSummary(end_state=end_state, avatars=avatar_summaries)


def _collect_command_files(command_path: str | Path) -> list[Path]:
    root = Path(command_path)
    if root.is_dir():
        return sorted(path for path in root.rglob("*") if path.is_file())
    return [root]


def _output_dir_for_command_path(command_root: Path, command_file: Path, output_root: Path, multi_mode: bool) -> Path:
    if not multi_mode:
        return output_root
    relative = command_file.relative_to(command_root)
    parts = list(relative.parts[:-1]) + [relative.stem]
    return output_root.joinpath(*parts)


def run_command_file(
    command_file: str | Path,
    *,
    output_dir: str | Path,
    max_steps: int = 30000,
    sampling_frequency: float = 10.0,
) -> list[TrialSummary]:
    summaries: list[TrialSummary] = []
    command_paths = _collect_command_files(command_file)
    command_root = Path(command_file) if Path(command_file).is_dir() else Path(command_file).parent
    multi_mode = len(command_paths) > 1
    output_root = Path(output_dir)

    for command_index, command_path in enumerate(command_paths):
        trials = parse_command_file(command_path)
        command_output_root = _output_dir_for_command_path(command_root, command_path, output_root, multi_mode)
        trace_dir = command_output_root / "traces"
        sampled_dir = command_output_root / "sampled_metrics"
        graph_dir = command_output_root / "graphs"
        tmp_dir = command_output_root / "tmp"
        (command_output_root / "videos").mkdir(parents=True, exist_ok=True)
        (command_output_root / "screenshots").mkdir(parents=True, exist_ok=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        file_summaries: list[TrialSummary] = []
        start_time = time.strftime("%Y%m%d_%H%M%S")
        for trial_index, trial in enumerate(trials):
            scheduler, summary = run_trial(
                trial,
                seed=trial.config.seed + trial_index,
                max_steps=max_steps,
                sampling_frequency=sampling_frequency,
            )
            summaries.append(summary)
            file_summaries.append(summary)
            for agent in scheduler.agents:
                export_trace_csv(trace_dir / f"trial_{trial_index}_agent_{agent.agent_id}.csv", agent.trace)
            export_sampled_metrics(sampled_dir, f"trialId_{trial_index}", summary.avatars)
            if scheduler.agents:
                environment = scheduler.agents[0].environment
                export_real_path_graph_png(
                    graph_dir / f"trial_{trial_index}_real_path.png",
                    tracking_space=environment.tracking_space,
                    obstacles=environment.obstacles,
                    user_real_paths=[
                        [Vector2(row.physical_x, row.physical_y) for row in agent.trace]
                        for agent in scheduler.agents
                    ],
                )
            (tmp_dir / f"{command_index}-{len(command_paths)} {trial_index}-{len(trials)}.txt").write_text("", encoding="utf-8")
        export_summary_results_scsv(command_output_root / "summary.csv", file_summaries, start_time)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenRDW command-file experiments in Python")
    parser.add_argument("--command-file", required=True)
    parser.add_argument("--output-dir", default="python_openrdw/out/experiments")
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument("--sampling-frequency", type=float, default=10.0)
    args = parser.parse_args()
    summaries = run_command_file(
        args.command_file,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        sampling_frequency=args.sampling_frequency,
    )
    print(f"Completed {len(summaries)} trial(s) into {args.output_dir}")


if __name__ == "__main__":
    main()
