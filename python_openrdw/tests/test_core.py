import tempfile
import time
import unittest
from math import isfinite
from pathlib import Path

from openrdw import (
    AgentState,
    ApfResetter,
    DeepLearningRedirector,
    DynamicApfRedirector,
    Environment,
    GainsConfig,
    MessingerApfRedirector,
    MultiAgentScheduler,
    NullResetter,
    OpenRDWSimulator,
    PassiveHapticApfRedirector,
    PathSeed,
    Pose2D,
    S2CRedirector,
    ScheduledAgent,
    SimulationConfig,
    ThomasApfRedirector,
    TwoOneTurnResetter,
    VisPolyRedirector,
    ZigZagRedirector,
    build_environment,
    build_scheduler,
    export_trace_csv,
    generate_cross_tracking_space,
    generate_initial_path_by_seed,
    generate_l_shape_tracking_space,
    generate_square_tracking_space,
    generate_t_shape_tracking_space,
    generate_trapezoid_tracking_space,
    load_sampling_intervals_from_file,
    load_tracking_space_from_file,
    load_waypoints_from_file,
    summarize_agent_trace,
)
from openrdw.experiments import (
    AvatarExperimentSpec,
    TrialSpec,
    build_scheduler_for_trial,
    parse_command_file,
    run_command_file,
)
from openrdw.geometry import Vector2, signed_angle
from openrdw.redirectors import _physical_to_virtual_point
from openrdw.ui import SimulationSession, build_index_html
from openrdw.visibility import compute_visibility_polygon


class _ThresholdResetter:
    def __init__(self, threshold: float):
        self.threshold = threshold

    def is_reset_required(self, state, environment, gains, other_agents):
        for other in other_agents:
            if getattr(other, "agent_index", None) == getattr(state, "agent_index", None):
                continue
            if (other.curr_pos_real - state.curr_pos_real).magnitude < self.threshold:
                return True
        return False

    def begin(self, state, environment, gains):
        return None

    def simulated_walker_update(self, state, environment, gains):
        return 0.0

    def inject_resetting(self, state, environment, gains, delta_rotation_deg):
        return type("Reset", (), {"plane_rotation_deg": 0.0, "user_rotation_deg": 0.0, "finished": True})()

    def step(self, state, environment, gains):
        return self.inject_resetting(state, environment, gains, 0.0)


class _PeerLoggingRedirector:
    def __init__(self):
        self.logged_peer_positions = []

    def inject(self, state, environment, gains, other_agents):
        self.logged_peer_positions.append(
            sorted(
                (
                    getattr(other, "agent_index", -1),
                    round(other.curr_pos.x, 4),
                    round(other.curr_pos.y, 4),
                )
                for other in other_agents
            )
        )
        return type(
            "Command",
            (),
            {
                "translation": Vector2(0.0, 0.0),
                "rotation_deg": 0.0,
                "curvature_deg": 0.0,
                "priority": 0.0,
                "translation_gain": 0.0,
                "rotation_gain": 0.0,
                "curvature_gain": 0.0,
                "debug": {},
            },
        )()

    def get_priority(self, state, environment, gains, other_agents):
        return 10.0 - state.agent_index


class GeometryTests(unittest.TestCase):
    def test_signed_angle_clockwise_is_negative_in_current_convention(self):
        angle = signed_angle(Vector2(0.0, 1.0), Vector2(1.0, 0.0))
        self.assertAlmostEqual(angle, -90.0, places=4)

    def test_square_tracking_space_has_four_corners(self):
        tracking, obstacles, initials = generate_square_tracking_space(20.0, obstacle_type=1)
        self.assertEqual(len(tracking), 4)
        self.assertEqual(len(obstacles), 1)
        self.assertGreaterEqual(len(initials), 1)

    def test_other_tracking_spaces_generate_vertices(self):
        self.assertGreaterEqual(len(generate_trapezoid_tracking_space()[0]), 4)
        self.assertGreaterEqual(len(generate_cross_tracking_space()[0]), 8)
        self.assertGreaterEqual(len(generate_l_shape_tracking_space()[0]), 6)
        self.assertGreaterEqual(len(generate_t_shape_tracking_space()[0]), 8)


class PathTests(unittest.TestCase):
    def test_seeded_path_generation_reaches_target_distance(self):
        points = generate_initial_path_by_seed(PathSeed.straight_line(), 40.0)
        self.assertGreaterEqual(len(points), 2)
        total = 0.0
        for start, end in zip(points, points[1:]):
            total += (end - start).magnitude
        self.assertAlmostEqual(total, 40.0, places=3)

    def test_waypoint_and_tracking_file_loaders_work(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            waypoint_file = tmp / "waypoints.txt"
            sampling_file = tmp / "sampling.txt"
            tracking_file = tmp / "tracking.txt"
            waypoint_file.write_text("0,0\n1,2\n3,4\n", encoding="utf-8")
            sampling_file.write_text("0.1\n0.2\n0.3\n", encoding="utf-8")
            tracking_file.write_text("1,1\n-1,1\n-1,-1\n1,-1\n\n0.2,0.2\n0.2,-0.2\n-0.2,-0.2\n-0.2,0.2\n", encoding="utf-8")
            self.assertEqual(len(load_waypoints_from_file(waypoint_file)), 3)
            self.assertEqual(len(load_sampling_intervals_from_file(sampling_file)), 3)
            tracking, obstacles = load_tracking_space_from_file(tracking_file)
            self.assertEqual(len(tracking), 4)
            self.assertEqual(len(obstacles), 1)


class RedirectorResetterTests(unittest.TestCase):
    def test_gains_config_uses_new_buffer_defaults(self):
        gains = GainsConfig()
        self.assertAlmostEqual(gains.body_collider_diameter, 0.1)
        self.assertAlmostEqual(gains.physical_space_buffer, 0.4)
        self.assertAlmostEqual(gains.obstacle_buffer, 0.4)

    def test_s2c_redirector_produces_nonzero_rotation_while_walking(self):
        tracking, obstacles, _ = generate_square_tracking_space(20.0)
        env = Environment(tracking, obstacles)
        state = AgentState(
            virtual_pose=Pose2D(Vector2(4.0, 0.0), 180.0),
            physical_pose=Pose2D(Vector2(4.0, 0.0), 180.0),
            base_delta_translation=Vector2(0.0, 0.1),
            base_delta_rotation_deg=5.0,
        )
        command = S2CRedirector().inject(state, env, GainsConfig(time_step=0.1), [state])
        self.assertNotEqual(command.rotation_deg + command.curvature_deg, 0.0)

    def test_apf_resetter_finishes_after_multiple_steps(self):
        tracking, obstacles, _ = generate_square_tracking_space(20.0)
        env = Environment(tracking, obstacles)
        state = AgentState(
            virtual_pose=Pose2D(Vector2(0.0, 0.0), 0.0),
            physical_pose=Pose2D(Vector2(0.0, 0.0), 0.0),
            total_force=Vector2(1.0, 0.0),
        )
        resetter = ApfResetter()
        gains = GainsConfig(time_step=0.1, rotation_speed=90.0)
        resetter.begin(state, env, gains)
        finished = False
        for _ in range(50):
            command = resetter.step(state, env, gains)
            if command.finished:
                finished = True
                break
        self.assertTrue(finished)

    def test_apf_resetter_rotates_toward_total_force(self):
        tracking, obstacles, _ = generate_square_tracking_space(20.0)
        env = Environment(tracking, obstacles)
        state = AgentState(
            virtual_pose=Pose2D(Vector2(0.0, 0.0), 0.0),
            physical_pose=Pose2D(Vector2(0.0, 0.0), 0.0),
            total_force=Vector2(1.0, 0.0),
        )
        resetter = ApfResetter()
        gains = GainsConfig(time_step=0.1, rotation_speed=90.0)
        resetter.begin(state, env, gains)
        for _ in range(50):
            command = resetter.step(state, env, gains)
            state.physical_pose = Pose2D(
                state.physical_pose.position,
                state.physical_pose.heading_deg + command.user_rotation_deg,
            )
            state.virtual_pose = Pose2D(
                state.virtual_pose.position,
                state.virtual_pose.heading_deg + command.user_rotation_deg - command.plane_rotation_deg,
            )
            if command.finished:
                break
        self.assertAlmostEqual(state.physical_pose.heading_deg % 360.0, 90.0, delta=2.0)

    def test_apf_resetter_handles_obtuse_heading_case(self):
        tracking, obstacles, _ = generate_square_tracking_space(20.0)
        env = Environment(tracking, obstacles)
        state = AgentState(
            virtual_pose=Pose2D(Vector2(0.0, 0.0), 58.5),
            physical_pose=Pose2D(Vector2(0.0, 0.0), 58.5),
            total_force=Vector2(0.0, -1.0),
        )
        resetter = ApfResetter()
        gains = GainsConfig(time_step=0.1, rotation_speed=90.0)
        resetter.begin(state, env, gains)
        for _ in range(50):
            command = resetter.step(state, env, gains)
            state.physical_pose = Pose2D(
                state.physical_pose.position,
                state.physical_pose.heading_deg + command.user_rotation_deg,
            )
            state.virtual_pose = Pose2D(
                state.virtual_pose.position,
                state.virtual_pose.heading_deg + command.user_rotation_deg - command.plane_rotation_deg,
            )
            if command.finished:
                break
        self.assertAlmostEqual(state.physical_pose.heading_deg % 360.0, 180.0, delta=2.0)

    def test_resetter_ignores_same_agent_by_agent_index_not_object_identity(self):
        tracking, obstacles, _ = generate_square_tracking_space(20.0)
        env = Environment(tracking, obstacles)
        state = AgentState(
            virtual_pose=Pose2D(Vector2(0.0, 0.0), 0.0),
            physical_pose=Pose2D(Vector2(0.0, 0.0), 0.0),
            agent_index=0,
        )
        mirrored_state = AgentState(
            virtual_pose=Pose2D(Vector2(0.0, 0.0), 0.0),
            physical_pose=Pose2D(Vector2(0.0, 0.0), 0.0),
            agent_index=0,
        )
        resetter = TwoOneTurnResetter()
        gains = GainsConfig(time_step=0.1, physical_space_buffer=0.4)
        self.assertFalse(resetter.is_reset_required(state, env, gains, [state, mirrored_state]))

    def test_messinger_apf_returns_force_metadata(self):
        tracking, obstacles, _ = generate_square_tracking_space(20.0)
        env = Environment(tracking, obstacles)
        state = AgentState(
            virtual_pose=Pose2D(Vector2(0.0, 0.0), 0.0),
            physical_pose=Pose2D(Vector2(0.0, 0.0), 0.0),
            base_delta_translation=Vector2(0.0, 0.1),
            base_delta_rotation_deg=2.0,
        )
        command = MessingerApfRedirector().inject(state, env, GainsConfig(time_step=0.1), [state])
        self.assertIn("force_x", command.debug)
        self.assertIn("force_y", command.debug)

    def test_visibility_polygon_computes_vertices(self):
        tracking, obstacles, _ = generate_square_tracking_space(20.0)
        poly = compute_visibility_polygon(Vector2(0.0, 0.0), [tracking, *obstacles])
        self.assertGreaterEqual(len(poly), 4)

    def test_thomas_apf_redirector_returns_repulsive_force(self):
        tracking, obstacles, _ = generate_square_tracking_space(20.0, obstacle_type=1)
        env = Environment(tracking, obstacles)
        state = AgentState(
            virtual_pose=Pose2D(Vector2(0.5, 0.0), 0.0),
            physical_pose=Pose2D(Vector2(0.5, 0.0), 0.0),
            base_delta_translation=Vector2(0.0, 0.1),
            base_delta_rotation_deg=2.0,
        )
        command = ThomasApfRedirector().inject(state, env, GainsConfig(time_step=0.1), [state])
        self.assertEqual(command.debug["mode"], "thomas_apf")
        self.assertIn("repulsive_force", command.debug)

    def test_zigzag_redirector_returns_target_metadata(self):
        tracking, obstacles, _ = generate_square_tracking_space(20.0)
        env = Environment(tracking, obstacles, physical_targets=[Vector2(0.0, 0.0), Vector2(3.0, 3.0)])
        state = AgentState(
            virtual_pose=Pose2D(Vector2(0.0, 0.0), 0.0),
            physical_pose=Pose2D(Vector2(0.0, 0.0), 0.0),
            active_waypoint=Vector2(3.0, 0.0),
            base_delta_translation=Vector2(0.0, 0.1),
            base_delta_rotation_deg=2.0,
        )
        command = ZigZagRedirector().inject(state, env, GainsConfig(time_step=0.1), [state])
        self.assertEqual(command.debug["mode"], "zigzag")
        self.assertIn("real_target_x", command.debug)

    def test_passive_haptic_redirector_uses_physical_targets(self):
        tracking, obstacles, _ = generate_square_tracking_space(20.0)
        env = Environment(tracking, obstacles, physical_targets=[Vector2(1.0, 1.0)])
        state = AgentState(
            virtual_pose=Pose2D(Vector2(0.0, 0.0), 0.0),
            physical_pose=Pose2D(Vector2(0.0, 0.0), 0.0),
            base_delta_translation=Vector2(0.0, 0.1),
            base_delta_rotation_deg=2.0,
        )
        command = PassiveHapticApfRedirector().inject(state, env, GainsConfig(time_step=0.1), [state])
        self.assertEqual(command.debug["mode"], "passive_haptic_apf")
        self.assertEqual(command.debug["target_x"], 1.0)

    def test_passive_haptic_redirector_uses_agent_specific_target_index(self):
        tracking, obstacles, _ = generate_square_tracking_space(20.0)
        env = Environment(tracking, obstacles, physical_targets=[Vector2(1.0, 1.0), Vector2(2.0, -1.0)])
        state0 = AgentState(
            virtual_pose=Pose2D(Vector2(0.0, 0.0), 0.0),
            physical_pose=Pose2D(Vector2(0.0, 0.0), 0.0),
            agent_index=0,
            base_delta_translation=Vector2(0.0, 0.1),
            base_delta_rotation_deg=2.0,
        )
        state1 = AgentState(
            virtual_pose=Pose2D(Vector2(0.5, 0.0), 0.0),
            physical_pose=Pose2D(Vector2(0.5, 0.0), 0.0),
            agent_index=1,
            base_delta_translation=Vector2(0.0, 0.1),
            base_delta_rotation_deg=2.0,
        )
        command = PassiveHapticApfRedirector().inject(state1, env, GainsConfig(time_step=0.1), [state0, state1])
        self.assertEqual(command.debug["target_x"], 2.0)
        self.assertEqual(command.debug["target_y"], -1.0)

    def test_passive_haptic_alignment_state_latches_true_when_condition_met(self):
        tracking, obstacles, _ = generate_square_tracking_space(20.0)
        env = Environment(
            tracking,
            obstacles,
            physical_targets=[Vector2(0.0, 3.0)],
            physical_target_forwards=[Vector2(0.0, 1.0)],
        )
        redirector = PassiveHapticApfRedirector()
        state = AgentState(
            virtual_pose=Pose2D(Vector2(0.0, 0.0), 0.0),
            physical_pose=Pose2D(Vector2(0.0, 0.0), 0.0),
            active_waypoint=Vector2(0.0, 3.1),
            final_waypoint=Vector2(0.0, 3.1),
            base_delta_translation=Vector2(0.0, 0.1),
            base_delta_rotation_deg=0.0,
        )
        command = redirector.inject(state, env, GainsConfig(time_step=0.1), [state])
        self.assertTrue(command.debug["alignment_state"])
        command = redirector.inject(state, env, GainsConfig(time_step=0.1), [state])
        self.assertTrue(command.debug["alignment_state"])

    def test_passive_haptic_alignment_requires_final_waypoint_like_unity(self):
        tracking, obstacles, _ = generate_square_tracking_space(20.0)
        env = Environment(
            tracking,
            obstacles,
            physical_targets=[Vector2(0.0, 3.0)],
            physical_target_forwards=[Vector2(0.0, 1.0)],
        )
        redirector = PassiveHapticApfRedirector()
        state = AgentState(
            virtual_pose=Pose2D(Vector2(0.0, 0.0), 0.0),
            physical_pose=Pose2D(Vector2(0.0, 0.0), 0.0),
            active_waypoint=Vector2(0.0, 3.1),
            final_waypoint=None,
            base_delta_translation=Vector2(0.0, 0.1),
            base_delta_rotation_deg=0.0,
        )
        command = redirector.inject(state, env, GainsConfig(time_step=0.1), [state])
        self.assertFalse(command.debug["alignment_state"])

    def test_physical_to_virtual_point_uses_tracking_space_heading(self):
        state = AgentState(
            virtual_pose=Pose2D(Vector2(10.0, 0.0), 0.0),
            physical_pose=Pose2D(Vector2(1.0, 0.0), 0.0),
            observed_virtual_pose=Pose2D(Vector2(10.0, 0.0), 0.0),
            observed_physical_pose=Pose2D(Vector2(1.0, 0.0), 0.0),
            root_pose=Pose2D(Vector2(0.0, 0.0), 10.0),
            tracking_space_pose=Pose2D(Vector2(0.0, 0.0), 40.0),
            observed_tracking_space_pose=Pose2D(Vector2(0.0, 0.0), 40.0),
        )
        mapped = _physical_to_virtual_point(state, Vector2(2.0, 0.0))
        expected = Vector2(10.0, 0.0) + Vector2(1.0, 0.0).rotate(40.0)
        self.assertAlmostEqual(mapped.x, expected.x, places=6)
        self.assertAlmostEqual(mapped.y, expected.y, places=6)

    def test_dynamic_apf_priority_uses_unity_angle_weighting(self):
        tracking, obstacles, _ = generate_square_tracking_space(20.0)
        env = Environment(tracking, obstacles)
        state = AgentState(
            virtual_pose=Pose2D(Vector2(4.0, 4.0), 180.0),
            physical_pose=Pose2D(Vector2(4.0, 4.0), 180.0),
            base_delta_translation=Vector2(0.0, 0.1),
            base_delta_rotation_deg=2.0,
        )
        redirector = DynamicApfRedirector()
        priority = redirector.get_priority(state, env, GainsConfig(time_step=0.1), [state])
        self.assertLess(priority, 0.0)

    def test_dynamic_apf_inject_uses_combined_force_for_debug_and_base_force_for_reset(self):
        tracking, obstacles, _ = generate_square_tracking_space(20.0)
        env = Environment(tracking, obstacles)
        state = AgentState(
            virtual_pose=Pose2D(Vector2(4.0, 4.0), 180.0),
            physical_pose=Pose2D(Vector2(4.0, 4.0), 180.0),
            base_delta_translation=Vector2(0.0, 0.1),
            base_delta_rotation_deg=2.0,
        )
        redirector = DynamicApfRedirector()
        raw_force = redirector.get_total_force(state, env, [state]).normalized()
        command = redirector.inject(state, env, GainsConfig(time_step=0.1), [state])
        self.assertEqual(command.debug["mode"], "dynamic_apf")
        self.assertAlmostEqual(command.debug["reset_force_x"], state.total_force.x, places=6)
        self.assertAlmostEqual(command.debug["reset_force_y"], state.total_force.y, places=6)
        self.assertAlmostEqual(state.total_force.x, raw_force.x, places=6)
        self.assertAlmostEqual(state.total_force.y, raw_force.y, places=6)

    def test_deep_learning_redirector_fails_softly_without_runtime(self):
        tracking, obstacles, _ = generate_square_tracking_space(20.0)
        env = Environment(tracking, obstacles)
        state = AgentState(
            virtual_pose=Pose2D(Vector2(0.0, 0.0), 0.0),
            physical_pose=Pose2D(Vector2(0.0, 0.0), 0.0),
            base_delta_translation=Vector2(0.0, 0.1),
            base_delta_rotation_deg=2.0,
        )
        command = DeepLearningRedirector().inject(state, env, GainsConfig(time_step=0.1), [state])
        self.assertIn(command.debug["mode"], {"deep_learning_unavailable", "deep_learning_warmup", "deep_learning"})

    def test_deep_learning_state_uses_unity_signed_angle_from_right(self):
        env = Environment(
            tracking_space=[Vector2(2.0, 2.0), Vector2(-2.0, 2.0), Vector2(-2.0, -2.0), Vector2(2.0, -2.0)],
            obstacles=[],
        )
        redirector = DeepLearningRedirector()
        state = AgentState(
            virtual_pose=Pose2D(Vector2(0.0, 0.0), 0.0),
            physical_pose=Pose2D(Vector2(0.0, 0.0), 0.0),
        )
        redirector._add_state(state, env)
        self.assertAlmostEqual(redirector.state_vectors[-1][2], 0.75, places=6)
        redirector.state_vectors.clear()
        state.observed_physical_pose = Pose2D(Vector2(0.0, 0.0), 270.0)
        redirector._add_state(state, env)
        self.assertAlmostEqual(redirector.state_vectors[-1][2], 1.0, places=6)

    def test_deep_learning_redirector_prefers_named_output_24(self):
        import openrdw.redirectors as redirector_module

        class _FakeOutput:
            def __init__(self, name):
                self.name = name

        class _FakeInput:
            def __init__(self, name):
                self.name = name

        class _FakeSession:
            def get_inputs(self):
                return [_FakeInput("input")]

            def get_outputs(self):
                return [_FakeOutput("24"), _FakeOutput("other")]

            def run(self, *_args, **_kwargs):
                import numpy as np

                return [
                    np.array([[0.25, -0.5, 0.75]], dtype=np.float32),
                    np.array([[9.0, 9.0, 9.0]], dtype=np.float32),
                ]

        env = Environment(
            tracking_space=[Vector2(2.0, 2.0), Vector2(-2.0, 2.0), Vector2(-2.0, -2.0), Vector2(2.0, -2.0)],
            obstacles=[],
        )
        state = AgentState(
            virtual_pose=Pose2D(Vector2(0.0, 0.0), 0.0),
            physical_pose=Pose2D(Vector2(0.0, 0.0), 0.0),
            base_delta_translation=Vector2(0.0, 0.1),
            base_delta_rotation_deg=2.0,
            if_just_end_reset=False,
        )
        redirector = DeepLearningRedirector(wait_time=1)
        redirector.session = _FakeSession()
        old_ort = redirector_module.ort
        redirector_module.ort = object()
        try:
            command = redirector.inject(state, env, GainsConfig(time_step=0.1), [state])
        finally:
            redirector_module.ort = old_ort
        self.assertAlmostEqual(redirector.action_mean[0], 0.25, places=6)
        self.assertAlmostEqual(command.translation_gain, redirector._convert(-1.0, 1.0, -0.14, 0.26, 0.25), places=6)

    def test_vispoly_redirector_returns_gradient_metadata(self):
        tracking, obstacles, _ = generate_square_tracking_space(20.0)
        virtual_obstacles = [
            [
                Vector2(-8.0, -8.0),
                Vector2(8.0, -8.0),
                Vector2(8.0, 8.0),
                Vector2(-8.0, 8.0),
            ],
            [
                Vector2(-1.0, -1.0),
                Vector2(-1.0, 1.0),
                Vector2(1.0, 1.0),
                Vector2(1.0, -1.0),
            ],
        ]
        env = Environment(tracking, obstacles, virtual_obstacles=virtual_obstacles)
        state = AgentState(
            virtual_pose=Pose2D(Vector2(0.0, -3.0), 0.0),
            physical_pose=Pose2D(Vector2(0.0, -3.0), 0.0),
            base_delta_translation=Vector2(0.0, 0.1),
            base_delta_rotation_deg=3.0,
        )
        command = VisPolyRedirector().inject(state, env, GainsConfig(time_step=0.1), [state])
        self.assertEqual(command.debug["mode"], "vispoly")
        self.assertIn("negative_gradient_x", command.debug)


class SchedulerTests(unittest.TestCase):
    def test_scheduler_uses_snapshot_peers_for_movement_phase(self):
        env = Environment(
            tracking_space=[Vector2(3.0, 3.0), Vector2(-3.0, 3.0), Vector2(-3.0, -3.0), Vector2(3.0, -3.0)],
            obstacles=[],
        )
        gains = GainsConfig(time_step=0.1, translation_speed=2.0, rotation_speed=180.0)
        state0 = AgentState(
            virtual_pose=Pose2D(Vector2(0.0, 0.0), 0.0),
            physical_pose=Pose2D(Vector2(0.0, 0.0), 0.0),
            agent_index=0,
            current_waypoint=1,
            active_waypoint=Vector2(0.0, 2.0),
            if_just_end_reset=True,
        )
        state1 = AgentState(
            virtual_pose=Pose2D(Vector2(0.0, 1.0), 180.0),
            physical_pose=Pose2D(Vector2(0.0, 1.0), 180.0),
            agent_index=1,
            current_waypoint=1,
            active_waypoint=Vector2(0.0, -1.0),
            if_just_end_reset=True,
        )
        scheduler = MultiAgentScheduler(
            [
                ScheduledAgent(
                    agent_id="0",
                    state=state0,
                    environment=env,
                    gains=gains,
                    redirector=_PeerLoggingRedirector(),
                    resetter=_ThresholdResetter(0.95),
                    waypoints=[Vector2(0.0, 0.0), Vector2(0.0, 2.0)],
                ),
                ScheduledAgent(
                    agent_id="1",
                    state=state1,
                    environment=env,
                    gains=gains,
                    redirector=_PeerLoggingRedirector(),
                    resetter=_ThresholdResetter(0.95),
                    waypoints=[Vector2(0.0, 1.0), Vector2(0.0, -1.0)],
                ),
            ]
        )
        scheduler.step(0)
        self.assertGreater(scheduler.agents[0].state.base_delta_translation.magnitude, 0.0)
        self.assertGreater(scheduler.agents[1].state.base_delta_translation.magnitude, 0.0)

    def test_scheduler_uses_post_movement_snapshot_for_redirection_phase(self):
        env = Environment(
            tracking_space=[Vector2(3.0, 3.0), Vector2(-3.0, 3.0), Vector2(-3.0, -3.0), Vector2(3.0, -3.0)],
            obstacles=[],
        )
        gains = GainsConfig(time_step=0.1, translation_speed=1.0, rotation_speed=180.0)
        redirector0 = _PeerLoggingRedirector()
        redirector1 = _PeerLoggingRedirector()
        scheduler = MultiAgentScheduler(
            [
                ScheduledAgent(
                    agent_id="0",
                    state=AgentState(
                        virtual_pose=Pose2D(Vector2(0.0, 0.0), 0.0),
                        physical_pose=Pose2D(Vector2(0.0, 0.0), 0.0),
                        agent_index=0,
                        current_waypoint=1,
                        active_waypoint=Vector2(0.0, 2.0),
                        if_just_end_reset=True,
                    ),
                    environment=env,
                    gains=gains,
                    redirector=redirector0,
                    resetter=NullResetter(),
                    waypoints=[Vector2(0.0, 0.0), Vector2(0.0, 2.0)],
                ),
                ScheduledAgent(
                    agent_id="1",
                    state=AgentState(
                        virtual_pose=Pose2D(Vector2(1.0, 0.0), 0.0),
                        physical_pose=Pose2D(Vector2(1.0, 0.0), 0.0),
                        agent_index=1,
                        current_waypoint=1,
                        active_waypoint=Vector2(1.0, 2.0),
                        if_just_end_reset=True,
                    ),
                    environment=env,
                    gains=gains,
                    redirector=redirector1,
                    resetter=NullResetter(),
                    waypoints=[Vector2(1.0, 0.0), Vector2(1.0, 2.0)],
                ),
            ]
        )
        scheduler.step(0)
        self.assertTrue(redirector0.logged_peer_positions)
        self.assertTrue(redirector1.logged_peer_positions)
        expected_y0 = round(scheduler.agents[0].state.curr_pos.y, 4)
        expected_y1 = round(scheduler.agents[1].state.curr_pos.y, 4)
        self.assertIn((0, 0.0, expected_y0), redirector0.logged_peer_positions[0])
        self.assertIn((1, 1.0, expected_y1), redirector0.logged_peer_positions[0])
        self.assertIn((0, 0.0, expected_y0), redirector1.logged_peer_positions[0])
        self.assertIn((1, 1.0, expected_y1), redirector1.logged_peer_positions[0])

    def test_scheduler_starts_with_first_waypoint_like_unity(self):
        scheduler = build_scheduler(
            SimulationConfig(
                redirector="none",
                resetter="none",
                path_mode="straight_line",
                tracking_space_shape="rectangle",
                agent_count=1,
                total_path_length=20.0,
            )
        )
        agent = scheduler.agents[0]
        self.assertEqual(agent.state.current_waypoint, 0)
        self.assertIsNotNone(agent.state.active_waypoint)
        self.assertAlmostEqual(agent.state.active_waypoint.x, agent.state.virtual_pose.position.x, places=4)
        self.assertAlmostEqual(agent.state.active_waypoint.y, agent.state.virtual_pose.position.y, places=4)

    def test_scheduler_advances_from_start_waypoint_before_first_motion_like_unity(self):
        scheduler = build_scheduler(
            SimulationConfig(
                redirector="none",
                resetter="none",
                path_mode="random_turn",
                tracking_space_shape="rectangle",
                agent_count=1,
                total_path_length=20.0,
                translation_speed=1.0,
                rotation_speed=180.0,
                time_step=1.0 / 60.0,
                seed=3041,
            )
        )
        agent = scheduler.agents[0]
        start_position = agent.state.virtual_pose.position
        scheduler.step(0)
        self.assertEqual(agent.state.current_waypoint, 1)
        self.assertGreater(agent.state.delta_pos.magnitude, 0.0)
        self.assertNotAlmostEqual(agent.state.virtual_pose.position.x, start_position.x, places=6)
        self.assertNotAlmostEqual(agent.state.virtual_pose.position.y, start_position.y, places=6)

    def test_waypoint_reflects_back_inside_virtual_rectangle(self):
        env = Environment(
            tracking_space=[
                Vector2(10.0, 10.0),
                Vector2(-10.0, 10.0),
                Vector2(-10.0, -10.0),
                Vector2(10.0, -10.0),
            ],
            virtual_obstacles=[
                [
                    Vector2(1.0, 1.0),
                    Vector2(-1.0, 1.0),
                    Vector2(-1.0, -1.0),
                    Vector2(1.0, -1.0),
                ]
            ],
        )
        simulator = OpenRDWSimulator(
            environment=env,
            gains=GainsConfig(time_step=0.1),
            redirector=S2CRedirector(),
            resetter=TwoOneTurnResetter(),
            waypoints=[Vector2(0.0, 0.0), Vector2(1.5, 0.5)],
            state=AgentState(
                virtual_pose=Pose2D(Vector2(0.0, 0.0), 0.0),
                physical_pose=Pose2D(Vector2(0.0, 0.0), 0.0),
            ),
        )
        simulator.step(0, [simulator.state])
        self.assertIsNotNone(simulator.state.active_waypoint)
        self.assertAlmostEqual(simulator.state.active_waypoint.x, 0.5, places=4)
        self.assertAlmostEqual(simulator.state.active_waypoint.y, 0.5, places=4)
        self.assertLessEqual(abs(simulator.state.active_waypoint.x), 1.0)
        self.assertLessEqual(abs(simulator.state.active_waypoint.y), 1.0)

    def test_waypoint_multiple_reflections_still_end_inside_virtual_rectangle(self):
        env = Environment(
            tracking_space=[
                Vector2(10.0, 10.0),
                Vector2(-10.0, 10.0),
                Vector2(-10.0, -10.0),
                Vector2(10.0, -10.0),
            ],
            virtual_obstacles=[
                [
                    Vector2(1.0, 1.0),
                    Vector2(-1.0, 1.0),
                    Vector2(-1.0, -1.0),
                    Vector2(1.0, -1.0),
                ]
            ],
        )
        simulator = OpenRDWSimulator(
            environment=env,
            gains=GainsConfig(time_step=0.1),
            redirector=S2CRedirector(),
            resetter=TwoOneTurnResetter(),
            waypoints=[Vector2(0.0, 0.0), Vector2(0.0, -5.0)],
            state=AgentState(
                virtual_pose=Pose2D(Vector2(0.0, 0.0), 0.0),
                physical_pose=Pose2D(Vector2(0.0, 0.0), 0.0),
            ),
        )
        simulator.step(0, [simulator.state])
        self.assertIsNotNone(simulator.state.active_waypoint)
        self.assertGreaterEqual(simulator.state.active_waypoint.x, -1.0)
        self.assertLessEqual(simulator.state.active_waypoint.x, 1.0)
        self.assertGreaterEqual(simulator.state.active_waypoint.y, -1.0)
        self.assertLessEqual(simulator.state.active_waypoint.y, 1.0)

    def test_build_scheduler_for_trial_applies_command_gains(self):
        trial = TrialSpec(
            avatars=(AvatarExperimentSpec(redirector="s2c", resetter="apf", path_mode="straight_line"),),
            config=SimulationConfig(
                tracking_space_shape="rectangle",
                obstacle_type=1,
                total_path_length=40.0,
                max_trans_gain=0.4,
                min_trans_gain=-0.3,
                max_rot_gain=0.7,
                min_rot_gain=-0.4,
                curvature_radius=8.0,
                reset_trigger_buffer=0.35,
            ),
        )
        scheduler, descriptors = build_scheduler_for_trial(trial, seed=42)
        self.assertEqual(len(scheduler.agents), 1)
        self.assertEqual(descriptors[0]["redirector"], "s2c")
        self.assertAlmostEqual(scheduler.agents[0].gains.max_trans_gain, 0.4)
        self.assertAlmostEqual(scheduler.agents[0].gains.reset_trigger_buffer, 0.35)

    def test_scheduler_manual_keyboard_input_moves_agent(self):
        scheduler = build_scheduler(
            SimulationConfig(
                redirector="none",
                resetter="none",
                movement_controller="keyboard",
                path_mode="straight_line",
                agent_count=1,
                time_step=0.1,
                translation_speed=1.0,
            )
        )
        start = scheduler.agents[0].state.virtual_pose.position
        scheduler.step(0, manual_inputs={"0": {"w": True, "a": False, "s": False, "d": False, "left": False, "right": False}})
        end = scheduler.agents[0].state.virtual_pose.position
        self.assertGreater((end - start).magnitude, 0.05)

    def test_command_file_runner_exports_summary_and_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            command_file = tmp / "commands.txt"
            command_file.write_text(
                "\n".join(
                    [
                        "newUser = 0",
                        "redirector = s2c",
                        "resetter = twoOneTurn",
                        "pathSeedChoice = straightLine",
                        "trackingSpaceChoice = rectangle",
                        "obstacleType = 1",
                        "max_trans_gain = 0.33",
                        "reset_trigger_buffer = 0.4",
                        "end = 1",
                    ]
                ),
                encoding="utf-8",
            )
            parsed = parse_command_file(command_file)
            self.assertEqual(len(parsed), 1)
            self.assertAlmostEqual(parsed[0].config.max_trans_gain, 0.33)
            self.assertAlmostEqual(parsed[0].config.reset_trigger_buffer, 0.4)
            summaries = run_command_file(command_file, output_dir=tmp / "out", max_steps=2000, sampling_frequency=5.0)
            self.assertEqual(len(summaries), 1)
            self.assertTrue((tmp / "out" / "summary.csv").exists())
            self.assertTrue((tmp / "out" / "traces" / "trial_0_agent_0.csv").exists())
            self.assertTrue((tmp / "out" / "sampled_metrics" / "trialId_0" / "userId_0" / "g_t.csv").exists())
            summary_text = (tmp / "out" / "summary.csv").read_text(encoding="utf-8")
            self.assertIn("execute_duration", summary_text)

    def test_command_file_parser_supports_sampling_and_trail_flags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            command_file = tmp / "commands.txt"
            command_file.write_text(
                "\n".join(
                    [
                        "newUser = 0",
                        "redirector = s2c",
                        "resetter = twoOneTurn",
                        "pathSeedChoice = straightLine",
                        "trackingSpaceChoice = rectangle",
                        "samplingFrequency = 5",
                        "useCustomSamplingFrequency = true",
                        "generatedPathLength = 60",
                        "firstWayPointIsStartPoint = false",
                        "alignToInitialForward = false",
                        "translationSpeed = 1.5",
                        "rotationSpeed = 120",
                        "drawRealTrail = false",
                        "drawVirtualTrail = true",
                        "trailVisualTime = 12",
                        "virtualWorldVisible = false",
                        "trackingSpaceVisible = true",
                        "bufferVisible = false",
                        "end = 1",
                    ]
                ),
                encoding="utf-8",
            )
            parsed = parse_command_file(command_file)
            self.assertEqual(len(parsed), 1)
            config = parsed[0].config
            self.assertAlmostEqual(config.sampling_frequency, 5.0)
            self.assertTrue(config.use_custom_sampling_frequency)
            self.assertAlmostEqual(config.total_path_length, 60.0)
            self.assertFalse(config.first_waypoint_is_start_point)
            self.assertFalse(config.align_to_initial_forward)
            self.assertAlmostEqual(config.translation_speed, 1.5)
            self.assertAlmostEqual(config.rotation_speed, 120.0)
            self.assertFalse(config.draw_real_trail)
            self.assertTrue(config.draw_virtual_trail)
            self.assertAlmostEqual(config.trail_visual_time, 12.0)
            self.assertFalse(config.virtual_world_visible)
            self.assertTrue(config.tracking_space_visible)
            self.assertFalse(config.buffer_visible)

    def test_command_directory_runner_creates_per_file_outputs_and_tmp_markers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            command_root = tmp / "commands"
            nested = command_root / "group_a"
            nested.mkdir(parents=True)
            for command_file in [command_root / "one.txt", nested / "two.txt"]:
                command_file.write_text(
                    "\n".join(
                        [
                            "newUser = 0",
                            "redirector = s2c",
                            "resetter = twoOneTurn",
                            "pathSeedChoice = straightLine",
                            "trackingSpaceChoice = rectangle",
                            "end = 1",
                        ]
                    ),
                    encoding="utf-8",
                )
            output_dir = tmp / "out"
            summaries = run_command_file(command_root, output_dir=output_dir, max_steps=800, sampling_frequency=5.0)
            self.assertEqual(len(summaries), 2)
            self.assertTrue((output_dir / "one" / "summary.csv").exists())
            self.assertTrue((output_dir / "group_a" / "two" / "summary.csv").exists())
            self.assertTrue(any((output_dir / "one" / "tmp").iterdir()))
            self.assertTrue(any((output_dir / "group_a" / "two" / "tmp").iterdir()))

    def test_summarize_agent_trace_respects_custom_sampling_frequency(self):
        trace = [
            type("Trace", (), {
                "virtual_x": 0.0, "virtual_y": 0.0, "virtual_heading_deg": 0.0,
                "physical_x": 0.0, "physical_y": 0.0, "physical_heading_deg": 0.0,
                "in_reset": False,
                "translation_injection_x": 0.0, "translation_injection_y": 0.0,
                "rotation_injection_deg": 0.0, "curvature_injection_deg": 0.0,
                "translation_gain": 0.0, "rotation_gain": 0.0, "curvature_gain": 0.0,
                "total_force_x": 0.0, "total_force_y": 0.0, "priority": 0.0,
            })(),
            type("Trace", (), {
                "virtual_x": 0.1, "virtual_y": 0.0, "virtual_heading_deg": 0.0,
                "physical_x": 0.1, "physical_y": 0.0, "physical_heading_deg": 0.0,
                "in_reset": True,
                "translation_injection_x": 0.0, "translation_injection_y": 0.0,
                "rotation_injection_deg": 0.0, "curvature_injection_deg": 0.0,
                "translation_gain": 0.0, "rotation_gain": 0.0, "curvature_gain": 0.0,
                "total_force_x": 0.0, "total_force_y": 0.0, "priority": 0.0,
            })(),
            type("Trace", (), {
                "virtual_x": 0.2, "virtual_y": 0.0, "virtual_heading_deg": 0.0,
                "physical_x": 0.2, "physical_y": 0.0, "physical_heading_deg": 0.0,
                "in_reset": False,
                "translation_injection_x": 0.0, "translation_injection_y": 0.0,
                "rotation_injection_deg": 0.0, "curvature_injection_deg": 0.0,
                "translation_gain": 0.0, "rotation_gain": 0.0, "curvature_gain": 0.0,
                "total_force_x": 0.0, "total_force_y": 0.0, "priority": 0.0,
            })(),
        ]
        env = Environment(
            tracking_space=[Vector2(1.0, 1.0), Vector2(-1.0, 1.0), Vector2(-1.0, -1.0), Vector2(1.0, -1.0)],
            obstacles=[],
        )
        summary = summarize_agent_trace(
            trace=trace,
            environment=env,
            waypoints=[Vector2(0.0, 0.0), Vector2(1.0, 0.0)],
            descriptor={},
            time_step=0.1,
            sampling_frequency=5.0,
            use_custom_sampling_frequency=True,
        )
        self.assertEqual(len(summary.two_dimensional_samples["user_real_positions"]), 1)
        self.assertAlmostEqual(summary.two_dimensional_samples["user_real_positions"][0].x, 0.2, places=4)

    def test_summarize_agent_trace_passive_haptics_angle_error_uses_observed_heading(self):
        trace = [
            type("Trace", (), {
                "virtual_x": 0.0, "virtual_y": 0.0, "virtual_heading_deg": 0.0,
                "physical_x": 0.0, "physical_y": 0.0, "physical_heading_deg": 90.0,
                "observed_physical_x": 0.0, "observed_physical_y": 0.0, "observed_physical_heading_deg": 0.0,
                "in_reset": False,
                "translation_injection_x": 0.0, "translation_injection_y": 0.0,
                "rotation_injection_deg": 0.0, "curvature_injection_deg": 0.0,
                "translation_gain": 0.0, "rotation_gain": 0.0, "curvature_gain": 0.0,
                "total_force_x": 0.0, "total_force_y": 0.0, "priority": 0.0,
            })(),
        ]
        env = Environment(
            tracking_space=[Vector2(1.0, 1.0), Vector2(-1.0, 1.0), Vector2(-1.0, -1.0), Vector2(1.0, -1.0)],
            obstacles=[],
        )
        summary = summarize_agent_trace(
            trace=trace,
            environment=env,
            waypoints=[Vector2(0.0, 0.0), Vector2(1.0, 0.0)],
            descriptor={},
            time_step=0.1,
            passive_target=Vector2(0.0, 0.0),
            passive_target_forward=Vector2(0.0, 1.0),
        )
        self.assertEqual(summary.values["angleError"], "0.0")

    def test_summarize_agent_trace_distance_sums_use_observed_virtual_motion(self):
        trace = [
            type("Trace", (), {
                "virtual_x": 0.0, "virtual_y": 0.0, "virtual_heading_deg": 0.0,
                "physical_x": 0.0, "physical_y": 0.0, "physical_heading_deg": 0.0,
                "observed_virtual_x": 0.0, "observed_virtual_y": 0.0,
                "observed_physical_x": 0.0, "observed_physical_y": 0.0,
                "in_reset": False,
                "translation_injection_x": 0.0, "translation_injection_y": 0.0,
                "rotation_injection_deg": 0.0, "curvature_injection_deg": 0.0,
                "translation_gain": 0.0, "rotation_gain": 0.0, "curvature_gain": 0.0,
                "total_force_x": 0.0, "total_force_y": 0.0, "priority": 0.0,
            })(),
            type("Trace", (), {
                "virtual_x": 10.0, "virtual_y": 0.0, "virtual_heading_deg": 0.0,
                "physical_x": 0.0, "physical_y": 0.0, "physical_heading_deg": 0.0,
                "observed_virtual_x": 0.5, "observed_virtual_y": 0.0,
                "observed_physical_x": 0.0, "observed_physical_y": 0.0,
                "in_reset": False,
                "translation_injection_x": 0.0, "translation_injection_y": 0.0,
                "rotation_injection_deg": 0.0, "curvature_injection_deg": 0.0,
                "translation_gain": 0.0, "rotation_gain": 0.0, "curvature_gain": 0.0,
                "total_force_x": 0.0, "total_force_y": 0.0, "priority": 0.0,
            })(),
        ]
        env = Environment(
            tracking_space=[Vector2(1.0, 1.0), Vector2(-1.0, 1.0), Vector2(-1.0, -1.0), Vector2(1.0, -1.0)],
            obstacles=[],
        )
        summary = summarize_agent_trace(
            trace=trace,
            environment=env,
            waypoints=[Vector2(0.0, 0.0), Vector2(1.0, 0.0)],
            descriptor={},
            time_step=0.1,
        )
        self.assertEqual(summary.values["sum_real_distance_travelled(IN METERS)"], "0.5")
        self.assertEqual(summary.values["sum_virtual_distance_travelled(IN METERS)"], "0.5")

    def test_summarize_agent_trace_samples_match_unity_event_buffer_semantics(self):
        trace = [
            type("Trace", (), {
                "virtual_x": 0.0, "virtual_y": 0.0, "virtual_heading_deg": 0.0,
                "physical_x": 0.0, "physical_y": 0.0, "physical_heading_deg": 0.0,
                "in_reset": False,
                "translation_injection_x": 0.1, "translation_injection_y": 0.0,
                "rotation_injection_deg": 0.0, "curvature_injection_deg": 0.0,
                "translation_gain": 0.2, "rotation_gain": 0.0, "curvature_gain": 0.0,
                "total_force_x": 0.0, "total_force_y": 0.0, "priority": 0.0,
            })(),
            type("Trace", (), {
                "virtual_x": 0.1, "virtual_y": 0.0, "virtual_heading_deg": 0.0,
                "physical_x": 0.1, "physical_y": 0.0, "physical_heading_deg": 0.0,
                "in_reset": False,
                "translation_injection_x": 0.0, "translation_injection_y": 0.0,
                "rotation_injection_deg": 0.0, "curvature_injection_deg": 0.0,
                "translation_gain": 0.0, "rotation_gain": 0.0, "curvature_gain": 0.0,
                "total_force_x": 0.0, "total_force_y": 0.0, "priority": 0.0,
            })(),
        ]
        env = Environment(
            tracking_space=[Vector2(1.0, 1.0), Vector2(-1.0, 1.0), Vector2(-1.0, -1.0), Vector2(1.0, -1.0)],
            obstacles=[],
        )
        summary = summarize_agent_trace(
            trace=trace,
            environment=env,
            waypoints=[Vector2(0.0, 0.0), Vector2(1.0, 0.0)],
            descriptor={},
            time_step=0.1,
            sampling_frequency=20.0,
        )
        self.assertEqual(summary.values["min_g_t"], "0.2")
        self.assertEqual(summary.values["max_g_t"], "0.2")
        self.assertAlmostEqual(summary.one_dimensional_samples["g_t"][0], 0.02, places=6)
        self.assertAlmostEqual(summary.one_dimensional_samples["injected_translations"][0], 0.01, places=6)

    def test_summarize_agent_trace_reports_na_gains_when_no_gain_events_happen(self):
        trace = [
            type("Trace", (), {
                "virtual_x": 0.0, "virtual_y": 0.0, "virtual_heading_deg": 0.0,
                "physical_x": 0.0, "physical_y": 0.0, "physical_heading_deg": 0.0,
                "in_reset": False,
                "translation_injection_x": 0.0, "translation_injection_y": 0.0,
                "rotation_injection_deg": 0.0, "curvature_injection_deg": 0.0,
                "translation_gain": 0.0, "rotation_gain": 0.0, "curvature_gain": 0.0,
                "total_force_x": 0.0, "total_force_y": 0.0, "priority": 0.0,
            })(),
        ]
        env = Environment(
            tracking_space=[Vector2(1.0, 1.0), Vector2(-1.0, 1.0), Vector2(-1.0, -1.0), Vector2(1.0, -1.0)],
            obstacles=[],
        )
        summary = summarize_agent_trace(
            trace=trace,
            environment=env,
            waypoints=[Vector2(0.0, 0.0)],
            descriptor={},
            time_step=0.1,
        )
        self.assertEqual(summary.values["min_g_t"], "N/A")
        self.assertEqual(summary.values["max_g_t"], "N/A")
        self.assertEqual(summary.values["min_g_r"], "N/A")
        self.assertEqual(summary.values["max_g_r"], "N/A")
        self.assertEqual(summary.values["min_g_c"], "N/A")
        self.assertEqual(summary.values["max_g_c"], "N/A")

    def test_multi_agent_scheduler_runs(self):
        tracking, obstacles, initials = generate_square_tracking_space(20.0)
        env = Environment(tracking, obstacles)
        waypoints = generate_initial_path_by_seed(PathSeed.straight_line(), 6.0)
        gains = GainsConfig(time_step=0.1)
        agent_a = ScheduledAgent(
            agent_id="a",
            state=AgentState(
                virtual_pose=Pose2D(initials[0].position, 0.0),
                physical_pose=Pose2D(initials[0].position, 0.0),
            ),
            environment=env,
            gains=gains,
            redirector=MessingerApfRedirector(),
            resetter=TwoOneTurnResetter(),
            waypoints=waypoints,
        )
        agent_b = ScheduledAgent(
            agent_id="b",
            state=AgentState(
                virtual_pose=Pose2D(initials[1].position, 0.0),
                physical_pose=Pose2D(initials[1].position, 0.0),
            ),
            environment=env,
            gains=gains,
            redirector=MessingerApfRedirector(),
            resetter=TwoOneTurnResetter(),
            waypoints=waypoints,
        )
        scheduler = MultiAgentScheduler([agent_a, agent_b])
        result = scheduler.run(5)
        self.assertEqual(len(result[0].trace), 5)
        self.assertEqual(len(result[1].trace), 5)

    def test_build_scheduler_respects_counts_and_modes(self):
        scheduler = build_scheduler(
            SimulationConfig(
                redirector="vispoly",
                resetter="apf",
                path_mode="circle",
                tracking_space_shape="cross",
                physical_width=18.0,
                physical_height=16.0,
                virtual_width=26.0,
                virtual_height=24.0,
                physical_obstacle_count=2,
                virtual_obstacle_count=3,
                agent_count=3,
            )
        )
        self.assertEqual(len(scheduler.agents), 3)
        env = scheduler.agents[0].environment
        self.assertEqual(len(env.obstacles), 2)
        self.assertEqual(len(env.all_virtual_polygons), 4)
        self.assertEqual(env.shape, "cross")

    def test_build_environment_supports_custom_physical_obstacles(self):
        env = build_environment(
            SimulationConfig(
                physical_width=5.0,
                physical_height=5.0,
                physical_obstacle_specs=(
                    {"shape": "square", "x": 0.0, "y": 0.0, "size": 1.2},
                    {"shape": "triangle", "x": 1.0, "y": -1.0, "width": 1.0, "height": 1.4},
                    {"shape": "circle", "x": -1.0, "y": 1.0, "radius": 0.4},
                ),
            )
        )
        self.assertEqual(len(env.obstacles), 3)
        self.assertEqual(len(env.obstacles[0]), 4)
        self.assertEqual(len(env.obstacles[1]), 3)
        self.assertEqual(len(env.obstacles[2]), 24)

    def test_redirector_matrix_smoke(self):
        redirectors = [
            "none",
            "s2c",
            "s2o",
            "zigzag",
            "thomas_apf",
            "messinger_apf",
            "dynamic_apf",
            "passive_haptic_apf",
            "vispoly",
        ]
        for redirector in redirectors:
            scheduler = build_scheduler(
                SimulationConfig(
                    redirector=redirector,
                    resetter="apf",
                    path_mode="ninety_turn",
                    tracking_space_shape="rectangle",
                    physical_width=20.0,
                    physical_height=20.0,
                    virtual_width=22.0,
                    virtual_height=22.0,
                    physical_obstacle_count=1,
                    virtual_obstacle_count=1,
                    agent_count=1,
                    total_path_length=20.0,
                    time_step=1.0 / 30.0,
                    translation_speed=1.0,
                    rotation_speed=90.0,
                )
            )
            agent = scheduler.agents[0]
            start = agent.state.virtual_pose.position
            for step in range(40):
                scheduler.step(step)
            end = agent.state.virtual_pose.position
            moved = (end - start).magnitude
            self.assertTrue(isfinite(end.x))
            self.assertTrue(isfinite(end.y))
            self.assertGreater(moved, 0.01, msg=redirector)

    def test_resetter_matrix_boundary_trigger_and_finish(self):
        env = Environment(
            tracking_space=[
                Vector2(2.0, 2.0),
                Vector2(-2.0, 2.0),
                Vector2(-2.0, -2.0),
                Vector2(2.0, -2.0),
            ],
            obstacles=[],
        )
        gains = GainsConfig(time_step=0.1, rotation_speed=90.0, reset_trigger_buffer=0.5)
        resetters = [TwoOneTurnResetter(), ApfResetter()]
        for resetter in resetters:
            state = AgentState(
                virtual_pose=Pose2D(Vector2(1.8, 0.0), 90.0),
                physical_pose=Pose2D(Vector2(1.8, 0.0), 90.0),
                total_force=Vector2(-1.0, 0.0),
            )
            self.assertTrue(resetter.is_reset_required(state, env, gains, [state]))
            resetter.begin(state, env, gains)
            finished = False
            for _ in range(40):
                command = resetter.step(state, env, gains)
                state.physical_pose = Pose2D(state.physical_pose.position, state.physical_pose.heading_deg + command.user_rotation_deg)
                state.virtual_pose = Pose2D(state.virtual_pose.position, state.virtual_pose.heading_deg + command.user_rotation_deg + command.plane_rotation_deg)
                if command.finished:
                    finished = True
                    break
            self.assertTrue(finished, msg=type(resetter).__name__)


class SimulatorTests(unittest.TestCase):
    def test_simulator_runs_and_exports_trace(self):
        tracking, obstacles, initials = generate_square_tracking_space(20.0)
        env = Environment(tracking, obstacles)
        waypoints = generate_initial_path_by_seed(PathSeed.ninety_turn(), 10.0)
        pose = Pose2D(initials[0].position, 0.0)
        state = AgentState(virtual_pose=pose, physical_pose=pose)
        sim = OpenRDWSimulator(
            environment=env,
            gains=GainsConfig(time_step=0.1),
            redirector=S2CRedirector(),
            resetter=TwoOneTurnResetter(),
            waypoints=waypoints,
            state=state,
        )
        trace = sim.run(10)
        self.assertEqual(len(trace), 10)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "trace.csv"
            export_trace_csv(out, trace)
            self.assertTrue(out.exists())

    def test_redirect_updates_tracking_space_pose(self):
        tracking, obstacles, initials = generate_square_tracking_space(20.0)
        env = Environment(tracking, obstacles)
        pose = Pose2D(initials[0].position, 0.0)
        state = AgentState(virtual_pose=pose, physical_pose=pose)
        sim = OpenRDWSimulator(
            environment=env,
            gains=GainsConfig(time_step=0.1),
            redirector=S2CRedirector(),
            resetter=TwoOneTurnResetter(),
            waypoints=[pose.position, pose.position + Vector2(0.0, 3.0)],
            state=state,
        )
        before_tracking = sim.state.tracking_space_pose
        for step in range(6):
            sim.step(step, [state])
        after_tracking = sim.state.tracking_space_pose
        self.assertIsNotNone(after_tracking)
        self.assertTrue(
            after_tracking.position != before_tracking.position
            or abs(after_tracking.heading_deg - before_tracking.heading_deg) > 1e-6
        )

    def test_tracking_space_pose_stays_consistent_with_virtual_and_physical_pose(self):
        tracking, obstacles, initials = generate_square_tracking_space(20.0)
        env = Environment(tracking, obstacles)
        pose = Pose2D(initials[0].position, 0.0)
        state = AgentState(virtual_pose=pose, physical_pose=pose)
        sim = OpenRDWSimulator(
            environment=env,
            gains=GainsConfig(time_step=0.1),
            redirector=ThomasApfRedirector(),
            resetter=ApfResetter(),
            waypoints=[pose.position, pose.position + Vector2(0.0, 8.0)],
            state=state,
        )
        for step in range(20):
            sim.step(step, [state])
        tracking_pose = sim.state.tracking_space_pose
        self.assertIsNotNone(tracking_pose)
        reconstructed = sim.state.physical_pose.position.rotate(tracking_pose.heading_deg) + tracking_pose.position
        self.assertAlmostEqual(reconstructed.x, sim.state.virtual_pose.position.x, places=5)
        self.assertAlmostEqual(reconstructed.y, sim.state.virtual_pose.position.y, places=5)

    def test_root_pose_and_tracking_space_pose_stay_consistent(self):
        tracking, obstacles, initials = generate_square_tracking_space(20.0)
        env = Environment(tracking, obstacles)
        pose = Pose2D(initials[0].position, 0.0)
        state = AgentState(virtual_pose=pose, physical_pose=pose)
        sim = OpenRDWSimulator(
            environment=env,
            gains=GainsConfig(time_step=0.1),
            redirector=S2CRedirector(),
            resetter=TwoOneTurnResetter(),
            waypoints=[pose.position, pose.position + Vector2(0.0, 4.0)],
            state=state,
        )
        for step in range(8):
            sim.step(step, [state])
        self.assertIsNotNone(sim.state.root_pose)
        self.assertIsNotNone(sim.state.tracking_space_pose)
        self.assertAlmostEqual(sim.state.root_pose.heading_deg, sim.state.tracking_space_pose.heading_deg, places=6)
        reconstructed_tracking = sim.state.root_pose.position + sim.state.tracking_space_local_position.rotate(sim.state.root_pose.heading_deg)
        self.assertAlmostEqual(reconstructed_tracking.x, sim.state.tracking_space_pose.position.x, places=6)
        self.assertAlmostEqual(reconstructed_tracking.y, sim.state.tracking_space_pose.position.y, places=6)
        self.assertIsNotNone(sim.state.prev_virtual_pose)
        self.assertIsNotNone(sim.state.prev_physical_pose)
        self.assertIsNotNone(sim.state.prev_root_pose)

    def test_tracking_space_local_heading_can_differ_from_root_heading(self):
        env = Environment(
            tracking_space=[Vector2(2.0, 2.0), Vector2(-2.0, 2.0), Vector2(-2.0, -2.0), Vector2(2.0, -2.0)],
            obstacles=[],
        )
        state = AgentState(
            virtual_pose=Pose2D(Vector2(1.0, 0.0), 90.0),
            physical_pose=Pose2D(Vector2(0.0, 1.0), 45.0),
            tracking_space_local_position=Vector2(0.5, -0.25),
            tracking_space_local_heading_deg=30.0,
        )
        sim = OpenRDWSimulator(
            environment=env,
            gains=GainsConfig(time_step=0.1),
            redirector=S2CRedirector(),
            resetter=TwoOneTurnResetter(),
            waypoints=[Vector2(1.0, 0.0), Vector2(1.0, 2.0)],
            state=state,
        )
        self.assertAlmostEqual(
            sim.state.tracking_space_pose.heading_deg,
            sim.state.root_pose.heading_deg + state.tracking_space_local_heading_deg,
            places=6,
        )
        self.assertAlmostEqual(sim.state.physical_pose.heading_deg, 45.0, places=6)
        reconstructed = sim.state.tracking_space_pose.position + sim.state.physical_pose.position.rotate(sim.state.tracking_space_pose.heading_deg)
        self.assertAlmostEqual(reconstructed.x, sim.state.virtual_pose.position.x, places=6)
        self.assertAlmostEqual(reconstructed.y, sim.state.virtual_pose.position.y, places=6)

    def test_same_pos_time_accumulates_when_agent_stops(self):
        tracking, obstacles, _ = generate_square_tracking_space(20.0)
        env = Environment(tracking, obstacles)
        state = AgentState(
            virtual_pose=Pose2D(Vector2(0.0, 0.0), 0.0),
            physical_pose=Pose2D(Vector2(0.0, 0.0), 0.0),
            mission_complete=False,
        )
        sim = OpenRDWSimulator(
            environment=env,
            gains=GainsConfig(time_step=0.1),
            redirector=S2CRedirector(),
            resetter=TwoOneTurnResetter(),
            waypoints=[],
            state=state,
        )
        for step in range(3):
            sim.step(step, [state])
        self.assertGreater(sim.state.same_pos_time, 0.0)
        self.assertAlmostEqual(sim.state.delta_virtual_translation.magnitude, 0.0, places=6)

    def test_same_pos_time_accumulates_during_reset_rotation_without_translation(self):
        tracking, obstacles, _ = generate_square_tracking_space(20.0)
        env = Environment(tracking, obstacles)
        state = AgentState(
            virtual_pose=Pose2D(Vector2(0.0, 0.0), 0.0),
            physical_pose=Pose2D(Vector2(0.0, 0.0), 0.0),
            in_reset=True,
        )
        sim = OpenRDWSimulator(
            environment=env,
            gains=GainsConfig(time_step=0.1),
            redirector=S2CRedirector(),
            resetter=TwoOneTurnResetter(remaining_plane_rotation_deg=180.0, remaining_user_rotation_deg=180.0),
            waypoints=[Vector2(0.0, 0.0), Vector2(1.0, 1.0)],
            state=state,
        )
        for step in range(3):
            sim.step(step, [state])
        self.assertGreater(sim.state.same_pos_time, 0.0)
        self.assertAlmostEqual(sim.state.delta_pos.magnitude, 0.0, places=6)
        self.assertNotEqual(sim.state.delta_dir, 0.0)

    def test_unity_style_curr_prev_aliases_match_pose_deltas(self):
        tracking, obstacles, initials = generate_square_tracking_space(20.0)
        env = Environment(tracking, obstacles)
        pose = Pose2D(initials[0].position, 0.0)
        state = AgentState(virtual_pose=pose, physical_pose=pose)
        sim = OpenRDWSimulator(
            environment=env,
            gains=GainsConfig(time_step=0.1),
            redirector=S2CRedirector(),
            resetter=TwoOneTurnResetter(),
            waypoints=[pose.position, pose.position + Vector2(0.0, 4.0)],
            state=state,
        )
        sim.step(0, [state])
        self.assertAlmostEqual(sim.state.curr_pos.x, sim.state.virtual_pose.position.x, places=6)
        self.assertAlmostEqual(sim.state.curr_pos_real.x, sim.state.physical_pose.position.x, places=6)
        self.assertAlmostEqual(sim.state.prev_pos.x, sim.state.prev_virtual_pose.position.x, places=6)
        self.assertAlmostEqual(sim.state.prev_pos_real.x, sim.state.prev_physical_pose.position.x, places=6)
        self.assertAlmostEqual(sim.state.delta_virtual_translation.x, sim.state.virtual_pose.position.x - sim.state.prev_virtual_pose.position.x, places=6)
        self.assertTrue(isfinite(sim.state.delta_virtual_rotation_deg))

    def test_unity_style_delta_aliases_fall_back_to_base_deltas_without_prev_pose(self):
        state = AgentState(
            virtual_pose=Pose2D(Vector2(0.0, 0.0), 0.0),
            physical_pose=Pose2D(Vector2(0.0, 0.0), 0.0),
            base_delta_translation=Vector2(0.1, 0.2),
            base_delta_rotation_deg=3.0,
        )
        self.assertAlmostEqual(state.delta_pos.x, 0.1, places=6)
        self.assertAlmostEqual(state.delta_pos.y, 0.2, places=6)
        self.assertAlmostEqual(state.delta_dir, 3.0, places=6)

    def test_unity_style_delta_aliases_track_base_motion_even_after_redirect_changes_pose(self):
        tracking, obstacles, _ = generate_square_tracking_space(20.0)
        env = Environment(tracking, obstacles)
        state = AgentState(
            virtual_pose=Pose2D(Vector2(0.0, 0.0), 0.0),
            physical_pose=Pose2D(Vector2(0.0, 0.0), 0.0),
        )
        sim = OpenRDWSimulator(
            environment=env,
            gains=GainsConfig(time_step=0.1),
            redirector=S2CRedirector(),
            resetter=TwoOneTurnResetter(),
            waypoints=[Vector2(0.0, 0.0), Vector2(0.0, 4.0)],
            state=state,
        )
        sim.step(0, [state])
        self.assertAlmostEqual(sim.state.delta_pos.x, sim.state.base_delta_translation.x, places=6)
        self.assertAlmostEqual(sim.state.delta_pos.y, sim.state.base_delta_translation.y, places=6)
        self.assertAlmostEqual(sim.state.delta_dir, sim.state.base_delta_rotation_deg, places=6)

    def test_curr_pos_alias_uses_observed_pre_injection_pose(self):
        tracking, obstacles, initials = generate_square_tracking_space(20.0)
        env = Environment(tracking, obstacles)
        pose = Pose2D(initials[0].position, 180.0)
        state = AgentState(virtual_pose=pose, physical_pose=pose)
        sim = OpenRDWSimulator(
            environment=env,
            gains=GainsConfig(time_step=0.1),
            redirector=S2CRedirector(),
            resetter=TwoOneTurnResetter(),
            waypoints=[pose.position, pose.position + Vector2(0.0, 4.0)],
            state=state,
        )
        sim.step(0, [state])
        self.assertIsNotNone(sim.state.observed_virtual_pose)
        self.assertAlmostEqual(sim.state.curr_pos.x, sim.state.observed_virtual_pose.position.x, places=6)
        self.assertAlmostEqual(sim.state.curr_pos.y, sim.state.observed_virtual_pose.position.y, places=6)

    def test_step_trace_carries_observed_and_final_positions(self):
        tracking, obstacles, initials = generate_square_tracking_space(20.0)
        env = Environment(tracking, obstacles)
        pose = Pose2D(initials[0].position, 180.0)
        state = AgentState(virtual_pose=pose, physical_pose=pose)
        sim = OpenRDWSimulator(
            environment=env,
            gains=GainsConfig(time_step=0.1),
            redirector=S2CRedirector(),
            resetter=TwoOneTurnResetter(),
            waypoints=[pose.position, pose.position + Vector2(0.0, 4.0)],
            state=state,
        )
        trace = sim.step(0, [state])
        self.assertIsNotNone(trace.observed_virtual_x)
        self.assertIsNotNone(trace.observed_physical_x)
        self.assertAlmostEqual(trace.observed_virtual_x, sim.state.observed_virtual_pose.position.x, places=6)
        self.assertAlmostEqual(trace.observed_physical_x, sim.state.observed_physical_pose.position.x, places=6)

    def test_collision_starts_reset_without_extra_forward_step(self):
        env = Environment(
            tracking_space=[
                Vector2(2.0, 2.0),
                Vector2(-2.0, 2.0),
                Vector2(-2.0, -2.0),
                Vector2(2.0, -2.0),
            ],
            obstacles=[],
        )
        state = AgentState(
            virtual_pose=Pose2D(Vector2(1.8, 0.0), 90.0),
            physical_pose=Pose2D(Vector2(1.8, 0.0), 90.0),
        )
        sim = OpenRDWSimulator(
            environment=env,
            gains=GainsConfig(time_step=0.1, rotation_speed=90.0, physical_space_buffer=0.4),
            redirector=S2CRedirector(),
            resetter=TwoOneTurnResetter(),
            waypoints=[Vector2(1.8, 0.0), Vector2(3.0, 0.0)],
            state=state,
        )
        start_virtual = state.virtual_pose.position
        start_physical = state.physical_pose.position
        sim.step(0, [state])
        self.assertTrue(sim.state.in_reset)
        self.assertAlmostEqual(sim.state.virtual_pose.position.x, start_virtual.x, places=4)
        self.assertAlmostEqual(sim.state.virtual_pose.position.y, start_virtual.y, places=4)
        self.assertAlmostEqual(sim.state.physical_pose.position.x, start_physical.x, places=4)
        self.assertAlmostEqual(sim.state.physical_pose.position.y, start_physical.y, places=4)

    def test_two_one_turn_reset_returns_virtual_heading_after_completion(self):
        tracking, obstacles, _ = generate_square_tracking_space(20.0)
        env = Environment(tracking, obstacles)
        state = AgentState(
            virtual_pose=Pose2D(Vector2(0.0, 0.0), 45.0),
            physical_pose=Pose2D(Vector2(0.0, 0.0), 45.0),
        )
        sim = OpenRDWSimulator(
            environment=env,
            gains=GainsConfig(time_step=0.1, rotation_speed=90.0),
            redirector=S2CRedirector(),
            resetter=TwoOneTurnResetter(),
            waypoints=[Vector2(0.0, 0.0), Vector2(1.0, 1.0)],
            state=state,
        )
        sim.resetter.begin(sim.state, sim.environment, sim.gains)
        before_heading = sim.state.virtual_pose.heading_deg
        for _ in range(40):
            sim._apply_reset(sim.resetter.step(sim.state, sim.environment, sim.gains))
            if sim.resetter.remaining_user_rotation_deg <= 1e-6 and sim.resetter.remaining_plane_rotation_deg <= 1e-6:
                break
        self.assertAlmostEqual(sim.state.virtual_pose.heading_deg, before_heading, places=4)

    def test_reset_step_clears_translation_deltas(self):
        tracking, obstacles, _ = generate_square_tracking_space(20.0)
        env = Environment(tracking, obstacles)
        state = AgentState(
            virtual_pose=Pose2D(Vector2(0.0, 0.0), 45.0),
            physical_pose=Pose2D(Vector2(0.0, 0.0), 45.0),
            in_reset=True,
            base_delta_translation=Vector2(0.1, 0.0),
            physical_delta_translation=Vector2(0.1, 0.0),
        )
        sim = OpenRDWSimulator(
            environment=env,
            gains=GainsConfig(time_step=0.1, rotation_speed=90.0),
            redirector=S2CRedirector(),
            resetter=TwoOneTurnResetter(remaining_plane_rotation_deg=180.0, remaining_user_rotation_deg=180.0),
            waypoints=[Vector2(0.0, 0.0), Vector2(1.0, 1.0)],
            state=state,
        )
        sim.step(0, [state])
        self.assertAlmostEqual(sim.state.base_delta_translation.magnitude, 0.0, places=6)
        self.assertAlmostEqual(sim.state.physical_delta_translation.magnitude, 0.0, places=6)


class UiServerTests(unittest.TestCase):
    def test_ui_html_contains_controls(self):
        html = build_index_html()
        self.assertIn("Python OpenRDW", html)
        self.assertIn("redirector", html)
        self.assertIn("Build Scene", html)
        self.assertIn("tracking_space_shape", html)
        self.assertIn("Current waypoint ball", html)
        self.assertIn("telemetry", html)
        self.assertIn("Physical Obstacle Editor", html)
        self.assertIn("RESET", html)
        self.assertIn('id="physical_width" type="number" step="0.5" value="5"', html)
        self.assertIn('id="physical_obstacle_count" type="number" min="0" max="24" value="0"', html)
        self.assertIn('id="physical_space_buffer" type="number" step="0.1" min="0" value="0.4"', html)
        self.assertIn('id="obstacle_buffer" type="number" step="0.1" min="0" value="0.4"', html)
        self.assertIn('id="sampling_frequency"', html)
        self.assertIn('id="trail_visual_time"', html)
        self.assertIn('id="draw_real_trail"', html)
        self.assertIn('id="draw_virtual_trail"', html)
        self.assertIn("const TRAIL_MIN_DIST = 0.1;", html)
        self.assertIn("Tracking Space Pos", html)
        self.assertIn("Tracking Space Local Pos", html)
        self.assertIn("Root Pos", html)
        self.assertIn("Same Pos Time", html)
        self.assertIn("fillSelect('path_mode', pathOptions, 'random_turn');", html)
        self.assertIn('id="body_collider_diameter"', html)

    def test_ui_session_builds_state(self):
        session = SimulationSession()
        snapshot = session.build(
            {
                "agent_count": 2,
                "physical_obstacle_count": 2,
                "virtual_obstacle_count": 1,
                "redirector": "s2c",
                "resetter": "apf",
                "path_mode": "random_turn",
                "physical_width": 20,
                "physical_height": 20,
                "virtual_width": 22,
                "virtual_height": 22,
                "total_path_length": 30,
                "time_step": 0.0333,
                "translation_speed": 1.2,
                "rotation_speed": 100,
                "physical_obstacle_specs": [
                    {"shape": "square", "x": 0.0, "y": 0.0, "size": 1.0},
                    {"shape": "circle", "x": 1.0, "y": 1.0, "radius": 0.5},
                ],
                "physical_space_buffer": 0.4,
                "obstacle_buffer": 0.4,
                "seed": 3041,
            }
        )
        self.assertEqual(snapshot["config"]["agent_count"], 2)
        self.assertEqual(len(snapshot["physical"]["obstacles"]), 2)
        self.assertEqual(len(snapshot["virtual"]["targets"]), 2)
        self.assertEqual(len(snapshot["agent_states"]), 2)
        self.assertIn("mission_complete", snapshot["agent_states"][0])
        self.assertIn("translation_gain", snapshot["agent_states"][0])
        self.assertIn("display_heading_deg", snapshot["agent_states"][0])
        self.assertIn("root_pose", snapshot["agent_states"][0])
        self.assertIn("tracking_space_pose", snapshot["agent_states"][0])
        self.assertIn("observed_tracking_space_pose", snapshot["agent_states"][0])
        self.assertIn("observed_physical_pose", snapshot["agent_states"][0])
        self.assertIn("tracking_space_local_position", snapshot["agent_states"][0])
        self.assertIn("tracking_space_local_heading_deg", snapshot["agent_states"][0])
        self.assertIn("prev_virtual_pose", snapshot["agent_states"][0])
        self.assertIn("prev_physical_pose", snapshot["agent_states"][0])
        self.assertIn("prev_root_pose", snapshot["agent_states"][0])
        self.assertIn("same_pos_time", snapshot["agent_states"][0])
        self.assertIn("delta_virtual_translation", snapshot["agent_states"][0])
        self.assertIn("delta_physical_translation", snapshot["agent_states"][0])
        self.assertIn("curr_pos", snapshot["agent_states"][0])
        self.assertIn("curr_pos_real", snapshot["agent_states"][0])
        self.assertIn("prev_pos", snapshot["agent_states"][0])
        self.assertIn("prev_pos_real", snapshot["agent_states"][0])
        self.assertIn("delta_pos", snapshot["agent_states"][0])
        self.assertIn("delta_dir", snapshot["agent_states"][0])
        self.assertIn("all_mission_complete", snapshot)
        self.assertAlmostEqual(snapshot["gains"]["body_collider_diameter"], 0.1)
        self.assertAlmostEqual(snapshot["gains"]["physical_space_buffer"], 0.4)
        self.assertAlmostEqual(snapshot["gains"]["obstacle_buffer"], 0.4)

    def test_ui_session_snapshot_traces_include_overlay_time_samples(self):
        session = SimulationSession()
        session.build(
            {
                "redirector": "s2c",
                "resetter": "two_one_turn",
                "path_mode": "random_turn",
                "physical_width": 5,
                "physical_height": 5,
                "virtual_width": 20,
                "virtual_height": 20,
                "total_path_length": 10,
                "time_step": 0.1,
            }
        )
        snapshot = session.step({})
        self.assertTrue(snapshot["physical"]["traces"][0])
        self.assertTrue(snapshot["virtual"]["traces"][0])
        self.assertIn("t", snapshot["physical"]["traces"][0][0])
        self.assertIn("t", snapshot["virtual"]["traces"][0][0])

    def test_ui_session_step_does_not_increase_after_completion(self):
        session = SimulationSession()
        session.build(
            {
                "redirector": "none",
                "resetter": "none",
                "path_mode": "straight_line",
                "physical_width": 20,
                "physical_height": 20,
                "virtual_width": 20,
                "virtual_height": 20,
                "total_path_length": 1,
                "translation_speed": 10.0,
                "rotation_speed": 360.0,
                "time_step": 1.0,
            }
        )
        before = None
        for _ in range(20):
            snapshot = session.step({})
            if snapshot["all_mission_complete"]:
                before = snapshot["step_index"]
                break
        self.assertIsNotNone(before)
        after = session.step({})
        self.assertEqual(after["step_index"], before)
        self.assertTrue(after["all_mission_complete"])

    def test_ui_session_command_file_job_runs_to_completion(self):
        session = SimulationSession()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            command_file = tmp / "commands.txt"
            output_dir = tmp / "out"
            command_file.write_text(
                "\n".join(
                    [
                        "newUser = 0",
                        "redirector = s2c",
                        "resetter = twoOneTurn",
                        "pathSeedChoice = straightLine",
                        "trackingSpaceChoice = rectangle",
                        "obstacleType = 1",
                        "end = 1",
                    ]
                ),
                encoding="utf-8",
            )
            snapshot = session.run_command_file(
                {
                    "command_file": str(command_file),
                    "output_dir": str(output_dir),
                    "max_steps": 1000,
                    "sampling_frequency": 5.0,
                }
            )
            self.assertIn(snapshot["command_job"]["status"], {"queued", "running", "completed"})
            deadline = time.time() + 5.0
            while time.time() < deadline:
                current = session.snapshot()
                if current["command_job"]["status"] == "completed":
                    break
                if current["command_job"]["status"] == "error":
                    self.fail(current["command_job"].get("error", "command file job failed"))
                time.sleep(0.05)
            self.assertEqual(session.snapshot()["command_job"]["status"], "completed")
            self.assertTrue((output_dir / "summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
