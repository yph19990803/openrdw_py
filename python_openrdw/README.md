# Python OpenRDW

This folder now contains an engine-independent Python port of the core OpenRDW simulation logic.

## What is implemented

- 2D geometry helpers and pose/state data models.
- Tracking-space generators for `rectangle`, `square`, `triangle`, `trapezoid`, `cross`, `l_shape`, `t_shape`, and file-loaded polygons.
- Procedural path generation for `_90Turn`, `RandomTurn`, `StraightLine`, `Sawtooth`, `Circle`, `FigureEight`, file paths, and real-user paths with sampling intervals.
- A headless simulator with Unity-style fixed-step updates, separate virtual/physical pose evolution, explicit tracking-space pose updates, reset triggering, and trajectory/stat trace fields.
- A multi-avatar scheduler that now follows Unity's frame structure more closely: update time/priority, move all avatars, then apply redirection/reset for all avatars.
- Redirectors: `none`, `s2c`, `s2o`, `zigzag`, `thomas_apf`, `messinger_apf`, `dynamic_apf`, `deep_learning`, `passive_haptic_apf`, `vispoly`.
- Resetters: `none`, `two_one_turn`, `apf`.
- Multi-avatar scheduling with priority-based ordering.
- CSV trace export and a small CLI demo entry point.
- command-file parsing, command-directory batch experiment running, wall-clock execution timing, summary-statistics export, sampled-metrics export, and real-path graph export
- A local web UI with dropdowns, numeric controls, file-path inputs, buttons, and a single combined overlay view.

## What is not implemented here

- Unity scene graph, rendering, avatars, VR devices, Photon networking, and editor tooling.
- The Python `vispoly` port expects explicit `virtual_obstacles` input instead of reading Unity meshes.
- `DeepLearning_Redirector` runs through Python `onnxruntime` instead of Unity Barracuda.
- Passive haptics logic is ported, but physical-target authoring and haptic hardware integration remain data/runtime concerns outside the pure 2D simulator.

See `PORT_STATUS.md` for the detailed portability map.
See `EQUIVALENCE_STATUS.md` for the stricter `Equivalent enough / Approximate / Not equivalent` status against the Unity project.

## Quick start

Run a small demo:

```bash
PYTHONPATH=python_openrdw python3 -m openrdw --steps 200 --redirector s2c --resetter two_one_turn --output python_openrdw/out/demo_trace.csv
```

Launch the interactive local web UI:

```bash
PYTHONPATH=python_openrdw python3 -m openrdw.ui
```

Then open [http://127.0.0.1:8765/](http://127.0.0.1:8765/).

For `deep_learning`, create the local project venv and use it to launch:

```bash
cd /Users/yph/Desktop/My_project/python_openrdw
python3 -m venv .venv
.venv/bin/python -m pip install onnxruntime numpy
PYTHONPATH=/Users/yph/Desktop/My_project/python_openrdw .venv/bin/python -m openrdw.ui
```

The UI provides:

- redirector and resetter dropdowns
- autopilot / keyboard controller selection
- tracking-space dropdown
- physical and virtual width/height inputs
- physical and virtual obstacle-count inputs
- agent-count input
- path selection
- optional tracking-space / waypoint / sampling-interval file inputs
- sampling-frequency, custom-sampling, and trail-visual-time controls
- body-diameter control for Unity-style reset-boundary visualization
- `Build Scene`, `Step`, `Start`, `Pause`, `Reset`, and `Export CSV` buttons
- a command-file runner with `command file / dir` and `output dir` inputs plus an experiment-complete panel
- a single combined canvas where the virtual space stays fixed and the physical tracking space moves via an explicit tracking-space pose transform
- a `Show physical buffer and reset boundary` toggle that overlays the physical reset-trigger zone
- live virtual and physical trail history overlays with TrailDrawer-style distance decimation and timestamp-based time clipping
- virtual-world / tracking-space / real-trail / virtual-trail visibility toggles
- live telemetry for virtual pose, physical pose, root pose, tracking-space pose, waypoint, gains, and reset state
- explicit tracking-space local position / local heading state so Python can model Unity's split `currPosReal`-vs-`currDirReal` reference frames more directly
- previous-frame poses, per-frame virtual/physical deltas, and `samePosTime`-style stagnation telemetry for frame-by-frame debugging
- Unity-style aliases in snapshot/debug state for `currPos`, `currPosReal`, `prevPos`, `prevPosReal`, `deltaPos`, and `deltaDir`
  `deltaPos` / `deltaDir` follow Unity `RedirectionManager` semantics and report the user's raw per-frame motion before the current frame's redirection injection, while `delta_virtual_translation` / `delta_virtual_rotation_deg` expose pose-to-pose deltas for debugging
- keyboard mode for 2D manual control with `W/A/S/D` and `Left/Right`

Run a Unity-style command file batch:

```bash
PYTHONPATH=/Users/yph/Desktop/My_project/python_openrdw python3 -m openrdw.experiments \
  --command-file /path/to/commands.txt \
  --output-dir /Users/yph/Desktop/My_project/python_openrdw/out/experiments
```

This writes:

- `summary.csv`
- per-agent trace CSVs
- sampled metric CSV directories
- `graphs/trial_*_real_path.png`

If `--command-file` points to a directory, the runner now walks it recursively and creates a separate output subtree for each command file, including Unity-style `tmp/` progress markers.

The Web UI also includes a command-file runner section. Start the UI with:

```bash
PYTHONPATH=/Users/yph/Desktop/My_project/python_openrdw python3 -m openrdw.ui
```

Then fill in `Command File` and `Output Dir`, click `Run Command File`, and monitor the completion panel on the right. The UI calls `POST /api/run_command_file` in the background and keeps the normal simulation controls available while the batch run is executing.

For a per-script coverage map against `Assets/OpenRDW/Scripts`, see `UNITY_SCRIPT_COVERAGE.md`.

The Unity implementation under `Assets/OpenRDW/` remains the behavioral reference.
