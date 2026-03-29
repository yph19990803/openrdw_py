# openrdw_py

`openrdw_py` is a Python reimplementation / port of the portable 2D core of OpenRDW.

It keeps the Unity project as the behavioral reference, but moves the simulation core into an engine-independent Python package that can run as:

- a headless simulator
- a local web UI
- a command-file batch runner
- a regression-tested research codebase

## Repository layout

- `python_openrdw/`
  The Python OpenRDW port, including simulator, redirectors, resetters, UI, experiment runner, exporters, and tests.
- `agent-harness/`
  The long-running multi-agent harness used during the porting process, including initialization script, feature list, and progress log.

## What the Python port currently covers

- Tracking-space generation and loading
- Waypoint / path generation and replay
- Redirectors:
  - `none`
  - `s2c`
  - `s2o`
  - `zigzag`
  - `thomas_apf`
  - `messinger_apf`
  - `dynamic_apf`
  - `deep_learning`
  - `passive_haptic_apf`
  - `vispoly`
- Resetters:
  - `none`
  - `two_one_turn`
  - `apf`
- Multi-agent fixed-step simulation
- CSV trace export
- Summary and sampled metrics export
- Command-file and command-directory batch execution
- Local 2D overview UI with physical / virtual overlay visualization

## What is still not 1:1 with Unity

The Python port does not replace Unity engine features such as:

- SteamVR / HMD integration
- Photon networking
- Unity scene graph and prefab lifecycle
- Unity colliders, trigger callbacks, meshes, materials, shaders, and cameras as engine primitives

For a stricter parity breakdown, see:

- [`python_openrdw/EQUIVALENCE_STATUS.md`](python_openrdw/EQUIVALENCE_STATUS.md)
- [`python_openrdw/PORT_STATUS.md`](python_openrdw/PORT_STATUS.md)
- [`python_openrdw/UNITY_SCRIPT_COVERAGE.md`](python_openrdw/UNITY_SCRIPT_COVERAGE.md)

## Quick start

Run the UI:

```bash
PYTHONPATH=python_openrdw python3 -m openrdw.ui
```

Run tests:

```bash
PYTHONPATH=python_openrdw python3 -m unittest discover -s python_openrdw/tests -v
```

Run a CLI simulation:

```bash
PYTHONPATH=python_openrdw python3 -m openrdw --steps 200 --redirector s2c --resetter two_one_turn
```

## Documentation

- English:
  - [`python_openrdw/README.md`](python_openrdw/README.md)
- 中文：
  - [`python_openrdw/README.zh-CN.md`](python_openrdw/README.zh-CN.md)

## Reference

The Unity OpenRDW project used as the reference during the port lives locally in:

`/Users/yph/Desktop/My_project/Unity/OpenRdw_vbdi`
