# Progress Log

## 2026-03-24

### Bootstrap

- Created a project-specific harness in `agent-harness/`.
- Bound the harness to the current Unity project facts:
  - Unity `2022.3.46f1`
  - Main scene `Assets/OpenRDW/Scenes/OpenRDW Scene.unity`
  - Results directory `Experiment Results/`
- Seeded the feature list from `RDW_Aware_Finetune_Plan.md`.

### Python OpenRDW port

- An engine-independent Python port now exists under `python_openrdw/`.
- Implemented core modules:
  - geometry and state models
  - tracking-space generators
  - path generation
  - headless simulator
  - redirectors: `none`, `s2c`, `s2o`, `messinger_apf`, `dynamic_apf`, `vispoly`
  - resetters: `none`, `two_one_turn`, `apf`
  - CSV trace export and demo CLI
  - multi-agent scheduler with priority-based ordering
  - local web UI with selectable strategies, dimensions, obstacle counts, and simulation controls
- Added a feature-gap document in `python_openrdw/PORT_STATUS.md`.
- Added unit tests in `python_openrdw/tests/test_core.py`.

### Current state

- The project root is not a git repository, so session continuity currently depends on this file plus `feature_list.json`.
- `init.sh` surfaces the expected Unity batchmode command and the missing-git warning.
- Batchmode smoke was attempted with Unity `2022.3.46f1` and reached the licensing stage, but exited with code `199` because the Unity Licensing Client IPC channel was not established within 60 seconds.

### Next recommended feature

- `BOOT-001`: initialize git and create a clean bootstrap commit, or explicitly decide to operate without git.
- `ENV-001`: resolve local Unity licensing so verifier sessions can use batchmode smoke checks.

### Handoff notes

- Treat `RDW_Aware_Finetune_Plan.md` as the product requirements baseline.
- Prefer one feature per session.
- When adding preprocessing or training code, record concrete evidence here:
  - command used
  - output location
  - shape/metric summary
  - unresolved blockers
- Latest validation evidence:
  - command: `"/Applications/Unity/Hub/Editor/2022.3.46f1/Unity.app/Contents/MacOS/Unity" -batchmode -projectPath "/Users/yph/Desktop/My_project/Unity/OpenRdw_vbdi" -quit -logFile "/Users/yph/Desktop/My_project/Unity/OpenRdw_vbdi/Logs/agent-smoke.log"`
  - result: failed before project import
  - blocker: `Logs/agent-smoke.log` ends with `Timed-out after 60.01s` and `IPC channel to LicensingClient doesn't exist; aborting`
  - command: `PYTHONPATH=python_openrdw python3 -m unittest discover -s python_openrdw/tests -v`
  - result: `10` tests passed
  - command: `PYTHONPATH=python_openrdw python3 -m openrdw --steps 20 --redirector messinger_apf --resetter apf --output python_openrdw/out/demo_trace.csv`
  - result: demo trace exported successfully
  - command: `PYTHONPATH=python_openrdw python3 -m openrdw --steps 20 --redirector vispoly --resetter apf --output python_openrdw/out/vispoly_demo_trace.csv`
  - result: vispoly demo trace exported successfully
  - command: `PYTHONPATH=python_openrdw python3 -m openrdw.ui`
  - result: verified with local HTTP requests that `/` serves the control page and `/api/build` returns a valid configured scene
