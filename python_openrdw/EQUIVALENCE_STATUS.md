# OpenRDW Python Equivalence Status

This file classifies the current Python port against the Unity OpenRDW project with a stricter equivalence lens.

Status legend:
- `Equivalent enough`: behavior is implemented in Python with the same practical role and near-source-level semantics for the 2D simulator.
- `Approximate`: behavior exists and is source-aligned, but runtime semantics are still adapted or not perfectly isomorphic to Unity.
- `Not equivalent`: Unity-specific engine, VR, rendering, editor, or networking behavior cannot be reproduced 1:1 in the pure-Python port.

## Equivalent enough

- 2D geometry helpers, headings, signed-angle conventions, polygon math, and visibility-polygon support.
- Tracking-space generators for built-in 2D shapes and file-loaded polygons.
- Waypoint/path generation for built-in path seeds, file paths, and real-user sampled paths.
- First-waypoint/start-point behavior and waypoint progression timing in the fixed-step simulator.
- Core simulated walking loop for autopilot and keyboard-style movement in 2D.
- Redirector set:
  - `none`
  - `s2c`
  - `s2o`
  - `zigzag`
  - `thomas_apf`
  - `messinger_apf`
  - `dynamic_apf`
  - `passive_haptic_apf`
  - `vispoly`
- Resetter set:
  - `none`
  - `two_one_turn`
  - `apf`
- Multi-agent fixed-step scheduling with Unity-style two-phase update order:
  - move all agents first
  - then apply redirection/reset for all agents
- Unity-style raw user-motion aliases in runtime state:
  - `currPos`
  - `currPosReal`
  - `prevPos`
  - `prevPosReal`
  - `deltaPos`
  - `deltaDir`
- CSV trace export, sampled-metrics export, summary export, and command-file batch running.
- Local 2D Web UI for selecting redirector, resetter, path, tracking space, sizes, buffers, obstacles, and running interactive simulations.
- Target-ball behavior in the 2D overview, including start waypoint and waypoint switching.

## Approximate

- `RedirectionManager` transform semantics.
  The Python port now has explicit `root_pose`, `tracking_space_pose`, observed-vs-final state, and Unity-style frame order, but Unity's actual `Transform` hierarchy is still represented as plain 2D math rather than engine transforms.

- `MovementManager`.
  The portable movement logic is covered, but Unity object lifecycle, scene object setup, camera hooks, and prefab wiring are replaced by Python-side state and Web UI rendering.

- `StatisticsLogger`.
  The Python port now mirrors most of the important metrics and uses observed-state semantics where Unity reads cached user state, but exact field-by-field parity with every Unity summary/sample metric is still being tightened.

- `TrailDrawer`.
  Trail history now follows Unity-style timestamp clipping and incremental `MIN_DIST` decimation semantics, but Unity mesh/material rendering is replaced by 2D canvas lines.

- `ResetTrigger`.
  Reset-trigger behavior is implemented geometrically through buffered polygons rather than Unity collider trigger events and scene callbacks. The overview now also models Unity's body-size-aware tracking-space shrink for the reset-boundary visualization.

- `UserInterfaceManager`.
  The Python Web UI covers operator controls, telemetry, command-file launching, command-directory launching, mission-complete state, and overview visualization, but Unity Canvas panels, file dialogs, and scene-driven UI behavior are replaced.

- `DynamicAPF`.
  The main redirector is implemented and source-aligned, but remaining edge cases in attraction-region search and frame-level handoff can still differ slightly from Unity.

- `PassiveHapticAPF`.
  The alignment state machine and physical target logic are now closer to Unity, including final-waypoint gating and per-agent target mapping, but physical-target authoring/runtime authoring is still adapter-based rather than Unity-scene-native.

- `DeepLearning_Redirector`.
  The Python version uses the original ONNX models, Unity-style signed-angle state encoding, and prefers the named action output tensor like the Unity implementation, but the runtime backend is `onnxruntime` rather than Unity Barracuda.

- Single-view overview rendering.
  The Python UI is intentionally a 2D top-down overlay. It is behaviorally useful and close to OpenRDW's overview mode, but it is still an adapted presentation layer, not the Unity camera stack itself.

## Not equivalent

- SteamVR / HMD device integration.
- Unity camera rig behavior and first-person / third-person camera switching as engine-rendered views.
- Photon networking, room lifecycle, and synchronized remote avatars.
- Unity scene graph, prefab hierarchy, GameObject lifecycle, layers, culling masks, meshes, materials, shaders, and colliders as engine primitives.
- Avatar animation controllers, mesh following, and body presentation scripts.
- Unity editor scripts and asset/tooling helpers.
- Unity-native video recording pipeline.

## Current practical conclusion

The current project under `python_openrdw/` is already a valid Python reimplementation of OpenRDW's portable 2D simulation core.

It is accurate to call it:
- `OpenRDW Python reimplementation`
- `Python port of OpenRDW`
- `OpenRDW 2D Python reconstruction`

It is not yet fully accurate to call it:
- `bit-for-bit equivalent to Unity OpenRDW`
- `a 100% identical replacement for the Unity runtime`

## Short gap list still worth tightening

- keep tightening the last `StatisticsLogger` field-level parity details
- keep reducing the last frame-edge differences around reset/render state
- keep tightening the last `RedirectionManager` transform semantics where Unity splits avatar-root and tracking-space references
- add more direct side-by-side regression checks for the hardest combinations such as `thomas_apf + apf`
