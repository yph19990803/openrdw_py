# OpenRDW Python Port Status

This document summarizes the Unity OpenRDW project and marks each subsystem by Python portability.

Portability legend:
- `Portable now`: core logic can be implemented in pure Python without waiting for a new engine.
- `Portable later`: the behavior can be ported, but it needs a runtime adapter, GUI, network backend, or 3D visualization layer.
- `Not equivalent in pure Python`: the Unity/SteamVR feature has no direct pure-Python equivalent and must be replaced by a different runtime stack.

## Unity Subsystems

- Experiment setup and control: `GlobalConfiguration`, `ExperimentSetup`, `InitialConfiguration`, command-file loading, trial iteration, controller mode selection.
- Tracking-space and path generation: `TrackingSpaceGenerator`, `VirtualPathGenerator`, `VisibilityPolygonCSharp`, `Segment`, `PointAdapter`, `Vector2Adapter`, `Utilities`.
- Redirection and reset logic: `RedirectionManager`, redirectors (`S2C`, `S2O`, `ZigZag`, APF variants, `VisPoly`, `DeepLearning`), resetters (`TwoOneTurn`, `APF`, `Null`).
- Movement and simulated walking: `MovementManager`, `SimulatedWalker`, `KeyboardController`, `SynchronizedByNet`.
- Avatar presentation: `AvatarAnimatorController`, `AvatarLayerManager`, `HeadFollower`, avatar prefabs and materials.
- Networking: `NetworkManager`, `AvatarInfoForNetworking`, Photon room creation and avatar synchronization.
- Analysis and logging: `StatisticsLogger`, `TrailDrawer`, `PoseFixer`, `VideoRecorder`.
- UI and editor glue: `UserInterfaceManager`, `EnsureAvatarLayer`, `OpenFileName`, editor-only scripts.
- VR and rendering runtime: SteamVR integration, camera rig logic, first/third-person view switching, prefabs, shaders, materials, scene objects.

## Portability Status

### Portable now

- Experiment parsing, trial scheduling, and configuration state.
- Path, waypoint, and tracking-space geometry generation.
- Core redirection math and reset decision logic.
- Visibility-polygon-based redirection when virtual obstacles are provided explicitly as polygons.
- Dataset logging, statistics aggregation, and trajectory export.
- DeepLearning redirector inference if the ONNX model is run through a Python runtime such as `onnxruntime`.
- Headless movement simulation and test harnesses that do not depend on Unity objects.

### Portable later

- Multiplayer synchronization and room lifecycle management.
- Runtime UI and operator controls.
- Avatar animation playback, head-follow behavior, and avatar visibility toggles.
- Optional video capture and rich visualization overlays.
- Unity-scene-derived virtual obstacle extraction for `VisPoly`.
- Any feature that currently assumes a live Unity scene but is otherwise algorithmic.

### Not equivalent in pure Python

- SteamVR/HMD device integration and Unity camera-rig behavior.
- Unity scene graph, prefabs, materials, shaders, and layer/culling-mask semantics.
- Unity editor extensions and build-time asset tooling.
- Exact first-person / third-person rendering behavior as implemented in Unity.

## Expected Python Architecture

The Python port should be split into a small engine-agnostic core plus adapters:

- `core/`: immutable config objects, trial state machine, movement update loop, redirection/reset algorithms, and geometry helpers.
- `core/pathing/`: waypoint generation, tracking-space generation, visibility polygon logic, and obstacle handling.
- `core/logging/`: statistics collection, trajectory export, and reproducible run metadata.
- `models/`: ONNX-backed inference for `DeepLearning_Redirector`, with model loading isolated from simulation state.
- `runtime/input/`: keyboard, autopilot, or external-tracker input adapters.
- `runtime/network/`: multiplayer state sync and room/session management.
- `runtime/visualization/`: optional 2D or 3D presentation layer that mirrors the Unity scene, but is not required by the simulation core.
- `app/`: CLI or service entry point that loads a command file, instantiates experiments, advances fixed timesteps, and writes outputs.

The key boundary is that the core simulation must not depend on rendering or device APIs. Unity-specific concepts such as `Transform`, `Camera`, `GameObject`, and `PhotonNetwork` should be represented as adapter interfaces or plain data structures in Python.

## Current Python package status

The current package under `python_openrdw/openrdw/` already includes:

- fixed-step headless simulation with separate virtual and physical pose updates
- explicit tracking-space pose state so the physical local pose, tracking-space transform, and virtual/world pose stay algebraically consistent
- explicit tracking-space local position/local heading state so `currPosReal` can stay relative to Tracking Space while `currDirReal` stays relative to the avatar root, matching Unity more closely
- previous-frame world/real/root poses plus `samePosTime`-style stagnation tracking to mirror Unity's `curr/prev` and invalid-run bookkeeping more closely
- Unity-style alias fields for `currPos`, `currPosReal`, `prevPos`, `prevPosReal`, `deltaPos`, and `deltaDir` in the runtime state/snapshot layer for direct source-code comparison, with `deltaPos` / `deltaDir` kept as raw pre-injection user motion like Unity's `RedirectionManager`
- redirectors: `none`, `s2c`, `s2o`, `zigzag`, `thomas_apf`, `messinger_apf`, `dynamic_apf`, `deep_learning`, `passive_haptic_apf`, `vispoly`
- resetters: `none`, `two_one_turn`, `apf`
- tracking-space generators for all built-in 2D shapes plus file-loaded polygons
- built-in path seeds, circle/figure-eight generation, file-path loading, and real-user-path loading with sampling intervals
- multi-agent scheduling with priority ordering
- multi-agent scheduling with Unity-style two-phase stepping: move all avatars first, then apply redirection/reset in priority order
- Unity-style sampled-metrics aggregation with reset-aware position sampling buffers
- CSV trace export, command-file parsing, command-directory batch running, wall-clock `execute_duration`, summary/sample metrics export, path-graph export, local web UI, and CLI demo
- UI trail/visibility toggles for real trail, virtual trail, trail time clipping, virtual-world visibility, tracking-space visibility, and buffer visibility, with TrailDrawer-style `MIN_DIST` decimation and timestamp-based trail clipping
- body-size-aware reset-boundary visualization for the tracking-space rectangle, closer to Unity `ResetTrigger`

The current `vispoly` implementation is intentionally adapted:

- physical obstacles come from `tracking_space` and `obstacles`
- virtual obstacles must be passed through `Environment.virtual_obstacles`
- Unity mesh extraction is not part of the pure-Python core

The current `deep_learning` implementation is also intentionally adapted:

- it uses the original `SRLNet_*.onnx` files from the Unity project
- it runs via Python `onnxruntime` instead of Unity Barracuda
- using it requires installing `onnxruntime` and `numpy` in the local project environment

## Remaining Adaptations

These parts are still intentionally not 1:1 with Unity even though the portable behavior exists:

- `ResetTrigger` is implemented as geometry checks rather than Unity collider trigger events, although the overview now includes Unity-style body-size-aware reset-boundary visualization.
- Trail rendering is a 2D canvas adaptation, not Unity mesh/material rendering, even though the stored trail samples now follow Unity-style time clipping and minimum-distance semantics more closely.
- `RedirectionManager` is much closer to Unity now via an explicit tracking-space pose and reset/redirect transform updates, but Unity's split `transform` vs `Tracking Space` hierarchy is still not perfectly isomorphic in plain 2D data.
- `DynamicAPF` and `PassiveHapticAPF` are now closer to Unity in gravitation search, reset-force handoff, and alignment latching, but remaining edge-case differences still come from Unity-specific transform/update ordering.
- First-person / overview camera switching is represented by 2D visibility toggles instead of Unity cameras.
