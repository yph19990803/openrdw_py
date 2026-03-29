# Unity Script Coverage

This file maps `Assets/OpenRDW/Scripts` to the Python port under `python_openrdw/openrdw`.

Status legend:
- `Implemented`: covered directly in the Python port.
- `Adapted`: the behavior is covered, but the runtime form differs from Unity.
- `Not equivalent`: Unity/VR/editor/network/rendering behavior has no direct pure-Python equivalent here.

## Analysis

| Unity script | Status | Python equivalent / note |
| --- | --- | --- |
| `Analysis/StatisticsLogger.cs` | Adapted | `openrdw/stats.py`, `openrdw/exporters.py`, `openrdw/experiments.py`; sample buffers, reset-aware sampling, sampled-metrics export, and wall-clock `execute_duration` are mirrored |
| `Analysis/TrailDrawer.cs` | Adapted | trace export plus live trail rendering in `openrdw/ui.py`, including real/virtual trail toggles, `trail_visual_time` clipping, and TrailDrawer-style `MIN_DIST` decimation; Unity mesh/material rendering is not ported |
| `Analysis/PoseFixer.cs` | Not equivalent | Unity transform-freezing helper is scene/runtime-specific |
| `Analysis/VideoRecorder.cs` | Not equivalent | Unity frame capture / AVI pipeline not ported in pure Python core |

## Avatar

| Unity script | Status | Python equivalent / note |
| --- | --- | --- |
| `Avatar/HeadFollower.cs` | Not equivalent | avatar body visualization is Unity-specific |
| `Avatar/AvatarAnimatorController.cs` | Not equivalent | Mecanim animation controller has no pure 2D equivalent |
| `Avatar/AvatarLayerManager.cs` | Not equivalent | Unity layer setup / culling behavior is engine-specific |

## Editor

| Unity script | Status | Python equivalent / note |
| --- | --- | --- |
| `Editor/EnsureAvatarLayer.cs` | Not equivalent | editor-only Unity asset/layer utility |

## Movement

| Unity script | Status | Python equivalent / note |
| --- | --- | --- |
| `Movement/VirtualPathGenerator.cs` | Implemented | `openrdw/paths.py` |
| `Movement/MovementManager.cs` | Adapted | `openrdw/simulator.py`, `openrdw/factory.py`, `openrdw/ui.py`; portable 2D movement, waypoint, and target-ball behavior are covered, but Unity scene/camera object wiring is replaced |
| `Movement/Motion Simulators/SimulatedWalker.cs` | Implemented | `openrdw/simulator.py` |
| `Movement/Motion Simulators/KeyboardController.cs` | Adapted | local Web UI keyboard mode covers forward/back/strafe/yaw; Unity pitch controls are not meaningful in 2D |
| `Movement/Motion Simulators/SynchronizedByNet.cs` | Not equivalent | Photon/network synchronization is outside the pure-Python simulator |

## Networking

| Unity script | Status | Python equivalent / note |
| --- | --- | --- |
| `Networking/NetworkManager.cs` | Not equivalent | Photon room/session lifecycle not ported |
| `Networking/AvatarInfoForNetworking.cs` | Not equivalent | Photon serialization component not ported |

## Others

| Unity script | Status | Python equivalent / note |
| --- | --- | --- |
| `Others/Utilities.cs` | Implemented | `openrdw/geometry.py`, file/CSV helpers in Python modules |
| `Others/InitialConfiguration.cs` | Implemented | `openrdw/models.py` |
| `Others/ExperimentSetup.cs` | Implemented | `openrdw/models.py`, `openrdw/experiments.py` |
| `Others/TrackingSpaceGenerator.cs` | Implemented | `openrdw/tracking.py` |
| `Others/VisibilityPolygonCSharp.cs` | Implemented | `openrdw/visibility.py` |
| `Others/Segment.cs` | Implemented | folded into `openrdw/visibility.py` geometry handling |
| `Others/PointAdapter.cs` | Implemented | folded into `openrdw/visibility.py` |
| `Others/Vector2Adapter.cs` | Implemented | folded into `openrdw/visibility.py` |
| `Others/GlobalConfiguration.cs` | Adapted | `openrdw/factory.py`, `openrdw/experiments.py`, `openrdw/ui.py`; portable experiment/path/trail/visibility settings and command parsing are covered, but Unity camera/view/network toggles are replaced or omitted |
| `Others/UserInterfaceManager.cs` | Adapted | local Web UI in `openrdw/ui.py`, including completion panel and command-file/directory runner; Unity Canvas/panel/file-dialog behavior is replaced |
| `Others/ResetTrigger.cs` | Adapted | reset-boundary logic in `openrdw/resetters.py`, overlay in `openrdw/ui.py`; collider trigger lifecycle is replaced by geometric boundary tests and explicit buffered polygons |
| `Others/HorizontalFollower.cs` | Not equivalent | Unity child-object follower for meshes/buffers |
| `Others/OpenFileName.cs` | Not equivalent | native Windows file dialog helper |
| `Others/VirtualSpaceManager.cs` | Adapted | empty in Unity project; virtual-space handling lives in `openrdw/ui.py` |

## Redirection

| Unity script | Status | Python equivalent / note |
| --- | --- | --- |
| `Redirection/RedirectionManager.cs` | Adapted | `openrdw/scheduler.py`, `openrdw/simulator.py`; reset sequencing, same-position invalidation, explicit root-pose plus tracking-space-pose state, and tracking-space-driven gain injection are implemented, but Unity transform-hierarchy semantics are still approximated in 2D |
| `Redirection/Redirectors/Redirector.cs` | Implemented | `openrdw/redirectors.py` |
| `Redirection/Redirectors/NullRedirector.cs` | Implemented | `openrdw/redirectors.py` |
| `Redirection/Redirectors/SteerToRedirector.cs` | Implemented | `openrdw/redirectors.py` |
| `Redirection/Redirectors/S2CRedirector.cs` | Implemented | `openrdw/redirectors.py` |
| `Redirection/Redirectors/S2ORedirector.cs` | Implemented | `openrdw/redirectors.py` |
| `Redirection/Redirectors/ZigZagRedirector.cs` | Implemented | `openrdw/redirectors.py` |
| `Redirection/Redirectors/APF_Redirector.cs` | Implemented | `openrdw/redirectors.py` |
| `Redirection/Redirectors/ThomasAPF_Redirector.cs` | Implemented | `openrdw/redirectors.py` |
| `Redirection/Redirectors/MessingerAPF_Redirector.cs` | Implemented | `openrdw/redirectors.py` |
| `Redirection/Redirectors/DynamicAPF_Redirector.cs` | Implemented | `openrdw/redirectors.py` |
| `Redirection/Redirectors/PassiveHapticAPF_Redirector.cs` | Implemented | `openrdw/redirectors.py` |
| `Redirection/Redirectors/VisPoly_Redirector.cs` | Implemented | `openrdw/redirectors.py`, `openrdw/visibility.py` |
| `Redirection/Redirectors/DeepLearning_Redirector.cs` | Adapted | `openrdw/redirectors.py` via `onnxruntime` instead of Unity Barracuda |
| `Redirection/Resetters/Resetter.cs` | Implemented | `openrdw/resetters.py` |
| `Redirection/Resetters/NullResetter.cs` | Implemented | `openrdw/resetters.py` |
| `Redirection/Resetters/TwoOneTurnResetter.cs` | Implemented | `openrdw/resetters.py` |
| `Redirection/Resetters/APF_Resetter.cs` | Implemented | `openrdw/resetters.py` |

## Net result

- The Python port now covers the portable 2D simulation core, command-file workflow, directory batch execution, sampled metrics export, path graphs, waypoint targets, trail controls, tracking-space generation, redirectors, and resetters.
- Several Unity modules remain `Adapted` rather than `Implemented` because their runtime form depends on Unity objects, colliders, meshes, cameras, or scene-hierarchy behavior.
- Rendering/VR/editor/network/avatar-presentation scripts remain outside the scope of a pure-Python 2D port.
