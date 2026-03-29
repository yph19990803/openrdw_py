# Python OpenRDW 使用说明

这个目录包含一个面向 OpenRDW 的 Python 重构 / 移植版本，重点覆盖 Unity 原项目中可迁移的二维仿真核心。

## 已实现内容

- 二维几何与位姿状态模型
- 追踪空间生成与文件加载
- 路径 / waypoint 生成与回放
- 固定步长模拟器
- 多 agent 调度
- 重定向算法：
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
- 重置算法：
  - `none`
  - `two_one_turn`
  - `apf`
- 轨迹导出、统计汇总、sampled metrics
- command file / command directory 批量实验运行
- 本地 Web UI

## 快速开始

### 1. 启动本地 UI

```bash
PYTHONPATH=python_openrdw python3 -m openrdw.ui
```

### 2. 运行测试

```bash
PYTHONPATH=python_openrdw python3 -m unittest discover -s python_openrdw/tests -v
```

### 3. 运行命令行仿真

```bash
PYTHONPATH=python_openrdw python3 -m openrdw --steps 200 --redirector s2c --resetter two_one_turn
```

## UI 功能

当前 UI 支持：

- 选择 redirector / resetter
- 选择 path / tracking space
- 设置物理空间与虚拟空间尺寸
- 设置物理障碍物与虚拟障碍物
- 设置 buffer 与 body diameter
- 单步、连续运行、重置、导出 CSV
- 查看虚拟空间固定、物理空间相对变化的叠加俯视图
- 查看虚拟轨迹与物理轨迹
- 查看实时位姿、增益、waypoint、reset 状态
- 运行 command file / command directory 实验

## 与 Unity 原版的关系

这个 Python 版以 Unity OpenRDW 为行为参考，但并不是 Unity 引擎本身的 1:1 替代。不能直接等价替代的部分主要包括：

- SteamVR / HMD
- Photon 联网
- Unity 的 GameObject / Transform / Collider / Camera / Mesh / Material / Shader
- Unity Editor 脚本和场景资源系统

更详细的对齐状态见：

- [`EQUIVALENCE_STATUS.md`](EQUIVALENCE_STATUS.md)
- [`PORT_STATUS.md`](PORT_STATUS.md)
- [`UNITY_SCRIPT_COVERAGE.md`](UNITY_SCRIPT_COVERAGE.md)
