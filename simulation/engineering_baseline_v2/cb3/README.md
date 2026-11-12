# UR10 CB3 离线几何与 A 组换向诊断（20261112）

历史 F 基点为 `archive/pre-ur10e-review-20261109`，提交 `79df355a931c2c497b2eeccf4af6df09a988ddd3`。旧 `run_slow.py` → `SimEnv` → `assets_ur10e/scene.xml` → `ur10e.xml` 实际加载 UR10e。新入口 `cb3/diagnose.py` 显式加载 `cb3/assets/ur10_cb3.xml`，用于修正历史代理的机型和诊断调用。

## 复现

在 `simulation/engineering_baseline_v2` 运行。输入采用原样的 A 组派生 CSV 和同一行记录六轴实际角、柜端 TCP 的实测 CSV。C 组日志可用于独立几何复核，来源身份随审查包保存。

```powershell
python cb3/diagnose.py --a sources_A/P05R01/D1_时序.csv sources_A/P05R02/D1_时序.csv --tcp <同帧关节角与柜端TCP的CSV> --legacy-scene assets_ur10e/scene.xml --out <新目录>
```

实测 CSV 使用 `robot_timestamp_s`、`actual_q_J1_rad` … `actual_q_J6_rad`（或六轴统一使用 `_deg`）及 `actual_tcp_x/y/z`（米）。如有 `stage=path_pre_motion` 则报告该静止段中位偏差与运动残差。样本严格按原顺序，同一行配对。A 表记录沿程最大值和实测派生速度由正转负的采样区间。普通段在原出程、回程标签内，以相应方向达到 2.5 mm/s 的首末样点为界，再从两端各去 0.25 s。这样可排除 R02 仍标为回程的停后静置。无采样重构、空间拟合或时间插值。结果包含完整原采样关键列和模型输出。验证环境为 Python 3.13.13、NumPy 2.4.4、MuJoCo 3.8.0。

## 参数和边界

- 运动学采用 UR 的 UR10 名义 DH。J1–J6 对应日志顺序，零位按名义 DH，柜端 Base 取恒等。TCP 暂取法兰零偏置。旧点在腕三局部 `[0,0.1,0]` m，两模型的比较包含不同末端定义。须取得 A 试次的安装 TCP 和控制器标定才能判断微米级几何一致性。没有用于 A 名义轨迹的 IK 输出。
- CB3 六个运动体采用 7.1、12.7、4.27、2.0、2.0、0.365 kg。质心、惯量使用 UR 发布的 URSim 名义数据，将惯量按 X 轴标注的角度转至链接坐标后写入 MJCF。固定基座质量不参与积分。模型不含渲染网格、接触、已知工具负载或柔性件。
- 新入口沿用 `SimEnv` 和 `ExecutionManager`。复核 FK、TCP 雅可比、力到关节力矩 `JᵀF`、以及同状态下按编译增益的 `Δq=τ/Kp`。保留历史位置伺服代理 Kp 5000 N·m/rad、Kd 500 N·m·s/rad，轴被动阻尼 10/10/5/2/2/2 N·m·s/rad，armature 每轴 0.1 kg·m²，代理力矩上限 330/330/150/56/56/56 N·m 及原肘部代理限位。这些值来自旧仿真资产，实机增益、动力学和执行限制待辨识。
- 原 SFC/线性律、误差分层、测量与执行日志及保护继续保留在历史代码中。当前只迁入按源时间轴读取 A 组数据和按既有阶段取事件窗口的逻辑。候选控制配置记录在 `validation.json`，对应 11 月 10 日 5 Hz 保守滤波链与 11 月 11 日 A 组旧对照。未来配对采用 1 倍线性阻尼。2 ms、20 Hz、短窗口和 ED 各自沿原证据归档。CB3 新入口未积分控制，故没有 A 无新增抑振、线性或 SFC 波形。

[UR 官方 DH 参数](https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics)

[UR ROS 2 描述的 UR10 质心和惯量，固定提交](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description/blob/0eb1a861f7772fceb38510d90fa2f98ec668070c/config/ur10/physical_parameters.yaml)
