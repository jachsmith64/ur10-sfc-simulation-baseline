# G 组三片 N／线性／SFC 对照

打开本包中的 G_NLS_segments_review.html 可直接离线阅读图文。总览后有原始试次时间轴上的启动、换向、稳定去程和稳定回程图。

已完成三片各一次 N、L、S，共九次连续物理积分。各片 N 的 87 个完整时序数组与归档逐字节一致。九次均无饱和和保护中止。SFC 相对线性在三片启动和换向减小了 RMS 与峰值，瞬态请求力 RMS 增至线性的 2.27–3.03 倍。10 mm/s 稳定回程的 RMS 高 4.75%、峰值高 26.36%。进入规划耦合前先审查等出力线性对照及该回程的峰值代价。

## 归档与复现

- Git 仓库为 https://github.com/jachsmith64/ur10-sfc-simulation-baseline.git。
- 已审代码为 57f6aa7dc498cdcd2398e8a33fbb8dc00720444c。
- 基准新提交为 251e1483577bb07b2589a55a84bc044c2a3c4603，标签为 g-cb3-a-reference-v1-20261116。
- 基准目录为 G/baselines/G_CB3_A_reference_v1_20261116。该目录有原始完整锁定输入、旧最佳 N 的力/公共表及复现身份。完整时序留在各自本机工作项的 local 下。
- 已审代码的实际 Git 直接父提交与旧交接所列祖先的区别、对照提交号、推送状态和恢复依据见 delivery.json。Git 分支包含源码、配置、基准锁定数据、评分及运行回执。审查 ZIP 包含可独立复算的原精度紧凑数组和 HTML。

在用于积分的 Git 检出中，从仓库根目录执行以下命令。绑定目录和输出目录使用新路径。runtime_local 与 runtime_science 由使用者指定为原版本运行库位置。

~~~powershell
$baseline = 'G/baselines/G_CB3_A_reference_v1_20261116'
$item = 'G/20261116_NLS_segments'
python -B -X utf8 "$baseline/archive_baseline.py" bind --destination "$baseline/local/rebound"
& "$item/reproduce.ps1" -Python <Python312.exe> -Bound "$baseline/local/rebound" -Output "$item/local/reproduced" -MuJoCoRuntime <runtime_local> -ScienceRuntime <runtime_science>
~~~

复现原单片 N 可加 -NOnly X_P05_R01 等。第一次 R01 N 积分后的回执写入遇到 NumPy 布尔序列化问题。使用该次已完成的原积分结果修复回执并完成逐字节回归，合计九次有效积分。恢复开关 -ResumeFirstReceipt 仅用于这种已有 R01 N 完整结果且其他模式未运行的场景。归档版本约束、数据身份和守卫仍需通过。

## 独立复算

从解包目录运行 python -B -X utf8 verify_review.py。只依赖 Python 与 NumPy，无需 MuJoCo 或原工作树。校验文件哈希，独立计算每窗 RMS、最大绝对值、百分比和出力指标，并复算原生采样参考、5 Hz 低通、力矩及命令执行证据。

- data/<trial>/A_source.npz 保留原 A 准备数组和有效帧。locked_input.npz 是完整归档 input.npz 中运行所需的原精度选列，逐数组哈希见 identity.json。
- data/shared_background.npz 是三片实证相同的执行背景与 pressure。每片 dq/dqd 读取与自身 1 ms 网格等长的原始前缀，pressure 使用完整表。评分与反馈参考同值网格只保留一份 shared.npz。每片对应的各采样时刻参考继续保留在传感器原生数组中。
- physical_1ms.npz 按 locked_input.npz 的原始时间读取有效前缀。command_native.npz 含 8 ms 时刻的原命令、有符号力分项、关节微补偿及力矩。command_events_native.npz 含发布、样本时刻、数据年龄和保持长度。各命令保持至下一命令，原生相机与编码器记录另存。物理输出、执行器总力矩和实现增量力矩均保留完整 1 ms 精度。
- display_mapping.npz 含 trial_id、original_time_s、display_time_s。0.55 s 空隙仅用于展示。任何评分、速度计算和力矩变化率均在独立试次内计算。
- source_evidence/ 提供相关源码切片、代码哈希和明确的 Git 原路径。完整绑定依赖从归档提交恢复。mode_overlay.patch 对应本轮的两行模式接入，inherited_command_channel.patch 对应已审的完整命令保持入口。
- metrics.json 是各窗原始指标，execution.json 和各模式回执包括真实物理模式、参数、有效步数、力路径、饱和及保护状态。

本轮只在 B 离线积分，A 未操作。等效 Fd 与方向基线适用于固定运动的三片对照。对新路径的激励及实体机械臂反应仍需单独验证。
