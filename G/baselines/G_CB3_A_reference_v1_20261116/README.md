# G CB3 三片 A 参考与 N 基准

已审代码为 57f6aa7dc498cdcd2398e8a33fbb8dc00720444c，直接父提交为 4daecbd0ec3fd9a7c6e89fbc6f7eb30b2c885dff。旧交接记录中的 0fd746f487cb9e5382b7e4249a3824c5c0c8bfe3 是该诊断提交的父提交，也是 Git bundle 的恢复前提。标签 g-cb3-a-reference-v1-20261116 指向本归档的新提交。已审包的身份见 reviewed_delivery.json 与 archive_baseline.py。

- X_P05_R01 为 N_fit_3，X_P05_R02 为 N_fit_4，X_P10_R01 为 N_fit_2。每片保留原始采样、原初态、时钟修正后的名义轨迹、方向参考、既有公共物理表、最佳 N 的等效 Fd 和正负微补偿证据。
- 原始物理横向 Y 用与 A 一致的共同零点表示。反馈和评分分别按各自时刻减同一锁定方向参考。公共慢纠偏在关节目标上施加，独立于参考扣除。
- 此基准适用于这三片的离线反馈抑振比较。真实机器人全动力学、未知方向偏置的在线估计、不同路径下的原始激励仍待单独验证。
- 本次保留完整原始时轴与开头静止记录，没有再次拟合、改参数或运行历史微补偿。正负脉冲的旧数组和证据见各片 inherited_pulse.npz 与 identity.json。

## 恢复入口

在包含标签历史的 Git 仓库中运行 python archive_baseline.py bind --destination <新目录>。这会从固定源码提交恢复 CB3 代码、模型资源及 V02 源码，并用 pinned_source_sha256.json 逐文件校验。V02 守卫与旧模块只从绑定目录加载。锁定的试次数组和运行配置在本归档，不依赖其他工作树。运行环境见 environment.json。对照运行入口及命令见 G/20261116_NLS_segments/README.md，可从任一片的 N 按同一入口独立复现。

完整旧输出在本机本目录的 local/archived_full，可用各片 identity.json 校验文件和每个数组。其他机器可用入口复算，并与每个归档 trace 数组的字节身份比对。历史报告和核心图在 reviewed_core_figures.html。

## 与上一个基准的差异

| 项目 | 已审基准实际使用 |
| --- | --- |
| 机械臂 | 对齐的 CB3，保留物理重力、惯量和被动阻尼 |
| 名义速度 | 保留 (Kd/Kp) qdot_nom，计入公共目标 |
| 时钟与字段 | 使用已审控制柜时间映射与清理后的相容目标位置/速度；X用于合理基础运动 |
| Y 物理重建 | 直接归档最佳 N 的等效力、公共物理表和结果 |
| 方向参考 | 保留来源窗口、事件内基线与过渡，物理表/反馈/评分各自保存 |
| 旧观测剩余项 | remaining 及其导数为零，保留已有 dq/dqd 和编码器 pressure |
