# B05 A 实测背景与控制响应增量

从本目录的上两级工程根运行。输入目录包含上一轮核验的 `input_audit.json`、两条 `X_P05_R0*_observed.npz`。另传入上一轮的 `model_probe/probe.json`。

```powershell
python -B -X utf8 cb3/turning/run_increment.py <input_audit> <probe.json> <新目录> --stationary-only
python -B -X utf8 cb3/turning/run_increment.py <input_audit> <probe.json> <另一新目录>
```

先验证 0.5 s 静止保持及两向脉冲。选定的局部对齐直接写入独立 MJCF 的 site，关节 FK、雅可比和外力作用点用该 site。重力支撑以名义角、零速的独立 MuJoCo 状态计算，经编译 Kp 换成公共关节命令偏置。名义物理目标仍来自六维柜端目标 TCP。

两个试次使用公共有效段 1.256–6.600 s。剔除两头的长柜端缺口，柜端短括点内重建名义输入。仿真初态为首个重建名义角和零速度，首个普通评价窗以前约有 0.32 s 模型推进。保存窗及保持时序来自 `increment_config.json` 和上一轮核验。

先积分无新增抑振 N。复算 N 接线，并按相同步读取 N 轨迹。相机位移代理为最近且已到达的 A 帧加模型末端响应差，编码器位置经同点 FK，关节输入是最近 A 实际角加模型关节响应差。两点因果差分及一阶 5 Hz 速度低通复用已归档 Channel。全指令每 8 ms 更新和保持。没有附加外扰、噪声合成、直接力矩注入或主状态重置。命令范围和执行器力矩使用原编译保护，发散量沿用历史阈值。

R01 先比较 N、L×1、L×4、SFC。以固定换向窗 RMS 从两个完成的历史线性候选中选择，对 R02 固定参数。额外的 R01 因果测试只改一个指定时刻及之后的视觉输入，用于验证先发布、再命令、再物理变化的顺序。输出数组含物理状态、命令、发布反馈、力矩及原始视觉帧上的误差估计。完整口径和列名在 summary.json。
