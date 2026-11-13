# A 组换向工作入口

在独立分支保留 2026-11-12 的名义 CB3 MJCF 与静态诊断。运行仅需离线文件。

```powershell
python -B -X utf8 cb3/turning/audit_inputs.py <已验交接包根> <新输入核查目录> --legacy-scene assets_ur10e/scene.xml
python -B -X utf8 cb3/turning/probe_model.py <输入核查目录> <新探针目录>
python -B -X utf8 cb3/turning/check_feedback.py
```

输入核查按柜端时刻去重，按主机时戳进行有上限的比较对齐。固定窗为首个相机主机时刻后 3.60–4.56 s。实际关节角仅用于 FK 及定点初值。六维位姿回代复用 B 端 C 组 `audit_c_path.py` 的位置及旋转向量最小二乘方法，比较两种工具假设并只保留回代和连续性指标。

定点探针用固定位置指令、TCP 世界系 Y 向 1 N 脉冲、经既有 `ExecutionManager` 的等效位置微补偿三种输入验证可控响应。脉冲来自预设常量。原控制器的 `gamma` 可选项和 `sfc.py` 取自交接包 V05 工程，默认 1。V05 反馈模块及短窗依赖按原字节放在 `feedback/`，独立复算保守的一阶 5 Hz 链路。历史通用积分入口、日志和保护沿用其原版本。

换向 N 仍需确定 flange/TCP 对应、跨缺口的输入边界及固定指令下的重力稳态偏差处理。名义轨迹和扰动只在 GPT 确认后进入正式换向积分。
