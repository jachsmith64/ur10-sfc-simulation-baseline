# F组理想执行基线_工程修复候选

本候选继承已审核F10-PRESCHEDULED-SLOW-v1的科研输入，修复新运行/历史查看与失败状态。它不是“仿真稳定版”发布，也未验证CB3实机微补偿。完整本机目录为本README所在目录；未新建远端或推送。

## 操作路径

1. **新仿真**：双击`新仿真.cmd`，或运行`python run_slow.py new`。每次生成唯一run_id和runs/run_id；三个方案各实际积分一次，没有输出存在即返回的分支。显式指定的run_id若冲突报错，失败重试用新的run_id及`--retry-of 原失败run_id`，不覆盖前次。此入口不是只重画，未来再次点击会产生三条新积分。
2. **验收**：每组返回后查真实日志的状态、10000有效步、全部必需字段/形状/有限性、积分前时间轴、来源/公共输入/参数与元数据。三组全过后生成本批报告，再原子写completed.json；否则写failure/state、非零退出，保留有效前缀，不继续后组、不调参或放宽阈值。
3. **查看历史**：双击本轮`查看第一批.cmd`或`查看第二批.cmd`，都绑定明确run_id，先核对日志及其独立输入副本/哈希再打开。也可`python run_slow.py show --run-id 明确ID`。缺失、损坏或未完成时报错，不选择别批成功数据。历史查看不依赖当前可变输入或模型XML，也不积分。
4. **仅重画指定批次**：`python run_slow.py redraw --run-id 明确ID`。只读该批日志与保存的窗口输入，输出仍在该批；另记录实际绘图源码哈希/版本，不把重画称为当前仿真重新积分。

本轮先验收的两批ID及六次实际积分在`review/accepted_run_ids.json`；复现检查在`review/reproducibility.json`和逐数组CSV。全部图及指标表关联该批run_id、数据SHA、单位、窗口、采样位置和绘图版本。日志NPZ及有符号完整1ms CSV保留实际q/qdot、各层目标、请求/实际力矩向量、physical/effective Y与速度、事件及限幅状态。静态HTML本身只显示既存文件；正式历史查看入口先校验，不应以直接打开旧HTML替代数据验收。

## 当前调用链与历史边界

`run_slow new → 显式读取已接受输入 → 新run_id → sweep.run_case三次 → PreciseLogger → validate_case → run_report → completed`。

当前链不调用`run_short.run/calibrate/formalize/main`、`get_q_traj`、`run_batch`或扰动反演；名义q与背景是明确冻结输入，不是仿真结果缓存。旧A0只作为已接受公共慢中心的历史来源，当前三条全部新跑。

旧B所有入口、旧F原件均不修改。副本中的`run_short.py`保留R026历史保护，CLI原已禁用；`sweep.get_q_traj/run_batch`、`short_prepare.py`、旧`short_report.py/slow_report.py/pack_slow.py/check_slow_review.py`保留历史用途但不属于新入口调用链，不运行它们来生成当前结果。原run_slow/report及有效旧输出另存historical。不把当前入口修复扩写成所有历史函数均已修复。

科研设置保持：q_nominal、g*dq、remaining/导数、Fd、已接受公共纠偏、初态、K/B0/n/μ、仿射伺服Kp/Kd和1ms时序。公共与抗抖的代价分列；强SFC较大作用仍如实报告。请求/实现力矩增量不是补偿位置精度；R017资料与建议见review/R017，未增加执行噪声或新模型。

运行生成身份覆盖源码、配置、模型XML及依赖资产、输入/参数、Python/NumPy/MuJoCo、步长/更新周期和Git提交/未提交状态。父仓库包含未跟踪F副本，不能仅用其HEAD指代此候选；实际文件清单记录在每批generation.json。记录用于追溯，不授予结果缓存复用许可。

后续发布前需决定执行模型及参数依据，验收新图，并明确已验证范围/未解决项。稳定只表示指定模型可复现，不能声称CB3已验证。本轮只准备差异和清单。
