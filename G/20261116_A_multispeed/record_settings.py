"""Append observed B results to every item of the supplied G01–G24 record."""
from pathlib import Path
import json

from report_data import HERE, RUNS, digest, read, read_csv, write_csv, dump

RUN_G = "run_g.py"
SOURCE = "source_a.py"
CORE = "v02_portable/run_campaign.py"
SENSE = "v02_portable/filter_sensing.py"
SWEEP = "v02_portable/source_A/01_工程/sweep.py"
ENV = "v02_portable/source_A/01_工程/sim_env.py"
IK = "../../simulation/engineering_baseline_v2/cb3/turning/increment_model.py"
MODEL = "../../simulation/engineering_baseline_v2/cb3/assets/ur10_cb3_b05_aligned.xml"
SCORE = "score_g.py"

DETAILS = [
("已复跑","完整冻结母版、入口、绑定输入以及双运行时逐文件校验，共 6096 件。三条 UR10e 锚点独立积分 10000 步，所有数组、输入标识、保护记录与历史一致。G 使用独立副本。","",[(CORE,"def initialize"),("evidence/regression_receipt.json",None),("evidence/v02_full_copy_manifest.json",None)]),
("N 已跑","旧积分循环保留 1 ms 物理推进。A 来源名义关节目标经位置伺服，物理实际运动由 MuJoCo 算出。反推 Y 力走原 TCP 点力接口。","L/S 待门禁。",[(SWEEP,"env.apply_disturbance_force_y"),(ENV,"def apply_disturbance_force_y"),(RUN_G,"def reconstruct_trial")]),
("N 已跑","新 G 使用 CB3 对齐 XML，旧回归独立用 UR10e。各轴 Kp/Kd 实际 5000/500，未调惯量、阻尼或增益。","",[(MODEL,'<mujoco'),(RUN_G,"mapping = compiled_mapping"),("06_实际参数与版本.json",None)]),
("N 已跑","沿 B05 固定映射的世界 Y 重建。沿 X 核对行程和时间。已恢复固定等效 TCP 对齐，按原始视觉映射取零点。独立工具标定未做。","",[(SOURCE,"inverse ="),(MODEL,"<site"),(RUN_G,"return aligned_model")]),
("N 已跑","按柜端 target_tcp_pose 解六维 IK。actual_q 首个有效值只用于数值初值。原记录 target_q 缺失。","",[(IK,"def reconstruct"),(IK,"q = obs"),(SOURCE,"return reconstruct")]),
("N 已跑","从原首个视觉帧到最后一帧，独立 1 ms 网格，实际启动前证实静止。仿真 q 初态为记录名义姿态，零速，无预热，保存初始沉降。柜端长空档保留并核对其名义 X 两端差及事件窗口。","10 mm/s 相同柜端时间戳的不同载荷经主机时间保留。",[(SOURCE,"grid ="),(SOURCE,"def archived_source"),(RUN_G,"initial_nominal_q_rad")]),
("N 已跑","复用 CB3 重新测量的复频响、旧 30/45 Hz 余弦带限和 2 秒填充。按 N 总观测残差迭代四次，边界采用固定端值以减小 FFT 边缘台阶。最终 force、慢纠偏和共享输入逐片锁定。","第 3、4 次对抖动层收益有限，未扩展控制扫描。",[(RUN_G,"def inverse"),(RUN_G,"for number, relaxation"),("v02_portable/source_A/01_工程/disturbance.py","def invert_series")]),
("N 已跑","保留原 C 统计 dq/dqd 非零切片，与名义和慢纠偏相加形成基底目标。覆盖各试次完整长度。","L/S 入口复用锁定数组。",[(RUN_G,'fields = ("dq"'),(SWEEP,"base_ctrl = q_traj"),("v02_portable/source_A/b132_run.py","def context")]),
("N 已跑","remaining 是同一旧观测项，进入 N 残差账目且走相机观测支路。无独立追加 A 曲线误差。同步保留物理输出来核对剩余项占比。","仍无法唯一分离 A 真实运动和测量成分。",[(SENSE,"camera_r ="),(RUN_G,"residual ="),(SCORE,"observed =")]),
("N 已跑","保留 V02 主压力 path-six-v1-r1 作为 8 ms 编码器代理读数输入，源自旧 C 返回直线。三片长度在 1250 点母版范围内，未重复或时间缩放。","",[(CORE,"def pressure"),(SENSE,"self.q.capture"),("v02_portable/frozen/pressure_identity.json",None)]),
("N 已跑","独立额外相机随机噪声数组为 0。图中 Y 仍含模型执行与 remaining 观测项。","历史 3 μm 只用于旧回归。",[(RUN_G,"ctx[\"pressure\"], np.zeros"),(SENSE,"self.c.capture")]),
("N 已跑","当前模型 TCP 派生视觉位置，编码器取物理位置代理加冻结压力。各自独立采样，命令年龄逐步记录。N 无补偿，控制入口绑定 Q 估速。","L/S 的闭环响应未运行。",[(SENSE,"def tick"),(SENSE,"def measure"),(CORE,"compute_cam_age_ms")]),
("历史锚点已复跑，G 控制待门禁","沿 V02 仅用相邻同源代理位置、实际时间差求速度。N 仍记录代理读数和差分历史。","G 控制组没走估速到力的动态闭环。",[(SENSE,"def capture"),(RUN_G,"case.update")]),
("配置已记录，G 控制待门禁","预设 5 / 20 Hz 速度后滤波，均一阶纯低通，零残差、无预测。旧 5 Hz 锚点已复跑。","本轮 G 5/20 Hz 抑振对照未积分。",[(RUN_G,"for cutoff"),("g_config.json",None),(SENSE,"def coefficients")]),
("配置已记录","禁用位置前滤波、短窗口与 ED。沿 V02 位置支路。","",[(SENSE,"self.c = Channel"),("g_config.json",None)]),
("N 已跑","1 ms 物理、8 ms 编码器、8 ms 完整六轴命令保持、132 Hz 模拟相机加 4 ms 可用延迟，编码器/执行附加延迟 0。原 A 时间仅用于离线来源和评分。各次保持和因果校验留存。","1 ms 是仿真积分格点。",[(RUN_G,"channel = core.CommandChannel"),(CORE,"class CommandChannel"),(SENSE,"def tick")]),
("N 已跑","CB3 前馈为零，物理重力沿 XML 编译为 [0,0,-9.81] m/s²。公共指令修正与原伺服承担慢偏移。","旧显式重力分支仅保留兼容调用。",[(SOURCE,"gravity_support=False"),(IK,"if gravity_support"),(RUN_G,'mapping["gravity_m_s2"]')]),
("N 已跑","重算各片启动前、稳定去程、稳定回程、停止后平台的 N−A 中心，用初态几何右逆换算到关节、C2 平滑过渡。无纠偏/仅纠偏的 N 保留作为独立证据。","G 控制入口直接复用表。",[(RUN_G,"def smooth_platforms"),(RUN_G,"levels ="),(RUN_G,"direction =")]),
("G 控制待门禁","登记每片 7 项。N 共用，L1 400、L4 1600 N·s/m，SFC 采用 V02 参数，各含 5/20 Hz。在时序与物理响应验收前自动阻断。","三片 L1/L4/SFC 均未积分。",[(RUN_G,"for law, gain"),(RUN_G,"if options.with_controls and coverage"),("results/control_gate.json",None)]),
("N 已跑","整片、启动、换向、去程、回程、前后静止分别出数。记录物理轨迹、同步总观测和延迟反馈、命令、力、增量力矩。保护有效前缀单独保留。","L/S 成本与指标需通过门禁后实算。",[(SOURCE,"windows ="),(SCORE,"def evaluate"),("05_重建验收与控制结果.html",None)]),
("完成三片 N 候选诊断","5 mm/s R01/R02 完整交接；10 mm/s R01 归档视觉逐帧与原 sidecar、RTDE 对齐。共两个真实速度、三次实际启动和换向。选片按对齐/名义位姿可用性，不依 FIT。","少于目标 4—6 片。剩余试次缺有六维 target_tcp_pose 的 RTDE 或完整相机原件，详见 03。",[(SOURCE,"def archived_source"),("03_A片段与拼合时间映射.csv",None)]),
("N 已跑","复制原保护前置路径和 20 ms 连续饱和保护。所有保留正式 N 重建迭代均完整。保留 10 mm/s 首次因柜端重复戳假设提前终止的队列记录。","微补偿验收和保护下的控制结果待 N 门禁。",[(CORE,"persistent_saturation"),(RUN_G,"class PulseCoordinator"),("evidence/A_P10_first_attempt.json",None)]),
("部分达标","整段、启动、换向的同步总观测两层 FIT 94.92–98.14。三片物理整段 FIT 69.65–73.98，启动 64.66–77.55，X 的三类事件落后 76–113 ms。未平移、未改验收分数。","50 ms 是本次预设的事件差行动阈值。验收未通过。",[(SCORE,"def metrics"),("results/FIT事件汇总.csv",None),("g_config.json",None)]),
("已归档","三片独立积分再拼成展示时间轴，间隔 0.35 s，所有显示时间和平移映射可查。独立本地分支、SHA 清单、代码 diff、A 接收及恢复入口放在交付目录。","正式 G 由 A 电脑接入。",[(RUN_G,"def main"),("03_A片段与拼合时间映射.csv",None),("07_以后如何恢复.md",None)])
]


def identify(p, needle):
    file = HERE/p
    sha = digest(file)
    line = None
    if needle:
        line = next(i for i,s in enumerate(file.read_text(encoding="utf-8-sig").splitlines(),1) if needle in s)
    return dict(file=p, line=line, sha256=sha)


def mapping():
    rows=read_csv(HERE/"instructions/04_A_registered_candidates.csv")
    archive=HERE/"local/A_handoff/data/analysis_20260902"
    inv=read_csv(archive/"inventory.csv")
    original_source={x["filename"]:x for x in inv}
    offset=0.
    selected={}
    for trial,batch in RUNS.items():
        record=read(HERE/"prepared_A"/trial/"source.json")
        sec=record["windows_s"]["full"][1]
        selected[trial]=dict(source_kind=record["source_kind"], source_start_s=0.0, source_end_s=sec,
                   host_origin_ns=record["source_clock_origin_ns"], actual_onset_host_s=record["first_half_speed_s"],
                   actual_turn_host_s=record["turn_actual_xmax_s"], actual_v_out_mm_s=record["source_median_vx_mm_s"]["outbound"],
                   actual_v_return_mm_s=record["source_median_vx_mm_s"]["return_straight"], missing_frames=record["missing_frame_count"],
                   display_start_s=offset, display_end_s=offset+sec, reset_at_display_s=offset,
                   sha256=digest(HERE/"prepared_A"/trial/"A_source.npz"),
                   reason="真实相机逐帧数据、柜端六维目标及记录事件可交叉核验")
        offset+=sec+.35
    out=[]
    for row in rows:
        stem=row["segment_base"]
        trial=stem.split("_B05_",1)[1]
        key=stem+"_UR10.txt"
        status=original_source.get(key,{})
        common=row|dict(trial=trial, original_rtde_expected_path=status.get("absolute_path"),
                        original_rtde_expected_sha256=status.get("sha256"))
        if trial in selected:
            out.append(common|dict(candidate_status="INCLUDED_FOR_N",**selected[trial]))
        else:
            reason = "D 盘批次归档存在视觉逐帧结果；B 侧未找到原 RTDE 六维名义目标。"
            if trial == "X_P01_R01":
                reason += "登记缺帧 26.0%，最长连续 5 帧。"
            out.append(common|dict(candidate_status="EXCLUDED_SOURCE_GAP", source_kind="archive_vision_only", reason=reason))
    write_csv(HERE/"03_A片段与拼合时间映射.csv",out)


def main():
    mapping()
    rows=read_csv(HERE/"instructions/G_decisions.csv")
    text=(HERE/"instructions/02_G01_G24_trace.md").read_text(encoding="utf-8-sig")
    out=[]
    appendix=["\n# B 电脑实际执行回执\n"]
    for i,(row,(stage,actual,why,sources)) in enumerate(zip(rows,DETAILS,strict=True),1):
        assert row["id"] == f"G{i:02}"
        refs=[identify(p,k) for p,k in sources]
        out.append(row | dict(actual_status=stage,actual_implementation=actual,divergence_or_limit=why,actual_evidence=refs))
        appendix.append(f'\n## {row["id"]} {row["item"]}\n\n**{stage}**。{actual} {why}\n\n')
        appendix.extend(f'- `{x["file"]}`'+(f' 第 {x["line"]} 行' if x["line"] else "")+f'，SHA-256 `{x["sha256"]}`\n'
                        for x in refs)
    dump(HERE/"01_G01-G24_全部设置与历史差异.json",out)
    write_csv(HERE/"01_G01-G24_全部设置与历史差异.csv",[r|dict(actual_evidence=json.dumps(r["actual_evidence"],ensure_ascii=False)) for r in out])
    (HERE/"01_G01-G24_全部设置与历史差异.md").write_text(text+"\n"+"".join(appendix),encoding="utf-8")


if __name__=="__main__":
    main()
