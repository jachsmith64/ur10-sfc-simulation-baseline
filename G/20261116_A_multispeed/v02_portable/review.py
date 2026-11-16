"""Package independently reviewable statistics, raw trajectories and an offline HTML report."""
import argparse
import csv
import difflib
from hashlib import file_digest
from html import escape
from io import BytesIO
import json
from pathlib import Path
import re
import shutil
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "filter_sfc_20261110"
OUT = ROOT / "results"
PACKAGE = ROOT / "review"
ZIP_NAME = "B_强滤波SFC_保守基线_审查包_20261110.zip"
COLORS = {"N": "#505960", "L": "#007bb6", "S": "#c74126", "H": "#8d6aa6"}
LABELS = {"N": "N 无新增抑振", "L": "同滤波线性", "S": "新 SFC", "H": "旧 10 Hz / ρ=0.25 SFC"}
ROW_METRICS = {
    "full_rms_um": "全程 RMS",
    "full_peak_um": "全程峰值",
    "ordinary_rms_um": "普通段 RMS",
    "ordinary_peak_um": "普通段峰值",
    "ordinary_y_band_15_60_um": "普通段 H",
    "ordinary_y_band_60_500_um": "普通段 60–500 Hz",
    "ordinary_tau_rms_Nm": "普通段力矩 RMS",
    "ordinary_tau_peak_Nm": "普通段力矩峰值",
    "ordinary_tau_rate_rms_Nm_s": "普通段力矩变化率 RMS",
    "ordinary_tau_rate_peak_Nm_s": "普通段力矩变化率峰值",
    "full_tau_rms_Nm": "全程力矩 RMS",
    "full_tau_peak_Nm": "全程力矩峰值",
    "full_tau_rate_rms_Nm_s": "全程力矩变化率 RMS",
    "full_tau_rate_peak_Nm_s": "全程力矩变化率峰值",
}
TRACE_FIELDS = (
    "t", "physical_y", "nominal_y", "physical_e_um", "physical_v",
    "enc_v", "enc_vf", "enc_event", "enc_sample", "enc_available",
    "cam_e", "cam_event", "cam_sample", "cam_available",
    "updated", "sample_e_m", "sample_v_m_s", "compute_ready",
    "compute_raw_v", "compute_filtered_v", "compute_cam_sample", "compute_cam_available",
    "compute_enc_sample", "compute_enc_available",
    "F_spring", "F_linear", "F_shear", "F_requested", "Fd",
    "exec_event", "applied_source_compute", "applied_force", "hold_until_exclusive",
    "actual_micro_q", "tau_realized_increment", "qdot",
    "ctrl_saturation", "force_saturation", "joint_saturation",
)
GATE_NAMES = ["完整与保护", "原始位移 RMS/峰值", "普通段 H", "普通段附加力矩 RMS", "全部事件峰值/IAE"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def table(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, keys)
        writer.writeheader()
        writer.writerows(rows)


def match(rows, stage, law, seed, filt=None):
    matches = [r for r in rows if (r["group"], r["law"], r["seed"]) == (stage, law, seed)
               and (filt is None or (r.get("filter_order", 1), r["fc"]) == tuple(filt))]
    assert len(matches) == 1, (stage, law, seed, filt)
    return matches[0]


def html_table(headings, rows):
    headings = "".join(f"<th>{escape(str(value))}</th>" for value in headings)
    body = "".join("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row) + "</tr>" for row in rows)
    return f'<div class="scroll"><table><thead><tr>{headings}</tr></thead><tbody>{body}</tbody></table></div>'


def figure(fig, caption):
    with BytesIO() as buffer:
        fig.savefig(buffer, format="svg", bbox_inches="tight", pad_inches=.16)
        svg = buffer.getvalue().decode("utf-8")
    plt.close(fig)
    svg = re.sub(r'id="([^"]+)"', lambda m: f'id="fig{figure.count}_{m[1]}"', svg)
    svg = re.sub(r'(href="#|url\(#)([\w.-]+)', lambda m: f'{m[1]}fig{figure.count}_{m[2]}', svg)
    figure.count += 1
    return f"<figure>{svg[svg.index('<svg'):]}<figcaption>{escape(caption)}</figcaption></figure>"


figure.count = 0


def curves(ax, traces, key, lo=None, hi=None, keys=("N", "H", "L", "S")):
    for k in keys:
        a = traces[k]
        t = a["t"]
        inside = (t >= (lo if lo is not None else t[0])) & (t <= (hi if hi is not None else t[-1]))
        y = a[key]
        if y.ndim > 1:
            y = np.linalg.norm(y, axis=1)
        ax.plot(t[inside], y[inside], lw=1 if k == "H" else 1.15, alpha=.55 if k == "H" else .94,
                color=COLORS[k], label=LABELS[k])
    ax.grid(alpha=.22)
    if lo is not None:
        ax.set_xlim(lo, hi)


def plots(traces, windows, seed, ranking, by_id, filt_name):
    fig, ax = plt.subplots(figsize=(11.5, 3.8), layout="constrained")
    curves(ax, traces, "physical_e_um")
    for w in windows:
        ax.axvspan(w["start_s"], w["end_s"], color="#dbe5ed", alpha=.45, zorder=0)
    ax.set(title=f"全程原始物理 Y 偏移 · {filt_name} · 种子 {seed} · +0 ms", xlabel="时间 (s)", ylabel="Y 偏移 (μm)")
    ax.legend(loc="upper right", ncol=2)
    yield figure(fig, "同一纵轴保留物理 Y 偏移的均值与峰值，阴影为预先冻结的事件窗。新 SFC 和同滤波线性均降低了相对 N 的启动与换向偏移，本图使用第一个留出种子。")

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 5.8), sharex=True, layout="constrained")
    curves(axes[0], traces, "physical_e_um", 3, 3.65)
    curves(axes[1], traces, "tau_realized_increment", 3, 3.65)
    axes[0].set(title=f"普通段 3.00–3.65 s · {filt_name} · 种子 {seed}", ylabel="Y 偏移 (μm)")
    axes[1].set(xlabel="时间 (s)", ylabel="附加力矩范数 (N·m)")
    axes[0].legend(loc="upper right", ncol=2)
    yield figure(fig, "上图是未平滑的物理位移，下图是实际生效六轴命令引起的模型内附加力矩范数。相对旧 SFC，两种纯低通控制的局部位移波动与出力均下降。")

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.2), sharex=True, sharey=True, layout="constrained")
    for ax, key in zip(axes, ("S", "L"), strict=True):
        a = traces[key]
        t = a["t"]
        mask = (t >= 3) & (t <= 3.65)
        for field, label, color, style in (
                ("enc_v", "已发布 Q 原始差分", "#6e757d", ":"),
                ("enc_vf", "已发布 Q 低通", "#02976c", "--"),
                ("physical_v", "物理 Y 速度", "#292f35", "-")):
            ax.plot(t[mask], a[field][mask] * 1e3, color=color, linestyle=style, lw=1.1, label=label,
                    drawstyle="steps-post" if field.startswith("enc_") else "default")
        upd = mask & a["updated"].astype(bool)
        ax.plot(t[upd], a["sample_v_m_s"][upd] * 1e3, ".", markersize=3, color=COLORS[key],
                label="控制更新实际使用")
        ax.set(title=LABELS[key], ylabel="Y 速度 (mm/s)")
        ax.grid(alpha=.22)
    axes[0].legend(loc="upper right", ncol=2)
    axes[1].set(xlabel="时间 (s)", xlim=(3, 3.65))
    fig.suptitle(f"Q 速度输入与物理速度 · {filt_name} · 种子 {seed}")
    yield figure(fig, "Q 差分、低通发布值和控制时刻的实际使用值分别绘制，深灰线为同一次闭环的真实 Y 速度。图可检查采样和滤波后的局部形状，单凭变号无法判定噪声来源。")

    fig, axes = plt.subplots(2, len(windows), figsize=(12.5, 5.5), sharey="row", layout="constrained")
    for i, w in enumerate(windows):
        lo, hi = w["start_s"], w["end_s"]
        curves(axes[0, i], traces, "physical_e_um", lo, hi)
        curves(axes[1, i], traces, "tau_realized_increment", lo, hi)
        axes[0, i].set_title(f'{w["label"]} · {lo:.3f}–{hi:.3f} s')
        axes[1, i].set_xlabel("时间 (s)")
    axes[0, 0].set_ylabel("Y 偏移 (μm)")
    axes[1, 0].set_ylabel("附加力矩范数 (N·m)")
    axes[0, 2].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"全部冻结事件 · {filt_name} · 种子 {seed}")
    yield figure(fig, "同一行共用纵轴。窗口来自名义轨迹，换向发生在直线路径上。各窗峰值、绝对误差积分和出力随逐事件表交付。")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
    for r in ranking:
        rows = [by_id[cid] for cid in r["ids"]]
        if not all(row["completed"] for row in rows):
            continue
        law = r["law"]
        tau = max(row["ordinary_tau_rms_Nm"] for row in rows)
        h = max(row["ordinary_y_band_15_60_um"] for row in rows)
        marker = "o" if r["passed"] else "x"
        for ax, y in zip(axes, (h, r["worst_iae"]), strict=True):
            ax.scatter(tau, y, marker=marker, color=COLORS[law], s=45, alpha=.85)
            if law == "S" and r["filter"] in ([1, 3], [1, 5]) and r["gain"] == 1:
                ax.annotate(f'{r["filter"][1]} Hz', (tau, y), xytext=(3, 5), textcoords="offset points", fontsize=9)
    for ax in axes:
        ax.set_xlabel("两校准种子较高的普通段附加力矩 RMS (N·m)")
        ax.grid(alpha=.22)
    axes[0].set_ylabel("两种子较高的普通段 H (μm)")
    axes[1].set_ylabel("最差种子总事件 IAE / 匹配 N")
    axes[1].axhline(1, lw=.7, color="#646464", ls="--")
    fig.suptitle("校准候选 · 红 SFC / 蓝线性 · 圆点两种子全通过 / 叉号未通过")
    failed = [f'{r["law"]} {r["filter"][1]} Hz 增益 {r["gain"]:g}' for r in ranking if r["complete"] < 2]
    note = "保护停止点仅进入失败表。" if failed else "图中各点均完成两次校准闭环。"
    yield figure(fig, f"横轴为模型内力矩代价。左图比较快速位移波动，右图比较事件积分。{note}逐门槛判定和各候选增益见汇总。")


def comparisons(rows, event_rows, selection, protocol):
    results = []
    events_by_id = {(r["id"], r["event"]): r for r in event_rows}
    for stage, seeds, setups in (("D", protocol["holdout_seeds"], selection["selected"]),
                                  ("E", protocol["holdout_seeds"], selection["selected"][:1])):
        for setup in setups:
            filt = setup["filter"]
            for seed in seeds:
                sfc = match(rows, stage, "S", seed, filt)
                for law in ("N", "L", "H"):
                    if law == "H":
                        ref = match(rows, "history", "S", seed)
                    else:
                        ref = match(rows, stage, law, seed, filt if law == "L" else None)
                    context = dict(stage=stage, order=filt[0], cutoff_hz=filt[1], seed=seed,
                                   sfc=sfc["id"], reference=ref["id"], reference_law=law,
                                   sfc_complete=sfc["completed"], reference_complete=ref["completed"])
                    entries = [("", k, sfc.get(k), ref.get(k)) for k in ROW_METRICS]
                    for name in ("start", "turn", "deceleration_stop"):
                        a, b = (events_by_id[(r["id"], name)] for r in (sfc, ref))
                        for k in ("peak_um", "iae_um_s", "tau_rms_Nm", "tau_peak_Nm"):
                            entries.append((name, k, float(a[k]) if a[k] else None, float(b[k]) if b[k] else None))
                    for event, metric, a, b in entries:
                        results.append(dict(context, event=event, metric=metric, sfc_value=a,
                                            reference_value=b, ratio=a / b if a is not None and b else None))
    return results


def assessment(selected, gates, comparisons, protocol):
    def passes(stage, setup):
        o, hz = setup["filter"]
        found = [r for r in gates if r["id"].startswith(f"{stage}_S_o{o}_f{hz}_")]
        assert len(found) == len(protocol["holdout_seeds"])
        return {int(r["seed"]): [name for i, name in enumerate(GATE_NAMES, 1) if r[f"gate{i}"] != "True"]
                for r in found}

    def wins(stage, setup):
        compared = [r for r in comparisons if r["stage"] == stage and r["reference_law"] == "L"
                    and [r["order"], r["cutoff_hz"]] == setup["filter"]]
        main_metrics = {"full_rms_um", "ordinary_rms_um", "ordinary_peak_um", "peak_um", "iae_um_s"}
        gain = []
        worst_effect = []
        costs = []
        for event, metric in dict.fromkeys((r["event"], r["metric"]) for r in compared):
            ratios = [r["ratio"] for r in compared if (r["event"], r["metric"]) == (event, metric)]
            if metric in main_metrics:
                if all(v is not None and v <= protocol["effect_gain_ratio"] for v in ratios):
                    gain.append(f"{event}/{metric}" if event else metric)
                worst_effect.extend(ratios)
            if not event and metric == "ordinary_tau_rms_Nm":
                costs = ratios
        similar = all(v is not None and v <= protocol["similar_effect_ratio"] for v in worst_effect)
        cost_gain = all(v is not None and v <= protocol["torque_gain_ratio"] for v in costs)
        cost_increase = any(v is not None and v > protocol["similar_effect_ratio"] for v in costs)
        return dict(repeat_effect_gain=gain, effects_within_5pct=similar, repeat_cost_gain=cost_gain,
                    cost_increase_any_seed=cost_increase,
                    tradeoff=bool(gain and (not similar or cost_increase)),
                    incremental_benefit=bool((gain and similar and not cost_increase) or (similar and cost_gain)))

    return [dict(setup, primary=i == 0, main=passes("D", setup), plus3=passes("E", setup) if i == 0 else None,
                 versus_linear_main=wins("D", setup), versus_linear_plus3=wins("E", setup) if i == 0 else None)
            for i, setup in enumerate(selected)]


def format_value(value, places=2):
    return "—" if value is None else f"{value:.{places}f}"


def package(destination):
    stats = read(OUT / "complete.json")
    protocol = read(OUT / "protocol.json")
    selection = read(OUT / "selection.json")
    windows = read(OUT / "event_windows.json")["windows"]
    rows = read(OUT / "metrics.json")
    manifest = read(OUT / "run_manifest.json")
    gates = csv_rows(OUT / "gates.csv")
    event_rows = csv_rows(OUT / "events.csv")
    by_id = {r["id"]: r for r in rows}
    assert len(by_id) == len(manifest) == stats["total"] == stats["new"] + stats["reused"]
    assert len(event_rows) == len(by_id) * len(windows)
    compared = comparisons(rows, event_rows, selection, protocol)
    decisions = assessment(selection["selected"], gates, compared, protocol)
    main = decisions[0]
    main_pass = sum(not failures for failures in main["main"].values())
    delay_pass = sum(not failures for failures in main["plus3"].values())
    n = len(protocol["holdout_seeds"])
    passed = [d for d in decisions if all(not f for f in d["main"].values())]
    if passed and delay_pass == n:
        kind = "A" if any(d["versus_linear_main"]["incremental_benefit"] for d in passed) else "B"
    elif passed or any(failures and all(f in ("全部事件峰值/IAE",) for f in failures)
                       for decision in decisions for failures in decision["main"].values()):
        kind = "C"
    else:
        kind = "D"

    summary_dir, trace_dir, reproduce_dir = [PACKAGE / p for p in ("02_汇总", "03_关键轨迹", "04_复现")]
    for directory in (summary_dir, trace_dir, reproduce_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for name in ("metrics.csv", "events.csv", "gates.csv", "filter_response.csv", "selection.json",
                 "event_windows.json", "protocol.json", "identity.json", "run_manifest.json",
                 "old_representatives.json", "complete.json"):
        shutil.copy2(OUT / name, summary_dir / name)
    table(summary_dir / "逐种子与事件配对.csv", compared)
    failures = [dict(id=r["id"], status=r["status"], valid_steps=r["valid_steps"],
                     note=r["note"], stop_s=r["stop_s"], guard=manifest[r["id"]]["guard"])
                for r in rows if not r["completed"]]
    write(summary_dir / "保护失败汇总.json", dict(count=len(failures), failures=failures))
    write(summary_dir / "留出与延迟判定.json", dict(conclusion=kind, decisions=decisions))
    shutil.copy2(ROOT / "source_receipt.json", summary_dir)
    for name in ("strong.py", "review.py", "reproduce_strong.ps1", "run_campaign.py", "filter_sensing.py",
                 "guarded_run.py", "offline.py", "setup_case.py", "protocol.json", "复现说明.md",
                 "force_interface_fix.patch"):
        shutil.copy2(ROOT / name, reproduce_dir)
    shutil.copy2(OUT / "command_channel.patch", reproduce_dir)
    a, b = [(p / "filter_sensing.py").read_text(encoding="utf-8").splitlines(keepends=True) for p in (PREVIOUS, ROOT)]
    (reproduce_dir / "filter_order.patch").write_text("".join(difflib.unified_diff(
        a, b, fromfile="a/filter_sensing.py", tofile="b/filter_sensing.py")), encoding="utf-8")

    seed = protocol["holdout_seeds"][0]
    filt = main["filter"]
    representative = {key: match(rows, "D", key, seed, filt if key != "N" else None) for key in ("N", "L", "S")}
    representative["H"] = match(rows, "history", "S", seed)
    representative["E"] = match(rows, "E", "S", seed, filt)
    traces = {}
    trace_index = []
    for key, row in representative.items():
        info = manifest[row["id"]]
        assert not info["cached"]
        folder = OUT / "runs" / row["id"]
        path = trace_dir / (row["id"] + ".npz")
        original = folder / "trace.npz"
        assert digest(original) == info["trace_sha256"]
        with np.load(original, allow_pickle=False) as z:
            selected_fields = {field: z[field] for field in TRACE_FIELDS}
        np.savez_compressed(path, **selected_fields)
        shutil.copy2(folder / "case.json", reproduce_dir / (row["id"] + ".json"))
        with np.load(path, allow_pickle=False) as z:
            for field, original_values in selected_fields.items():
                np.testing.assert_array_equal(z[field], original_values)
            traces[key] = {field: z[field] for field in
                           ("t", "physical_e_um", "tau_realized_increment", "physical_v", "enc_v",
                            "enc_vf", "sample_v_m_s", "updated")}
            assert len(z["t"]) == row["valid_steps"] and z["actual_micro_q"].shape == (row["valid_steps"], 6)
            if row["completed"]:
                assert abs(np.sqrt(np.mean(np.square(z["physical_e_um"]))) - row["full_rms_um"]) < 1e-10
        trace_index.append(dict(id=row["id"], file=path.name, sha256=digest(path),
                                source_trace_sha256=info["trace_sha256"], unchanged_fields=list(TRACE_FIELDS),
                                condition=LABELS.get(key, "主 SFC +3 ms"), seed=seed))
    write(trace_dir / "索引.json", trace_index)

    first_name = f"{filt[0]} 阶 {filt[1]} Hz"
    gains = f'γ={main["sfc_gamma"]:g}，线性 B0×{main["linear_multiplier"]:g}'
    descriptive = {
        "A": "找到基础版，SFC 相对同滤波线性出现可重复收益",
        "B": "找到基础版，SFC 相对同滤波线性的额外收益尚不清楚或存在出力取舍",
        "C": "普通段改善，事件或额外执行延迟条件未全部通过",
        "D": "有限范围内未找到合适设置",
    }[kind]
    benefit = main["versus_linear_main"]
    delta_rows = [r for r in compared if r["stage"] == "D" and r["reference_law"] == "L"
                  and [r["order"], r["cutoff_hz"]] == filt]
    effect_names = {"peak_um": "峰值", "iae_um_s": "IAE"}
    events_label = {w["event"]: w["label"] for w in windows}
    repeat_gain = []
    for key in benefit["repeat_effect_gain"]:
        if "/" in key:
            event, metric = key.split("/")
            repeat_gain.append(f"{events_label[event]} {effect_names[metric]}")
        else:
            repeat_gain.append(ROW_METRICS[key])
    def ratio_text(metric, event=""):
        values = [r["ratio"] for r in delta_rows if r["metric"] == metric and r["event"] == event and r["ratio"] is not None]
        if not values:
            return "N/A"
        return f"{(min(values)-1)*100:+.1f}% 至 {(max(values)-1)*100:+.1f}%"

    residuals = [r for r in compared if r["stage"] == "D" and r["reference_law"] == "N"
                 and [r["order"], r["cutoff_hz"]] == filt and r["metric"] == "ordinary_y_band_60_500_um"
                 and r["sfc_value"] is not None and r["reference_value"] is not None]
    residual_note = ""
    if residuals:
        a, b = [[r[field] for r in residuals] for field in ("sfc_value", "reference_value")]
        residual_note = (f"主 SFC 普通段 60–500 Hz 残余为 {min(a):.4f}–{max(a):.4f} μm，"
                         f"匹配 N 为 {min(b):.4f}–{max(b):.4f} μm，相对同滤波线性 {ratio_text('ordinary_y_band_60_500_um')}。")
    msg = [
        f"结论 {kind}。{descriptive}。",
        f"本轮新增闭环 {stats['new']} 条，复用 {stats['reused']} 条，共 {stats['total']} 条。保护或有效前缀停止 {len(failures)} 条。",
        f"主设置为纯低通 {first_name}，{gains}。主时序 8 ms 整条六轴命令保持，额外执行延迟 0 ms。",
        f"主 SFC 基础门槛通过 {main_pass}/{n} 个留出种子，额外执行 +3 ms 通过 {delay_pass}/{n}。",
        f"主 SFC 相对同滤波线性，全程 RMS {ratio_text('full_rms_um')}，普通段 RMS {ratio_text('ordinary_rms_um')}，"
        f"普通段附加力矩 RMS {ratio_text('ordinary_tau_rms_Nm')}，力矩峰值 {ratio_text('ordinary_tau_peak_Nm')}。",
        f"主 SFC 相对同滤波线性的直线换向 IAE {ratio_text('iae_um_s', 'turn')}。",
        "三种子重复达到 5% 改善的主指标为" + ("、".join(repeat_gain) if repeat_gain else "无") + "。",
        "重复改善项伴随其他位移指标或出力代价的变化，归类为性能与出力之间的取舍。" if benefit["tradeoff"] else
        "SFC 额外收益的逐项判定见留出与延迟判定表。",
        residual_note,
        "验证范围是固定轨迹、冻结外扰和测量扰动种子。真实转角、其他轨迹与实机高频振动尚未验证。",
        "解压后在浏览器打开 01_结果图解.html。五组图使用第一个留出种子，逐种子及逐事件数值见 02_汇总。",
    ]
    if not selection["ran_C"]:
        msg.append("一阶 SFC 已有校准通过设置，按预案跳过二阶矩阵。")
    for check in selection["boundary"]:
        if check["x8_tested"]:
            x8 = [r for r in selection["ranking"] if r["filter"] == check["filter"]
                  and r["law"] == "L" and r["gain"] == 8][0]
            msg.append(f'{check["filter"][1]} Hz 线性 ×8 边界复核共通过 {x8["gates_passed"]}/10 个种子门槛，'
                       "完整排序见 selection.json。")
    for d in decisions:
        scopes = [("主时序", d["main"])] + ([("+3 ms", d["plus3"])] if d["plus3"] else [])
        for label, result in scopes:
            failed = [f'{s}（{"、".join(f)}）' for s, f in result.items() if f]
            msg.append(f'{d["filter"][0]} 阶 {d["filter"][1]} Hz，{label}，{sum(not f for f in result.values())}/{n} 通过。'
                       + ("未通过种子与门槛为 " + "，".join(failed) + "。" if failed else ""))
    for held_seed, failures_at_delay in main["plus3"].items():
        if GATE_NAMES[2] in failures_at_delay:
            sfc = match(rows, "E", "S", held_seed, filt)
            base = match(rows, "E", "N", held_seed)
            key = "ordinary_y_band_15_60_um"
            threshold = base[key] + protocol["h_margin_um"]
            msg.append(f"+3 ms 种子 {held_seed} 的 H 为 {sfc[key]:.5f} μm，门槛 {threshold:.5f} μm，"
                       f"超出 {sfc[key]-threshold:.5f} μm。")
    (PACKAGE / "00_先看结论.txt").write_text("\n".join(msg) + "\n", encoding="utf-8-sig")

    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "svg.fonttype": "path", "font.size": 10.5})
    body = "".join(f"<p>{escape(line)}</p>" for line in msg[1:] if line and "解压后" not in line)
    body += '<h2>留出与延迟逐种子</h2>'
    audit_rows = []
    for g in gates:
        if g["id"].split("_")[0] not in ("D", "E"):
            continue
        r = by_id[g["id"]]
        audit_rows.append([r["execution_delay_ms"], r["seed"], f'{r.get("filter_order",1)} 阶 {r["fc"]} Hz',
                           "SFC" if r["law"] == "S" else "线性",
                           "通过" if g["passed"] == "True" else "、".join(
                               label for i, label in enumerate(GATE_NAMES, 1) if g[f"gate{i}"] != "True"),
                           format_value(r.get("full_rms_um")), format_value(r.get("ordinary_rms_um")),
                           format_value(r.get("ordinary_y_band_15_60_um"), 3),
                           format_value(r.get("ordinary_y_band_60_500_um"), 4),
                           format_value(r.get("ordinary_tau_rms_Nm"), 4)])
    body += html_table(["额外延迟 ms", "种子", "滤波", "控制", "门槛", "全程 RMS μm", "普通段 RMS μm",
                        "H μm", "60–500 Hz μm", "附加力矩 RMS N·m"], audit_rows)
    body += '<h2>主代表原始轨迹</h2>' + "".join(plots(traces, windows, seed, selection["ranking"], by_id, first_name))
    body += '<h2>主设置逐事件对照</h2>'
    event_lookup = {(r["id"], r["event"]): r for r in event_rows}
    event_display = []
    for stage in ("D", "E"):
        for held_seed in protocol["holdout_seeds"]:
            for w in windows:
                for law in ("N", "L", "S"):
                    row = match(rows, stage, law, held_seed, filt if law != "N" else None)
                    event = event_lookup[(row["id"], w["event"])]
                    fmt = lambda k: format_value(float(event[k]) if event[k] else None)
                    event_display.append([row["execution_delay_ms"], held_seed, w["label"], LABELS[law],
                                          fmt("duration_s"), fmt("peak_um"), fmt("iae_um_s"), fmt("tau_peak_Nm")])
    body += html_table(["额外延迟 ms", "种子", "事件", "控制", "时长 s", "峰值 μm", "IAE μm·s", "力矩峰值 N·m"],
                       event_display)
    body += """<h2>评分口径</h2><p>全程和普通段时域 RMS 保留均值。普通段采用 [2.226,4.381) 与 [5.943,8.668) s。
    频带使用 Hann Welch 估计，2 s 分段、1 s 重叠并去除各段常数，普通窗按段数加权，短窗频带为估计值。
    H 是 15–60 Hz 带内物理位移 RMS。60–500 Hz 残余单列。事件恢复时间为 N/A。</p>
    <p>力矩是同一物理状态下实际生效命令与保持基线命令的模型反事实差，单位 N·m。变化率对六轴力矩向量作 1 ms 差分后取范数。
    +3 ms 位移门槛以同延迟 N 配对，力矩仍与同种子 0 ms 历史 SFC 比较。</p>
    <p>幅值、相位与群延迟按 125 Hz 数字滤波器单独计算，采样差分、观测发布和整条命令保持的时序留在关键轨迹。</p>"""
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1"><title>强滤波 SFC 保守基线</title>
    <style>body{{font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif;margin:auto;padding:2rem;max-width:1200px;color:#253747;background:#f4f7f9}}
    h1,h2{{line-height:1.25;color:#152a3c}}h2{{margin-top:2rem}}figure{{margin:1.2rem 0;padding:1rem;background:white;border:1px solid #dce5eb;border-radius:8px}}
    svg{{width:100%;height:auto;display:block}}figcaption{{border-top:1px solid #e3eaef;padding:.65rem .3rem 0;color:#394b59}}
    table{{border-collapse:collapse;width:100%;background:white}}th,td{{border:1px solid #d8e2e9;padding:.45rem .58rem;text-align:left}}
    th{{background:#e2edf4}}.scroll{{overflow:auto}}.badge{{color:#4a6375;font-size:14px}}</style></head>
    <body><div class="badge">离线 UR10e · 8 ms 控制 · 2026-11-10</div>
    <h1>{escape(msg[0])}</h1>{body}</body></html>"""
    assert html.count("<svg ") == 5 and html.count("<figcaption>") == 5
    (PACKAGE / "01_结果图解.html").write_text(html, encoding="utf-8")
    files = sorted(p for p in PACKAGE.rglob("*") if p.is_file() and p.name != "文件SHA256.json")
    hashes = {p.relative_to(PACKAGE).as_posix(): digest(p) for p in files}
    write(summary_dir / "文件SHA256.json", hashes)
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=6) as z:
        for p in files + [summary_dir / "文件SHA256.json"]:
            z.write(p, p.relative_to(PACKAGE).as_posix())
    with ZipFile(destination) as z:
        assert z.testzip() is None and len(z.namelist()) == len(files) + 1
        assert all(digest(PACKAGE / p) == digest_from_zip(z, p) == h for p, h in hashes.items())
    assert destination.stat().st_size <= 30_000_000
    print("\n".join(msg))
    print(destination, destination.stat().st_size)


def digest_from_zip(z, name):
    with z.open(name) as file:
        return file_digest(file, "sha256").hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=ROOT.parents[1] / ZIP_NAME)
    package(parser.parse_args().zip.resolve())
