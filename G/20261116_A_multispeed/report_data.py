"""Produce G review evidence from completed offline receipts and float64 arrays."""
import base64
import csv
from hashlib import sha256
import html
from io import BytesIO
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = {"X_P05_R01":"A_portable_replay", "X_P05_R02":"A_portable_replay", "X_P10_R01":"A_P10_host"}
LABELS = {"full":"整段", "start":"启动", "turn":"换向", "outbound":"稳定去程",
          "return_straight":"稳定回程", "pre_static":"启动前静止", "post_static":"停止后静止"}


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def dump(p, x):
    Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def read_csv(p):
    with Path(p).open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(p, items):
    keys = list(dict.fromkeys(k for row in items for k in row))
    with Path(p).open("w", encoding="utf-8-sig", newline="") as file:
        csv_writer = csv.DictWriter(file, fieldnames=keys)
        csv_writer.writeheader()
        csv_writer.writerows(items)


def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def digest(p):
    return sha256(Path(p).read_bytes()).hexdigest()


def figure_img(fig, label):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=125, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f'<img alt="{html.escape(label)}" src="data:image/png;base64,{b64}">'


def event_times(t,x,v,source):
    at = source["turn_actual_xmax_s"]
    near = (t>=at-.3)&(t<=at+.3)
    half = source["speed_mm_s"]/2
    return dict(start_s=float(t[(t>.75)&(v>half)][0]),
                turn_s=float(t[near][np.argmax(x[near])]),
                return_s=float(t[(t>=at-.15)&(v<-half)][0]))


def save_trial(trial, batch, public):
    folder = HERE/"runs"/batch/trial
    source = read(folder/"source.json")
    source.setdefault("source_kind","original_vision_cabinet_runlog_and_sidecar")
    status = read(folder/"status.json")
    attempts = read(folder/"attempts.json")
    best = status["best_attempt"]
    best_path = folder/("N_common_only" if best==0 else f"N_fit_{best}")
    target = public/trial
    target.mkdir(parents=True, exist_ok=True)
    out = HERE/"prepared_A"/trial
    out.mkdir(parents=True, exist_ok=True)
    a = load(folder/"original_A.npz")
    x = load(best_path/"trace.npz")
    locked = load(folder/"locked_inputs.npz")
    np.savez_compressed(out/"A_source.npz", **a)
    dump(out/"source.json",source)
    compact = ("t","physical_xyz_m","physical_e_um","effective_e_um","physical_v","remaining",
               "enc_v","enc_vf","sample_v_m_s","sample_e_m","compute_cam_age_ms","compute_enc_age_ms",
               "F_requested","Fd","q_nominal","dq_C","q","qdot","final_ctrl","actual_micro_q",
               "tau_realized_increment","updated","qfrc_disturbance","qfrc_applied")
    np.savez_compressed(target/"best_N.npz",**{k:x[k] for k in compact})
    np.savez_compressed(target/"locked_inputs.npz", **locked)
    ts = locked["t"]
    ft = a["frame_host_time_s"]
    y = x["physical_e_um"]
    physical = []
    for step in range(len(attempts)):
        path = folder/("N_common_only" if step==0 else f"N_fit_{step}")
        expected = load(path/"trace.npz")
        physical.append(expected["physical_e_um"])
        dump(target/f"N_{step}_case.json",read(path/"case.json"))
        dump(target/f"N_{step}_status.json",dict(row=read(path/"row.json"), validation=read(path/"validation.json"), guard=read(path/"guard.json")))
    alone = load(folder/"N_without_common"/"trace.npz")
    common = load(folder/"N_common_only"/"trace.npz")
    dump(target/"N_without_common_case.json",read(folder/"N_without_common/case.json"))
    dump(target/"N_without_common_status.json",dict(row=read(folder/"N_without_common/row.json"), validation=read(folder/"N_without_common/validation.json"),
                                                     guard=read(folder/"N_without_common/guard.json")))
    np.savez_compressed(target/"N_components_and_attempts.npz", t=ts,
                        without_common_physical_um=alone["physical_e_um"], common_only_physical_um=common["physical_e_um"],
                        fit_physical_um=np.stack(physical), remaining_um=x["remaining"]*1e6)
    driver = a["actual_tcp_pose"][:,0]
    timing = dict(source=event_times(a["host_time_s"],driver,a["actual_tcp_speed"][:,0]*1000,source),
                  target=event_times(ts,locked["nominal_xyz"][:,0],np.gradient(locked["nominal_xyz"][:,0],ts)*1000,source))
    for series, trace in (("N_without_common",alone),("N_common_only",common),("N_best",x)):
        vx = np.gradient(trace["physical_xyz_m"][:,0],trace["t"])*1000
        timing[series]=event_times(trace["t"],trace["physical_xyz_m"][:,0],vx,source)
    nz = x["t"] < .15
    pre = a["frame_valid"] & (a["frame_phase"]=="pre_motion")
    status.update(timing_detail=timing,
                  A_pre_motion_centered_rms_um=float(np.std(a["A_raw_um"][pre])),
                  remaining_peak_to_peak_um=float(np.ptp(x["remaining"])*1e6),
                  remaining_demeaned_energy_um2=float(np.var(x["remaining"]*1e6)),
                  physical_demeaned_energy_um2=float(np.var(y)),
                  baseline_cold_first150ms_max_abs_um=float(np.max(abs(alone["physical_e_um"][nz]))),
                  best_cold_first150ms_max_abs_um=float(np.max(abs(y[nz]))),
                  provenance=dict(original_rundir=f"runs/{batch}/{trial}",
                                  best_trace_full_sha256=digest(best_path/"trace.npz"),
                                  plant_sha256=digest(folder/"plant.npz")))
    dump(target/"status.json", status)
    dump(target/"attempts.json", attempts)
    return dict(trial=trial,a=a,x=x,locked=locked,source=source,status=status,
                best=attempts[best]["scores"], ts=ts,ft=ft)


def make_figures(records):
    fig, axes = plt.subplots(2,1,figsize=(11,6),sharex=True)
    offset = 0.
    for item in records:
        a,x,ts,ft,locked = (item[k] for k in ("a","x","ts","ft","locked"))
        remaining = x["remaining"]*1e6
        label = item is records[0]
        axes[0].plot(ft+offset, a["A_raw_um"], color="#b15b13", lw=.8,label="A, raw" if label else None)
        axes[0].plot(ts+offset, x["physical_e_um"]+remaining+item["source"]["raw_visual_zero_um"], color="#285296", lw=.8,
                     label="N reconstructed observation" if label else None)
        axes[1].plot(ft+offset,a["A_centered_um"]-np.interp(ft,ts,locked["A_trend_um"]),color="#b15b13",lw=.8,label="A, drift removed" if label else None)
        axes[1].plot(ts+offset,x["physical_e_um"]-locked["A_trend_um"],color="#16765a",lw=.85,
                     label="N physical, same reference" if label else None)
        axes[0].text(offset+.13,.94,item["trial"],fontweight="bold",transform=axes[0].get_xaxis_transform(),va="top")
        for ax in axes:
            ax.axvline(offset,color="#617081",linestyle="--",linewidth=.75)
        offset += ft[-1]+.35
    for ax in axes:
        ax.set_ylabel("Y [µm]")
        ax.grid(alpha=.25)
        ax.legend(loc="lower center",ncol=2,fontsize=8)
    axes[-1].set_xlabel("Overview time [s], dashed lines = independent initial states")
    overview=figure_img(fig,"全段拼合原始观测与物理输出")
    local = []
    for item in records:
        a,x,ts,ft,locked = (item[k] for k in ("a","x","ts","ft","locked"))
        fig,axes=plt.subplots(2,2,figsize=(11,6))
        for col,name in enumerate(("start","turn")):
            lo,hi=item["source"]["windows_s"][name]
            f=(ft>=lo)&(ft<=hi)&a["frame_valid"]
            g=(ts>=lo)&(ts<=hi)
            h=(a["host_time_s"]>=lo)&(a["host_time_s"]<=hi)
            trend=locked["A_trend_um"]
            y=x["physical_e_um"]
            ax=axes[0,col]
            ax.plot(ft[f],a["A_centered_um"][f]-np.interp(ft[f],ts,trend),".-",ms=2,color="#b15b13",label="A frame")
            ax.plot(ts[g],(y+x["remaining"]*1e6-trend)[g],lw=1,color="#285296",label="N observation")
            ax.plot(ts[g],(y-trend)[g],lw=.9,color="#16765a",label="N physical")
            ax.set_ylabel("Y after common reference [µm]")
            ax.set_title("Start" if col==0 else "Reversal")
            ax.legend(loc="best",fontsize=8)
            ax=axes[1,col]
            r=a["actual_tcp_pose"][0,0]
            ax.plot(a["host_time_s"][h],(a["actual_tcp_pose"][h,0]-r)*1000,".",ms=2,color="#b15b13",label="A cabinet X")
            ax.plot(ts[g],(locked["nominal_xyz"][g,0]-r)*1000,"--",lw=1,color="#617081",label="TCP target X")
            ax.plot(ts[g],(x["physical_xyz_m"][g,0]-r)*1000,lw=1,color="#285296",label="N physical X")
            ax.set_ylabel("X relative to source start [mm]")
            ax.set_xlabel("Original host time [s]")
            ax.legend(loc="best",fontsize=8)
        for ax in axes.flat:
            ax.grid(alpha=.25)
        fig.tight_layout()
        local.append(figure_img(fig,item["trial"]+" 启动与换向局部"))
    return overview,local


def report(records):
    overview, locals_ = make_figures(records)
    fmt=lambda x: "不适用" if x is None else f"{x:.2f}"
    sections=[]
    for item,img in zip(records,locals_,strict=True):
        txt=[]
        index = item["status"]["best_attempt"]
        delay=item["status"]["source_timing"]
        for label, data in item["best"].items():
            values = [LABELS[label], data["raw"]["samples"], fmt(data["raw"]["fit"]), fmt(data["oscillation"]["fit"]),
                      fmt(data["physical_oscillation"]["fit"]), fmt(data["raw"]["rmse_um"]), fmt(data["physical_oscillation"]["rmse_um"])]
            txt.append("<tr>"+"".join(f"<td>{x}</td>" for x in values)+"</tr>")
        stat=f'最佳 N 第 {index} 次反推。X 启动、换向、回程相对 A 晚 {1000*delay["start_speed_s"]:.1f}、{1000*delay["turn_xmax_s"]:.1f}、{1000*delay["return_speed_s"]:.1f} ms。'
        section=f'<section><h2>{item["trial"]}</h2><p>{stat}</p>{img}<table><thead><tr><th>窗口</th><th>有效帧</th><th>原始层 FIT</th><th>抖动层 FIT</th><th>物理抖动 FIT</th><th>观测 RMSE μm</th><th>物理抖动 RMSE μm</th></tr></thead><tbody>{"".join(txt)}</tbody></table></section>'
        sections.append(section)
    status_rows = "".join(f"<tr><td>{r['trial']}</td><td>已运行</td>"+"<td>门禁未通过</td>"*6+"</tr>" for r in records)
    text=f'''<!DOCTYPE html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>G 组 A 重建验收</title><style>
body{{max-width:1150px;margin:36px auto;padding:0 18px;font:18px/1.62 SimSun,"Times New Roman",serif;color:#16232c}}
h1,h2{{font-family:SimHei,Arial,sans-serif;line-height:1.25}} section{{margin:40px 0}}
img{{width:100%;height:auto}} table{{border-collapse:collapse;width:100%;font-size:15px}} td,th{{padding:8px;border-bottom:1px solid #d3dce2;text-align:right}} td:first-child,th:first-child{{text-align:left}}
thead{{background:#edf2f5}} .highlight{{padding:10px 16px;background:#edf2f5;border-left:4px solid #285296}}
</style><main><h1>G 组 A 实测重建验收</h1>
<p class="highlight">5 与 10 mm/s 的三段 A 数据完成 N 重建。横向观测拟合与物理响应分列，沿 X 的启动和换向时差触发验收门禁。线性与 SFC 单元保留未运行状态。</p>
<h2>独立试次总览</h2><p>横向单位 μm。竖线表示模型与采样器重置，段间留 0.35 s。原始图通过固定零点还原测量坐标。物理图和 A 去同一条参考趋势。A 有效帧按原主机时间取点。</p>{overview}
{"".join(sections)}
<h2>七项实验状态</h2><table><thead><tr><th>试次</th><th>N</th><th>L1 5 Hz</th><th>L4 5 Hz</th><th>SFC 5 Hz</th><th>L1 20 Hz</th><th>L4 20 Hz</th><th>SFC 20 Hz</th></tr></thead><tbody>{status_rows}</tbody></table>
<h2>复核口径</h2><p>FIT = 100 × (1 − RMSE / 扣除均值后的 A RMS)。阈值 80，目标 90。原始层使用原零点，抖动层对 A 与模型使用同一固定四平台慢趋势。图线仅连接有效帧，评分没有新增实测点。频带诊断另以 132 Hz 统一插值、Hann 窗计算，评分仍使用原始时间戳。完整事件和频带数值见每片 attempts.json。静止参考为 ST05 历史约 0.9 μm，判近乎平坦取 1.0 μm。</p>
<p>原件来源与哈希见 prepared_A 下 source.json。锁定的扰动力、误差背景、慢纠偏、时序和运行保护见 <code>results/</code>。紧凑数组保留原数据类型，可用 <code>verify_review.py</code> 复算评分。</p></main></html>'''
    (HERE/"05_重建验收与控制结果.html").write_text(text,encoding="utf-8")


def main():
    public = HERE/"results"
    records=[save_trial(trial,batch,public) for trial,batch in RUNS.items()]
    evidence = HERE/"evidence"
    manifest = {str(p.relative_to(HERE).as_posix()):dict(bytes=p.stat().st_size,sha256=digest(p))
                for dirpath in (HERE/"v02_portable",HERE/"prepared_A") for p in sorted(dirpath.rglob("*")) if p.is_file()}
    dump(evidence/"ready_inputs_sha256.json",manifest)
    write_csv(public/"FIT事件汇总.csv",[
        dict(trial=r["trial"], window=k, **{layer:d[layer]["fit"] for layer in ("raw","oscillation","physical_oscillation")},
             RMSE_obs_um=d["raw"]["rmse_um"], samples=d["raw"]["samples"], requires_gate=k in ("full","start","turn"))
        for r in records for k,d in r["best"].items()])
    report(records)


if __name__=="__main__":
    main()
