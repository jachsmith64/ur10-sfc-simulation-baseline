"""Build the offline G review from independent source windows and measured runs."""
from base64 import b64encode
import argparse
import csv
from hashlib import sha256
import html
from io import BytesIO
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from windows import TRIALS, PLATFORMS, EVENT_TO_TRANSITION


ROOT=Path(__file__).resolve().parent
LABELS={"full":"整段","pre_static":"启动前静止","start":"启动及余振","start_transition":"启动后参考过渡",
        "outbound":"稳定去程","turn":"换向及余振","turn_transition":"换向后参考过渡",
        "return_straight":"稳定回程","stop":"停止及响应","stop_transition":"停止后参考过渡",
        "post_static":"停止后静止"}
AXIAL={"start10":"启动 10%","start50":"启动 50%","start90":"启动 90%","decel90":"换向减速始 90%",
       "zero":"换向速度过零","turn_xmax":"X 最大位置","return50":"反向 50%","return90":"反向 90%",
       "stop90":"停止减速始 90%","stop10":"停止 10%"}
EVENT="#f4e0d2"
TRANS="#eae8f2"
STABLE="#e3f0e8"
A_COLOR="#b15726"
N_COLOR="#234e88"
REF_COLOR="#267455"
plt.rcParams.update({"font.size":8.5,"axes.spines.top":False,"axes.spines.right":False})


def js(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def npz(path):
    with np.load(path,allow_pickle=False) as a:
        return {k:a[k] for k in a.files}


def blob(fig,alt):
    out=BytesIO()
    fig.savefig(out,format="png",dpi=116,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{b64encode(out.getvalue()).decode()}">'


def table(header,rows):
    head="".join(f"<th>{html.escape(str(s))}</th>" for s in header)
    body="".join("<tr>"+"".join(f"<td>{html.escape(str(s))}</td>" for s in row)+"</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def number(x,dec=2):
    return "—" if x is None else f"{x:.{dec}f}"


def bands(ax,rec,labels=False):
    chosen={"start":"Start","turn":"Reversal","stop":"Stop",
            "outbound":"Steady out","return_straight":"Steady back"}
    for key,(lo,hi) in rec["windows_s"].items():
        color=EVENT if key in EVENT_TO_TRANSITION else TRANS if key.endswith("transition") else STABLE
        ax.axvspan(lo,hi,facecolor=color,edgecolor="none",zorder=-10)
        if labels and key in chosen:
            ax.text((lo+hi)/2,1.02,chosen[key],ha="center",va="bottom",
                    transform=ax.get_xaxis_transform(),fontsize=7)
    ax.grid(alpha=.22,zorder=-9)


def summary_plot(records):
    fig, axs=plt.subplots(3,2,figsize=(13,10),layout="constrained")
    for (trial,s,r,meta,a,z),row in zip(records,axs,strict=True):
        ft=a["frame_host_time_s"]; ht=a["host_time_s"]; m=a["frame_valid"]; t=z["t"]
        zero=meta["raw_visual_zero_um"]
        ax=row[0]
        bands(ax,r,True)
        ax.plot(ft[m],a["A_raw_um"][m],".-",lw=.65,ms=1.7,color=A_COLOR,label="A native visual Y")
        ax.plot(t,z["direction_reference_um"]+zero,color=REF_COLOR,lw=1.3,label="Direction reference")
        ax.plot(t,z["slow_trend_um"],":",color="#8c6099",lw=1,label="Separate drift [0]")
        for name,(lo,hi) in r["original_stage_labels_s"].items():
            ax.axvline(lo,color="#5f6467",lw=.48,alpha=.7,linestyle="--")
        ax.set_title(trial+" · source windows",loc="left",pad=17,fontweight="bold")
        ax.set_ylabel("Visual Y [µm]")
        ax.legend(loc="upper right",fontsize=7)
        ax=row[1]
        bands(ax,r)
        ax.plot(ht,1e3*a["actual_tcp_speed"][:,0],".",ms=1.5,color=A_COLOR,label="A cabinet X")
        ax.plot(t,z["physical_vx_m_s"]*1e3,color=N_COLOR,lw=1.1,label="New N X")
        ax.plot(t,np.gradient(z["nominal_x_m"],t)*1e3,color="#658271",lw=.75,alpha=.85,label="RTDE target")
        for lo,hi in r["cabinet_gaps_s"]:
            ax.axvspan(lo,hi,facecolor="none",edgecolor="#9a9a9a",lw=.5,hatch="///")
        ax.set_title(trial+" · axial speed",loc="left",fontweight="bold")
        ax.set_ylabel("Vx [mm/s]")
        ax.legend(loc="lower left",fontsize=7)
        for ax in row:
            ax.set_xlabel("Original host time [s]")
            ax.set_xlim(0,ft[-1])
    return blob(fig,"三片原始 A 和窗口总览，右栏为未平移的 X 速度")


def details(rec):
    trial,status,r,meta,a,z=rec
    ft=a["frame_host_time_s"]; ht=a["host_time_s"]; valid=a["frame_valid"]; t=z["t"]
    zero=meta["raw_visual_zero_um"]
    aligned=np.interp(ft,t,z["scoring_reference_um"])
    fig,axs=plt.subplots(3,3,figsize=(13,9),layout="constrained")
    for col,(event,name) in enumerate((("start","Startup"),("turn","Reversal"),("stop","Stop"))):
        lo=r["windows_s"][event][0]-.10
        hi=r["windows_s"][EVENT_TO_TRANSITION[event]][1]+.10
        keep=(ft>=lo)&(ft<=hi)&valid
        robot=(ht>=lo)&(ht<=hi)
        sim=(t>=lo)&(t<=hi)
        for ax in axs[:,col]:
            bands(ax,r)
            ax.set_xlim(lo,hi)
        ax=axs[0,col]
        ax.set_title(name+" · Y with locked reference",loc="left",fontweight="bold")
        ax.plot(ft[keep],a["A_raw_um"][keep],".",color=A_COLOR,ms=2.6,label="A raw")
        ax.plot(t[sim],z["physical_y_um"][sim]+zero,color=N_COLOR,lw=1,label="N physical, fixed origin")
        ax.plot(ft[keep],a["A_centered_um"][keep]-aligned[keep],".",ms=1.5,color="#d28e4f",label="A minus reference")
        ax.plot(t[sim],z["physical_y_um"][sim]-z["scoring_reference_um"][sim],color="#416ea5",lw=.8,
                ls="--",label="N minus reference")
        ax.plot(t[sim],z["direction_reference_um"][sim]+zero,color=REF_COLOR,lw=1,label="Reference")
        ax.set_ylabel("Y [µm]")
        ax.legend(loc="best",fontsize=6.3)
        ax=axs[1,col]
        x0=a["actual_tcp_pose"][0,0]
        ax.plot(ht[robot],(a["actual_tcp_pose"][robot,0]-x0)*1000,".",ms=2,color=A_COLOR,label="A cabinet X")
        ax.plot(t[sim],(z["nominal_x_m"][sim]-x0)*1000,"--",lw=.8,color=REF_COLOR,label="RTDE target")
        ax.plot(t[sim],(z["physical_xyz_m"][sim,0]-x0)*1000,color=N_COLOR,lw=1,label="N physical X")
        ax.set_ylabel("X [mm]")
        ax.legend(loc="best",fontsize=6.5)
        ax=axs[2,col]
        ax.plot(ht[robot],1000*a["actual_tcp_speed"][robot,0],".",ms=2.2,color=A_COLOR,label="A cabinet Vx")
        ax.plot(t[sim],z["physical_vx_m_s"][sim]*1000,color=N_COLOR,lw=1.1,label="N fitted Vx")
        ax.plot(t[sim],z["initial_measured_vx_m_s"][sim]*1000,color="#6d7aa2",lw=.85,
                ls="--",label="N before Y fit")
        ax.plot(t[sim],np.gradient(z["nominal_x_m"],t)[sim]*1000,color=REF_COLOR,lw=.65,label="Target derivative")
        t_rec=status["timing"]["best_N"]
        wanted={"start":["start10","start50","start90"],
                "turn":["decel90","zero","return90"],"stop":["stop90","stop10"]}[event]
        for key in wanted:
            for series,color in (("source",A_COLOR),("candidate",N_COLOR)):
                stamp=t_rec[series][key]["time_s"]
                if stamp is not None:
                    ax.axvline(stamp,color=color,lw=.63,ls=":",alpha=.8)
        for a0,b0 in r["cabinet_gaps_s"]:
            ax.axvspan(a0,b0,facecolor="none",edgecolor="#969696",lw=.4,hatch="///")
        ax.set_ylabel("Vx [mm/s]")
        ax.set_xlabel("Original host time [s]")
        ax.legend(loc="best",fontsize=6.5)
    return blob(fig,trial+" 的启动、换向、停止事件局部和时序")


def old_new_plot(records):
    fig,axs=plt.subplots(3,2,figsize=(13,9),layout="constrained")
    for (trial,status,r,meta,a,z),row in zip(records,axs,strict=True):
        ft=a["frame_host_time_s"]; t=z["t"]; valid=a["frame_valid"]
        ref=z["scoring_reference_um"]
        ax=row[0]
        bands(ax,r)
        ax.plot(ft[valid],(a["A_centered_um"]-np.interp(ft,t,ref))[valid],".",ms=1.9,color=A_COLOR,label="A minus locked reference")
        ax.plot(t,z["old_physical_y_um"]-ref,lw=.75,color="#8b8395",label="Prior N physical")
        ax.plot(t,z["physical_y_um"]-ref,lw=.95,color=N_COLOR,label="New N physical")
        ax.plot(t,z["old_remaining_um"],lw=.65,color="#6c86ad",alpha=.65,label="Prior additive observation term")
        ax.set_title(trial+" · physical output comparison",loc="left",fontweight="bold")
        ax.set_ylabel("Y [µm]")
        ax.legend(loc="best",fontsize=7)
        ax=row[1]
        bands(ax,r)
        ax.plot(t,z["common_command_y_m"]*1e6,lw=1.15,color=REF_COLOR,label="Commanded equivalent Y")
        ax.plot(t,z["actual_common_effect_y_um"],lw=.9,color=N_COLOR,label="Paired physical Y change")
        a2=ax.twinx()
        a2.plot(t,z["reference_rate_um_s"],color="#956b38",lw=.75,label="Feedback reference rate")
        a2.set_ylabel("Reference rate [µm/s]")
        h,l=ax.get_legend_handles_labels(); hh,ll=a2.get_legend_handles_labels()
        ax.legend(h+hh,l+ll,loc="best",fontsize=6.8)
        ax.set_title(trial+" · separate command correction",loc="left",fontweight="bold")
        ax.set_ylabel("Y [µm]")
        for ax in row:
            ax.set_xlabel("Original host time [s]")
    return blob(fig,"关闭 remaining 前后物理输出及独立公共指令纠偏")


def report(records,out):
    start=summary_plot(records)
    compare=old_new_plot(records)
    sections=[]
    for item in records:
        trial,st,rec,meta,a,z=item
        timing=st["timing"]["best_N"]
        phase_table=table(["区段","源时刻 s","有效帧","A 中位数 μm","原始 FIT","处理后 FIT",
                           "原始 RMSE μm","处理后 RMSE μm","A / N 峰值 μm（处理后）","峰值时差 ms（处理后）"],
            [(LABELS[k],("全片" if k=="full" else "–".join(number(x,3) for x in rec["windows_s"][k])),
              values["raw"]["samples"],number(rec["source_statistics"].get(k,{}).get("median_um")),
              number(values["raw"].get("fit")),number(values["processed"].get("fit")),
              number(values["raw"].get("rmse_um")),number(values["processed"].get("rmse_um")),
              f'{number(values["processed"].get("peak_A_um"))} / {number(values["processed"].get("peak_N_um"))}',
              number((values["processed"].get("absolute_peak_delay_s") or 0)*1000,1))
             for k,values in st["scores"].items()])
        timing_tbl=table(["事件","A 原时刻 s","N 原时刻 s","N 晚于 A ms","纯名义网格晚于 A ms"],
            [(label,number(timing["source"][key]["time_s"],4),number(timing["candidate"][key]["time_s"],4),
              number(timing["delay_ms"][key],1),number(st["timing"]["reconstructed_nominal"]["delay_ms"][key],1))
             for key,label in AXIAL.items()])
        duration_tbl=table(["过程","A 时长 ms","N 时长 ms"],[
            (label,number(timing["durations"][key]["A_ms"],1),number(timing["durations"][key]["candidate_ms"],1))
             for key,label in (("rise_10_90","启动 10→90%"),("reversal_90_90","换向 +90→−90%"),("stop_90_10","停止 90→10%"))])
        detail=details(item)
        text_sources=" ".join(rec["rationale"])
        bounds="，".join(f'{k} {v[0]:.3f}–{v[1]:.3f} s' for k,v in enumerate(rec["cabinet_gaps_s"],1))
        sections.append(f"<section><h2>{trial}</h2><p>{text_sources}</p>"
            f"<p>长柜端间隔 {bounds}。停止阈值若落在长空档，表中保留空值，完整包记录原区间。</p>{detail}"
            f"<h3>窗口和物理重建</h3>{phase_table}<h3>未平移的 X 时序和过程时长</h3>{timing_tbl}{duration_tbl}</section>")
    overview=table(["试次","最低物理 FIT","启动半速晚 ms","换向位置峰晚 ms","回程半速晚 ms","结论"],
        [(trial,number(st["best_min_fit"]),*[number(st["timing"]["best_N"]["delay_ms"][k],1)
            for k in ("start50","turn_xmax","return50")],"沿程时序未通过") for trial,st,*_ in records])
    doc=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>G 组窗口、基线和时序离线审查</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{{margin:32px auto;max-width:1220px;padding:0 22px;color:#192733;font:18px/1.58 SimSun,"Times New Roman",serif}}
h1,h2,h3{{font-family:SimHei,Arial,sans-serif;line-height:1.2}} h1{{margin-bottom:.5em}} section{{margin:44px 0}}
img{{width:100%;height:auto}} table{{width:100%;border-collapse:collapse;font-size:15px;margin:12px 0 24px}}
th,td{{padding:7px 9px;border-bottom:1px solid #d8e3ea;text-align:right}} th{{background:#ecf2f6}}
td:first-child,th:first-child{{text-align:left}} .note{{border-left:4px solid #285287;padding:8px 14px;background:#f2f6fa}}
</style></head><body><main>
<h1>G 组窗口、基线和 X 时序</h1>
<p class="note">三片从原始 A 冻结窗口，关闭旧 C 观测剩余量后完成离线 N 物理重建。保留真实横向事件及独立过渡，沿 X 的事件时序仍未通过，三片 N 均为诊断性重建。</p>
{overview}
<h2>三片原始 A 与窗口总览</h2>
<p>橙色底为保护事件，紫灰色底为参考过渡，绿色底为稳定区段。细灰虚线是源阶段标签，斜纹是长柜端采样空档。横向时间始终为该试次从首个原始相机帧量起的主机时间。不同试次独立初始化。</p>
{start}
<h2>物理输出和公共关节指令</h2>
<p>左图把同一个评分参考应用到 A、两版物理输出。右图对比施加到关节目标的计划修正、同力有无公共表的物理响应差，以及独立反馈参考变化率。参考变化只发生在紫灰过渡。旧 remaining 只为历史账目显示，新版值与导数均为零。</p>
{compare}
{"".join(sections)}
<h2>采样和评价口径</h2>
<p>相机与编码器分别在各自采样时刻从当前模型位置扣除反馈参考。编码器从已扣参考且含原压力的相邻读数求差商，再作 5 Hz 一阶低通。参考变化率随差分自动计入一次。相机 132 Hz、发布延迟 4 ms，编码器和整组六轴命令更新 8 ms，模型积分 1 ms。采样年龄差留在结果数组，未做预测或重同步。</p>
<p>原始层以 A 原测量零点显示同坐标物理位移。处理层对 A 和物理位移扣相同参考。FIT 原口径为 100 × (1 − RMSE / A 去均值 RMS)。近乎平坦使用 1 μm 分母门槛，低于门槛时表中显示空值。评分只用原始有效视觉帧。方向平台转移和运动瞬态在保护窗内一并保留。</p>
<p>50 ms 门槛沿用启动半速、X 最大点和回程半速三项。10/90% 和速度过零用于形状诊断。速度 RMSE、峰值差、完整窗计数、邻界变化、原始求解与保护日志见 JSON 和 CSV。</p>
</main></body></html>'''
    (out/"G_preprocess_timing_review.html").write_text(doc,encoding="utf-8")


def build(run, prepared, out):
    receipt=js(run/"receipt.json")
    frozen=js(ROOT/"windows_frozen.json")
    assert receipt["frozen_windows_sha256"]==sha256((ROOT/"windows_frozen.json").read_bytes()).hexdigest()
    assert list(receipt["results"])==list(TRIALS)
    public=out/"results"
    public.mkdir(exist_ok=True)
    records=[]
    fit_rows=[]
    timing_rows=[]
    for trial in TRIALS:
        case=run/trial
        final=public/trial
        final.mkdir(exist_ok=True)
        for name in ("attempts.json","status.json","plant_receipt.json","axial_before_fit.json","review.npz","locked.npz"):
            shutil.copy2(case/name,final/name)
        st=js(case/"status.json")
        for replay in sorted(case.glob("*/case.json")):
            folder=final/"run_configs"/replay.parent.name
            folder.mkdir(parents=True,exist_ok=True)
            for f in ("case.json","row.json","checks.json","guard.json"):
                shutil.copy2(replay.parent/f,folder/f)
        inputs=prepared/trial
        record=(trial,st,frozen["trials"][trial],js(inputs/"source.json"),npz(inputs/"A_source.npz"),npz(case/"review.npz"))
        records.append(record)
        for label,window in st["scores"].items():
            for layer,r in window.items():
                fit_rows.append(dict(trial=trial,window=label,layer=layer,fit=r["fit"],rmse_um=r["rmse_um"],samples=r["samples"],
                peak_A_um=r.get("peak_A_um"),peak_N_um=r.get("peak_N_um"),peak_delay_s=r.get("absolute_peak_delay_s")))
        for comparison,data in st["timing"].items():
            if comparison=="code_clock":
                continue
            for k in AXIAL:
                timing_rows.append(dict(trial=trial,comparison=comparison,marker=k,
                      source_s=data["source"][k]["time_s"],candidate_s=data["candidate"][k]["time_s"],delay_ms=data["delay_ms"][k]))
    for name,rows in (("fit_windows.csv",fit_rows),("x_event_timing.csv",timing_rows)):
        with (public/name).open("w",encoding="utf-8-sig",newline="") as target:
            writer=csv.DictWriter(target,fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    shutil.copy2(run/"receipt.json",public/"actual_execution.json")
    shutil.copy2(run/"integration_source"/"command_channel.patch",public/"inherited_command_channel.patch")
    report(records,out)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--prepared",type=Path,default=ROOT.parent/"20261116_A_multispeed/prepared_A")
    p.add_argument("--out",type=Path,default=ROOT)
    a=p.parse_args()
    build(a.run,a.prepared,a.out)
