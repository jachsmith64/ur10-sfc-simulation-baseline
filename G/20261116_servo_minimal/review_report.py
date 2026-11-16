"""Create the reduced, independently reviewable six-case evidence and report."""
import argparse
from base64 import b64encode
import csv
from hashlib import sha256
import html
from io import BytesIO
import json
from pathlib import Path
from shutil import copy2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
PRECEDING=ROOT.parent/"20261116_preprocess_timing"
PREPARED=ROOT.parent/"20261116_A_multispeed/prepared_A"
TRIALS=("X_P05_R01","X_P05_R02","X_P10_R01")
ARMS=("original","nominal_velocity")
AXIS={"start50":"启动半速","turn_xmax":"最大 X","return50":"回程半速"}
EVENT={"start":"启动","turn":"换向","stop":"停止","outbound":"稳定去程","return_straight":"稳定回程"}
COLORS={"A":"#a35228","original":"#748192","nominal_velocity":"#154a85","nominal":"#408066"}
plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False})


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def arrays(path):
    with np.load(path,allow_pickle=False) as a:
        return {k:a[k] for k in a.files}


def write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")


def compact(src,dest):
    found=arrays(src)
    both={"t","nom_x_m","q_nom","qdot_nom_rad_s","knot_t_s","knot_q_rad","knot_target_pose","interval_index",
          "tick_index","target_twist_at_ticks","jacobian_twist_at_ticks","dq","dqd","pressure","feedback_reference_um"}
    full=("t","vx_m_s","physical_xyz_m","updated","actual_micro_q","ctrl_saturation","joint_saturation",
          "force_saturation","qfrc_applied","Fd","remaining","remaining_dot")
    sampled=("q","qdot","final_ctrl","base_ctrl","qfrc_actuator","qdot_nom_command_rad_s","q_velocity_ff_rad")
    chosen={name:found[name] for name in sorted(both)}
    for arm in ARMS:
        for key in full:
            chosen[f"{arm}_{key}"]=found[f"{arm}_{key}"]
        ticks=found["tick_index"]
        keep=ticks[ticks<len(found[f"{arm}_t"])]
        for key in sampled:
            chosen[f"{arm}_{key}_at_ticks"]=found[f"{arm}_{key}"][keep]
    np.savez_compressed(dest,**chosen)
    return chosen


def number(x,digits=2):
    return "—" if x is None else f"{x:.{digits}f}"


def table(names,rows):
    h="".join(f"<th>{html.escape(str(x))}</th>" for x in names)
    items="".join("<tr>"+"".join(f"<td>{html.escape(str(x))}</td>" for x in row)+"</tr>" for row in rows)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{items}</tbody></table>"


def img(fig,alt):
    buf=BytesIO()
    fig.savefig(buf,format="png",dpi=125,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{b64encode(buf.getvalue()).decode()}">'


def shade(ax,rec):
    for key,color in (("start","#fff0e6"),("turn","#f1ecf8"),("stop","#fff0e6")):
        lo,hi=rec["windows_s"][key]
        ax.axvspan(lo,hi,color=color,zorder=-5)
    for lo,hi in rec["cabinet_gaps_s"]:
        ax.axvspan(lo,hi,facecolor="none",edgecolor="#adadad",hatch="//",lw=.5,zorder=-4)
    ax.grid(alpha=.2)


def axis_plot(ax,item,kind,xlim=None,metrics=None):
    trial,rec,data,source,status=item
    h=source["host_time_s"]
    t=data["t"]
    origin=source["actual_tcp_pose"][0,0]
    shade(ax,rec)
    a=(source["actual_tcp_pose"][:,0]-origin)*1000 if kind=="X" else source["actual_tcp_speed"][:,0]*1000
    ax.plot(h,a,".",ms=2.5,color=COLORS["A"],label="A cabinet")
    if kind=="X":
        ax.plot(t,(data["nom_x_m"]-origin)*1000,"--",lw=.8,color=COLORS["nominal"],label="Nominal X")
    for arm,name in zip(ARMS,("Original","Nominal velocity"),strict=True):
        y=(data[f"{arm}_physical_xyz_m"][:,0]-origin)*1000 if kind=="X" else data[f"{arm}_vx_m_s"]*1000
        ax.plot(data[f"{arm}_t"],y,lw=1.05,color=COLORS[arm],label=name)
    if kind=="Vx":
        ticks=data["tick_index"]
        v=data["target_twist_at_ticks"][:,0]*1000
        ax.plot(t[ticks],v,lw=.7,color=COLORS["nominal"],alpha=.7,drawstyle="steps-post",label="Native nominal Vx")
    if metrics:
        for marker in metrics:
            sec=status["candidate"]["axis"]["source"][marker]["time_s"]
            if sec is not None:
                ax.axvline(sec,ls=":",lw=.7,color=COLORS["A"])
    ax.set(xlabel="Original host time [s]",ylabel="X [mm]" if kind=="X" else "Vx [mm/s]")
    if xlim:
        ax.set_xlim(*xlim)
    ax.legend(loc="best",fontsize=7)


def overview_fig(items):
    fig,axes=plt.subplots(3,2,figsize=(13,9.4),layout="constrained")
    for item,row in zip(items,axes,strict=True):
        for ax,kind in zip(row,("X","Vx"),strict=True):
            axis_plot(ax,item,kind)
            ax.set_title(f"{item[0]}   {kind}",loc="left",fontweight="bold")
    return img(fig,"3 个试次原版与候选 X 位置、速度总览")


def detail_fig(item):
    trial,rec,*_=item
    fig,axes=plt.subplots(2,3,figsize=(13,6.4),layout="constrained")
    for col,(key,name) in enumerate((("start","Startup"),("turn","Reversal"),("stop","Stop"))):
        lo,hi=rec["windows_s"][key]
        axes[0,col].set_title(name,loc="left",fontweight="bold")
        for row,kind in enumerate(("X","Vx")):
            metrics={"start":("start10","start50","start90"),"turn":("decel90","zero","return90"),"stop":("stop90","stop10")}[key]
            axis_plot(axes[row,col],item,kind,(lo-.07,hi+.07),metrics if kind=="Vx" else ())
    return img(fig,trial+" 启动、换向与停止局部")


def plot_report(items):
    overview=overview_fig(items)
    detail=[]
    for item in items:
        trial,rec,_,_,s=item
        a=s["original"]; b=s["candidate"]; dr=s["derivative"]
        gate=s["gate"]
        events=table(["标记","A 时刻 s","原版时刻 s","候选时刻 s","原版晚 ms","候选晚 ms"],[
            (label,number(a["axis"]["source"][k]["time_s"],4),number(a["axis"]["candidate"][k]["time_s"],4),
                    number(b["axis"]["candidate"][k]["time_s"],4),number(a["axis"]["delay_ms"][k]),number(b["axis"]["delay_ms"][k]))
              for k,label in AXIS.items()])
        durations=table(["过程","A 时长 ms","原版时长 ms","候选时长 ms","原版绝对误差 ms","候选绝对误差 ms"],[
            (label,number(a["axis"]["durations"][k]["A_ms"]),number(a["axis"]["durations"][k]["candidate_ms"]),
             number(b["axis"]["durations"][k]["candidate_ms"]),number(gate["duration_errors"][k]["original_ms"]),
             number(gate["duration_errors"][k]["candidate_ms"]))
             for k,label in (("rise_10_90","启动 10→90%"),("reversal_90_90","换向 +90→−90%"))])
        shaped=table(["区段","原版速度 RMSE mm/s","候选速度 RMSE mm/s","原版 X RMSE mm","候选 X RMSE mm"],[
            (label,number(a["axis"]["shape"][k]["v_RMSE_mm_s"],3),number(b["axis"]["shape"][k]["v_RMSE_mm_s"],3),
             number(a["axis"]["shape"][k]["x_RMSE_um"]/1000,3),number(b["axis"]["shape"][k]["x_RMSE_um"]/1000,3))
             for k,label in EVENT.items()])
        stability=table(["区段","原版稳态跟踪尺度 ms","候选稳态跟踪尺度 ms","A 速度峰 mm/s","原版速度峰 mm/s","候选速度峰 mm/s"],[
            (EVENT[k],
             number(a["stable"][k]["nominal_tracking_scale_ms"]) if k in a["stable"] else "—",
             number(b["stable"][k]["nominal_tracking_scale_ms"]) if k in b["stable"] else "—",
             number(a["overshoot"]["speed"][k]["A_abs_peak_mm_s"],3) if k in a["overshoot"]["speed"] else "—",
             number(a["overshoot"]["speed"][k]["model_abs_peak_mm_s"],3) if k in a["overshoot"]["speed"] else "—",
             number(b["overshoot"]["speed"][k]["model_abs_peak_mm_s"],3) if k in b["overshoot"]["speed"] else "—")
             for k in ("outbound","return_straight","start","turn","stop")])
        extra=(f'8 ms 读取的名义 X 速度范围 {number(dr["raw_nominal_vx_tick_mm_s_minmax"][0])} 至 '
               f'{number(dr["raw_nominal_vx_tick_mm_s_minmax"][1])} mm/s。'
               f'新增远端/近端位置越界 {number(gate["spatial_overshoot_change_mm"]["far_X_excess_over_A_mm"],3)} / '
               f'{number(gate["spatial_overshoot_change_mm"]["near_X_excess_over_A_mm"],3)} mm。'
               '两臂均无硬保护或执行器饱和。')
        detail.append(f'<section><h2>{trial}</h2><p>{extra}</p>{detail_fig(item)}'
                      f'<h3>时刻与过程时长</h3>{events}{durations}<h3>速度与位置误差</h3>{shaped}{stability}</section>')
    quick=table(["试次","主时刻 ≤50 ms","时长误差均改善","事件速度 RMSE 均改善","过冲和保护","整片"],[
        (trial,*("通过" if g[k] else "未通过" for k in ("timing_pass","duration_pass","shape_pass")),
         "通过" if g["overshoot_pass"] and g["protection_pass"] else "未通过","通过" if g["pass_all"] else "未通过")
        for trial,_,_,_,s in items for g in [s["gate"]]])
    text=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>G 组底层速度参考最小对照</title>
<style>body{{max-width:1200px;margin:34px auto;padding:0 20px;font:18px/1.6 SimSun,"Times New Roman",serif;color:#182630}}
h1,h2,h3{{font-family:SimHei,Arial,sans-serif;line-height:1.2}} .main{{background:#edf4f8;padding:12px 16px;border-left:4px solid #154a85}}
img{{width:100%;height:auto}} section{{margin:45px 0}} table{{border-collapse:collapse;width:100%;font-size:15px;margin:12px 0 25px}}
th,td{{padding:7px;border-bottom:1px solid #d9e2e8;text-align:right}} td:first-child,th:first-child{{text-align:left}} th{{background:#edf1f5}}
</style></head><body><main><h1>G 组底层速度参考最小对照</h1>
<p class="main">约 100 ms 的模型阻尼跟踪尺度获得支持。三片主时刻均达标，形状与过冲联合门槛未通过，Y 重建未启动。本轮为离线隔离诊断，尚未实机验证。</p>
{quick}
<h2>三片原轴总览</h2>
<p>橙色与紫色背景标出冻结的启动、换向、停止窗；斜纹标出柜端采样空档。X 位置以对应实测起点表示，时间为各次原主机轴。绿线为原名义 X 或在原 8 ms 结点可读的导数。六次分别积分。R01、R02、P10 的原版均重新跑过，与既有 axis_measured_init 的 24 个字段逐数组相等。</p>
{overview}
<h2>唯一候选和速度来源</h2>
<p>执行器限幅前保留 Kp(qcmd−q)−Kd·qdot。只在最终目标加入 (Kd/Kp)·qdot_nom，将名义速度项按原 8 ms 六轴命令一起更新与保持。旧 dq、参考、重力、惯量、阻尼、限幅和原保护保持。等效拟合力、公共指令表及新增反馈控制都为零。</p>
<p>B 的五份原 RTDE/并行运行日志未发现目标速度字段。速度只来自既有六维逆解的分段线性关节目标，按相邻原主机时间结点的斜率构造。内结点取右导数，最后端点取左导数。原名义文件保留；未做平滑或重规划。三片首尾读数为零。六维雅可比回代误差与逐轴斜率变化在 JSON 中。</p>
<h2>为什么停止</h2>
<p>原版稳定去回程空间偏差折算为约 99.9–109.9 ms，候选缩小到约 −0.2–10.0 ms。这个变化支持原模型绝对速度阻尼造成主要沿程偏差的假设。R01 的 +90%→−90% 换向时间从 373.2 ms 缩到 91.2 ms，而 A 是 259.5 ms，绝对误差增大。10 mm/s 的目标差分在冻结结点达到约 ±20 mm/s，候选换向绝对速度峰 18.30 mm/s，高于 A 的 10.79 mm/s。启动速度 RMSE 从 2.439 增到 3.313 mm/s。原时钟下的一些名义速度峰会经这一支路进入伺服。</p>
<p>在继续走同一路径之前，最小的下一步是核对已有名义目标位姿记录的发送时刻与状态出现时刻，确认 10 mm/s 的双幅斜率来自任务生成还是非同步记录；有可信名义速度证据后再按原门槛审查。报告中不推断真实柜端的控制结构。</p>
{"".join(detail)}
<h2>复算</h2><p><code>verify_review.py</code> 可从紧凑数组复核两臂命令合成、原始主机时刻、误差和门槛。完整 1 ms 执行器状态、保护和采样日志保留在本机。此包用于 GPT 审查。</p>
</main></body></html>'''
    (ROOT/"G_servo_minimal_review.html").write_text(text,encoding="utf-8")


def build(run):
    out=ROOT/"outputs"
    out.mkdir(exist_ok=True)
    r=read(run/"receipt.json")
    assert r["formal_control_runs"]==0 and r["completed_pairs"]==list(TRIALS)
    assert not r["x_pass"] and r["physical_y_reconstruction"]=="not started"
    copy2(run/"receipt.json",out/"receipt.json")
    copy2(PRECEDING/"windows_frozen.json",ROOT/"windows_frozen.json")
    copy2(run/"integration_source"/"command_channel.patch",out/"inherited_command_channel.patch")
    windows=read(ROOT/"windows_frozen.json")["trials"]
    items=[]
    rows=[]
    for trial in TRIALS:
        source=run/trial
        dest=out/trial
        dest.mkdir(exist_ok=True)
        z=compact(source/"comparison.npz",dest/"comparison.npz")
        source_hash=sha256((source/"comparison.npz").read_bytes()).hexdigest()
        write(dest/"trace_reference.json",dict(rundirectory_relative=f"runs/{run.name}/{trial}",
             full_array_sha256=source_hash,velocity_state_actuation_archived_at_8ms=True))
        s=read(source/"status.json")
        copy2(source/"status.json",dest/"status.json")
        for arm in ARMS:
            log=dest/arm
            log.mkdir(exist_ok=True)
            for f in ("case.json","row.json","guard.json","checks.json"):
                if (source/arm/f).is_file():
                    copy2(source/arm/f,log/f)
        items.append((trial,windows[trial],z,arrays(PREPARED/trial/"A_source.npz"),s))
        for label,k in ((k,label) for k,label in AXIS.items()):
            rows.append(dict(trial=trial,metric=label,description=k,units="ms",
                       original=s["original"]["axis"]["delay_ms"][label],candidate=s["candidate"]["axis"]["delay_ms"][label]))
        for label,k in EVENT.items():
            rows.append(dict(trial=trial,metric=label,description=k+"速度 RMSE",units="mm/s",
                       original=s["original"]["axis"]["shape"][label]["v_RMSE_mm_s"],
                       candidate=s["candidate"]["axis"]["shape"][label]["v_RMSE_mm_s"]))
    with (out/"key_x_metrics.csv").open("w",encoding="utf-8-sig",newline="") as file:
        w=csv.DictWriter(file,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot_report(items)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,required=True)
    opts=parser.parse_args()
    build(opts.run)
