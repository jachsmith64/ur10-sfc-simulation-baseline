"""Build the Chinese offline review from byte-pinned preflight and paired results."""
from base64 import b64encode
import argparse
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
PREPARED=ROOT.parent/"20261116_A_multispeed/prepared_A"
TRIALS=("X_P05_R01","X_P05_R02","X_P10_R01")
ARMS=("velocity_off","velocity_on")
MARKERS={"start50":"启动半速","turn_xmax":"最大 X","return50":"回程半速"}
SPANS={"rise_10_90":"启动 10→90%","reversal_90_90":"换向 +90→−90%","reversal_80_80":"换向 ±80% 诊断","reversal_50_50":"换向 ±50% 诊断"}
COLORS=("#778398","#135487","#ad5229","#318167")
plt.rcParams.update({"font.size":8.5,"axes.spines.top":False,"axes.spines.right":False})


def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def data(p):
    with np.load(p,allow_pickle=False) as a:
        return {k:a[k] for k in a.files}


def enc(b):
    return b64encode(b).decode("ascii")


def plot_img(fig,label):
    buf=BytesIO()
    fig.savefig(buf,format="png",dpi=118,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    return f'<img alt="{html.escape(label)}" src="data:image/png;base64,{enc(buf.getvalue())}">'


def num(x,n=1):
    return "不确定" if x is None else f"{x:.{n}f}"


def table(header,rows):
    cells=lambda fields,tag:"".join(f"<{tag}>{html.escape(str(c))}</{tag}>" for c in fields)
    return "<table><thead><tr>"+cells(header,"th")+"</tr></thead><tbody>"+"".join(
        "<tr>"+cells(r,"td")+"</tr>" for r in rows)+"</tbody></table>"


def draw(ax,item,kind,span=None,events=()):
    trial,win,source,inp,trace,s,pre=item
    ht=source["host_time_s"]
    mapped=inp["actual_mapped_host_s"]
    keep=inp["actual_index"]
    x0=source["actual_tcp_pose"][0,0]
    start=win["windows_s"]["start"]
    turn=win["windows_s"]["turn"]
    stop=win["windows_s"]["stop"]
    for pair,color in ((start,"#fdebde"),(turn,"#eeebf7"),(stop,"#fdebde")):
        ax.axvspan(*pair,facecolor=color,edgecolor="none",zorder=-5)
    for g in pre["clock"]["actual_gaps"]:
        ax.axvspan(*g["mapped_s"],facecolor="none",edgecolor="#a9a9a9",hatch="//",lw=.45,zorder=-4)
    if kind=="X":
        raw=(source["actual_tcp_pose"][:,0]-x0)*1000
        ax.plot(inp["t"],(inp["nom_x_m"]-x0)*1000,"--",color=COLORS[3],lw=.8,label="Mapped nominal")
    else:
        raw=source["actual_tcp_speed"][:,0]*1000
        t=inp["target_mapped_host_s"]
        x=inp["derived_target_tcp_pose"][:,0]
        ax.step(t[:-1],np.diff(x)/np.diff(t)*1000,where="post",color=COLORS[3],lw=.75,label="Mapped target slope")
    if span:
        ax.plot(ht,raw,".",ms=1.4,alpha=.44,color="#bc9c82",label="A points on arrival clock")
    ax.plot(mapped,raw[keep],".",ms=2.2,color=COLORS[2],label="A mapped raw")
    for arm,color,label in zip(ARMS,COLORS,("Velocity off","Velocity on")):
        y=(trace[arm+"_physical_xyz_m"][:,0]-x0)*1000 if kind=="X" else trace[arm+"_vx_m_s"]*1000
        ax.plot(trace[arm+"_t"],y,color=color,lw=.9,label=label)
    if events:
        a=pre["input"]["mapped_actual"]["events"]
        old=pre["input"]["legacy_actual_events"]
        for i,key in enumerate(events):
            stamp=old[key]["time_s"]
            if stamp is not None:
                ax.axvline(stamp,color="#8b6546",ls="--",lw=.7,label="Legacy A marker" if i==0 else None)
        for i,key in enumerate(events):
            for sign,color,status in (("A",COLORS[2],a),("B",COLORS[1],s["conditions"]["velocity_on"]["robust"]["events"])):
                tm=status[key]["time_s"]
                if tm is not None:
                    ax.axvline(tm,color=color,ls=":",lw=.7,label="Robust "+sign if i==0 else None)
    if span:
        ax.set_xlim(*span)
    ax.set(xlabel="Common host time [s]",ylabel="X [mm]" if kind=="X" else "Vx [mm/s]")
    ax.grid(alpha=.18)
    ax.legend(loc="best",fontsize=6.5)


def figures(items):
    f,a=plt.subplots(3,2,figsize=(13,9.5),layout="constrained")
    for item,row in zip(items,a,strict=True):
        for ax,kind in zip(row,("X","Vx"),strict=True):
            draw(ax,item,kind)
            ax.set_title(f"{item[0]}  {kind}",loc="left",fontweight="bold")
    overview=plot_img(f,"三片全程 X 位置和速度")
    details=[]
    for item in items:
        f,a=plt.subplots(2,2,figsize=(13,6),layout="constrained")
        for col,(phase,title,markers) in enumerate((
                ("start","Startup",("start10","start50","start90")),
                ("turn","Reversal",("decel90","zero","return90")))):
            lo,hi=item[1]["windows_s"][phase]
            a[0,col].set_title(title,loc="left",fontweight="bold")
            draw(a[0,col],item,"X",(lo-.07,hi+.08))
            draw(a[1,col],item,"Vx",(lo-.07,hi+.08),markers)
        details.append(plot_img(f,item[0]+" 启动和换向局部"))
    return overview,details



def y_report(root,items):
    receipt=read(root/"receipt.json")
    fig,axs=plt.subplots(3,3,figsize=(14,9),layout="constrained")
    conclusion=[]
    details=[]
    for item,row in zip(items,axs,strict=True):
        trial,rec,source,_,_,_,_=item
        status=receipt["results"][trial]
        a=data(root/trial/"refit_review.npz")
        meta=read(PREPARED/trial/"source.json")
        t=a["t"]
        f=source["frame_host_time_s"]
        valid=source["frame_valid"]
        zero=meta["raw_visual_zero_um"]
        r=a["score_reference_um"]
        refs=(np.full_like(r,-zero),r)
        for ax,reference,title in zip(row[:2],refs,("Original visual origin","Common reference removed"),strict=True):
            for phase,color in (("start","#fdebde"),("turn","#eeebf7"),("stop","#fdebde")):
                ax.axvspan(*rec["windows_s"][phase],facecolor=color,edgecolor="none",zorder=-3)
            ar=np.interp(f,t,reference)
            ax.plot(f[valid],(source["A_centered_um"]-ar)[valid],".",ms=1.7,color=COLORS[2],label="A camera valid")
            ax.plot(t,a["fit_y_um"]-reference,color=COLORS[1],lw=.9,label="N physical")
            ax.set(title=trial+"  "+title,xlabel="Original camera host time [s]",ylabel="Y [µm]")
            ax.legend(loc="best",fontsize=6.5)
            ax.grid(alpha=.18)
        ax=row[2]
        ax.plot(t,a["plus_delta_y_um"],color=COLORS[1],lw=.9,label="+1 N")
        ax.plot(t,a["minus_delta_y_um"],color=COLORS[2],lw=.9,label="−1 N")
        ax.axvspan(.800,.856,facecolor="#e5ede7",zorder=-3)
        ax.set(xlim=(.76,1.18),title=trial+"  Micro-compensation",xlabel="Original time [s]",ylabel="Physical Y change [µm]")
        ax.legend(loc="best",fontsize=7)
        ax.grid(alpha=.18)
        attempt=status["attempts"][status["best_index"]]
        conclusion.append((trial,num(attempt["scores"]["full"]["raw"]["fit"],2),
             num(attempt["scores"]["full"]["processed"]["fit"],2),num(status["best_min_fit"],2),
             num(status["initial_stationary_y_um"],3),
            num((max(a["fitted_xyz_m"][:,0])-max(source["actual_tcp_pose"][:,0]))*1000,3),
             num(status["positive_micro_peak_um"],2)+" / "+num(status["negative_micro_peak_um"],2)))
        details.append(f"<h3>{trial} 的所有 Y 窗口</h3>"+table(["窗口","原始 FIT","去共同参考 FIT","原始 RMSE μm","去参考 RMSE μm"],[
            (k,num(v["raw"]["fit"],2),num(v["processed"]["fit"],2),num(v["raw"]["rmse_um"],3),num(v["processed"]["rmse_um"],3))
            for k,v in attempt["scores"].items()]))
    caption=table(["试次","整段原始 FIT","整段抖动 FIT","主要窗口最低 FIT","初始物理 Y μm","拟合后远端误差 mm","正/负峰 μm"],conclusion)
    return caption+plot_img(fig,"三片 Y 物理重建与正负微补偿响应")+"".join(details)


def report(items,prepared,y_root):
    overview,details=figures(items)
    summary=[]
    parts=[]
    for item,img in zip(items,details,strict=True):
        name,_,_,_,_,s,p=item
        g=s["gate"]; a=s["conditions"]["velocity_off"]; b=s["conditions"]["velocity_on"]
        summary.append((name,*[num(g["markers"][k]["delay_ms"],1) for k in MARKERS],"通过" if g["pass_all"] else "未通过"))
        spans=table(["时长","A ms","速度项关 ms","速度项开 ms","关误差 ms","开误差 ms"],[
            (text,*[num(g["spans"][k][j],1) for j in ("measured_ms","original_ms","candidate_ms","original_error_ms","candidate_error_ms")])
            for k,text in SPANS.items()])
        shapes=table(["原始窗","速度项关 RMSE mm/s","速度项开 RMSE mm/s","速度项关 X RMSE μm","速度项开 X RMSE μm"],[
            (k,num(u["v_RMSE_mm_s"],3),num(b["axis"]["shape"][k]["v_RMSE_mm_s"],3),
               num(u["x_RMSE_um"]),num(b["axis"]["shape"][k]["x_RMSE_um"]))
            for k,u in a["axis"]["shape"].items()])
        point_order=("start10","start50","start90","decel90","zero","return50","return90")
        pts=table(["标记","原 A arrival 时刻 s","新 A 物理时间估计 s","候选时刻 s"],[
            (k,num(p["input"]["legacy_actual_events"][k]["time_s"],4),
               num(p["input"]["mapped_actual"]["events"][k]["time_s"],4),
               num(b["robust"]["events"][k]["time_s"],4)) for k in point_order])
        peaks=table(["过冲","关相对 A 的绝对超量","开相对 A 的绝对超量"],[
            (k+" mm/s",num(a["overshoot"]["speed"][k]["excess_over_A_mm_s"],3),
                         num(b["overshoot"]["speed"][k]["excess_over_A_mm_s"],3)) for k in ("start","turn","stop")]+[
            (k+" mm",num(a["overshoot"][k],4),num(b["overshoot"][k],4)) for k in ("far_X_excess_over_A_mm","near_X_excess_over_A_mm")])
        old=table(["历史指标原样","A/历史关/历史开 （ms）","本轮同轴 legacy 关/开（ms）"],[
            ("+90→−90%"," / ".join(num(p["historical"][k]["axis"]["durations"]["reversal_90_90"][j])
                  for k,j in (("original","A_ms"),("original","candidate_ms"),("candidate","candidate_ms"))),
                  " / ".join(num(x["axis"]["durations"]["reversal_90_90"]["candidate_ms"]) for x in (a,b)))])
        flags=[("主时刻",g["timing_pass"]),("时长",g["durations_pass"]),("原波形速度",g["shape_pass"]),
               ("稳定速度",g["stable_shape_pass"]),("过冲",g["overshoot_pass"]),("保护",g["protection_pass"])]
        verdict="，".join(k+("通过" if v else "未通过") for k,v in flags)
        uncertainty=table(["标记","候选时差 ms","原采样括号宽度加最大时钟残差 ms"],[
            (name,num(x["delay_ms"],2),num(x["reported_resolution_ms"],2)) for name,x in g["markers"].items()])
        parts.append(f"<section><h2>{name}</h2><p>{verdict}。</p>{img}{pts}{spans}{shapes}{peaks}{uncertainty}{old}</section>")
    input_overlay=f'<img alt="积分前原名义输入和修正输入叠图" src="data:image/png;base64,{enc((prepared/"input_overlay.png").read_bytes())}">'
    verdict="按原联合门槛回放，P10 新增端点过冲仍超出限值。按用户更新的研究用途，三片 X 参考足够进入 Y 物理重建。"
    doc=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>G 组时钟与事件修正</title>
<style>body{{max-width:1210px;margin:32px auto;padding:0 22px;font:18px/1.56 SimSun,"Times New Roman",serif;color:#192a37}}
h1,h2,h3{{font-family:SimHei,Arial,sans-serif;line-height:1.23}} img{{width:100%;height:auto}}
table{{border-collapse:collapse;width:100%;margin:10px 0 26px;font-size:14px}}
th,td{{padding:7px 9px;border-bottom:1px solid #d4e0e7;text-align:right}} th{{background:#ecf3f8}} th:first-child,td:first-child{{text-align:left}}
section{{margin:45px 0}} .card{{background:#edf4f8;border-left:4px solid #225686;padding:8px 16px}}
</style></head><body><main>
<h1>G 组名义时间和事件识别</h1><p class="card">三片输入预检通过。{verdict} 同一 G 冻结窗口、实际静止初态和原保护下完成六次新积分。</p>
{table(["试次","候选启动半速差 ms","候选最大 X 差 ms","候选回程半速差 ms","旧联合诊断"],summary)}
<h2>三片全程位置和速度</h2>
<p>实验 A 关闭速度参考，实验 B 开启唯一名义速度参考。横坐标以相机首帧为零，共同主机时间未对某一臂平移。以下 X 时长、速度和过冲表保留旧诊断口径。斜纹是柜端长采样空档，原到达时刻与映射时刻均留存。局部图淡点为实测原到达时间，橙点为映射后原值；虚线为旧事件点，点线为稳健事件点。RMSE 直接使用映射后原样点。</p>{overview}
<h2>积分前名义时钟检查</h2>
<p>P10 同一跳变 160 μm，主机间隔 7.946 ms 给出 20.14 mm/s，柜端 16 ms 给出 10 mm/s。原相邻读取调用先取 actual 字段，再写主机时戳和柜端戳，最后单独取目标位姿，记录未证明字段原子一致。固定偏移的精度受未知采集到达延迟限制。相同柜端戳有冲突时排除整组目标，实际位置与速度独立核对。新图使用同一位置曲线及其导数，未限幅或滤掉原运动。</p>
<p>初次源数据图另发现固定稳定窗之外三处残余双幅尖峰。前后等速斜率与短桥一致，便排除孤立中点。该处理完成于动力学积分前。未在未知的长空档内重建实际起停过程；可疑停车段仍保留为空档。各行决定和长短跨接见证据 JSON。</p>
{input_overlay}
<h2>P10 端点局限</h2><p>P10 在修正后未出现上轮速度过冲，稳定速度 RMSE 约 0.31–0.32 mm/s，仍有 0.10875 mm 的远端新增位置过冲。原实测峰与映射后实测峰相同，候选也高于纠正名义峰 0.07335 mm。Kp/Kd、初态和其余条件成对保持，只有名义速度项开关变化。详细峰值见原始数组和 <code>P10_peak_crosscheck.json</code>。</p>
<h2>事件锁定与测量口径</h2>
<p>只为检测将原速度在有证据的短括号内线性插值到 1 ms，用半宽 12 ms 的居中中值和至少 24 ms 的持续性。围绕唯一主过零取最后一个连通的减速穿越；R01 的提前回落后还回到了正向平台，因此落在主段之外。90% 为主，50% 与 80% 仅用于敏感性呈现。数值表中的小数是可复算插值，物理精度仍受原样点括号、固定时钟残差和检测窗影响。报告不据此声明亚毫秒同步。</p>
{"".join(parts)}
<h2>相机 Y 的物理 N 重建</h2>
<p>Y 使用相机原有幅值、有效帧和时刻。仅施加由静止与稳定平台定标的公共物理纠偏，以及通过 30/45 Hz 反推的 Y 力。原方向反馈参考和同源评分参考单独封存；事件窗内保留预事件基线。旧观测 remaining 及其导数为零。正负 1 N 在相同最佳 N 上由原关节指令通道施加，0.800 s 开始，第一处可分辨物理响应在 0.801 s。三片保持原 8 ms 速度项，未覆盖实际关节状态。</p>
{y_report(y_root,items)}
<p>运动窗 FIT 的范围只衡量对这三片源 Y 的离线拟合。静止与过渡窗单列，较低 FIT 的结果保留。此包供 GPT 审查，未执行正式线性/SFC 对照。</p>
</main></body></html>'''
    (ROOT/"G_clock_event_review.html").write_text(doc,encoding="utf8")


def build(prepared,pairs,y_root):
    public=ROOT/"results"
    public.mkdir(exist_ok=True)
    windows=read(ROOT.parent/"20261116_preprocess_timing/windows_frozen.json")
    copy2(ROOT.parent/"20261116_preprocess_timing/windows_frozen.json",ROOT/"windows_frozen.json")
    for name in ("lock.json","preflight_plot_and_rules.json","derived_command_event_check.json"):
        copy2(prepared/name,public/name)
    copy2(pairs/"receipt.json",public/"paired_receipt.json")
    copy2(y_root/"receipt.json",public/"y_receipt.json")
    copy2(pairs/"integration_source"/"command_channel.patch",public/"inherited_command_channel.patch")
    items=[]
    comparison=[]
    for trial in TRIALS:
        p=prepared/trial
        g=pairs/trial
        out=public/trial
        out.mkdir(exist_ok=True)
        for name in ("input.npz","clock_decisions.json","input.json","ik.json"):
            copy2(p/name,out/name)
        for name in ("status.json","comparison.npz"):
            copy2(g/name,out/name)
        for arm in ARMS:
            path=out/arm
            path.mkdir(exist_ok=True)
            for f in ("row.json","case.json","checks.json","guard.json"):
                copy2(g/arm/f,path/f)
        y=y_root/trial
        copy2(y/"status.json",out/"y_status.json")
        copy2(y/"refit_review.npz",out/"refit_review.npz")
        copy2(y/"plant.npz",out/"plant.npz")
        audit=out/"y_run_configs"
        audit.mkdir(exist_ok=True)
        for config in sorted(y.glob("*/case.json")):
            place=audit/config.parent.name
            place.mkdir(exist_ok=True)
            for name in ("case.json","row.json","checks.json","guard.json"):
                copy2(config.parent/name,place/name)
        s=read(g/"status.json")
        old_receipt=ROOT.parent/"20261116_servo_minimal/outputs"/trial/"status.json"
        history=read(old_receipt)
        copy2(old_receipt,out/"historical_servo_status.json")
        source=data(PREPARED/trial/"A_source.npz")
        original_x=source["actual_tcp_pose"][:,0]
        fit_x=data(y/"refit_review.npz")["fitted_xyz_m"][:,0]
        before=data(g/"comparison.npz")["velocity_on_physical_xyz_m"][:,0]
        result=dict(fitted_far_endpoint_signed_error_mm=float((max(fit_x)-max(original_x))*1000),
             fitted_near_endpoint_signed_error_mm=float((min(fit_x)-min(original_x))*1000),
             fitted_span_mm=float(np.ptp(fit_x)*1000),
             reference_before_Y_far_endpoint_signed_error_mm=float((max(before)-max(original_x))*1000))
        (out/"Y_X_endpoints.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")

        evidence=dict(input=read(p/"input.json"),clock=read(p/"clock_decisions.json"),historical=history)
        items.append((trial,windows["trials"][trial],source,data(p/"input.npz"),data(g/"comparison.npz"),s,evidence))
        for k,v in s["gate"]["markers"].items():
            comparison.append(dict(trial=trial,item=k,candidate_delay_ms=v["delay_ms"],sample_resolution_plus_residual_ms=v["reported_resolution_ms"]))
    with (public/"key_x.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=comparison[0].keys())
        w.writeheader()
        w.writerows(comparison)
    report(items,prepared,y_root)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--prepared",type=Path,required=True)
    p.add_argument("--pairs",type=Path,required=True)
    p.add_argument("--y",type=Path,required=True)
    a=p.parse_args()
    build(a.prepared,a.pairs,a.y)
