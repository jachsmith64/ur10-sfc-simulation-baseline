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


def report(items,prepared):
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
    verdict="三片通过 X 联合验收。" if all(item[5]["gate"]["pass_all"] for item in items) else "至少一片未通过 X 联合验收。"
    doc=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>G 组时钟与事件修正</title>
<style>body{{max-width:1210px;margin:32px auto;padding:0 22px;font:18px/1.56 SimSun,"Times New Roman",serif;color:#192a37}}
h1,h2,h3{{font-family:SimHei,Arial,sans-serif;line-height:1.23}} img{{width:100%;height:auto}}
table{{border-collapse:collapse;width:100%;margin:10px 0 26px;font-size:14px}}
th,td{{padding:7px 9px;border-bottom:1px solid #d4e0e7;text-align:right}} th{{background:#ecf3f8}} th:first-child,td:first-child{{text-align:left}}
section{{margin:45px 0}} .card{{background:#edf4f8;border-left:4px solid #225686;padding:8px 16px}}
</style></head><body><main>
<h1>G 组名义时间和事件识别</h1><p class="card">三片输入预检通过。{verdict} 同一 G 冻结窗口、实际静止初态和原保护下完成六次新积分。</p>
{table(["试次","候选启动半速差 ms","候选最大 X 差 ms","候选回程半速差 ms","联合门槛"],summary)}
<h2>三片全程位置和速度</h2>
<p>实验 A 关闭速度参考，实验 B 开启唯一名义速度参考。横坐标以相机首帧为零，共同主机时间未对某一臂平移。斜纹是柜端长采样空档，原到达时刻与映射时刻均留存。局部图淡点为实测原到达时间，橙点为映射后原值；虚线为旧事件点，点线为稳健事件点。RMSE 直接使用映射后原样点。</p>{overview}
<h2>积分前名义时钟检查</h2>
<p>P10 同一跳变 160 μm，主机间隔 7.946 ms 给出 20.14 mm/s，柜端 16 ms 给出 10 mm/s。原相邻读取调用先取 actual 字段，再写主机时戳和柜端戳，最后单独取目标位姿，记录未证明字段原子一致。固定偏移的精度受未知采集到达延迟限制。相同柜端戳有冲突时排除整组目标，实际位置与速度独立核对。新图使用同一位置曲线及其导数，未限幅或滤掉原运动。</p>
<p>初次源数据图另发现固定稳定窗之外三处残余双幅尖峰。前后等速斜率与短桥一致，便排除孤立中点。该处理完成于动力学积分前。未在未知的长空档内重建实际起停过程；可疑停车段仍保留为空档。各行决定和长短跨接见证据 JSON。</p>
{input_overlay}
<h2>为何没有继续 N</h2><p>P10 在修正后未出现上轮速度过冲，稳定速度 RMSE 约 0.31–0.32 mm/s，仍有 0.10875 mm 的远端新增位置过冲。原实测峰与映射后实测峰相同，候选也高于纠正名义峰 0.07335 mm。Kp/Kd、初态和其余条件成对保持，只有名义速度项开关变化。详细峰值见原始数组和 <code>P10_peak_crosscheck.json</code>。</p>
<h2>事件锁定与测量口径</h2>
<p>只为检测将原速度在有证据的短括号内线性插值到 1 ms，用半宽 12 ms 的居中中值和至少 24 ms 的持续性。围绕唯一主过零取最后一个连通的减速穿越；R01 的提前回落后还回到了正向平台，因此落在主段之外。90% 为主，50% 与 80% 仅用于敏感性呈现。数值表中的小数是可复算插值，物理精度仍受原样点括号、固定时钟残差和检测窗影响。报告不据此声明亚毫秒同步。</p>
{"".join(parts)}
<h2>条件 N 重建</h2><p>联合 X 门槛未通过。Y 从零重建、公共关节纠偏重算以及新条件微补偿未启动。</p>
</main></body></html>'''
    (ROOT/"G_clock_event_review.html").write_text(doc,encoding="utf8")


def build(prepared,pairs):
    public=ROOT/"results"
    public.mkdir(exist_ok=True)
    windows=read(ROOT.parent/"20261116_preprocess_timing/windows_frozen.json")
    copy2(ROOT.parent/"20261116_preprocess_timing/windows_frozen.json",ROOT/"windows_frozen.json")
    for name in ("lock.json","preflight_plot_and_rules.json","derived_command_event_check.json"):
        copy2(prepared/name,public/name)
    copy2(pairs/"receipt.json",public/"paired_receipt.json")
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
        s=read(g/"status.json")
        old_receipt=ROOT.parent/"20261116_servo_minimal/outputs"/trial/"status.json"
        history=read(old_receipt)
        copy2(old_receipt,out/"historical_servo_status.json")
        source=data(PREPARED/trial/"A_source.npz")
        evidence=dict(input=read(p/"input.json"),clock=read(p/"clock_decisions.json"),historical=history)
        items.append((trial,windows["trials"][trial],source,data(p/"input.npz"),data(g/"comparison.npz"),s,evidence))
        for k,v in s["gate"]["markers"].items():
            comparison.append(dict(trial=trial,item=k,candidate_delay_ms=v["delay_ms"],sample_resolution_plus_residual_ms=v["reported_resolution_ms"]))
    with (public/"key_x.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=comparison[0].keys())
        w.writeheader()
        w.writerows(comparison)
    report(items,prepared)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--prepared",type=Path,required=True)
    p.add_argument("--pairs",type=Path,required=True)
    a=p.parse_args()
    build(a.prepared,a.pairs)
