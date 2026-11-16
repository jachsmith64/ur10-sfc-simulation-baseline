"""Create static scientific figures and an offline Chinese review from scored data."""
import argparse
from base64 import b64encode
from html import escape
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from collect import CASES, MAIN_WINDOWS, MODES, load, read

COLORS = dict(N="#546174", L="#196bc4", S="#bf5428")
TRIAL_LABELS = dict(X_P05_R01="5 mm/s · R01", X_P05_R02="5 mm/s · R02", X_P10_R01="10 mm/s · R01")
LABELS = dict(full="整段",pre_static="开头静止",start="启动",start_transition="启动后参考过渡",
              outbound="稳定去程",turn="换向",turn_transition="换向后参考过渡",return_straight="稳定回程",
              stop="停止",stop_transition="停止后参考过渡",post_static="停止后静止")


def style(ax):
    ax.grid(axis="y",color="#e0e6ee",lw=.7)
    ax.axhline(0,color="#adb9c8",linewidth=.6)
    ax.spines[["right","top"]].set_visible(False)


def draw(root, trials):
    plt.rcParams.update({"font.family":"DejaVu Serif","font.size":9,"axes.labelcolor":"#2c384a",
                         "axes.edgecolor":"#8b98aa","figure.facecolor":"white","axes.axisbelow":True,
                         "path.simplify":False})
    figures = root/"figures"
    figures.mkdir(exist_ok=True)

    def save(fig,name):
        path=figures/f"{name}.png"
        fig.savefig(path,dpi=135,bbox_inches="tight",pad_inches=.18)
        plt.close(fig)
        return path

    fig,ax=plt.subplots(figsize=(12,3.6),layout="constrained")
    for trial,(rec,d,t,display,ref,phy,cmd,a,_) in trials.items():
        for stage,hue in (("start","#f0deca"),("turn","#dbe9f7")):
            ax.axvspan(*(np.array(rec["windows_s"][stage])+display[0]),color=hue,alpha=.75,lw=0)
        for m in MODES:
            ax.plot(display[:len(phy[m]["physical_e_um"])],phy[m]["physical_e_um"]-ref[:len(phy[m]["physical_e_um"])],
                    label=m if trial==CASES[0] else None,color=COLORS[m],lw=.9)
        ax.text((display[0]+display[-1])*.5,.97,TRIAL_LABELS[trial],ha="center",va="top",
                transform=ax.get_xaxis_transform(),fontsize=10,fontweight="bold")
        if trial!=CASES[-1]:
            ax.axvline(display[-1]+.275,color="#8794a4",linestyle=(0,(2,3)),lw=.8)
    style(ax)
    ax.set(xlabel="Display time (s) · independent trials with gaps",ylabel="Y − fixed reference (µm)")
    ax.margins(y=.18)
    ax.legend(loc="lower right",ncols=3,frameon=False)
    paths=[save(fig,"01_overview")]

    for name,pair in (("02_events",("start","turn")),("03_stable",("outbound","return_straight"))):
        fig,axes=plt.subplots(3,2,figsize=(11,6.7),layout="constrained",sharey=True)
        for i,(trial,(rec,d,t,display,ref,phy,cmd,a,_)) in enumerate(trials.items()):
            for j,stage in enumerate(pair):
                ax=axes[i,j]
                lo,hi=rec["windows_s"][stage]
                ax.set_xlim(lo,hi)
                for m in MODES:
                    n=len(phy[m]["physical_e_um"])
                    ax.plot(t[:n],phy[m]["physical_e_um"]-ref[:n],color=COLORS[m],lw=1)
                style(ax)
                ax.set_title(f"{TRIAL_LABELS[trial]} · {stage.replace('_',' ')}",fontsize=9)
                if j==0: ax.set_ylabel("Y − ref. (µm)")
                if i==2: ax.set_xlabel("Original trial time (s)")
        paths.append(save(fig,name))

    fig,axes=plt.subplots(3,2,figsize=(11,6.9),layout="constrained")
    for i,(trial,(rec,d,t,display,ref,phy,cmd,a,locked)) in enumerate(trials.items()):
        ax=axes[i,0]
        for m in MODES:
            y=phy[m]["physical_e_um"]
            ax.plot(t[:len(y)],y,color=COLORS[m],lw=.8,label=m)
        good=a["frame_valid"]
        ax.plot(a["frame_host_time_s"][good],a["A_centered_um"][good],".",ms=1,color="#111826",alpha=.5,label="Camera A")
        ax.plot(t,ref,"--",lw=1,color="#39836b",label="Fixed reference")
        ax.set_title(TRIAL_LABELS[trial]); ax.set_ylabel("Physical Y (µm)")
        if i==0: ax.legend(loc="lower left",frameon=False,ncols=3,fontsize=7)
        style(ax)
        ax=axes[i,1]
        origin=locked["nom_x_m"][0]
        measured=(a["actual_tcp_pose"][:,0]-origin)*1000
        ax.plot(a["host_time_s"],measured,color="#8996a4",lw=.8,label="A cabinet position")
        ax.plot(t,(locked["nom_x_m"]-origin)*1000,"--",color="#39836b",lw=1,label="Nominal")
        for m in MODES:
            x=phy[m]["physical_x_m"]
            ax.plot(t[:len(x)],(x-origin)*1000,color=COLORS[m],lw=.75,alpha=.75)
        if i==0: ax.legend(loc="lower left",frameon=False,fontsize=7)
        ax.set_ylabel("X position (mm)"); style(ax)
    for ax in axes[-1]: ax.set_xlabel("Original trial time (s)")
    paths.append(save(fig,"04_physical_y_x"))

    fig,axes=plt.subplots(3,2,figsize=(11,6.6),layout="constrained")
    for i,(trial,(rec,d,t,display,ref,phy,cmd,a,locked)) in enumerate(trials.items()):
        ax=axes[i,0]
        for m in MODES:
            v=phy[m]["vx_m_s"]
            ax.plot(t[:len(v)],1000*v,color=COLORS[m],lw=.8,label=m)
        ax.axhline(rec["speed_mm_s"],ls=":",color="#39836b",lw=1)
        ax.axhline(-rec["speed_mm_s"],ls=":",color="#39836b",lw=1)
        ax.set_title(TRIAL_LABELS[trial]); ax.set_ylabel("Physical X speed (mm/s)"); style(ax)
        ax=axes[i,1]
        for m in "LS":
            u=cmd[m]
            ax.step(t[u["step"]],u["F_requested"],where="post",color=COLORS[m],lw=.85,label=f"{m} request")
        u=cmd["S"]
        ax.step(t[u["step"]],u["F_shear"],where="post",color="#805a8e",lw=.85,alpha=.7,label="S nonlinear")
        ax.set_ylabel("Y force request (N)"); style(ax)
        if i==0: ax.legend(loc="upper right",frameon=False,ncols=3,fontsize=7)
    for ax in axes[-1]: ax.set_xlabel("Original trial time (s)")
    paths.append(save(fig,"05_speed_force"))
    return paths


def pct(x):
    return "—" if x is None else f"{x:+.2f}%"


def table(headers,rows):
    head="".join(f"<th>{escape(str(x))}</th>" for x in headers)
    body="".join("<tr>"+"".join(f"<td>{escape(str(x))}</td>" for x in row)+"</tr>" for row in rows)
    return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def describe(root, paths):
    result=read(root/"metrics.json")
    windows=read(root/"windows_frozen.json")["trials"]
    rows={(w["trial"],w["mode"],w["window"]):w for w in result["windows"]}
    ratios={(w["trial"],w["window"]):w for w in result["reductions_percent"]}
    ex=read(root/"execution.json")
    def fig(i,cap):
        url="data:image/png;base64,"+b64encode(paths[i].read_bytes()).decode("ascii")
        return f"<figure><img src='{url}' alt='{escape(cap)}'><figcaption>{escape(cap)}</figcaption></figure>"
    def line(t,w):
        r=ratios[t,w]
        vals=[rows[t,m,w]["score_um"] for m in MODES]
        return (t,LABELS[w],*(f"{v['rms']:.2f} / {v['peak']:.2f}" for v in vals),
                *(f"{pct(r[k]['rms'])} / {pct(r[k]['peak'])}" for k in ("LN","SN","SL")))
    def summary(w):
        return (LABELS[w],*(f"{median(a):.2f}% / {min(a):.2f}%" for a in
            ([ratios[t,w][k][field] for t in CASES] for k in ("SN","SL") for field in ("rms","peak"))))
    def effort(t,w,m):
        a=rows[t,m,w]
        fields=("requested_N","spring_N","linear_N","nonlinear_N","micro_q_rad","request_tau_Nm",
                "realized_increment_tau_Nm","actuator_total_tau_Nm","realized_increment_tau_rate_Nm_s")
        return (t,LABELS[w],m,*(f"{a[k]['rms']:.3g} / {a[k]['peak']:.3g}" for k in fields))
    h=["试次","窗口","N RMS / 峰值","L RMS / 峰值","S RMS / 峰值","L 比 N","S 比 N","S 比 L"]
    intro = """
<p class="lead"><strong>SFC 的启动和换向表现更好，普通段存在反例，出力也明显增加。</strong></p>
<ul>
<li>启动 RMS 相对 N 降低 23.71–25.08%，相对 L 降低 7.65–8.70%。换向相对 N 降低 24.10–28.45%，相对 L 降低 6.56–11.91%。三片启动和换向的峰值也均低于 L。</li>
<li>六个稳定窗口中有五个 RMS 低于 L。10 mm/s 稳定回程 RMS 比 L 高 4.75%，峰值高 26.36%。5 mm/s R01 稳定去程峰值比 L 高 9.67%。</li>
<li>S 的请求力 RMS 为 L 的 2.27–3.03 倍（启动和换向），稳定段为 3.71–4.62 倍。九次积分均完成，限幅饱和和保护中止均为零。</li>
<li>进入规划耦合前，下一项最值得验证的是等出力线性对照，并用 10 mm/s 回程核对峰值代价。当前数据适用于本轮同运动、同扰动的离线比较。</li>
</ul>"""
    text = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>G 组三片 N／线性／SFC 对照</title><style>
*{{box-sizing:border-box}}body{{margin:0;color:#19263b;background:#f0f4f8;font:16px/1.68 SimSun,"宋体","Times New Roman",serif}}
main{{max-width:1260px;margin:auto;background:white;padding:clamp(16px,3.4vw,50px);min-height:100vh}}
h1,h2,h3,strong,th{{font-family:SimHei,"黑体","Times New Roman",serif}}h1{{font-size:clamp(27px,4vw,43px);line-height:1.2;margin:.2em 0 .5em}}
h2{{border-top:1px solid #dbe2eb;padding-top:.8em;margin:1.4em 0 .5em;font-size:1.46rem}}.lead{{font-size:1.2em;margin-bottom:.35em}}
li{{margin:.22em 0}}.tag{{color:#42536c;font-size:.96em;overflow-wrap:anywhere}}a{{color:#1466ac}}figure{{margin:1em 0 1.5em}}
img{{display:block;max-width:100%;height:auto;margin:auto}}figcaption{{color:#4c5870;margin-top:.4em}}
.scroll{{overflow-x:auto;margin:.7em 0 1em}}table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}
td,th{{padding:.38em .65em;border-bottom:1px solid #e2e8f0;text-align:left;white-space:nowrap}}th{{background:#eaf0f7}}
tbody tr:nth-child(even){{background:#f8fafc}}code{{overflow-wrap:anywhere}}details{{margin:1em 0}}
</style><main><h1>G 组三片 N／线性／SFC 对照</h1>
<p class="tag">已归档 <code>G/baselines/G_CB3_A_reference_v1_20261116</code> · <code>{ex["archive_commit"]}</code> · 标签 <code>g-cb3-a-reference-v1-20261116</code>。远端或本机恢复状态见 <a href="delivery.json">交付回执</a>。</p>
{intro}
<h2>三片总览</h2>{fig(0,"图 1　独立仿真，拼合展示。缝隙为 0.55 s，只出现在展示轴；浅褐为启动，浅蓝为换向。")}
<p>N 为无新增抑振，L 为线性，S 为 SFC。误差单位为 μm，取物理 Y 减同一方向参考。表内每格为 RMS / 最大绝对偏差，小于零的降低率表示恶化。</p>
<h2>启动和换向</h2>{fig(1,"图 2　启动和换向，使用各片原时间轴，峰值未对齐。")}
{table(h,[line(t,w) for t in CASES for w in MAIN_WINDOWS[:2]])}
<h2>稳定去程和回程</h2>{fig(2,"图 3　稳定去程和回程，采用 A 来源数据预先划定的固定窗口。")}
{table(h,[line(t,w) for t in CASES for w in MAIN_WINDOWS[2:]])}
{table(["跨片比较","S 比 N RMS 中位/最差","S 比 N 峰值中位/最差","S 比 L RMS 中位/最差","S 比 L 峰值中位/最差"],[summary(w) for w in MAIN_WINDOWS])}
<details><summary>过渡、停止、原开头静止与整段诊断</summary>
{table(h,[line(t,w) for t in CASES for w in ("start_transition","turn_transition","stop","stop_transition","pre_static","post_static","full")])}</details>
<h2>原始物理 Y、方向参考与运动背景</h2>
{fig(3,"图 4　左为共同零点下物理 Y、A 有效视觉帧与方向参考，右为 X 位置背景。折线无时间平移。")}
{fig(4,"图 5　左为物理 X 速度与速度量级，右为横向请求力及 S 的非线性分项。X 速度只作为运动背景。")}
<p>物理模型各片从原初态连续积分。方向参考、公共物理表和 Fd 来自归档 N，三模式按试次共享。基准使用同一事件保护和 C² 过渡，过渡单列。反馈的位置取本次模型的模拟相机采样，速度取本次模型编码器误差差分的一阶 5 Hz 低通。各传感器按自身时刻减参考。相机帧在已就绪控制时刻的年龄为 4.00–11.52 ms，编码器为 0 ms。</p>
<h2>控制作用和代价</h2>
<p>1 ms 物理步长，8 ms 完整六轴目标保持，132 Hz 模拟相机，采样到可用采用原 4 ms 设置，执行附加延迟为零。力按雅可比转成力矩请求，再除以 Kp 加入目标；外加广义力通道只接收归档扰动力。微补偿和实现的增量力矩与含名义跟踪的总执行器力矩分别报告。</p>
{table(["试次","窗口","模式","请求 N","弹簧 N","线性 N","非线性 N","微补偿 rad","请求 Nm","实现增量 Nm","执行器总 Nm","增量变化率 Nm/s"],
 [effort(t,w,m) for t in CASES for w in MAIN_WINDOWS for m in "LS"])}
<p>关节量先取六轴向量范数，再取 RMS 与峰值。既有增量力矩变化率按相邻 1 ms 样本差分，仅在窗口内计算。此路径没有独立定义的位置、速度、总力矩及变化率平方积分代价，也没有预测后残余运动。“全部窗口和出力字段”见 <a href="metrics.json">评分数据</a>。</p>
<h2>归档、运行和复核</h2>
<p>源版本 <code>{ex["source_commit"]}</code>。三片各自的 N 均与归档的 87 个完整时间序列数组逐字节一致，最大绝对差异为 0。S 在三片分别有 969、958、914 个非零非线性控制时刻。三模式的原 1 ms、8 ms 和传感器原生数据已随包提供。</p>
<p>参数、复现及失败回执见 <a href="README.md">复现说明</a>、<a href="configuration_notes.md">设置差异</a>、<a href="execution.json">九次运行回执</a>和 <a href="delivery.json">交付回执</a>。</p>
<p>所用方向参考是离线校准资料。后续改变 S 曲线时，需要另行验证轨迹变化怎样改变实际激励。当前结果仅覆盖三片已校准轨迹。</p>
</main></html>"""
    (root/"G_NLS_segments_review.html").write_text(text,encoding="utf-8",newline="\n")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    a=p.parse_args()
    root=a.root
    trials={}
    for trial,rec in read(root/"windows_frozen.json")["trials"].items():
        d=root/"data"/trial
        t=load(d/"display_mapping.npz")
        trials[trial]=(rec,d,t["original_time_s"],t["display_time_s"],load(d/"shared.npz")["score_reference_um"],
                       {m:load(d/m/"physical_1ms.npz") for m in MODES},
                       {m:load(d/m/"command_native.npz") for m in MODES},
                       load(d/"A_source.npz"), load(d/"locked_input.npz"))
    describe(root,draw(root,trials))


if __name__=="__main__":
    main()
