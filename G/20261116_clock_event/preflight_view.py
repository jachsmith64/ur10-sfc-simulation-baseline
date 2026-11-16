"""Make the source-only check figure before dynamic outputs are accessible."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from robust_events import detect, spans

ROOT=Path(__file__).resolve().parent


def main(prepared):
    fig, axes=plt.subplots(3,2,figsize=(13,9),layout="constrained")
    metrics={}
    derived_events={}
    rules=json.loads((ROOT/"clock_event_config.json").read_text(encoding="utf8"))
    windows=json.loads((ROOT.parent/"20261116_preprocess_timing/windows_frozen.json").read_text(encoding="utf8"))["trials"]
    trials=("X_P05_R01","X_P05_R02","X_P10_R01")
    for trial,row in zip(trials,axes,strict=True):
        source=ROOT.parent/"20261116_A_multispeed/prepared_A"/trial/"A_source.npz"
        new=prepared/trial/"input.npz"
        with np.load(source,allow_pickle=False) as a, np.load(new,allow_pickle=False) as b:
            t=a["host_time_s"]; x=a["target_tcp_pose"][:,0]
            m=b["target_mapped_host_s"]; q=b["derived_target_tcp_pose"][:,0]
            origin=x[0]
            row[0].plot(t,(x-origin)*1000,color="#aa5522",lw=.8,label="Legacy host target")
            row[0].plot(m,(q-origin)*1000,color="#245583",lw=.8,ls="--",label="Cabinet-mapped target")
            row[1].step(t[:-1],np.diff(x)/np.diff(t)*1000,where="post",lw=.7,color="#aa5522",label="Legacy slope")
            row[1].step(m[:-1],np.diff(q)/np.diff(m)*1000,where="post",lw=.9,color="#245583",label="Mapped slope")
            f=json.loads((prepared/trial/"input.json").read_text(encoding="utf8"))
            grid=b["t"]
            sample_index=np.minimum(np.searchsorted(m,grid,side="right")-1,len(q)-2)
            target_velocity=(np.diff(q)/np.diff(m))[sample_index]*1000
            mark,_,_=detect(grid,b["nom_x_m"],target_velocity,windows[trial],rules["events"],rules["clock"]["long_gap_s"])
            derived_events[trial]={"markers":mark,"spans_ms":spans(mark),
                "scope":"command grid explicitly includes the bounded nominal hold; actual RTDE still has missing samples"}
            for k in ("legacy_nominal","mapped_nominal"):
                metrics[trial+"/"+k]={"span_x_mm":f[k]["span_x_mm"],"x_maximum_s":f[k]["x_maximum_s"],
                                     "spans_ms":f[k]["source_velocity_on_knot_rule"]["spans_ms"],
                                     "stable":f[k]["stable"]}
        for ax,name,unit in zip(row,("X target position","X target velocity"),("X [mm]","Vx [mm/s]"),strict=True):
            ax.set_title(trial+"   "+name,loc="left",fontweight="bold")
            ax.set(xlabel="Common host time [s]",ylabel=unit)
            ax.grid(alpha=.22); ax.legend(loc="best",fontsize=8)
    (prepared/"derived_command_event_check.json").write_text(json.dumps(derived_events,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    pic=prepared/"input_overlay.png"
    fig.savefig(pic,dpi=113)
    ids={k:sha256((ROOT/k).read_bytes()).hexdigest() for k in ("clock_map.py","robust_events.py","run_clock_event.py","clock_event_config.json")}
    (prepared/"preflight_plot_and_rules.json").write_text(json.dumps(dict(source_only=True,
       png_sha256=sha256(pic.read_bytes()).hexdigest(),algorithm_sha256=ids,nominal_checks=metrics),
       ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    for k,d in metrics.items():
        print(k,"span",round(d["span_x_mm"],5),"events",d["spans_ms"], "spikes", [j["exceeding_15_mm_s"] for j in d["stable"].values()])


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("prepared",type=Path)
    main(p.parse_args().prepared)
