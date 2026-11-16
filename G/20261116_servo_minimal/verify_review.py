"""Verify the unpacked archive, independently recomputing the paired comparison."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

TRIALS=("X_P05_R01","X_P05_R02","X_P10_R01")
ARMS=("original","nominal_velocity")


def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def sha(bs):
    return hashlib.sha256(bs).hexdigest()


def close(a,b,tol=1e-9):
    np.testing.assert_allclose(a,b,atol=tol,rtol=0)


def crossing(entry,t,v,f):
    a=entry["time_s"]
    bracket=entry.get("bracket_s")
    if a is None:
        if entry["unresolved_gap"]:
            assert bracket[1]-bracket[0]>.04
        return None
    i=np.searchsorted(t,bracket[0])
    assert t[i]==bracket[0] and t[i+1]==bracket[1] and bracket[1]-bracket[0]<=.04
    z=t[i]+(t[i+1]-t[i])*(f-v[i])/(v[i+1]-v[i])
    close(z,a,1e-12)
    return float(z)


def timing(trace,t,x,v,h,a_x,a_v,w,speed):
    out={}
    for name,scale in (("start10",.1),("start50",.5),("start90",.9),("decel90",.9),("zero",0),
                       ("return50",-.5),("return90",-.9),("stop90",-.9),("stop10",-.1)):
        ts=crossing(trace["source"][name],h,a_v,scale*speed)
        tm=crossing(trace["candidate"][name],t,v,scale*speed)
        out[name]=(ts,tm)
        delay=trace["delay_ms"][name]
        if ts is None or tm is None:
            assert delay is None
        else:
            close(1000*(tm-ts),delay)
    lo,hi=w["turn"][0]-.03,w["turn"][0]+.56
    mm=(t>=lo)&(t<=hi)
    am=(h>=lo)&(h<=hi)
    ts=h[am][np.argmax(a_x[am])]
    tm=t[mm][np.argmax(x[mm])]
    out["turn_xmax"]=(float(ts),float(tm))
    close(1000*(tm-ts),trace["delay_ms"]["turn_xmax"])
    durations={}
    for group,left,right in (("rise_10_90","start10","start90"),("reversal_90_90","decel90","return90")):
        dur=[]
        for side,tag in enumerate(("A_ms","candidate_ms")):
            p,q=out[left][side],out[right][side]
            expected=(q-p)*1000 if p is not None and q is not None and q>=p else None
            if expected is None:
                assert trace["durations"][group][tag] is None
            else:
                close(expected,trace["durations"][group][tag])
            dur.append(expected)
        durations[group]=dur
    return out,durations


def verify(root):
    manifest=read(root/"SHA256SUMS.json")
    present={p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    assert present==set(manifest)|{"SHA256SUMS.json"}
    for key,attr in manifest.items():
        bs=(root/key).read_bytes()
        assert len(bs)==attr["bytes"] and sha(bs)==attr["sha256"],key
    windows=read(root/"windows_frozen.json")
    config=read(root/"servo_config.json")
    receipt=read(root/"outputs/receipt.json")
    assert sha((root/"windows_frozen.json").read_bytes())==receipt["inherited_window_sha256"]
    assert sha((root/"servo_config.json").read_bytes())==receipt["config_sha256"]
    assert sha((root/"velocity_source.json").read_bytes())==receipt["source_check_sha256"]
    assert receipt["formal_control_runs"]==0 and receipt["physical_y_reconstruction"]=="not started"
    count=0
    completed=[]
    for trial in TRIALS:
        folder=root/"outputs"/trial
        rec=windows["trials"][trial]
        upstream=root/"source_arrays"/trial/"A_source.npz"
        assert sha(upstream.read_bytes())==rec["source_npz_sha256"]
        with np.load(upstream,allow_pickle=False) as a:
            h=a["host_time_s"]; x_a=a["actual_tcp_pose"][:,0]; v_a=a["actual_tcp_speed"][:,0]*1000
        s=read(folder/"status.json")
        assert s["regression"]["all_selected_exact"] and all(s["regression"]["compared"].values())
        with np.load(folder/"comparison.npz",allow_pickle=False) as raw:
            z={key:raw[key] for key in raw.files}
        t=z["t"]; host=z["knot_t_s"]; ik=z["knot_q_rad"]; q=z["q_nom"]; ticks=z["tick_index"]
        assert sha(q.tobytes())==s["nominal_q_sha256"]
        assert sha(z["dq"].tobytes())==s["dq_sha256"] and sha(z["pressure"].tobytes())==s["pressure_sha256"]
        close(q,np.column_stack([np.interp(t,host,v) for v in ik.T]),1e-14)
        slopes=(ik[1:]-ik[:-1])/(host[1:,None]-host[:-1,None])
        ids=np.searchsorted(host,t,side="right")-1
        ids[t==host[-1]]=len(slopes)-1
        assert np.array_equal(z["interval_index"],ids)
        close(z["qdot_nom_rad_s"],slopes[ids],1e-15)
        assert np.array_equal(ticks,np.arange(0,len(t),8))
        for arm,ref,condition in zip(ARMS,("original","candidate"),(0,1),strict=True):
            prefix=arm+"_"
            result=s[ref]
            assert result["status"]=="OK" and result["guard_clear"]
            assert read(folder/arm/"checks.json")["held_command_exact"]
            assert read(folder/arm/"guard.json") is None
            assert np.array_equal(z[prefix+"t"],t)
            assert np.array_equal(np.flatnonzero(z[prefix+"updated"]),ticks)
            assert not any(z[prefix+name].any() for name in ("actual_micro_q","Fd","remaining","remaining_dot","qfrc_applied",
                "ctrl_saturation","joint_saturation","force_saturation"))
            gain=np.array(s["compiled"]["Kp_Nm_per_rad"])
            damping=np.array(s["compiled"]["Kd_Nms_per_rad"])
            command=z[prefix+"final_ctrl_at_ticks"]
            v_nom=condition*slopes[ids[ticks]]
            add=v_nom*damping/gain
            close(z[prefix+"qdot_nom_command_rad_s_at_ticks"],v_nom,1e-15)
            close(z[prefix+"q_velocity_ff_rad_at_ticks"],add,1e-15)
            close(command,q[ticks]+z["dq"][ticks]+add,1e-15)
            close(command,z[prefix+"base_ctrl_at_ticks"],1e-15)
            close(gain*(command-z[prefix+"q_at_ticks"])-damping*z[prefix+"qdot_at_ticks"],
                  z[prefix+"qfrc_actuator_at_ticks"],1e-9)
            x=z[prefix+"physical_xyz_m"][:,0]
            v=z[prefix+"vx_m_s"]*1000
            marks,length=timing(result["axis"],t,x,v,h,x_a,v_a,rec["windows_s"],rec["speed_mm_s"])
            count+=sum(a is not None and b is not None for a,b in marks.values())
            shape={}
            for name in ("start","turn","outbound","return_straight","stop"):
                lo,hi=rec["windows_s"][name]
                keep=(h>=lo)&(h<hi)
                matched=h[keep]
                error_v=np.interp(matched,t,v)-v_a[keep]
                shape[name]=np.linalg.norm(error_v)/np.sqrt(len(error_v))
                close(result["axis"]["shape"][name]["v_RMSE_mm_s"],shape[name])
                if name in ("outbound","return_straight"):
                    es=np.interp(matched,t,x)-np.interp(matched,t,z["nom_x_m"])
                    sign=1 if name=="outbound" else -1
                    scale=-1e6*sign*np.median(es)/rec["speed_mm_s"]
                    close(scale,result["stable"][name]["nominal_tracking_scale_ms"])
                    e=np.interp(matched,t,x)-x_a[keep]
                    close(np.linalg.norm(e)*1000/np.sqrt(len(e)),result["stable"][name]["actual_x_RMSE_mm"])
            peaks={}
            for ev in ("start","turn","stop"):
                lo,hi=rec["windows_s"][ev]
                ref_peak=np.abs(v_a[(h>=lo)&(h<hi)]).max()
                act_peak=np.abs(v[(t>=lo)&(t<hi)]).max()
                peaks[ev]=max(0.,act_peak-ref_peak)
                close(peaks[ev],result["overshoot"]["speed"][ev]["excess_over_A_mm_s"])
            spaces=np.maximum(0,1000*np.array([x.max()-x_a.max(),x_a.min()-x.min()]))
            close(spaces,[result["overshoot"]["far_X_excess_over_A_mm"],result["overshoot"]["near_X_excess_over_A_mm"]])
            timing_ok=all(abs((marks[key][1]-marks[key][0])*1000)<=config["episode_gate_delay_ms"]
                          for key in ("start50","turn_xmax","return50"))
            if not condition:
                first=length,shape,peaks,spaces
            else:
                prev,val,early,exc=first
                duration_good=all(abs(b-a)<abs(prev[key][1]-prev[key][0]) for key,(a,b) in length.items())
                shape_good=all(shape[key]<val[key] for key in ("start","turn"))
                overshoot_good=(np.max(spaces-exc)<=config["maximum_new_spatial_overshoot_mm"] and
                    max(peaks[key]-early[key] for key in peaks)<=rec["speed_mm_s"]*config["maximum_new_speed_overshoot_fraction_nominal"])
                checks=dict(timing_pass=timing_ok,duration_pass=duration_good,
                            shape_pass=shape_good,overshoot_pass=bool(overshoot_good),protection_pass=True)
                assert all(v==s["gate"][k] for k,v in checks.items())
                completed.append(all(checks.values()))
    assert completed==[False,True,False] and receipt["x_pass"]==all(completed)
    print(f"{len(manifest)} hashes, {count} paired event markers, 6 command and actuator comparisons verified. Combined X gate failed.")


if __name__=="__main__":
    verify(Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent)
