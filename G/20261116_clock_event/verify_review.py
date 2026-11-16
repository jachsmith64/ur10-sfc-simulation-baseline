"""Independent unpacked-review verification from raw timestamps and physical traces."""
import base64
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

import numpy as np

TRIALS=("X_P05_R01","X_P05_R02","X_P10_R01")
ARMS=("velocity_off","velocity_on")
MARKERS=("start50","turn_xmax","return50")


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def arrays(path):
    with np.load(path,allow_pickle=False) as b:
        return {k:b[k] for k in b.files}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def same(a,b,tol=1e-10):
    np.testing.assert_allclose(a,b,rtol=0,atol=tol,equal_nan=True)


def check_clock(a,d,k,policy):
    h=a["host_time_s"]
    c=a["cabinet_time_s"]
    M=k["all_mapped_host_s"]
    stamps, n=np.unique(c,return_counts=True)
    singles=stamps[n==1]
    flag=np.isin(c,singles)
    offset=np.median(h[flag]-(c[flag]-c[0]))
    same(M,(c-c[0])+offset)
    same(offset,d["offset_s"])
    same(np.percentile((h[flag]-M[flag])*1e3,[0,5,25,50,75,95,100]),d["residual_quantile_ms"])
    idx=k["target_index"]; obs=k["actual_index"]
    same(k["target_mapped_host_s"],M[idx])
    same(k["actual_mapped_host_s"],M[obs])
    same(k["actual_original_host_s"],h[obs])
    decisions=d["decisions"]
    assert [z["index"] for z in decisions]==list(range(len(c)))
    for tag,used,fields in (("target",idx,a["target_tcp_pose"]),
                           ("actual",obs,np.c_[a["actual_tcp_pose"],a["actual_tcp_speed"]])):
        kept=set(used.tolist())
        for stamp in stamps[n>1]:
            indices=np.flatnonzero(c==stamp)
            conflict=not np.all(fields[indices]==fields[indices[0]])
            if conflict:
                assert not any(i in kept for i in indices)
                assert all(decisions[i][tag+"_action"]=="excluded_conflicting_timestamp" for i in indices)
            else:
                assert any(decisions[i][tag+"_action"].startswith("merged") for i in indices)
    v=k["derived_target_tcp_pose"].copy()
    holds={g["right_source_index"]:g for g in d["long_target_gaps"]}
    for i,source in enumerate(idx):
        if int(source) in holds:
            g=holds[int(source)]
            assert g["accepted_bounded_target_hold"] and g["actual_capture_gap_not_filled"]
            assert g["observed_tip_displacement_um"] <= 2 and not g["intersects_critical"]
            same(v[i],v[i-1])
        else:
            np.testing.assert_array_equal(v[i],a["target_tcp_pose"][source])
    raw_ixs=d["field_consistency_exclusions"]
    for e in raw_ixs:
        center=e["excluded_source_index"]
        lo,hi=e["trusted_source_endpoints"]
        assert center not in idx and M[hi]-M[lo]<=policy["long_gap_s"]+1e-9
        x=a["target_tcp_pose"][:,0]
        left=(x[center]-x[lo])/(M[center]-M[lo])*1000
        right=(x[hi]-x[center])/(M[hi]-M[center])*1000
        same([left,right],e["interior_velocities_mm_s"])
    return obs


def recorded_detector(t,v,g,observed,policy,max_gap):
    # Independently reconstruct the detector waveform with an elementary sliding median.
    expected_grid=np.arange(int(np.ceil(t[0]*1000)),int(np.floor(t[-1]*1000))+1)/1000
    same(expected_grid,g,1e-14)
    # Use the published grid for raw bracket edge equality, just as the rule does.
    i=np.searchsorted(t,g,side="right")
    l=np.clip(i-1,0,len(t)-1); r=np.clip(i,0,len(t)-1)
    good=(t[r]-t[l]<=max_gap+1e-9)|(g==t[l])
    y=np.interp(g,t,v)
    generated=np.full_like(g,np.nan)
    ix=np.flatnonzero(good)
    half=int(round(policy["median_half_width_s"]/policy["grid_s"]))
    for seg in np.split(ix,np.flatnonzero(np.diff(ix)!=1)+1):
        if not len(seg):
            continue
        a,b=int(seg[0]),int(seg[-1])
        lo=a if a==0 else a+half
        hi=b if b==len(g)-1 else b-half
        for k in range(lo,hi+1):
            generated[k]=np.median(y[np.clip(np.arange(k-half,k+half+1),a,b)])
    same(generated,observed)
    return generated


def event_times(t,x,v,g,f,record,settings,spec):
    y=recorded_detector(t,v,g,f,settings,.04)
    speed=record["speed_mm_s"]
    check={"start10":.1,"start50":.5,"start80":.8,"start90":.9,"decel50":.5,"decel80":.8,
           "decel90":.9,"zero":0.,"return50":-.5,"return80":-.8,"return90":-.9}
    result={}
    for name,rel in check.items():
        e=spec["events"][name]
        mark=e["time_s"]
        assert mark is not None
        p,q=e["bracket_s"]
        idx=np.searchsorted(g,p)
        same(g[idx:idx+2],[p,q],1e-13)
        a,b=y[idx:idx+2]
        target=rel*speed
        got=p+(q-p)*(target-a)/(b-a)
        same(got,mark)
        if name.startswith("decel"):
            zero=spec["events"]["zero"]["time_s"]
            rest=y[(g>=got)&(g<=zero)]
            assert np.all(rest<=target+speed*settings["persistence_hysteresis_fraction"])
            later=(g[1:]<zero)&(g[:-1]>got)&(y[:-1]>target)&(y[1:]<=target)
            assert not np.any(later)
        result[name]=got
    anchor=record["windows_s"]["turn"][0]
    m=(t>=anchor-.03)&(t<=anchor+.56)
    result["turn_xmax"]=t[m][np.argmax(x[m])]
    same(result["turn_xmax"],spec["events"]["turn_xmax"]["time_s"])
    span={}
    for k,begin,end in (("rise_10_90","start10","start90"),("reversal_90_90","decel90","return90"),
                        ("reversal_50_50","decel50","return50"),("reversal_80_80","decel80","return80")):
        span[k]=(result[end]-result[begin])*1e3
        same(span[k],spec["spans_ms"][k])
    return result,span


def raw_scores(t,x,v,h,xa,va,w,result):
    score={}
    for name in ("start","turn","outbound","return_straight","stop"):
        lo,hi=w[name]
        m=(h>=lo)&(h<hi)&(h>=t[0])&(h<=t[-1])
        rx=np.interp(h[m],t,x)-xa[m]
        rv=np.interp(h[m],t,v)-va[m]
        score[name]=np.linalg.norm(rv)/np.sqrt(len(rv))
        same(score[name],result["axis"]["shape"][name]["v_RMSE_mm_s"])
        same(np.linalg.norm(rx)/np.sqrt(len(rx))*1e6,result["axis"]["shape"][name]["x_RMSE_um"])
    peaks={}
    for name in ("start","turn","stop"):
        lo,hi=w[name]
        am=(h>=lo)&(h<hi); bm=(t>=lo)&(t<hi)
        peaks[name]=max(0.,np.max(abs(v[bm]))-np.max(abs(va[am])))
        same(peaks[name],result["overshoot"]["speed"][name]["excess_over_A_mm_s"])
    distance=[max(0.,(x.max()-xa.max())*1e3),max(0.,(xa.min()-x.min())*1e3)]
    same(distance,[result["overshoot"][k] for k in ("far_X_excess_over_A_mm","near_X_excess_over_A_mm")])
    return score,peaks,np.array(distance)


def verify(root):
    manifest=load(root/"SHA256SUMS.json")
    members={p.relative_to(root).as_posix():p for p in root.rglob("*") if p.is_file()}
    assert members.keys()==manifest.keys()|{"SHA256SUMS.json"}
    for name,info in manifest.items():
        p=members[name]
        assert p.stat().st_size==info["bytes"] and digest(p)==info["sha256"],name
    settings=load(root/"clock_event_config.json")
    windows=load(root/"windows_frozen.json")
    locked=load(root/"results/lock.json")
    proof=load(root/"results/preflight_plot_and_rules.json")
    assert locked["config_sha256"]==digest(root/"clock_event_config.json")
    assert locked["windows_sha256"]==digest(root/"windows_frozen.json")
    for name,hash_ in proof["algorithm_sha256"].items():
        assert digest(root/name)==hash_, name
    htm=(root/"G_clock_event_review.html").read_text(encoding="utf-8")
    image=re.findall(r'data:image/png;base64,([^"]+)',htm)[1]
    assert sha256(base64.b64decode(image,validate=True)).hexdigest()==proof["png_sha256"]
    outcomes=[]
    n_marks=0
    for trial in TRIALS:
        path=root/"results"/trial
        record=windows["trials"][trial]
        source=root/"source_arrays"/trial/"A_source.npz"
        assert digest(source)==record["source_npz_sha256"]
        assert digest(path/"input.npz")==locked["input_hashes"][trial]
        a,z,b=arrays(source),arrays(path/"input.npz"),arrays(path/"comparison.npz")
        status=load(path/"status.json")
        inputs=load(path/"input.json")
        mapped=load(path/"clock_decisions.json")
        obs=check_clock(a,mapped,z,settings["clock"])
        h=z["actual_mapped_host_s"]; xa=a["actual_tcp_pose"][obs,0]; va=a["actual_tcp_speed"][obs,0]*1e3
        t,kt,kq=z["t"],z["knot_t_s"],z["knot_q_rad"]
        ix=np.minimum(np.searchsorted(kt,t,side="right")-1,len(kt)-2)
        sl=np.diff(kq,axis=0)/np.diff(kt)[:,None]
        same(z["qdot_nom_rad_s"],sl[ix],1e-16)
        same(z["q_nom"],np.column_stack([np.interp(t,kt,x) for x in kq.T]),1e-15)
        same(np.sum(np.diff(kt)[:,None]*sl,axis=0),kq[-1]-kq[0],1e-13)
        tick=z["tick_index"]
        assert np.array_equal(tick,np.arange(0,len(t),8))
        nominal_delta=np.diff(z["derived_target_tcp_pose"][:,0])*1e3
        native_time=z["target_mapped_host_s"]
        native_speed=nominal_delta/np.diff(native_time)
        same(np.max(abs(native_speed)),inputs["mapped_nominal"]["max_nominal_abs_vx_mm_s"])
        assert inputs["input_pass"] and locked["input_gates"][trial]
        assert np.max(abs(native_speed))<=record["speed_mm_s"]*settings["gates"]["nominal_max_physical_speed_fraction"]
        if trial=="X_P10_R01":
            for name,earlier in (("outbound",17),("return_straight",12)):
                lo,hi=record["windows_s"][name]
                rh=a["host_time_s"]
                raw_speed=np.diff(a["target_tcp_pose"][:,0])*1e3/np.diff(rh)
                assert int(np.count_nonzero((rh[:-1]>=lo)&(rh[1:]<hi)&(abs(raw_speed)>15)))==earlier
                assert not np.any((native_time[:-1]>=lo)&(native_time[1:]<hi)&(abs(native_speed)>15))
            assert len(mapped["field_consistency_exclusions"])==3 and np.max(xa)==np.max(a["actual_tcp_pose"][:,0])
        source_ev,source_len=event_times(h,xa,va,z["source_detector_t_s"],z["source_detector_v_mm_s"],
                                        record,settings["events"],inputs["mapped_actual"])
        axis=[]
        for condition,arm in enumerate(ARMS):
            p=arm+"_"
            qff=b[p+"q_velocity_ff_rad_at_ticks"]
            velocity=z["qdot_nom_rad_s"][tick]*condition
            same(qff,velocity*.1,1e-16)
            same(b[p+"qdot_nom_command_rad_s_at_ticks"],velocity,1e-16)
            ctrl=b[p+"final_ctrl_at_ticks"]
            same(ctrl,(z["q_nom"]+z["dq"])[tick]+qff,1e-15)
            same(5000*(ctrl-b[p+"q_at_ticks"])-500*b[p+"qdot_at_ticks"],b[p+"qfrc_actuator_at_ticks"],1e-9)
            assert load(path/arm/"checks.json")["held_command_exact"] and load(path/arm/"guard.json") is None
            assert not any(b[p+k].any() for k in ("Fd","qfrc_applied","actual_micro_q","remaining","remaining_dot",
                           "ctrl_saturation","joint_saturation","force_saturation"))
            assert np.array_equal(np.flatnonzero(b[p+"updated"]),tick)
            x=b[p+"physical_xyz_m"][:,0]; v=b[p+"vx_m_s"]*1000
            actual=status["conditions"][arm]
            ev,durations=event_times(t,x,v,b[p+"detector_t_s"],b[p+"detector_v_mm_s"],
                                     record,settings["events"],actual["robust"])
            raw=raw_scores(t,x,v,h,xa,va,record["windows_s"],actual)
            axis.append((ev,durations,*raw))
            n_marks+=len(ev)
        ev0,du0,v0,pk0,x0=axis[0]; ev1,du1,v1,pk1,x1=axis[1]
        gates=settings["gates"]
        amounts={name:(ev1[name]-source_ev[name])*1000 for name in MARKERS}
        for name,x in amounts.items():
            same(x,status["gate"]["markers"][name]["delay_ms"])
        durs=all(abs(du1[k]-source_len[k])<abs(du0[k]-source_len[k]) for k in ("rise_10_90","reversal_90_90"))
        v_improve=all(v1[key]<v0[key] for key in ("start","turn"))
        stable=all(v1[k]-v0[k]<=gates["max_added_stable_speed_rmse_mm_s"] for k in ("outbound","return_straight"))
        peak=(max(pk1[k]-pk0[k] for k in pk0)<=gates["max_new_speed_peak_fraction"]*record["speed_mm_s"] and
              np.max(x1-x0)<=gates["max_new_axial_endpoint_excess_mm"])
        test=[all(abs(x)<=gates["main_marker_ms"] for x in amounts.values()),durs,v_improve,stable,bool(peak),True]
        for key,val in zip(("timing_pass","durations_pass","shape_pass","stable_shape_pass","overshoot_pass","protection_pass"),test,strict=True):
            assert status["gate"][key]==val
        outcomes.append(all(test))
    assert outcomes==[True,True,False]
    receipt=load(root/"results/paired_receipt.json")
    assert not receipt["x_pass"] and receipt["n_refit"]=="not started" and receipt["formal_L_SFC"]==0
    print(f"{len(manifest)} hashes, {n_marks} physical-event evaluations, six command/torque pairs, raw shape and the failed P10 overshoot gate verified.")


if __name__=="__main__":
    verify(Path(sys.argv[1]).resolve())
