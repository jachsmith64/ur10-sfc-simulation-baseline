"""Independently verify extracted review data, including FIT and sampling math."""
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

TRIALS=("X_P05_R01","X_P05_R02","X_P10_R01")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def data(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def subset(t,span):
    a,b=span
    return (a<=t)&((t<b)|((b==t[-1])&(t==b)))


def close(got, want, atol=1e-10):
    np.testing.assert_allclose(got,want,atol=atol,rtol=0)


def reference(t,windows,levels):
    shifts=np.array([levels[k] for k in ("pre_static","outbound","return_straight","post_static")])
    output=np.full(np.size(t),shifts[0])
    rate=np.zeros(np.size(t))
    for j,name in enumerate(("start_transition","turn_transition","stop_transition")):
        lo,hi=windows[name]
        s=np.clip((t-lo)/(hi-lo),0.,1.)
        output+=np.diff(shifts)[j]*(6*s**5-15*s**4+10*s**3)
        rate+=np.diff(shifts)[j]*30*s**2*(1-s)**2/(hi-lo)
    return output,rate


def exact_time(event,level,t,v):
    reported=event["time_s"]
    pair=event["bracket_s"]
    if event["unresolved_gap"]:
        assert reported is None and pair[1]-pair[0]>.04
        return None
    if pair is None:
        assert reported is None
        return None
    i=np.searchsorted(t,pair[0])
    assert t[i]==pair[0] and t[i+1]==pair[1] and pair[1]-pair[0]<=.04
    crossing=pair[0]+(pair[1]-pair[0])*(level-v[i])/(v[i+1]-v[i])
    close(crossing,reported,1e-11)
    return crossing


def verify(root):
    manifest=read(root/"SHA256SUMS.json")
    files={p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    assert files==set(manifest)|{"SHA256SUMS.json"}
    for k,v in manifest.items():
        b=(root/k).read_bytes()
        assert len(b)==v["bytes"] and sha256(b).hexdigest()==v["sha256"],k
    frozen=read(root/"windows_frozen.json")
    config=read(root/"preprocess_config.json")
    upstream=read(root/"inputs/upstream_errors.json")
    fit_count,mark_count=0,0
    receipt=read(root/"results/actual_execution.json")
    assert receipt["formal_control_runs"]==0
    for trial in TRIALS:
        i=root/"source_arrays"/trial
        path=root/"results"/trial
        s=read(path/"status.json")
        a=data(i/"A_source.npz")
        meta=read(i/"source.json")
        z=data(path/"review.npz")
        locked=data(path/"locked.npz")
        plan=frozen["trials"][trial]
        w=plan["windows_s"]
        assert sha256((i/"A_source.npz").read_bytes()).hexdigest()==plan["source_npz_sha256"]
        assert sha256((i/"source.json").read_bytes()).hexdigest()==plan["source_record_sha256"]
        t=z["t"]; f=a["frame_host_time_s"]
        assert np.array_equal(t,locked["t"])
        assert np.array_equal(locked["force_N"],z["force_N"])
        assert np.all(a["frame_valid"].astype(int)==sum((a["frame_valid"]&subset(f,q)).astype(int) for q in w.values()))
        assert all(np.count_nonzero(a["frame_valid"]&subset(f,span))==plan["source_statistics"][name]["valid_frames"]
                   for name,span in w.items())
        assert all(np.array_equal(locked[name],np.zeros_like(locked[name]))
                   for name in ("remaining","remaining_dot"))
        assert not any(z[name].any() for name in ("remaining_m","remaining_rate_m_s","slow_trend_um"))
        for key in ("dq","dqd","pressure"):
            assert sha256(locked[key].tobytes()).hexdigest()==upstream[trial][key]["sha256"]
        close(meta["raw_visual_zero_um"]+a["A_centered_um"],a["A_raw_um"])
        nominal=read(path/"run_configs"/s["best_case"]/"case.json")["nominal_y_world_m"]
        close((z["physical_xyz_m"][:,1]-nominal)*1e6,z["physical_y_um"])
        expected,rate=reference(t,w,plan["bases_um"])
        close(z["direction_reference_um"],expected)
        close(z["scoring_reference_um"],expected)
        close(z["reference_rate_um_s"],rate,1e-9)
        for name in ("start","turn","stop"):
            inside=subset(t,w[name])
            assert np.all(z["reference_rate_um_s"][inside]==0)
            assert np.ptp(z["common_command_y_m"][inside])==0
        for key in ("camera","encoder"):
            ts=z[f"{key}_sample"]/1000
            actual_ref=reference(ts,w,plan["bases_um"])[0]*1e-6
            close(z[f"{key}_reference_m"],actual_ref,1e-15)
            y=np.interp(ts,t,z["physical_y_um"])/1e6
            pressure=locked["pressure"][:len(ts)] if key=="encoder" else 0.
            error=y+pressure-actual_ref
            close(z[f"{key}_e"],error,1e-13)
            close(z[f"{key}_v"][1:],np.diff(error)/np.diff(ts),1e-12)
            assert np.all(z[f"{key}_available"]>=z[f"{key}_sample"])
        v=z["encoder_v"]
        cutoff=config["encoder_observer_cutoff_hz"]
        k=np.tan(np.pi*cutoff/125.)
        b,alpha=k/(1+k),(k-1)/(1+k)
        filt=np.zeros(len(v))
        filt[1]=v[1]
        for n in range(2,len(v)):
            filt[n]=b*(v[n]+v[n-1])-alpha*filt[n-1]
        close(z["encoder_vf"],filt,1e-15)
        tt=z["encoder_sample"][:-1]
        assert np.all(np.diff(tt)==8.)
        assert s["new_remaining_rms_um"]==0.
        assert np.any(z["old_remaining_um"])
        score_ref=reference(f,w,plan["bases_um"])[0]
        attempts=read(path/"attempts.json")
        for j,attempt in enumerate(attempts):
            physical=np.interp(f,t,z["attempt_y_um"][j])
            for label,metrics in attempt["scores"].items():
                keep=a["frame_valid"]&subset(f,([0.,f[-1]] if label=="full" else w[label]))
                src=a["A_centered_um"]
                zero=meta["raw_visual_zero_um"]
                for layer,at,mt in (("raw",a["A_raw_um"],physical+zero),
                                     ("processed",src-score_ref,physical-score_ref)):
                    av,mv=at[keep],mt[keep]
                    rmse=np.linalg.norm(av-mv)/np.sqrt(len(av))
                    spread=np.std(av)
                    fit=(1-rmse/spread)*100 if spread>1 else None
                    want=metrics[layer]
                    assert len(av)==want["samples"]
                    close(rmse,want["rmse_um"])
                    if fit is None:
                        assert want["fit"] is None
                    else:
                        close(fit,want["fit"],1e-8)
                    ai,mi=np.argmax(abs(av)),np.argmax(abs(mv))
                    close((av[ai],mv[mi],f[keep][mi]-f[keep][ai]),
                          (want["peak_A_um"],want["peak_N_um"],want["absolute_peak_delay_s"]))
                    fit_count+=1
        proof=s["timing"]["best_N"]
        ht=a["host_time_s"]
        speed=plan["speed_mm_s"]
        for key,fraction in (("start10",.1),("start50",.5),("start90",.9),("decel90",.9),
                             ("zero",0.),("return50",-.5),("return90",-.9),("stop90",-.9),("stop10",-.1)):
            source=exact_time(proof["source"][key],speed*fraction,ht,a["actual_tcp_speed"][:,0]*1000)
            model=exact_time(proof["candidate"][key],speed*fraction,t,z["physical_vx_m_s"]*1000)
            if source is not None and model is not None:
                close((model-source)*1000,proof["delay_ms"][key],1e-7)
                mark_count+=1
        lo=w["turn"][0]-.03
        hi=w["turn"][0]+.56
        section=(t>=lo)&(t<=hi)
        from_model=t[section][np.argmax(z["physical_xyz_m"][section,0])]
        on_a=(ht>=lo)&(ht<=hi)
        from_a=ht[on_a][np.argmax(a["actual_tcp_pose"][on_a,0])]
        close((from_model-from_a)*1000,proof["delay_ms"]["turn_xmax"])
        mark_count+=1
        keys=("start50","turn_xmax","return50")
        gate=all(proof["delay_ms"][k] is not None and abs(proof["delay_ms"][k])<=50 for k in keys)
        assert gate==proof["gate_50ms"]
        if trial==TRIALS[0]:
            assert not np.any(z["plus_delta_y_um"][:801]) and not np.any(z["minus_delta_y_um"][:801])
            assert z["plus_delta_y_um"][801]>0>z["minus_delta_y_um"][801]
            assert np.any(z["plus_micro_q_rad"]) and np.any(z["minus_micro_q_rad"])
    print(f"{len(manifest)} members verified. Recomputed {fit_count} FIT/RMSE/peak rows and {mark_count} paired X timings; "
          "reference, filtering, zero removal, legacy pressure/background and signed pulse passed.")


if __name__=="__main__":
    verify(Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent)
