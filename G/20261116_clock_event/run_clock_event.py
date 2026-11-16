"""Prepare fixed-clock replay knots, then run the inherited zero-force X comparison."""
import argparse
from hashlib import sha256
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parent
SERVO=ROOT.parent/"20261116_servo_minimal"
PREVIOUS=ROOT.parent/"20261116_preprocess_timing"
sys.path.insert(0,str(SERVO))
import run_servo as servo

from clock_map import map_clocks
from robust_events import detect, spans
from increment_model import reconstruct

prior,old,core=servo.prior,servo.old,servo.core
CONFIG=core.read(ROOT/"clock_event_config.json")
POLICY=CONFIG["events"]
GATES=CONFIG["gates"]
CLOCK=CONFIG["clock"]
TRIALS=prior.TRIALS
PAST_ROOT=SERVO/"outputs"
WINDOW_PATH=PREVIOUS/"windows_frozen.json"
CASE=core.read(old.PROJECT/"evidence/D_N_o1_fraw_r1_gain1_seed1001_d0.json")|dict(
    id="N_physical",estimator="Q",fc=5,rho=0.,filter_order=1)


def write(path,obj):
    core.write(path,core.clean(obj))


def source_data(trial,record):
    path=old.PROJECT/"prepared_A"/trial/"A_source.npz"
    if core.sha(path)!=record["source_npz_sha256"]:
        raise ValueError("Prepared raw source hash mismatch")
    return path


def series(src,clock_arrays):
    i=clock_arrays["actual_index"]
    return dict(host_time_s=clock_arrays["actual_mapped_host_s"],
                actual_tcp_pose=src["actual_tcp_pose"][i],
                actual_tcp_speed=src["actual_tcp_speed"][i])


def analyze(t,x,v,record):
    events,g,f=detect(t,x,v,record,POLICY,CLOCK["long_gap_s"])
    native={}
    for key,item in events.items():
        a=item["time_s"]
        if a is None:
            native[key]=None
            continue
        p=np.searchsorted(t,a)
        native[key]=t[max(0,p-1):min(len(t),p+1)].tolist()
    return dict(events=events,spans_ms=spans(events),native_brackets_s=native),g,f


def analyze_src(src,record):
    return analyze(src["host_time_s"],src["actual_tcp_pose"][:,0],src["actual_tcp_speed"][:,0]*1000,record)


def nominal_record(t,pose,record):
    rates=np.diff(pose[:,0])/np.diff(t)*1000
    stable={}
    for key,sign in (("outbound",1),("return_straight",-1)):
        lo,hi=record["windows_s"][key]
        valid=(t[:-1]>=lo)&(t[1:]<hi)
        vx=rates[valid]
        stable[key]=dict(count=len(vx),min=float(vx.min()),median=float(np.median(vx)),max=float(vx.max()),
                         exceeding_15_mm_s=int(np.count_nonzero(abs(vx)>15)),
                         max_error_mm_s=float(np.max(abs(vx-sign*record["speed_mm_s"]))))
    candidate,_,_=analyze(t,pose[:,0],np.r_[rates,rates[-1]],record)
    return dict(stable=stable,max_nominal_abs_vx_mm_s=float(np.max(abs(rates))),source_velocity_on_knot_rule=candidate,first_x_mm=pose[0,0]*1000,
                last_x_mm=pose[-1,0]*1000,span_x_mm=np.ptp(pose[:,0])*1000,x_maximum_s=float(t[np.argmax(pose[:,0])]))


def replace_nominal(ctx,q,nom_xyz):
    c=ctx["c"]|dict(q=q,along=(nom_xyz[:,0]-nom_xyz[0,0])*1000)
    old.core.aconfig.Y_NOMINAL=float(nom_xyz[0,1])
    return ctx|dict(c=c,scaled=ctx["scaled"]|c,nominal_xyz=nom_xyz,input_id=servo.array_id(q))


def spike_fix_example(src):
    h=src["host_time_s"]
    i=int(np.argmin(abs(h-2.272432)))
    x=src["target_tcp_pose"][:,0]
    c=src["cabinet_time_s"]
    dx=x[i+1]-x[i]
    assert abs(h[i]-2.272432)<1e-5 and dx>.00015
    return dict(row=i,old_host_s=float(h[i]),dx_um=float(dx*1e6),host_dt_ms=float((h[i+1]-h[i])*1000),
        cabinet_dt_ms=float((c[i+1]-c[i])*1000),host_speed_mm_s=float(dx/(h[i+1]-h[i])*1000),
        cabinet_speed_mm_s=float(dx/(c[i+1]-c[i])*1000))


def prepare_trial(dest,trial,record,frozen):
    dest.mkdir()
    source=source_data(trial,record)
    src,legacy_kin,_,ctx,initial_q,_=servo.case_inputs(trial,record,source,frozen)
    clock,clock_meta=map_clocks(src,record,CLOCK)
    t=ctx["c"]["t"]
    mapped=series(src,clock)
    actual_events,detect_t,detect_v=analyze_src(mapped,record)
    legacy_events=prior.axial.events(src["host_time_s"],src["actual_tcp_pose"][:,0],
                                     src["actual_tcp_speed"][:,0]*1000,record)
    old_nominal=nominal_record(src["host_time_s"],src["target_tcp_pose"],record)
    new_nominal=nominal_record(clock["target_mapped_host_s"],clock["derived_target_tcp_pose"],record)
    stable_valid=all(x["max_error_mm_s"]<=GATES["nominal_stable_speed_error_fraction"]*record["speed_mm_s"]
                     for x in new_nominal["stable"].values())
    required=("start10","start50","start90","decel90","zero","return50","return90")
    speed_clean=new_nominal["max_nominal_abs_vx_mm_s"] <= record["speed_mm_s"]*GATES["nominal_max_physical_speed_fraction"]
    input_pass=(not clock_meta["blocking_gaps"] and stable_valid and speed_clean and
                all(actual_events["events"][k]["time_s"] is not None for k in required))
    write(dest/"clock_decisions.json",clock_meta)
    write(dest/"input.json",dict(input_pass=input_pass,legacy_nominal=old_nominal,mapped_nominal=new_nominal,
            legacy_actual_events=legacy_events,mapped_actual=actual_events,spike_example=spike_fix_example(src) if trial=="X_P10_R01" else None,
            source_hash=core.sha(source),observed_actual_stop_gaps=clock_meta["actual_gaps"],
            frozen_window_sha256=core.sha(WINDOW_PATH),locked_rule_sha256=core.sha(ROOT/"clock_event_config.json")))
    if not input_pass:
        return False
    # Reuse the inherited six-dimensional IK. Original actual_q is supplied solely as the initial inverse seed.
    idx=clock["target_index"]
    target_clock=clock["target_mapped_host_s"]
    conditioned=dict(host_time_s=target_clock,cabinet_time_s=src["cabinet_time_s"][idx],
                     target_tcp_pose=clock["derived_target_tcp_pose"],actual_q_rad=src["actual_q_rad"][idx])
    cfg=old.CB3|dict(max_source_host_bracket_ms=float(np.max(np.diff(target_clock))*1000+1e-6),
          max_source_cabinet_gap_ms=float(np.max(np.diff(conditioned["cabinet_time_s"]))*1000+1e-6))
    kin,ik=reconstruct(cfg,conditioned,t,gravity_support=False)
    q=kin["nominal_q_rad"]
    xyz=kin["nominal_tcp_m_rotvec_rad"][:,:3]
    new_ctx=replace_nominal(ctx,q,xyz)
    vel,derivative,packed=servo.derivative_receipt(kin,new_ctx,CASE)
    derivative["method"]=CONFIG["clock"]["offset"]+"; analytic slope of the same re-timed linear q knots"
    recovered=np.diff(packed["knot_t_s"])[:,None]*(np.diff(packed["knot_q_rad"],axis=0)/np.diff(packed["knot_t_s"])[:,None])
    displacement=np.diff(packed["knot_q_rad"],axis=0)
    np.testing.assert_allclose(recovered,displacement,atol=1e-18,rtol=0)
    np.testing.assert_allclose(np.sum(recovered,axis=0),packed["knot_q_rad"][-1]-packed["knot_q_rad"][0],atol=1e-13,rtol=0)
    if np.max(abs(packed["jacobian_twist_at_ticks"]-packed["target_twist_at_ticks"]))>1e-5:
        raise ValueError("Six-dimensional velocity back-substitution is inconsistent")
    fields=packed|clock|dict(initial_q_rad=initial_q,nominal_xyz_m=xyz,
              source_detector_t_s=detect_t,source_detector_v_mm_s=detect_v,
              previous_nominal_x_m=legacy_kin["nominal_tcp_m_rotvec_rad"][:,0])
    np.savez_compressed(dest/"input.npz",**fields)
    write(dest/"ik.json",dict(ik=ik,derivative=derivative,
        integrated_q_displacement_error_max_rad=float(np.max(abs(np.sum(recovered,0)-
                                             (packed["knot_q_rad"][-1]-packed["knot_q_rad"][0])))),
        source_trajectory_ends_delta_old_to_new_mm=np.round((xyz[[0,-1],0]-
                                   legacy_kin["nominal_tcp_m_rotvec_rad"][[0,-1],0])*1e3,12),
        knot_position_interpolation="piecewise linear for q and target request, no extra planner or smoothing"))
    return True


def run_arm(sweep,path,ctx,initial_q,data,src,rec):
    t=ctx["c"]["t"]
    name=path.name
    candidate=(name=="velocity_on")
    trace=servo.run_case(sweep,path,ctx,CASE,initial_q,data["qdot_nom_rad_s"] if candidate else None)
    base=servo.summarize(src,rec,ctx,trace,path)
    if base["axis"] is None:
        base["robust"]=None
        return base,trace,{}
    x=trace["physical_xyz_m"][:,0]
    v=trace["vx_m_s"]*1000
    info,detector_t,detector_v=analyze(t,x,v,rec)
    base["robust"]=info
    tick=data["tick_index"]
    q_cmd=trace["final_ctrl"]
    expected=(data["q_nom"]+data["dq"])[np.arange(len(t))//8*8]
    if candidate:
        qd_ref=data["qdot_nom_rad_s"][np.arange(len(t))//8*8]
        np.testing.assert_array_equal(trace["qdot_nom_command_rad_s"],qd_ref)
        np.testing.assert_allclose(trace["q_velocity_ff_rad"]*5000,qd_ref*500,rtol=0,atol=1e-13)
        expected=expected+trace["q_velocity_ff_rad"]
    np.testing.assert_array_equal(q_cmd,expected)
    np.testing.assert_allclose(5000*(q_cmd-trace["q"])-500*trace["qdot"],trace["qfrc_actuator"],atol=1e-9,rtol=0)
    full_fields=("t","physical_xyz_m","vx_m_s","Fd","qfrc_applied","updated","ctrl_saturation","joint_saturation",
                 "force_saturation","remaining","remaining_dot","actual_micro_q")
    tick_fields=("final_ctrl","q","qdot","base_ctrl","qfrc_actuator","qdot_nom_command_rad_s","q_velocity_ff_rad")
    payload={key:trace[key] for key in full_fields}|{key+"_at_ticks":trace[key][tick] for key in tick_fields}
    return base,trace,payload|dict(detector_t_s=detector_t,detector_v_mm_s=detector_v)


def gate(rec,original,candidate,actual,clock_uncertainty):
    if not all(d["robust"] for d in (original,candidate)):
        return dict(pass_all=False,reason="Retained protected prefix")
    a=actual["events"]
    b=candidate["robust"]["events"]
    markers={}
    for k in ("start50","turn_xmax","return50"):
        if a[k]["time_s"] is None or b[k]["time_s"] is None:
            markers[k]=dict(delay_ms=None,reported_resolution_ms=None)
            continue
        delay=1000*(b[k]["time_s"]-a[k]["time_s"])
        raw_bracket=actual["native_brackets_s"][k]
        sample_width=1000*(raw_bracket[-1]-raw_bracket[0]) if len(raw_bracket)>1 else 0.
        # Empirical width for interpretation rather than an absolute latency bound.
        markers[k]=dict(delay_ms=delay,reported_resolution_ms=sample_width+clock_uncertainty)
    main=all(v["delay_ms"] is not None and abs(v["delay_ms"])<=GATES["main_marker_ms"] for v in markers.values())
    spans={}
    spans_ok=True
    for k in ("rise_10_90","reversal_90_90","reversal_50_50","reversal_80_80"):
        o,c=original["robust"]["spans_ms"][k],candidate["robust"]["spans_ms"][k]
        sample=actual["spans_ms"][k]
        o_err=abs(o-sample) if o is not None and sample is not None else None
        c_err=abs(c-sample) if c is not None and sample is not None else None
        spans[k]=dict(measured_ms=sample,original_ms=o,candidate_ms=c,original_error_ms=o_err,candidate_error_ms=c_err)
        if k in ("rise_10_90","reversal_90_90"):
            spans_ok &= o_err is not None and c_err is not None and c_err<o_err
    stable={k:[p["stable"][k]["actual_v_RMSE_mm_s"] for p in (original,candidate)] for k in ("outbound","return_straight")}
    no_degrade=all(y-x<=GATES["max_added_stable_speed_rmse_mm_s"] for x,y in stable.values())
    raw=servo.x_metrics.gate(original,candidate,rec["speed_mm_s"],dict(
        episode_gate_delay_ms=GATES["main_marker_ms"],maximum_new_speed_overshoot_fraction_nominal=GATES["max_new_speed_peak_fraction"],
        maximum_new_spatial_overshoot_mm=GATES["max_new_axial_endpoint_excess_mm"]))
    passes=dict(timing_pass=main,durations_pass=spans_ok,shape_pass=raw["shape_pass"],
                stable_shape_pass=no_degrade,overshoot_pass=raw["overshoot_pass"],protection_pass=raw["protection_pass"])
    return passes|dict(pass_all=all(passes.values()),markers=markers,spans=spans,raw_gate_details=raw,stable_velocity_RMSE_mm_s=stable)


def pair_trial(dest,trial,rec,frozen,prepared,sweep):
    pre=prepared/trial
    p=core.read(pre/"input.json")
    assert p["locked_rule_sha256"]==core.sha(ROOT/"clock_event_config.json") and p["frozen_window_sha256"]==core.sha(WINDOW_PATH)
    data=core.data_arrays(pre/"input.npz")
    path=source_data(trial,rec)
    src,_,_,ctx,q0,_=servo.case_inputs(trial,rec,path,frozen)
    dest.mkdir()
    mapped=series(src,data)
    np.testing.assert_array_equal(q0,data["initial_q_rad"])
    for k in ("dq","dqd"):
        np.testing.assert_array_equal(ctx["scaled"][k],data[k])
    np.testing.assert_array_equal(ctx["pressure"],data["pressure"])
    ctx=replace_nominal(ctx,data["q_nom"],data["nominal_xyz_m"])
    base={}
    arrays={}
    for arm in ("velocity_off","velocity_on"):
        result,_,packed=run_arm(sweep,dest/arm,ctx,q0,data,mapped,rec)
        base[arm]=result
        arrays.update({arm+"_"+k:v for k,v in packed.items()})
    post=core.read(pre/"clock_decisions.json")
    past=core.read(PAST_ROOT/trial/"status.json")
    summary=dict(trial=trial,conditions=base,gate=gate(rec,*base.values(),p["mapped_actual"],post["largest_abs_residual_ms"]),
            legacy_previous_gate=past["gate"],mapped_input_sha256=core.sha(pre/"input.npz"),actual_model=ctx["compiled"],
            reference="No movement of frozen Y windows or feedback reference, no external fitted force or physical correction")
    write(dest/"status.json",summary)
    np.savez_compressed(dest/"comparison.npz",**arrays)
    print(trial,summary["gate"],flush=True)
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--prepare",type=Path)
    p.add_argument("--use-prepared",type=Path)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    frozen=core.context()
    windows=core.read(WINDOW_PATH)["trials"]
    if args.prepare:
        args.prepare.mkdir()
        outcomes={trial:prepare_trial(args.prepare/trial,trial,windows[trial],frozen) for trial in TRIALS}
        write(args.prepare/"lock.json",dict(input_gates=outcomes,
            input_hashes={k:core.sha(args.prepare/k/"input.npz") for k,v in outcomes.items() if v},
            config_sha256=core.sha(ROOT/"clock_event_config.json"),windows_sha256=core.sha(WINDOW_PATH),
            stage="Locked using source evidence before any paired simulation"))
        print("Prepared",outcomes,flush=True)
        if not all(outcomes.values()):
            raise SystemExit(2)
        return
    prepared=args.use_prepared
    lock=core.read(prepared/"lock.json")
    if not all(lock["input_gates"].values()) or lock["config_sha256"]!=core.sha(ROOT/"clock_event_config.json"):
        raise ValueError("Prepared-input gate failed")
    for trial in TRIALS:
        assert core.sha(prepared/trial/"input.npz")==lock["input_hashes"][trial]
    args.output.mkdir()
    d=args.output/"integration_source"
    d.mkdir()
    sweep=core.integration_module(d)
    status={}
    for trial in TRIALS:
        status[trial]=pair_trial(args.output/trial,trial,windows[trial],frozen,prepared,sweep)
        x_pass=list(status)==list(TRIALS) and all(x["gate"]["pass_all"] for x in status.values())
        write(args.output/"receipt.json",dict(x_pass=x_pass,conditions=status,n_refit="pending" if x_pass else "not started",
            formal_L_SFC=0,config_sha256=lock["config_sha256"],input_lock=lock,python=sys.version,
            mujoco=old.mujoco.__version__,numpy=np.__version__,scipy=core.scipy.__version__))


if __name__=="__main__":
    main()
