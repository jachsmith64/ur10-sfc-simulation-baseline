"""Run the V02/CB3 integration in N mode with an independent frozen reference."""
import argparse
from hashlib import sha256
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent/"20261116_A_multispeed"
sys.path.insert(0,str(PREVIOUS))

import run_g as old

import numpy as np
from b132_common import save
import axial
from reference import Schedule, ReferencedMeasurement
from windows import EVENT_TO_TRANSITION, PLATFORMS, TRIALS, mask
from score_g import metrics
from position_execution import compiled_mapping

core = old.core
PLAN = old.PLAN
CONFIG = core.read(ROOT/"preprocess_config.json")
FIT_WINDOWS = ("full","start","outbound","turn","return_straight","stop")


class TimingLogger(core.Logger):
    def __init__(self,path,ctx,case,sensor,channel):
        super().__init__(path,ctx,case,sensor,channel)
        self.arr["vx_m_s"]=np.empty(self.n)

    def sample(self,i,env,*args):
        super().sample(i,env,*args)
        self.arr["vx_m_s"][i]=env.jacp()[0] @ env.data.qvel


def initial_factory(initial_q):
    def factory():
        env=old.model()
        original_reset=env.reset
        calls=0

        def reset(nominal):
            nonlocal calls
            calls+=1
            if calls != 1:
                raise RuntimeError("Physical state may only be initialized once")
            original_reset(initial_q)

        env.reset=reset
        return env
    return factory


def integrate(sweep,path,ctx,case,fd_values,initial_q,pulse=0.,*,execution=None):
    path.mkdir()
    t,p=ctx["c"]["t"],ctx["parameters"]
    noise=np.zeros(int(np.ceil((t[-1]+.001)*PLAN["camera_hz"]))+1)
    sensor=ReferencedMeasurement(core.CONFIG,case,core.aconfig.Y_NOMINAL,ctx["pressure"],noise,ctx["reference"])
    coordinator=old.PulseCoordinator(ctx,case,sensor,pulse) if pulse else core.Coordinator(ctx,case,sensor)
    if execution is None:
        channel=core.CommandChannel(case["execution_delay_ms"],case["command_ms"],coordinator)
        logger=TimingLogger(path/"trace.npz",ctx,case,sensor,channel)
    else:
        channel,logger=execution(path,ctx,case,sensor,coordinator)
    fd_sha=sha256(fd_values.tobytes()).hexdigest()
    fd=old.disturbance.FrozenDisturbance(Path("physical_only"),t,fd_values,{"frozen":{"sha256":fd_sha}})
    sweep.SimEnv=initial_factory(initial_q)
    row=sweep.run_case(path.name,"A0",p["K"],p["B0"],p["n"],p["mu"],p["r"],p["v_ref"],fd,
                       ctx["c"]["along"],ctx["c"]["q"],phase_code=ctx["c"]["phase"],c_input=ctx["scaled"],
                       execution_mode=old.POSITION_EQUIV,control_update_steps=case["command_ms"],
                       precise_logger=logger,common_position_correction=ctx["corr"]["common_delta_q"],
                       e_extension=coordinator,measurement_adapter=sensor,final_command_hold=True,
                       command_channel=channel)
    save(path/"row.json",row)
    save(path/"guard.json",coordinator.rejection)
    trace=core.data_arrays(path/"trace.npz")
    frames=core.as_frames(sensor)
    for kind, values in (("camera",sensor.camera_reference_m),("encoder",sensor.encoder_reference_m)):
        for frame,ref in zip(frames[kind],values,strict=True):
            frame["reference_m"]=ref
    save(path/"sample_receipt.json",frames)
    save(path/"command_events.json",channel.events)
    save(path/"case.json",dict(case=case,parameters=p,initial_q_rad=initial_q,initial_qdot_rad_s=[0.]*6,
        nominal_y_world_m=core.aconfig.Y_NOMINAL,Fd_sha256=fd_sha,pulse_N=pulse,
        physical_command_correction_sha256=sha256(ctx["corr"]["common_delta_q"].tobytes()).hexdigest(),
        remaining_active=False,camera_noise_m=0.))
    if row["status"]!="OK" or len(trace["t"])!=len(t):
        raise RuntimeError(f"{path.name}: invalid integration prefix {len(trace['t'])}/{len(t)}")
    step=np.arange(len(t))
    updates=step[step%case["command_ms"]==0]
    interior=step[step%case["command_ms"]!=0]
    assert np.array_equal(trace["q"][0],initial_q) and not np.any(trace["qdot"][0])
    assert np.array_equal(trace["t"],t) and np.array_equal(trace["Fd"],fd_values)
    assert not np.any(trace["remaining"]) and not np.any(trace["remaining_dot"])
    assert np.array_equal(trace["qfrc_applied"],trace["qfrc_disturbance"])
    assert np.array_equal(trace["final_ctrl"][interior],trace["final_ctrl"][interior-1])
    assert [a["execute"] for a in channel.events]==updates.tolist()
    assert all(a["c_available"]<=a["compute"] and a["q_available"]<=a["compute"] for a in channel.events)
    qframes,cframes=sensor.q.frames,sensor.c.frames
    enc_t=np.array([f.sample for f in qframes])/1000
    enc_y=np.array([f.e for f in qframes])
    np.testing.assert_allclose(enc_y,[f.physical_e + pr-ref for f,pr,ref in
        zip(qframes,ctx["pressure"][:len(qframes)],sensor.encoder_reference_m,strict=True)],atol=1e-18,rtol=0)
    np.testing.assert_allclose([f.v for f in qframes][1:],np.diff(enc_y)/np.diff(enc_t),atol=1e-14,rtol=0)
    np.testing.assert_allclose([f.e for f in cframes],
        np.interp([f.sample/1000 for f in cframes],t,trace["physical_e_um"]*1e-6)-sensor.camera_reference_m,
        atol=2e-13,rtol=0)
    if not pulse:
        assert not np.any(trace["F_requested"]) and not np.any(trace["actual_micro_q"])
    save(path/"checks.json",dict(complete=True,pre_step_time_exact=True,remaining_identically_zero=True,
        remaining_dot_identically_zero=True,frame_reference_checked=True,held_command_exact=True,
        command_publication_causal=True,external_force_path_exact=True,observer_replay=False))
    print(path.parent.name,path.name,len(t),"steps OK",flush=True)
    return trace,frames


def common_correction(grid, schedule, direction):
    b,rate,accel=schedule.at(grid)
    return dict(common_delta_q=-1e-6*b[:,None]*direction,
                common_delta_q_dot=-1e-6*rate[:,None]*direction,
                common_delta_q_ddot=-1e-6*accel[:,None]*direction,
                common_baseline_um=b,common_delta_y_m=-1e-6*b)


def score(src,rec,t,y,reference,optional_windows=None):
    f=src["frame_host_time_s"]
    a=src["A_centered_um"]
    xr=np.interp(f,t,y)
    zero=rec["raw_visual_zero_um"]
    r=reference.at(f)[0]
    end=f[-1]
    win={"full":[0.,float(end)]}|rec["windows_s"] if optional_windows is None else optional_windows
    result={}
    for name,span in win.items():
        keep=src["frame_valid"] & mask(f,span,end)
        model_y,source_y=xr[keep],a[keep]
        raw=metrics(f[keep],source_y+zero,model_y+zero,PLAN["flat_reference_um"])
        processed=metrics(f[keep],source_y-r[keep],model_y-r[keep],PLAN["flat_reference_um"])
        for layer,data_a,data_m in ((raw,source_y+zero,model_y+zero),(processed,source_y-r[keep],model_y-r[keep])):
            if keep.any():
                stamps=f[keep]
                ai,mi=np.argmax(abs(data_a)),np.argmax(abs(data_m))
                layer.update(peak_A_um=float(data_a[ai]),peak_N_um=float(data_m[mi]),
                             absolute_peak_delay_s=float(stamps[mi]-stamps[ai]),
                             peak_A_time_s=float(stamps[ai]),peak_N_time_s=float(stamps[mi]))
        result[name]=dict(raw=raw,processed=processed)
    return result


def evaluate_timing(trace,src,rec):
    return axial.measure(src,rec,trace["t"],trace["physical_xyz_m"][:,0],trace["vx_m_s"]*1000)


def run_trial(sweep, output, prepared, trial, rec, frozen, prior_results):
    dest=output/trial
    dest.mkdir()
    orig=prepared/trial
    src=core.data_arrays(orig/"A_source.npz")
    meta=core.read(orig/"source.json")
    assert core.sha(orig/"A_source.npz")==rec["source_npz_sha256"]
    assert core.sha(orig/"source.json")==rec["source_record_sha256"]
    t=np.arange(int(np.ceil(src["frame_host_time_s"][-1]*1000))+1)*.001
    kin,ik=old.nominal(old.CB3,src,t)
    q=kin["nominal_q_rad"]
    xyz=kin["nominal_tcp_m_rotvec_rad"][:,:3]
    ht=src["host_time_s"]
    source_q0=np.array([np.interp(0.,ht,x) for x in src["actual_q_rad"].T])
    source_v0=np.array([np.interp(0.,ht,x) for x in src["actual_tcp_speed"][:,:3].T])
    if np.linalg.norm(source_v0)>5e-5:
        raise ValueError("The t=0 record does not support stationary initialization")
    core.aconfig.Y_NOMINAL=float(xyz[0,1])
    phase=np.zeros(len(t),dtype=int)
    for i,span in enumerate(rec["windows_s"].values(),1):
        phase[mask(t,span,src["frame_host_time_s"][-1])]=i
    c=dict(t=t,q=q,along=(xyz[:,0]-xyz[0,0])*1000,phase=phase)
    scaled=c|{key:frozen["scaled"][key][:len(t)].copy() for key in ("dq","dqd")}
    scaled.update(remaining=np.zeros(len(t)),remaining_dot=np.zeros(len(t)))
    assert np.any(scaled["dq"]) and np.any(scaled["dqd"])
    pressure=core.pressure(PLAN["encoder_pressure"])
    reference=Schedule(rec,rec["bases_um"])
    score_ref=Schedule(rec,rec["bases_um"])  # independent receipt, same fixed calibration
    env=old.model()
    env.reset(q[0])
    j=env.jacp()
    direction=j.T @ np.linalg.solve(j@j.T,[0.,1.,0.])
    np.testing.assert_allclose(j@direction,[0.,1.,0.],rtol=0,atol=1e-12)
    params=frozen["parameters"]
    mapping=compiled_mapping(env.model)
    assert mapping["Kp_Nm_per_rad"]==[5000]*6 and mapping["Kd_Nms_per_rad"]==[500]*6
    assert mapping["gravity"]==[0.,0.,-9.81]
    no_common=common_correction(t,Schedule(rec,dict.fromkeys(PLATFORMS,0.)),direction)
    ctx=dict(c=c,scaled=scaled,reference=reference,corr=no_common,pressure=pressure,
             nominal_xyz=xyz,input_id=sha256(q.tobytes()).hexdigest(),parameters=params,compiled=mapping)
    base_case=core.read(PREVIOUS/"evidence/D_N_o1_fraw_r1_gain1_seed1001_d0.json")
    case=base_case|dict(id="N_physical",estimator="Q",fc=5,rho=0.,filter_order=1)
    force=np.zeros(len(t))
    old_init,_=integrate(sweep,dest/"axis_nominal_init",ctx,case,force,q[0])
    seed,_=integrate(sweep,dest/"axis_measured_init",ctx,case,force,source_q0)
    measured_timing=evaluate_timing(seed,src,rec)
    source_x=src["target_tcp_pose"][:,0]
    t_hold=t[::8]
    x_hold=xyz[::8,0]
    axis=dict(old_initial=evaluate_timing(old_init,src,rec),
              measured_initial=measured_timing,
              RTDE_target_host=axial.measure(src,rec,ht,source_x,np.gradient(source_x,ht)*1e3),
              reconstructed_nominal=axial.measure(src,rec,t,xyz[:,0],np.gradient(xyz[:,0],t)*1e3),
              command_grid_nominal=axial.measure(src,rec,t_hold,x_hold,np.gradient(x_hold,t_hold)*1e3),
              code_clock=dict(physical_dt_s=.001,command_period_s=.008,execution_extra_s=0.,
                  sampling="model q and J qdot logged before integration step; command at same pre-step tick",
                  input="original host axis, unchanged samples; elementwise linear IK nominal interpolation",
                  cabinet_max_gap_s=float(np.max(np.diff(ht))),source_initial_speed_m_s=source_v0.tolist(),
                  initial_actual_minus_nominal_rad=(source_q0-q[0]).tolist()))
    save(dest/"axial_before_fit.json",axis)

    ft=src["frame_host_time_s"]
    valid=src["frame_valid"]
    a=src["A_centered_um"]
    differences=np.interp(ft,t,seed["physical_e_um"])-a
    centers={k:float(np.median(differences[valid & mask(ft,rec["windows_s"][k],ft[-1])])) for k in PLATFORMS}
    correction_ref=Schedule(rec,centers)
    ctx["corr"]=common_correction(t,correction_ref,direction)
    initial,_=integrate(sweep,dest/"common_zero_force",ctx,case,force,source_q0)
    # Reuse the preceding CB3 frequency probe on exactly the same nominal trajectory/model.
    plant_path=prior_results/trial/"plant.npz"
    prior_status=core.read(prior_results/trial/"status.json")
    if core.sha(plant_path)!=prior_status["provenance"]["plant_sha256"]:
        raise ValueError("Prior unmodified model plant probe hash mismatch")
    np.testing.assert_array_equal(q, core.data_arrays(prior_results/trial/"locked_inputs.npz")["q"])
    plant=old.disturbance.load_plant(plant_path)
    if plant is None:
        raise RuntimeError("The verified prior CB3 plant probe is unreadable")
    save(dest/"plant_receipt.json",dict(sha256=core.sha(plant_path),
            pass_hz=PLAN["pass_hz"],stop_hz=PLAN["stop_hz"],same_nominal_q_verified=True,
            source_trial=trial,solver="original V02 inverse, two-second edge padding"))
    trace=initial
    trials_y=[]
    attempts=[]
    best_quality=-np.inf
    quiet=0
    last_significant=-np.inf
    best_trace=initial
    best_force=force.copy()
    best_frames=None
    best_name="common_zero_force"
    for i in range(len(PLAN["iterations"])+1):
        if i:
            residual=np.interp(t,ft[valid],a[valid])-trace["physical_e_um"]
            force=force+PLAN["iterations"][i-1]*old.inverse(residual,plant)
            name=f"N_fit_{i}"
            trace,frames=integrate(sweep,dest/name,ctx,case,force,source_q0)
        else:
            name="common_zero_force"
            frames=core.read(dest/name/"sample_receipt.json")
        fit=score(src,meta|rec,t,trace["physical_e_um"],score_ref)
        values=[fit[k][layer]["fit"] for k in FIT_WINDOWS for layer in ("raw","processed")]
        quality=min(x for x in values if x is not None)
        attempts.append(dict(case=name,iteration=i,force_max_N=float(np.max(abs(force))),
            force_rms_N=float(np.sqrt(np.mean(force**2))),force_sha256=sha256(force.tobytes()).hexdigest(),
            min_fit=quality,scores=fit))
        trials_y.append(trace["physical_e_um"])
        if quality>best_quality:
            best_quality,best_trace,best_force,best_frames,best_name=quality,trace,force.copy(),frames,name
        if quality-last_significant>=CONFIG["minimum_material_fit_gain"]:
            quiet,last_significant=0,quality
        else:
            quiet+=1
        if best_quality>=PLAN["fit_target"] or quiet>=CONFIG["consecutive_iterations_without_material_gain"]:
            break
    save(dest/"attempts.json",attempts)
    axis["best_N"]=evaluate_timing(best_trace,src,rec)
    pure=dict(ctx,corr=no_common)
    without,_=integrate(sweep,dest/"best_force_without_common",pure,case,best_force,source_q0)
    common_actual=best_trace["physical_e_um"]-without["physical_e_um"]
    common_effect={}
    directions={}
    for event in EVENT_TO_TRANSITION:
        win=rec["windows_s"][event]
        keep=mask(t,win,ft[-1])
        akeep=valid & mask(ft,win,ft[-1])
        assert np.ptp(reference.at(ft[akeep])[0])==0.
        assert np.ptp(ctx["corr"]["common_baseline_um"][keep])==0.
        np.testing.assert_allclose(np.diff(a[akeep]),np.diff((a-score_ref.at(ft)[0])[akeep]),rtol=0,atol=1e-12)
        common_effect[event]=dict(mean_Y_um=float(np.mean(common_actual[keep])),
             peak_to_peak_Y_um=float(np.ptp(common_actual[keep])),
             peak_to_peak_X_um=float(np.ptp((best_trace["physical_xyz_m"]-without["physical_xyz_m"])[keep,0])*1e6))
        directions[event]=dict(reference_fixed_to_pre_event_um=float(score_ref.at(ft[akeep])[0][0]),
                          source_peak_um=float(a[akeep][np.argmax(abs(a[akeep]))]),
                          source_peak_s=float(ft[akeep][np.argmax(abs(a[akeep]))]))
    alternatives={}
    for event,variants in rec["alternatives"].items():
        alternatives[event]=[]
        for alt in variants:
            next_phase=PLATFORMS[list(EVENT_TO_TRANSITION).index(event)+1]
            changed=dict(rec["windows_s"])
            transition=EVENT_TO_TRANSITION[event]
            changed[event]=[changed[event][0],alt["transition"][0]]
            changed[transition]=alt["transition"]
            changed[next_phase]=alt["stable_window"]
            levels=rec["bases_um"]|{next_phase:alt["stable_source"]["median_um"]}
            alt_fit=score(src,meta|rec,t,best_trace["physical_e_um"],
                          Schedule(rec|dict(windows_s=changed),levels),
                          {event:changed[event],next_phase:changed[next_phase]})
            alternatives[event].append(dict(delta_s=alt["delta_s"],shifted_phase=next_phase,
                   center_delta_um=levels[next_phase]-rec["bases_um"][next_phase],scores=alt_fit))
    old_case=core.data_arrays(prior_results/trial/"best_N.npz")
    old_fit=score(src,meta|rec,t,old_case["physical_e_um"],score_ref)
    pulse_receipt=None
    pulse_payload={}
    if trial==TRIALS[0]:
        for label,value in (("plus",1.),("minus",-1.)):
            p,_=integrate(sweep,dest/f"micro_{label}",ctx,case,best_force,source_q0,value)
            delta=p["physical_e_um"]-best_trace["physical_e_um"]
            assert not np.any(delta[:801])
            nonzero=np.flatnonzero(delta)
            assert nonzero[0]==801 and np.sign(delta[801])==np.sign(value)
            assert np.any(p["actual_micro_q"][800:856])
            pulse_payload[label+"_delta_y_um"]=delta
            pulse_payload[label+"_micro_q_rad"]=p["actual_micro_q"][800:856]
        pulse_receipt=dict(channel="old POSITION_EQUIV pulse ±1 N",command_window_s=[.8,.856],
                           first_motion_s=.801,plus_peak_um=float(pulse_payload["plus_delta_y_um"].max()),
                           minus_peak_um=float(pulse_payload["minus_delta_y_um"].min()),causal_signed_response=True)

    b,rate,acc=reference.at(t)
    score_b=score_ref.at(t)[0]
    fields=dict(t=t,nominal_x_m=xyz[:,0],initial_nominal_x_m=old_init["physical_xyz_m"][:,0],
                initial_nominal_vx_m_s=old_init["vx_m_s"],initial_measured_x_m=seed["physical_xyz_m"][:,0],
                initial_measured_vx_m_s=seed["vx_m_s"],physical_xyz_m=best_trace["physical_xyz_m"],
                physical_vx_m_s=best_trace["vx_m_s"],physical_y_um=best_trace["physical_e_um"],
                old_physical_y_um=old_case["physical_e_um"],old_remaining_um=old_case["remaining"]*1e6,
                actual_common_effect_y_um=common_actual,force_N=best_force,
                attempt_y_um=np.stack(trials_y),direction_reference_um=b,reference_rate_um_s=rate,
                reference_acceleration_um_s2=acc,slow_trend_um=np.zeros(len(t)),scoring_reference_um=score_b,
                common_command_y_m=ctx["corr"]["common_delta_y_m"],remaining_m=scaled["remaining"],
                remaining_rate_m_s=scaled["remaining_dot"],actual_phase_code=phase,
                command_cam_age_ms=best_trace["compute_cam_age_ms"][::8],
                command_enc_age_ms=best_trace["compute_enc_age_ms"][::8],**pulse_payload)
    for channel in ("camera","encoder"):
        for name in ("sample","available","e","v","vf","physical_e","additive","reference_m"):
            fields[f"{channel}_{name}"]=np.asarray([frame[name] for frame in best_frames[channel]])
    np.savez_compressed(dest/"review.npz",**fields)
    np.savez_compressed(dest/"locked.npz",**(scaled|ctx["corr"]),
                        pressure=pressure,force_N=best_force,scoring_reference_um=score_b,
                        feedback_reference_um=b,feedback_reference_rate_um_s=rate,initial_q_rad=source_q0)
    cam_age=fields["command_cam_age_ms"]; enc_age=fields["command_enc_age_ms"]
    valid_age=(cam_age>=0)&(enc_age>=0)
    result=dict(trial=trial,best_case=best_name,best_min_fit=best_quality,scores=next(a["scores"] for a in attempts if a["case"]==best_name),
        timing=axis,new_gate=bool(best_quality>=PLAN["fit_min"] and measured_timing["gate_50ms"] and axis["best_N"]["gate_50ms"]),
        outcome="diagnostic_only_axial_gate_failed" if not axis["best_N"]["gate_50ms"] else "review_shape_before_acceptance",
        public_platform_residual_um=centers,physical_correction_event_effect=common_effect,event_references=directions,
        old_physical_scores_same_windows=old_fit,old_remaining_rms_um=float(np.sqrt(np.mean(fields["old_remaining_um"]**2))),
        new_remaining_rms_um=0.,boundary_sensitivity=alternatives,command_age_difference_ms=np.percentile(cam_age[valid_age]-enc_age[valid_age],[0,50,100]).tolist(),
        pulse=pulse_receipt,parameters=params,compiled=mapping,ik=ik,source_time_zero_ns=rec["source_clock_origin_ns"])
    save(dest/"status.json",result)
    print(trial,"FIT",best_quality,"X timing",axis["best_N"]["delay_ms"],flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepared",type=Path,default=PREVIOUS/"prepared_A")
    parser.add_argument("--prior-results",type=Path,default=ROOT/"inputs/previous")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--trials",nargs="+",choices=TRIALS,default=TRIALS)
    a=parser.parse_args()
    required=core.read(PREVIOUS/"evidence/regression_receipt.json")
    assert len(required)==3 and all(x["all_arrays_equal"] and x["inputs_equal"] for x in required.values())
    window=core.read(ROOT/"windows_frozen.json")
    assert window["source_only"]
    a.output.mkdir(parents=True)
    source_dir=a.output/"integration_source"
    source_dir.mkdir()
    sweep=core.integration_module(source_dir)
    frozen=core.context()
    results={}
    for trial in a.trials:
        results[trial]=run_trial(sweep,a.output,a.prepared,trial,window["trials"][trial],frozen,a.prior_results)
        save(a.output/"receipt.json",dict(results=results,frozen_windows_sha256=core.sha(ROOT/"windows_frozen.json"),
              formal_control_runs=0,python=sys.version,mujoco=old.mujoco.__version__,numpy=np.__version__,
              scipy=core.scipy.__version__,previous_commit="0f611217b3806442a44f57ec337d0a438c5432c9"))


if __name__=="__main__":
    main()
