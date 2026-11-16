"""Six authorized paired short integrations, using only the original model and window schedule."""
import argparse
from hashlib import sha256
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/"20261116_preprocess_timing"
sys.path.insert(0,str(PREVIOUS))
import run_preprocess as prior

from velocity_reference import VelocityReference, nominal_derivative
import x_metrics

core=prior.core
PLAN=prior.PLAN
old=prior.old
CONFIG=core.read(ROOT/"servo_config.json")
FIELDS=("t","vx_m_s","physical_xyz_m","q","qdot","final_ctrl","base_ctrl","actual_micro_q","updated",
        "qfrc_actuator","ctrl_saturation","force_saturation","joint_saturation","qfrc_applied","Fd",
        "remaining","remaining_dot","q_velocity_ff_rad","qdot_nom_command_rad_s")
REGRESSION=("t","vx_m_s","physical_xyz_m","q","qdot","final_ctrl","base_ctrl","actual_micro_q","updated",
            "qfrc_actuator","qfrc_applied","Fd","remaining","remaining_dot","original_base_ctrl","dq_C",
            "compute_cam_age_ms","compute_enc_age_ms","sample_e_m","sample_v_m_s","cam_e","enc_e","enc_v","enc_vf")


def array_id(x):
    return sha256(x.tobytes()).hexdigest()


def case_inputs(trial,rec,source_path,frozen):
    src=core.data_arrays(source_path)
    if core.sha(source_path)!=rec["source_npz_sha256"]:
        raise ValueError("Source NPZ differs from frozen window receipt")
    t=np.arange(int(np.ceil(src["frame_host_time_s"][-1]*1000))+1)*.001
    kin,ik=old.nominal(old.CB3,src,t)
    q, xyz=kin["nominal_q_rad"],kin["nominal_tcp_m_rotvec_rad"][:,:3]
    nominal_receipt=core.data_arrays(PREVIOUS/"inputs/previous"/trial/"locked_inputs.npz")
    np.testing.assert_array_equal(q,nominal_receipt["q"])
    ht=src["host_time_s"]
    q0=np.array([np.interp(0,ht,x) for x in src["actual_q_rad"].T])
    v0=np.array([np.interp(0,ht,x) for x in src["actual_tcp_speed"][:,:3].T])
    if np.linalg.norm(v0)>5e-5:
        raise ValueError("Original t=0 cabinet state is not stationary")
    phase=np.zeros(len(t),dtype=int)
    for i,span in enumerate(rec["windows_s"].values(),1):
        phase[prior.mask(t,span,src["frame_host_time_s"][-1])]=i
    c=dict(t=t,q=q,along=(xyz[:,0]-xyz[0,0])*1000,phase=phase)
    scaled=c|{k:frozen["scaled"][k][:len(t)].copy() for k in ("dq","dqd")}
    scaled.update(remaining=np.zeros(len(t)),remaining_dot=np.zeros(len(t)))
    pressure=core.pressure(PLAN["encoder_pressure"])
    upstream=core.read(PREVIOUS/"inputs/upstream_errors.json")[trial]
    for key,data in (("dq",scaled["dq"]),("dqd",scaled["dqd"]),("pressure",pressure)):
        assert array_id(data)==upstream[key]["sha256"]
    env=old.model()
    env.reset(q[0])
    j=env.jacp()
    direction=j.T @ np.linalg.solve(j@j.T,[0.,1.,0.])
    zero_corr=prior.common_correction(t,prior.Schedule(rec,dict.fromkeys(prior.PLATFORMS,0.)),direction)
    model=prior.compiled_mapping(env.model)
    assert model["Kp_Nm_per_rad"]==[5000]*6 and model["Kd_Nms_per_rad"]==[500]*6
    ref=prior.Schedule(rec,rec["bases_um"])
    ctx=dict(c=c,scaled=scaled,reference=ref,corr=zero_corr,pressure=pressure,nominal_xyz=xyz,
             input_id=array_id(q),parameters=frozen["parameters"],compiled=model)
    core.aconfig.Y_NOMINAL=float(xyz[0,1])
    return src,kin,ik,ctx,q0,direction


def derivative_receipt(kin,ctx,case):
    t=ctx["c"]["t"]
    knots_t,knots_q=kin["target_knot_time_s"],kin["target_knot_q_rad"]
    vel,slopes,seq=nominal_derivative(t,knots_t,knots_q)
    ticks=np.arange(0,len(t),case["command_ms"])
    env=old.model()
    jac_p=np.empty((3,6))
    jac_r=np.empty((3,6))
    fk_velocity=[]
    for i in ticks:
        env.data.qpos[:]=ctx["c"]["q"][i]
        old.mujoco.mj_kinematics(env.model,env.data)
        old.mujoco.mj_comPos(env.model,env.data)
        old.mujoco.mj_jacSite(env.model,env.data,jac_p,jac_r,env.site_id)
        fk_velocity.append(np.r_[jac_p @ vel[i],jac_r @ vel[i]])
    fk_velocity=np.asarray(fk_velocity)
    target_slopes=np.diff(kin["target_knot_tcp"],axis=0)/np.diff(knots_t)[:,None]
    expected=target_slopes[seq[ticks]]
    gap_diff=np.diff(knots_t)
    changing=np.flatnonzero(np.linalg.norm(np.diff(knots_q,axis=0),axis=1)>1e-10)
    stats=dict(method=CONFIG["nominal_velocity_rule"],first_qdot_rad_s=vel[0])
    stats.update(final_qdot_rad_s=vel[-1],native_dt_ms=np.percentile(gap_diff*1000,[0,50,95,100]),
        max_abs_qdot_rad_s=np.max(abs(vel[ticks]),axis=0),
        max_abs_slope_change_rad_s=np.max(abs(np.diff(slopes,axis=0)),axis=0),
        max_abs_held_speed_step_rad_s=np.max(abs(np.diff(vel[ticks],axis=0)),axis=0),
        max_abs_offset_rad=np.max(abs(vel[ticks]*.1),axis=0),
        max_abs_equivalent_torque_Nm=np.max(abs(vel[ticks]*500),axis=0),
        raw_nominal_vx_tick_mm_s_minmax=[float(expected[:,0].min()*1000),float(expected[:,0].max()*1000)],
        max_kinematic_twist_error=np.max(abs(fk_velocity-expected),axis=0),
        nominal_movement_native_intervals=int(len(changing)),shape_rad_s=vel.shape)
    fields=dict(t=t,q_nom=ctx["c"]["q"],nom_x_m=ctx["nominal_xyz"][:,0],
                qdot_nom_rad_s=vel,knot_t_s=knots_t,knot_q_rad=knots_q,knot_target_pose=kin["target_knot_tcp"],
                interval_index=seq,tick_index=ticks,target_twist_at_ticks=expected,jacobian_twist_at_ticks=fk_velocity,
                dq=ctx["scaled"]["dq"],dqd=ctx["scaled"]["dqd"],pressure=ctx["pressure"],
                feedback_reference_um=ctx["reference"].at(t)[0])
    return vel,stats,fields


def summarize(src,rec,ctx,trace,path):
    row=core.read(path/"row.json")
    guard=core.read(path/"guard.json")
    t,x,v=trace["t"],trace["physical_xyz_m"][:,0],trace["vx_m_s"]*1000
    sat={k:int(np.count_nonzero(trace[k].any(axis=1)))
         for k in ("ctrl_saturation","joint_saturation","force_saturation")}
    guard_info=dict(status=row["status"],guard_clear=guard is None,guard=guard,saturation_rows=sat,
                    steps=len(t),max_abs_actuator_Nm=np.max(abs(trace["qfrc_actuator"]),axis=0).tolist() if len(t) else None)
    if len(t)!=len(ctx["c"]["t"]):
        return guard_info|dict(axis=None,stable=None,overshoot=None)
    return guard_info|dict(axis=prior.axial.measure(src,rec,t,x,v),
        stable=x_metrics.track(src,rec,t,x,v,ctx["nominal_xyz"][:,0]),overshoot=x_metrics.overshoot(src,rec,t,x,v))


def run_case(sweep,path,ctx,case,q0,vel=None):
    hook=VelocityReference(core,prior.TimingLogger,vel,case["command_ms"]) if vel is not None else None
    fd=np.zeros(len(ctx["c"]["t"]))
    try:
        arrays,_=prior.integrate(sweep,path,ctx,case,fd,q0,execution=hook)
    except RuntimeError:
        if not (path/"guard.json").is_file() or core.read(path/"guard.json") is None:
            raise
        arrays=core.data_arrays(path/"trace.npz")
    # Applied offset and nominal speed are zero for the off arm, even on a valid prefix.
    for k in ("qdot_nom_command_rad_s","q_velocity_ff_rad"):
        if k not in arrays:
            arrays[k]=np.zeros((len(arrays["t"]),6))
    return arrays


def run_trial(sweep,out,trial,record,source_path,frozen,old_runs):
    out.mkdir()
    src,kin,ik,ctx,q0,_=case_inputs(trial,record,source_path,frozen)
    bcase=core.read(old.PROJECT/"evidence/D_N_o1_fraw_r1_gain1_seed1001_d0.json")
    case=bcase|dict(id="N_physical",estimator="Q",fc=5,rho=0.,filter_order=1)
    vel,dt_info,data=derivative_receipt(kin,ctx,case)
    traces={}
    outcomes={}
    for name,d in (("original",None),("nominal_velocity",vel)):
        path=out/name
        trace=run_case(sweep,path,ctx,case,q0,d)
        outcomes[name]=summarize(src,record,ctx,trace,path)
        traces[name]=trace
        for field in FIELDS:
            data[name+"_"+field]=trace[field]
    source_dir=old_runs/trial/"axis_measured_init" if old_runs else None
    regression=None
    if source_dir:
        prior_trace=core.data_arrays(source_dir/"trace.npz")
        equal={k:bool(np.array_equal(traces["original"][k],prior_trace[k],equal_nan=True)) for k in REGRESSION}
        assert all(equal.values()),equal
        assert core.read(out/"original"/"case.json")==core.read(source_dir/"case.json")
        regression=dict(all_selected_exact=True,compared=equal,upstream_trace_sha256=core.sha(source_dir/"trace.npz"),
                        prior_path_relative_to_supplied_run_root=f"{trial}/axis_measured_init/trace.npz")
    if all(x["axis"] for x in outcomes.values()):
        gate=x_metrics.gate(outcomes["original"],outcomes["nominal_velocity"],record["speed_mm_s"],CONFIG)
    else:
        gate=dict(pass_all=False,reason="At least one integration retained a guarded prefix")
    if outcomes["nominal_velocity"]["status"]=="OK":
        kp=np.array(ctx["compiled"]["Kp_Nm_per_rad"])
        kd=np.array(ctx["compiled"]["Kd_Nms_per_rad"])
        candidate=traces["nominal_velocity"]
        a=candidate["qdot_nom_command_rad_s"]
        np.testing.assert_allclose(candidate["q_velocity_ff_rad"]*kp,a*kd,rtol=0,atol=1e-13)
        held=np.arange(len(a))//8*8
        np.testing.assert_array_equal(a,vel[held])
        expected=(ctx["c"]["q"]+ctx["scaled"]["dq"])[held]+candidate["q_velocity_ff_rad"]
        np.testing.assert_array_equal(candidate["final_ctrl"],expected)
        reconstructed=kp*(candidate["final_ctrl"]-candidate["q"])-kd*candidate["qdot"]
        np.testing.assert_allclose(reconstructed,candidate["qfrc_actuator"],rtol=0,atol=1e-9)
    np.savez_compressed(out/"comparison.npz",**data)
    stats=dict(trial=trial,original=outcomes["original"],candidate=outcomes["nominal_velocity"],
               gate=gate,derivative=dt_info,ik=ik,regression=regression,compiled=ctx["compiled"],
               nominal_q_sha256=array_id(ctx["c"]["q"]),dq_sha256=array_id(ctx["scaled"]["dq"]),
               pressure_sha256=array_id(ctx["pressure"]))
    core.write(out/"status.json",core.clean(stats))
    print(trial,"gate",gate,flush=True)
    return stats


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--baseline-runs",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--trials",choices=prior.TRIALS,nargs="+",default=prior.TRIALS)
    args=parser.parse_args()
    assert CONFIG["episode_gate_delay_ms"]==PLAN["event_timing_tolerance_s"]*1000
    frozen=core.context()
    all_windows=core.read(PREVIOUS/"windows_frozen.json")
    args.output.mkdir()
    integration=args.output/"integration_source"
    integration.mkdir()
    sweep=core.integration_module(integration)
    all_={}
    for trial in args.trials:
        a=old.PROJECT/"prepared_A"/trial/"A_source.npz"
        all_[trial]=run_trial(sweep,args.output/trial,trial,all_windows["trials"][trial],a,frozen,args.baseline_runs)
        x_pass=list(all_)==list(prior.TRIALS) and all(v["gate"]["pass_all"] for v in all_.values())
        core.write(args.output/"receipt.json",core.clean(dict(x_pass=x_pass,results=all_,
                completed_pairs=list(all_),formal_control_runs=0,physical_y_reconstruction="awaits X gate" if x_pass else "not started",
                inherited_window_sha256=core.sha(PREVIOUS/"windows_frozen.json"),
                source_check_sha256=core.sha(ROOT/"velocity_source.json"),
                config_sha256=core.sha(ROOT/"servo_config.json"),environment=dict(python=sys.version,mujoco=old.mujoco.__version__,
                                                                                numpy=np.__version__,scipy=core.scipy.__version__),
                source_parent=CONFIG["parent_commit"])))


if __name__=="__main__":
    main()
