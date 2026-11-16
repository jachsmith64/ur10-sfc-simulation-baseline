"""Reuse the authorized clock-corrected X reference for physical-only N reconstruction."""
import argparse
from pathlib import Path

import numpy as np
from run_clock_event import (ROOT, core, servo, prior, old, TRIALS, WINDOW_PATH, CASE,
                             replace_nominal, source_data, write, analyze)


def integrate(sweep,folder,ctx,initial_q,force,hook,pulse=0.):
    trace,_=prior.integrate(sweep,folder,ctx,CASE,force,initial_q,pulse,execution=hook)
    assert not np.any(trace["remaining"]) and not np.any(trace["remaining_dot"])
    return trace


def plant_for(poses,folder):
    code=old.disturbance
    original_q=core.aconfig.INIT_Q
    original_dc=code.measure_dc_compliance
    try:
        # Re-estimate the inherited static-position frequency response. Here qdot_nom=0.
        core.aconfig.INIT_Q=poses[len(poses)//2].copy()
        code.measure_dc_compliance=lambda: original_dc(old.model())
        plant=code.measure_plant(old.model())
    finally:
        core.aconfig.INIT_Q=original_q
        code.measure_dc_compliance=original_dc
    code.save_plant(plant,folder/"plant.npz")
    return plant


def trial_refit(folder,trial,rec,frozen,sweep,pre,pair_path,decision):
    folder.mkdir()
    src,_,_,ctx,q0,_=servo.case_inputs(trial,rec,source_data(trial,rec),frozen)
    timebase=core.data_arrays(pre/trial/"input.npz")
    pair=core.read(pair_path/trial/"status.json")
    input_hash=core.sha(pre/trial/"input.npz")
    assert decision["sufficient_for_current_Y_baseline"] and decision["unchanged_protection_pass"]
    assert input_hash==pair["mapped_input_sha256"]==decision["input_sha256"]
    np.testing.assert_array_equal(q0,timebase["initial_q_rad"])
    for key in ("dq","dqd"):
        np.testing.assert_array_equal(ctx["scaled"][key],timebase[key])
    np.testing.assert_array_equal(ctx["pressure"],timebase["pressure"])
    ctx=replace_nominal(ctx,timebase["q_nom"],timebase["nominal_xyz_m"])
    t=ctx["c"]["t"]
    env=old.model()
    env.reset(timebase["q_nom"][0])
    j=env.jacp()
    direction=j.T@np.linalg.solve(j@j.T,[0.,1.,0.])
    ref=prior.Schedule(rec,rec["bases_um"])
    hook=servo.VelocityReference(core,prior.TimingLogger,timebase["qdot_nom_rad_s"],CASE["command_ms"])
    no_fit=core.data_arrays(pair_path/trial/"velocity_on"/"trace.npz")
    assert not np.any(no_fit["Fd"]) and not np.any(no_fit["common_delta_y_m"])
    np.testing.assert_array_equal(no_fit["q_nominal"],timebase["q_nom"])
    ft=src["frame_host_time_s"]
    valid=src["frame_valid"]
    a=src["A_centered_um"]
    residual=np.interp(ft,t,no_fit["physical_e_um"])-a
    medians={name:float(np.median(residual[valid & prior.mask(ft,rec["windows_s"][name],ft[-1])]))
             for name in prior.PLATFORMS}
    ctx["corr"]=prior.common_correction(t,prior.Schedule(rec,medians),direction)
    plant=plant_for(timebase["q_nom"],folder)
    force=np.zeros(len(t))
    y_attempt=[]
    attempts=[]
    best=None
    reference=-np.inf
    quiet=0
    metadata=core.read(old.PROJECT/"prepared_A"/trial/"source.json")
    for i in range(len(prior.PLAN["iterations"])+1):
        if i:
            residual=np.interp(t,ft[valid],a[valid])-trace["physical_e_um"]
            force=force+prior.PLAN["iterations"][i-1]*old.inverse(residual,plant)
        tag=f"N_fit_{i}"
        trace=integrate(sweep,folder/tag,ctx,q0,force,hook)
        scores=prior.score(src,metadata|rec,t,trace["physical_e_um"],ref)
        y_attempt.append(trace["physical_e_um"])
        values=[scores[name][layer]["fit"] for name in prior.FIT_WINDOWS for layer in ("raw","processed")]
        fit=min(v for v in values if v is not None)
        attempts.append(dict(name=tag,min_fit=fit,force_sha256=servo.array_id(force),
                             force_peak_N=float(np.max(abs(force))),scores=scores))
        if best is None or fit>best[0]:
            best=(fit,trace,force.copy(),i)
        if fit-reference>=prior.CONFIG["minimum_material_fit_gain"]:
            reference,quiet=fit,0
        else:
            quiet+=1
        if best[0]>=prior.PLAN["fit_target"] or quiet>=prior.CONFIG["consecutive_iterations_without_material_gain"]:
            break
    fit,chosen,force,index=best
    pulses={}
    for label,amplitude in (("plus",1.),("minus",-1.)):
        probe=integrate(sweep,folder/("micro_"+label),ctx,q0,force,hook,amplitude)
        delta=probe["physical_e_um"]-chosen["physical_e_um"]
        micro=probe["actual_micro_q"]
        np.testing.assert_array_equal(probe["q_velocity_ff_rad"],chosen["q_velocity_ff_rad"])
        np.testing.assert_array_equal(probe["Fd"],chosen["Fd"])
        assert not delta[:801].any() and np.sign(delta[801])==np.sign(amplitude)
        assert np.any(micro[800:856]) and not micro[:800].any() and not micro[856:].any()
        pulses[label+"_delta_y_um"]=delta
        pulses[label+"_micro_q_rad"]=micro[800:856:8]
    zero_corr=prior.common_correction(t,prior.Schedule(rec,dict.fromkeys(prior.PLATFORMS,0.)),direction)
    no_public=integrate(sweep,folder/"same_force_no_public",ctx|dict(corr=zero_corr),q0,force,hook)
    fitted_axis,g,f=analyze(t,chosen["physical_xyz_m"][:,0],chosen["vx_m_s"]*1000,rec)
    measured=core.read(pre/trial/"input.json")["mapped_actual"]["events"]
    markers={}
    for k in ("start50","turn_xmax","return50"):
        a0,b0=measured[k]["time_s"],fitted_axis["events"][k]["time_s"]
        markers[k]=None if a0 is None or b0 is None else (b0-a0)*1000
    lock_fields=dict(t=t,fit_y_um=chosen["physical_e_um"],all_attempt_y_um=np.stack(y_attempt),force_N=force,
        fitted_xyz_m=chosen["physical_xyz_m"],vx_m_s=chosen["vx_m_s"],remaining_m=chosen["remaining"],
        remaining_dot_m_s=chosen["remaining_dot"],guarded_updated=chosen["updated"],
        common_command_y_m=ctx["corr"]["common_delta_y_m"],common_delta_q=ctx["corr"]["common_delta_q"],
        physical_common_effect_y_um=chosen["physical_e_um"]-no_public["physical_e_um"],
        no_fit_y_um=no_fit["physical_e_um"],score_reference_um=ref.at(t)[0],
        feedback_reference_um=ctx["reference"].at(t)[0],
        velocity_ff_at_ticks_rad=chosen["q_velocity_ff_rad"][::8],**pulses)
    np.savez_compressed(folder/"refit_review.npz",**lock_fields)
    status=dict(best_index=index,best_min_fit=fit,physical_Y_major_windows_gate=fit>=prior.PLAN["fit_min"],
           fitted_X_events_ms=markers,common_residual_platform_um=medians,
           initial_stationary_y_um=float(chosen["physical_e_um"][0]),
           frequency_response_sha256=core.sha(folder/"plant.npz"),plant_qdot_nom_rad_s=[0.]*6,
           attempts=attempts,positive_micro_peak_um=float(pulses["plus_delta_y_um"].max()),
           negative_micro_peak_um=float(pulses["minus_delta_y_um"].min()),pulse_first_response_s=.801,
           pulse_pass=True,paired_X_historical_gate=pair["gate"]["pass_all"],
           mapped_X_input_sha256=input_hash)
    write(folder/"status.json",status)
    print(trial,"N FIT",fit,"signed pulse",status["positive_micro_peak_um"],status["negative_micro_peak_um"],flush=True)
    return status


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--prepared",type=Path,required=True)
    p.add_argument("--pairs",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    x_policy=ROOT/"x_reference_for_y.json"
    decision=core.read(x_policy)
    assert decision["continuation_authorized_by_current_user"] and not decision["parallel_controls"]
    paired=core.read(a.pairs/"receipt.json")
    assert list(paired["conditions"])==list(TRIALS)
    windows=core.read(WINDOW_PATH)["trials"]
    a.output.mkdir()
    integration=a.output/"integration_source"
    integration.mkdir()
    sweep=core.integration_module(integration)
    frozen=core.context()
    result={}
    for k in TRIALS:
        result[k]=trial_refit(a.output/k,k,windows[k],frozen,sweep,a.prepared,a.pairs,decision["results"][k])
        write(a.output/"receipt.json",dict(results=result,formal_L_SFC=0,historical_X_gate=paired["x_pass"],
            Y_baseline_motion_windows_pass=len(result)==3 and all(s["physical_Y_major_windows_gate"] for s in result.values()),
            signed_micro_compensation_verified=len(result)==3 and all(s["pulse_pass"] for s in result.values()),
            rule_sha256=core.sha(x_policy),source_window_sha256=core.sha(WINDOW_PATH),
            paired_receipt_sha256=core.sha(a.pairs/"receipt.json")))


if __name__=="__main__":
    main()
