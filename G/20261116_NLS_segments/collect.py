"""Export original-precision run arrays and score the locked windows."""
import argparse
from hashlib import file_digest
import json
from pathlib import Path
from shutil import copyfile

import numpy as np

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT.parent/"baselines/G_CB3_A_reference_v1_20261116"
CASES = ("X_P05_R01","X_P05_R02","X_P10_R01")
MODES = "NLS"
MAIN_WINDOWS = ("start","turn","outbound","return_straight")
PHYSICAL = ("physical_e_um","physical_v","vx_m_s","q","qdot","qfrc_actuator",
            "tau_realized_increment","ctrl_saturation","force_saturation","joint_saturation")
COMMAND = ("sample_e_m","sample_v_m_s","F_spring","F_linear","F_shear","F_requested",
           "tau_requested","J_y_m_rad","actual_micro_q","final_ctrl","base_ctrl","prehold_tau",
           "qfrc_applied")
LOCKED_INPUT = ("t","q_nom","nominal_xyz_m","qdot_nom_rad_s","initial_q_rad","nom_x_m")
BACKGROUND = ("dq","dqd","pressure")
RECEIPTS = ("case.json","row.json","guard.json","checks.json","audit.json","trace.json")


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream,"sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n",
                    encoding="utf-8",newline="\n")


def load(path):
    with np.load(path,allow_pickle=False) as data:
        return dict(data)


def window_indices(t, win, recorded_end):
    yield "full", np.arange(len(t))
    for key,(lo,hi) in win.items():
        right = (t<=hi) if hi == recorded_end else (t<hi)
        yield key,np.flatnonzero((t>=lo)&right)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else None


def magnitude(a):
    return np.linalg.norm(a,axis=1) if a.ndim == 2 else np.abs(a)


def stats(a):
    return {"rms":rms(a),"peak":float(np.max(np.abs(a))) if len(a) else None}


def metric_rows(trial, mode, t, p, commands, reference, win, recorded_end, complete):
    e = p["physical_e_um"] - reference[:len(t)]
    hold = np.searchsorted(commands["step"],np.arange(len(t)),side="right")-1
    for key,idx in window_indices(t,win,recorded_end):
        peak = int(idx[np.argmax(np.abs(e[idx]))]) if len(idx) else None
        entry = dict(trial=trial,mode=mode,window=key,n=len(idx),complete_run=complete,
                     score_um=stats(e[idx]),peak_time_s=float(t[peak]) if peak is not None else None)
        for field,label in (("F_requested","requested_N"),("F_spring","spring_N"),("F_linear","linear_N"),
                            ("F_shear","nonlinear_N"),("actual_micro_q","micro_q_rad"),("tau_requested","request_tau_Nm")):
            entry[label] = stats(magnitude(commands[field][hold[idx]]))
        entry["actuator_total_tau_Nm"] = stats(magnitude(p["qfrc_actuator"][idx]))
        inc = p["tau_realized_increment"]
        entry["realized_increment_tau_Nm"] = stats(magnitude(inc[idx]))
        pair = idx[1:][np.diff(idx)==1]
        entry["realized_increment_tau_rate_Nm_s"] = stats(magnitude((inc[pair]-inc[pair-1])/.001))
        entry["saturation_steps"] = {k:int(np.count_nonzero(np.any(p[k][idx],axis=1)))
                                     for k in ("ctrl_saturation","force_saturation","joint_saturation")}
        yield entry


def reduction(n, other, floor):
    return 100*(n-other)/n if n is not None and other is not None and n>floor else None


def comparison_rows(rows, floor):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["trial"],row["window"]),{})[row["mode"]] = row
    result = []
    for (trial,window), modes in grouped.items():
        entry = {"trial":trial, "window":window}
        for candidate,reference in ("LN","SN","SL"):
            c,b = modes.get(candidate),modes.get(reference)
            available = c and b and c["complete_run"] and b["complete_run"] and c["n"] == b["n"]
            entry[candidate+reference] = {kind:reduction(b["score_um"][kind],c["score_um"][kind],floor)
                                if available else None for kind in ("rms","peak")}
        result.append(entry)
    return result


def native_table(rows):
    return {key:np.array([row[key] for row in rows]) for key in rows[0]} if rows else {}


def export(root, archive, stage):
    config = read(ROOT/"comparison.json")
    windows = read(archive/"windows_frozen.json")
    source = root/"execution.json"
    ledger = read(source)
    assert [(r["trial"],r["law"]) for r in ledger["runs"]] == [(t,m) for t in CASES for m in MODES]
    data = stage/"data"
    data.mkdir(parents=True)
    for filename in ("windows_frozen.json","environment.json","pinned_source_sha256.json","reviewed_input_lock.json"):
        copyfile(archive/filename,stage/filename)
    copyfile(source,stage/"execution.json")
    copyfile(root/"mode_overlay.patch",stage/"mode_overlay.patch")
    copyfile(root/"integration_source/command_channel.patch",stage/"inherited_command_channel.patch")
    metrics = []
    background = load(archive/"trials"/CASES[0]/"input.npz")
    np.savez_compressed(data/"shared_background.npz",**{k:background[k] for k in BACKGROUND})
    offset = 0.
    for trial in CASES:
        target = data/trial
        target.mkdir()
        prep = archive/"trials"/trial
        for filename in ("A_source.npz","source.json","shared.npz","identity.json",
                         "inherited_pulse.npz","N_case.json","N_checks.json"):
            copyfile(prep/filename,target/filename)
        with np.load(prep/"input.npz",allow_pickle=False) as inputs:
            t = inputs["t"]
            for k in BACKGROUND:
                np.testing.assert_array_equal(inputs[k],background[k][:len(inputs[k])])
            np.savez_compressed(target/"locked_input.npz",**{k:inputs[k] for k in LOCKED_INPUT})
        with np.load(prep/"A_source.npz",allow_pickle=False) as original:
            recorded_end = original["frame_host_time_s"][-1]
        with np.load(prep/"shared.npz",allow_pickle=False) as shared:
            reference,fd = shared["score_reference_um"],shared["Fd"]
        np.savez_compressed(target/"display_mapping.npz",trial_id=np.array(trial),
                            original_time_s=t,display_time_s=t+offset)
        offset = float(t[-1]+offset+config["display_gap_s"])
        for mode in MODES:
            src = root/trial/mode
            dest = target/mode
            dest.mkdir()
            for filename in RECEIPTS:
                if (src/filename).exists():
                    copyfile(src/filename,dest/filename)
            with np.load(src/"trace.npz",allow_pickle=False) as a:
                valid = len(a["t"])
                np.testing.assert_array_equal(a["t"],t[:valid])
                physical = {k:a[k] for k in PHYSICAL}
                physical["physical_x_m"] = a["physical_xyz_m"][:,0]
                tick = np.flatnonzero(a["updated"])
                command = {k:a[k][tick] for k in COMMAND}
                command["step"] = tick
                if mode == "N":
                    np.savez_compressed(target/"velocity_nominal_native.npz",step=tick,
                        qdot_nom_command_rad_s=a["qdot_nom_command_rad_s"][tick],
                        q_velocity_ff_rad=a["q_velocity_ff_rad"][tick])
                with np.load(target/"velocity_nominal_native.npz",allow_pickle=False) as vel:
                    for k in ("qdot_nom_command_rad_s","q_velocity_ff_rad"):
                        np.testing.assert_array_equal(a[k][tick],vel[k][:len(tick)])
                np.savez_compressed(dest/"physical_1ms.npz",**physical)
                np.savez_compressed(dest/"command_native.npz",**command)
                complete = read(src/"row.json")["status"]=="OK"
                metrics.extend(metric_rows(trial,mode,t[:valid],physical,command,reference,
                    windows["trials"][trial]["windows_s"],recorded_end,complete))
                write(dest/"trace_provenance.json", {
                    "full_trace_sha256":digest(src/"trace.npz"),"valid_physics_samples":valid,
                    "all_step_Fd_exact":np.array_equal(a["Fd"],fd[:valid]),
                    "all_step_external_applied_equals_only_disturbance":np.array_equal(a["qfrc_applied"],a["qfrc_disturbance"]),
                    "all_step_remaining_and_derivative_zero":not (a["remaining"].any() or a["remaining_dot"].any()),
                    "control_ticks":len(tick),"dtype_or_precision_conversion":False,
                    "physics_time":"shared input.npz t, truncated only at the actual valid prefix",
                    "full_results_B_relative_to_work_item":f"local/integrals/{trial}/{mode}",
                })
            frames = read(src/"sample_receipt.json")
            for kind in ("camera","encoder"):
                np.savez_compressed(dest/f"{kind}_native.npz",**native_table(frames[kind]))
            events = read(src/"command_events.json")
            np.savez_compressed(dest/"command_events_native.npz",**native_table(events))
    ratios = comparison_rows(metrics,config["reduction_denominator_min_um"])
    write(stage/"metrics.json",{"windows":metrics,"reductions_percent":ratios,
          "historical_costs":{"implemented":"V02 strong.py: per-sample six-joint realized incremental torque "
             "vector norm RMS/peak, and adjacent 1 ms finite difference vector norm RMS/peak. "
             "Joint actuator total RMS/peak and requested effort are separately labeled.",
             "integral_position_velocity_total_torque_and_rate_costs":"Unavailable as separately defined cost "
             "integrals for this archived controller path. Physical 1 ms Y/Y velocity/total actuator torque "
             "and native command e/v are exported at original precision.",
             "predicted_residual":"Not applicable; no prediction runs."}})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--integrals",type=Path,required=True)
    p.add_argument("--archive",type=Path,default=ARCHIVE)
    p.add_argument("--stage",type=Path,required=True)
    a=p.parse_args()
    export(a.integrals,a.archive,a.stage)


if __name__ == "__main__":
    main()
