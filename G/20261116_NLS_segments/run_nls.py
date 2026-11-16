"""Run the fixed, independent three-trial N/L/S comparison offline."""
import argparse
import difflib
import inspect
import os
from pathlib import Path
import sys

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parent
ARCHIVE_NAME = "G_CB3_A_reference_v1_20261116"
DEFAULT_ARCHIVE = ROOT.parent/"baselines"/ARCHIVE_NAME


def select_integrator(prior, output):
    original = inspect.getsource(prior.integrate)
    replacements = (
        ('sweep.run_case(path.name,"A0",', 'sweep.run_case(path.name,core.MODES[case["law"]],'),
        ('    if not pulse:\n', '    if case["law"] == "N" and not pulse:\n'),
    )
    revised = original
    for needle, replacement in replacements:
        if revised.count(needle) != 1:
            raise ValueError(f"Inherited integrator changed at {needle!r}")
        revised = revised.replace(needle, replacement)
    file_source = Path(prior.__file__).read_text(encoding="utf-8")
    if file_source.count(original) != 1:
        raise ValueError("Unable to anchor the mode overlay to the source file")
    path = "G/20261116_preprocess_timing/run_preprocess.py"
    patch = "".join(difflib.unified_diff(file_source.splitlines(True),
        file_source.replace(original,revised).splitlines(True),"a/"+path,"b/"+path))
    (output/"mode_overlay.patch").write_text(patch, encoding="utf-8", newline="\n")
    namespace = vars(prior).copy()
    exec(compile(revised,"<pinned integrate with N/L/S mode overlay>","exec"),namespace)
    return namespace["integrate"]


def compare_regression(data, identity, archived_full, file_sha, array_identity):
    expected = identity["trace_arrays"]
    observed = {key: array_identity(data[key]) for key in expected}
    exact = {key: observed[key] == value for key, value in expected.items()}
    maxima = {key: 0.0 if ok else None for key, ok in exact.items()}
    present = archived_full.is_file()
    if present:
        assert file_sha(archived_full) == identity["full_files_sha256"]["trace.npz"]
        with np.load(archived_full,allow_pickle=False) as z:
            for key in maxima:
                if data[key].shape == z[key].shape:
                    delta = np.abs(data[key].astype(float) - z[key].astype(float))
                    maxima[key] = float(np.max(delta)) if np.isfinite(delta).all() else None
    return {"full_time_arrays": len(exact), "all_arrays_byte_identical": all(exact.values()),
            "full_archive_locally_compared": present, "absolute_tolerance": 0.0,
            "relative_tolerance": 0.0, "byte_identity_by_array": exact,
            "max_absolute_difference_by_array": maxima,
            "maximum_absolute_difference": max((v for v in maxima.values() if v is not None),default=None)}


def checked_environment(archive, read):
    import mujoco
    import ruckig

    actual = {
        "python": ".".join(map(str,sys.version_info[:3])), "numpy": np.__version__,
        "scipy": scipy.__version__, "mujoco": mujoco.__version__, "ruckig": ruckig.__version__,
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
    }
    assert actual == read(archive/"environment.json")["offline_integrator"], actual
    return actual


def evidence_logger(prior):
    class AuditLogger(prior.TimingLogger):
        def __init__(self,path,ctx,case,sensor,channel):
            super().__init__(path,ctx,case,sensor,channel)
            self.arr["J_y_m_rad"] = np.empty((self.n,6))

        def sample(self,i,env,*args):
            super().sample(i,env,*args)
            self.arr["J_y_m_rad"][i] = env.jacp()[1]

    return AuditLogger


def setup_trial(archive, trial, prior, config, read, file_sha, array_identity, correction):
    core = prior.core
    path = archive/"trials"/trial
    window = read(archive/"windows_frozen.json")["trials"][trial]
    inputs = core.data_arrays(path/"input.npz")
    shared = core.data_arrays(path/"shared.npz")
    identity = read(path/"identity.json")
    assert file_sha(path/"A_source.npz") == window["source_npz_sha256"]
    assert {k: array_identity(v) for k,v in inputs.items()} == identity["input_arrays"]
    assert {k: array_identity(v) for k,v in shared.items()} == identity["shared_arrays"]
    t,q,xyz = (inputs[k] for k in ("t","q_nom","nominal_xyz_m"))
    c = dict(t=t,q=q,along=(xyz[:,0]-xyz[0,0])*1000,phase=shared["phase"])
    scaled = c | {k: inputs[k] for k in ("dq","dqd")}
    scaled.update(remaining=np.zeros(len(t)), remaining_dot=np.zeros(len(t)))
    model = prior.compiled_mapping(prior.old.model().model)
    receipt = read(path/"N_case.json")
    assert {k: receipt["parameters"][k] for k in config["parameter_check"]} == config["parameter_check"]
    assert {k: receipt["case"][k] for k in config["case_check"]} == config["case_check"]
    assert model["Kp_Nm_per_rad"] == [5000]*6 and model["Kd_Nms_per_rad"] == [500]*6
    assert model["dt"] == 0.001
    assert prior.PLAN["camera_hz"] == core.CONFIG["camera_hz"] == 132
    np.testing.assert_array_equal(core.pressure(receipt["case"]["pressure"]),inputs["pressure"])
    assert not receipt["remaining_active"] and receipt["camera_noise_m"] == 0
    prior.core.aconfig.Y_NOMINAL = receipt["nominal_y_world_m"]
    ref = prior.Schedule(window,window["bases_um"])
    np.testing.assert_array_equal(ref.at(t)[0],shared["score_reference_um"])
    ctx = dict(c=c,scaled=scaled,reference=ref,corr={k:shared[k] for k in correction},
               nominal_xyz=xyz,pressure=inputs["pressure"],input_id=array_identity(q)["sha256"],
               parameters=receipt["parameters"],compiled=model)
    return ctx, inputs, shared, receipt, identity


def check_control(a, frames, ctx, case):
    law = case["law"]
    p = ctx["parameters"]
    step = np.arange(len(a["t"]))
    tick = step[::case["command_ms"]]
    assert np.array_equal(np.flatnonzero(a["updated"]),tick)
    assert np.array_equal(np.flatnonzero(a["exec_event"]),tick)
    assert np.array_equal(a["applied_source_compute"],tick[step//case["command_ms"]])
    assert np.array_equal(a["hold_until_exclusive"],a["applied_source_compute"]+case["command_ms"])
    ready = a["compute_ready"][tick].astype(bool)
    e = a["sample_e_m"][tick]
    v = a["sample_v_m_s"][tick]
    np.testing.assert_array_equal(v[ready],a["compute_filtered_v"][tick][ready])
    np.testing.assert_array_equal(a["F_requested"],a["applied_force"])
    components = np.column_stack([a[k][tick] for k in ("F_spring","F_linear","F_shear")])
    expected = np.zeros_like(components)
    if law != "N":
        expected[:,0] = -p["K"]*e
        expected[:,1] = -p["B0"]*v
    if law == "S":
        expected[:,2] = -case["gamma"]*p["mu"]*np.abs(v)**(p["n"]-1)*v
        assert np.any(components[:,2] != 0)
    np.testing.assert_allclose(components,expected,atol=1e-12,rtol=0)
    np.testing.assert_allclose(components.sum(1),a["F_requested"][tick],atol=1e-12,rtol=0)
    np.testing.assert_allclose(a["tau_requested"][tick],a["J_y_m_rad"][tick]*a["F_requested"][tick,None],
                               atol=1e-12,rtol=0)
    np.testing.assert_allclose(a["actual_micro_q"][tick],a["tau_requested"][tick]/5000,atol=1e-15,rtol=0)
    base = a["q_nominal"][tick]+a["dq_C"][tick]+a["common_delta_q"][tick]+a["q_velocity_ff_rad"][tick]
    np.testing.assert_array_equal(a["base_ctrl"][tick],base)
    for role in ("camera","encoder"):
        values = frames[role]
        sample = np.array([f["sample"] for f in values])
        delay = case["camera_delay_ms"] if role == "camera" else 0
        assert np.all(np.array([f["available"]-f["model_step"] for f in values]) == delay)
        np.testing.assert_allclose(sample,np.arange(len(sample))*1000/(132 if role=="camera" else 125),atol=1e-10,rtol=0)
        np.testing.assert_array_equal([f["reference_m"] for f in values],
                                     [ctx["reference"].value(t/1000)*1e-6 for t in sample])
    sat = {k: int(np.count_nonzero(np.any(a[k],axis=1)))
           for k in ("ctrl_saturation","force_saturation","joint_saturation")}
    if law != "N":
        assert np.any(a["actual_micro_q"]) and np.any(a["tau_realized_increment"])
    return {"control_formula_max_abs_N": float(np.max(np.abs(components-expected))),
            "nonzero_nonlinear_tick_count": int(np.count_nonzero(components[:,2])),
            "nonzero_micro_sample_count": int(np.count_nonzero(np.any(a["actual_micro_q"],axis=1))),
            "saturated_physics_step_counts": sat,
            "camera_age_ms_range_at_ready_commands": [float(np.min(a["compute_cam_age_ms"][tick][ready])),
                                                     float(np.max(a["compute_cam_age_ms"][tick][ready]))],
            "encoder_age_ms_range_at_ready_commands": [float(np.min(a["compute_enc_age_ms"][tick][ready])),
                                                       float(np.max(a["compute_enc_age_ms"][tick][ready]))],
            "whole_command_hold_and_causality_verified": True, "physical_force_route_verified": True,
            "separately_initialized": True, "planner_enabled": False,
            "camera_and_encoder_sample_own_reference_times": True}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--archive",type=Path,default=DEFAULT_ARCHIVE)
    cli.add_argument("--bound",type=Path,required=True)
    cli.add_argument("--output",type=Path,required=True)
    cli.add_argument("--resume-first-receipt",action="store_true",
                     help="Recover a completed R01 N after receipt-writing failure, without rerunning it")
    cli.add_argument("--n-only",choices=["X_P05_R01","X_P05_R02","X_P10_R01"],
                     help="Reproduce one archived N in a separate output directory")
    args = cli.parse_args()
    archive, bound, output = (p.resolve(strict=strict) for p,strict in
                              ((args.archive,True),(args.bound,True),(args.output,False)))
    sys.path.insert(0,str(archive))
    from archive_baseline import (TRIALS,CORRECTION, read,write,file_sha,array_identity,PREPROCESS,SERVO)
    for name, expected in read(archive/"SHA256SUMS.json").items():
        if file_sha(archive/name) != expected:
            raise ValueError(f"Archive member differs: {name}")
    source = bound/"source"
    manifest = read(archive/"pinned_source_sha256.json")
    for name, expected in manifest["files"].items():
        if file_sha(source/name) != expected:
            raise ValueError(f"Pinned source member differs: {name}")
    sys.path.insert(0,str(source/PREPROCESS))
    sys.path.insert(0,str(source/SERVO))
    import run_preprocess as prior
    from velocity_reference import VelocityReference
    config = read(ROOT/"comparison.json")
    assert prior.core.MODES == config["mode_check"]
    actual_env = checked_environment(archive,read)
    output.mkdir(parents=True,exist_ok=args.resume_first_receipt)
    integration_source = output/"integration_source"
    integration_source.mkdir(exist_ok=args.resume_first_receipt)
    sweep = prior.core.integration_module(integration_source)
    integrate = select_integrator(prior,output)
    ledger = {"archive_commit":config["archive_commit"], "source_commit":manifest["source_commit"],
              "environment":actual_env, "runs":[]}
    write(output/"execution.json",ledger)
    trials = [args.n_only] if args.n_only else TRIALS
    for trial in trials:
        ctx,inp,shared,config_receipt,identity = setup_trial(
            archive,trial,prior,config,read,file_sha,array_identity,CORRECTION)
        (output/trial).mkdir(exist_ok=args.resume_first_receipt and trial == next(iter(TRIALS)))
        for law in ("N" if args.n_only else "NLS"):
            folder = output/trial/law
            case = config_receipt["case"]|{"law":law,"id":f"{trial}_{law}"}
            hook = VelocityReference(prior.core,evidence_logger(prior),inp["qdot_nom_rad_s"],case["command_ms"])
            entry = {"trial":trial,"law":law,"physical_mode":prior.core.MODES[law],
                     "input_file_sha256":file_sha(archive/"trials"/trial/"input.npz"),
                     "shared_file_sha256":file_sha(archive/"trials"/trial/"shared.npz"),
                     "windows_sha256":file_sha(archive/"windows_frozen.json"),
                     "compiled":prior.core.clean(ctx["compiled"]),"planning":False}
            ledger["runs"].append(entry)
            try:
                recover = args.resume_first_receipt and trial == "X_P05_R01" and law == "N"
                if recover:
                    assert read(folder/"case.json")["case"] == case
                    assert read(folder/"checks.json")["complete"] and read(folder/"guard.json") is None
                    assert [p.name for p in (output/trial).iterdir()] == ["N"]
                    a = prior.core.data_arrays(folder/"trace.npz")
                    frames = read(folder/"sample_receipt.json")
                    entry["integration_reused_after_receipt_serialization_fix"] = True
                    print(trial,"N full integration retained from first launch",flush=True)
                else:
                    a,frames = integrate(sweep,folder,ctx,case,shared["Fd"],inp["initial_q_rad"],execution=hook)
            except RuntimeError:
                guard = read(folder/"guard.json")
                row = read(folder/"row.json")
                if guard is None and row["status"]=="OK":
                    raise
                entry.update(status="protection_or_invalid_prefix",guard=guard,row=row)
                write(output/"execution.json",ledger)
                if law == "N":
                    raise
                continue
            assert read(folder/"row.json")["mode"] == prior.core.MODES[law]
            entry.update(status="complete",steps=len(a["t"]),audit=check_control(a,frames,ctx,case))
            if law == "N":
                entry["regression"] = compare_regression(a,identity,archive/"local/archived_full"/trial/"trace.npz",
                                                         file_sha,array_identity)
            else:
                with np.load(output/trial/"N/trace.npz",allow_pickle=False) as n:
                    delta = a["physical_e_um"]-n["physical_e_um"]
                    assert np.any(delta) and not np.array_equal(a["q"],n["q"])
                    first = np.flatnonzero(delta)[0]
                    entry["observed_closed_loop_change"] = {"first_physical_Y_change_s":float(a["t"][first]),
                         "max_abs_Y_change_um":float(np.max(np.abs(delta)))}
            write(folder/"audit.json",entry)
            write(output/"execution.json",ledger)
            if law == "N":
                reg = entry["regression"]
                print(trial,"N archive regression:",reg["full_time_arrays"],"arrays, max diff",
                      reg["maximum_absolute_difference"],flush=True)
                assert reg["all_arrays_byte_identical"], "N regression differs from reviewed archive"


if __name__ == "__main__":
    main()
