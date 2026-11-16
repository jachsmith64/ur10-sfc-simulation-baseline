"""Offline G adapter for the copied V02 integration, guard and measurement loop."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import run_campaign as core

PROJECT = Path(__file__).resolve().parent
REPO = PROJECT.parents[1]
BASELINE = REPO / "simulation/engineering_baseline_v2"
TURNING = BASELINE / "cb3/turning"
sys.path.insert(0, str(TURNING))

import mujoco
import numpy as np
import disturbance
from b132_common import save
from filter_sensing import Measurement
from increment_model import aligned_model
from position_execution import POSITION_EQUIV, compiled_mapping
from score_g import evaluate
from source_a import TRIALS, load_source, nominal, speed

PLAN = core.read(PROJECT / "g_config.json")
CB3 = core.read(TURNING / "increment_config.json")


def model():
    return aligned_model(CB3)


def smooth_platforms(t, windows, values):
    # F historical run_slow.slow_center: constant platforms, C2 quintic connections.
    b = np.full_like(t, values[0])
    bd = np.zeros_like(t)
    bdd = np.zeros_like(t)
    for (_, lo), (hi, _), a, z in zip(windows[:-1], windows[1:], values[:-1], values[1:], strict=True):
        inside = (t >= lo) & (t <= hi)
        span = hi - lo
        u = (t[inside] - lo) / span
        b[inside] = a + (z-a) * u**3 * (10 - 15*u + 6*u*u)
        bd[inside] = (z-a) * 30*u*u*(1-u)**2 / span
        bdd[inside] = (z-a) * (60*u - 180*u*u + 120*u**3) / span**2
        b[t > hi] = z
    return b, bd, bdd


def correction(t, windows, levels, direction):
    b, d, dd = smooth_platforms(t, windows, levels)
    return dict(common_delta_q=(-1e-6*b[:, None])*direction,
                common_delta_q_dot=(-1e-6*d[:, None])*direction,
                common_delta_q_ddot=(-1e-6*dd[:, None])*direction,
                common_baseline_um=b, common_delta_y_m=-1e-6*b)


class PulseCoordinator(core.Coordinator):
    def __init__(self, ctx, case, sensor, pulse):
        super().__init__(ctx, case, sensor)
        self.pulse = pulse

    def command(self, i, env, command, out, actuation):
        if 800 <= i < 856:
            actuation.apply_requested_torque(env, self.pulse * env.jacp()[1])
            command = command + actuation.delta
        return super().command(i, env, command, out, actuation)


def run_case(sweep, folder, ctx, case, force, pulse=0.):
    folder.mkdir(parents=True)
    t = ctx["c"]["t"]
    p = ctx["parameters"]
    sensor = Measurement(core.CONFIG, case, core.aconfig.Y_NOMINAL,
                         ctx["pressure"], np.zeros(int(np.ceil((t[-1]+.001)*PLAN["camera_hz"]))+1))
    ext = PulseCoordinator(ctx, case, sensor, pulse) if pulse else core.Coordinator(ctx, case, sensor)
    channel = core.CommandChannel(case["execution_delay_ms"], case["command_ms"], ext)
    logger = core.Logger(folder / "trace.npz", ctx, case, sensor, channel)
    force_sha = sha256(force.tobytes()).hexdigest()
    fd = disturbance.FrozenDisturbance(Path("G_equivalent_force"), t, force, {"frozen":{"sha256":force_sha}})
    save(folder/"case.json", dict(case=case, parameters=p | {"B0":p["B0"]*case["linear_multiplier"]},
                                 diagnostic_pulse_N=pulse, plan=PLAN, compiled=ctx["compiled"],
                                 nominal_y_m=core.aconfig.Y_NOMINAL, force_sha256=force_sha,
                                 input_hashes={k: sha256(np.asarray(v).tobytes()).hexdigest() for k, v in
                                               (ctx["scaled"] | ctx["corr"] | {"pressure":ctx["pressure"]}).items()}))
    row = sweep.run_case(folder.name, core.MODES[case["law"]], p["K"], p["B0"]*case["linear_multiplier"],
                         p["n"], p["mu"], p["r"], p["v_ref"], fd, ctx["c"]["along"], ctx["c"]["q"],
                         phase_code=ctx["c"]["phase"], c_input=ctx["scaled"], execution_mode=POSITION_EQUIV,
                         control_update_steps=case["command_ms"], precise_logger=logger,
                         common_position_correction=ctx["corr"]["common_delta_q"], e_extension=ext,
                         measurement_adapter=sensor, final_command_hold=True, command_channel=channel)
    save(folder/"row.json", row)
    save(folder/"guard.json", ext.rejection)
    save(folder/"frames.json", core.as_frames(sensor))
    save(folder/"command_events.json", channel.events)
    trace = core.data_arrays(folder/"trace.npz")
    size = len(trace["t"])
    step = np.arange(size)
    check = dict(status=row["status"], planned_steps=len(t), retained_steps=size,
                 full=bool(row["status"]=="OK" and size==len(t)), held=bool(np.array_equal(
                 trace["final_ctrl"][step % 8 != 0], trace["final_ctrl"][step[step % 8 != 0]-1])),
                 external_only=bool(np.array_equal(trace["qfrc_applied"], trace["qfrc_disturbance"])),
                 no_future=all(item["c_available"] <= item["compute"] and item["q_available"] <= item["compute"]
                               for item in channel.events))
    save(folder/"validation.json", check)
    assert check["held"] and check["external_only"] and check["no_future"]
    print(folder.name, row["status"], size, row.get("note"), flush=True)
    return trace, check


def axial_timing(trace, src, grid, nominal_pose, receipt):
    t = trace["t"]
    if len(t) < 2:
        return {"pass": False}
    x = trace["physical_xyz_m"][:, 0]
    v = np.gradient(x, t)*1000
    actual_host = src["host_time_s"]
    actual_v = src["actual_tcp_speed"][:, 0]*1000
    turn_source = receipt["turn_actual_xmax_s"]
    near_turn = (t >= turn_source-.3) & (t <= turn_source+.3)
    half = receipt["speed_mm_s"] / 2
    start = t[(t > .75) & (v > half)]
    return_v = t[(t >= turn_source-.15) & (v < -half)]
    source_return_v = actual_host[(actual_host >= turn_source-.15) & (actual_v < -half)]
    if not len(start) or not near_turn.any() or not len(return_v) or not len(source_return_v):
        return {"pass": False, "reason": "retained prefix has no paired start, turn or return"}
    model_turn = t[near_turn][np.argmax(x[near_turn])]
    shifts = dict(start_speed_s=float(start[0]-receipt["first_half_speed_s"]),
                  turn_xmax_s=float(model_turn-turn_source), return_speed_s=float(return_v[0]-source_return_v[0]))
    actual_x = np.interp(t, actual_host, src["actual_tcp_pose"][:, 0])
    speeds = {k: float(np.median(v[(t >= receipt["windows_s"][k][0]) &
                                  (t <= receipt["windows_s"][k][1])])) for k in ("outbound", "return_straight")}
    return dict(shifts, along_speed_model_mm_s=speeds, rmse_x_model_vs_A_um=float(np.sqrt(np.mean((x-actual_x)**2))*1e6),
                rmse_x_model_vs_nominal_um=float(np.sqrt(np.mean((x-nominal_pose[:len(t), 0])**2))*1e6),
                **{"pass": max(abs(x) for x in shifts.values()) <= PLAN["event_timing_tolerance_s"]})


def inverse(target, plant):
    pad = 2000
    padded = np.pad(target, (pad, pad), mode="edge")
    return disturbance.invert_series(padded, plant, PLAN["pass_hz"], PLAN["stop_hz"])[pad:-pad]


def reconstruct_trial(sweep, out, src_root, prepared_root, trial, frozen):
    if src_root is not None:
        src, receipt, t = load_source(src_root, trial)
    else:
        src = core.data_arrays(prepared_root/trial/"A_source.npz")
        receipt = core.read(prepared_root/trial/"source.json")
        t = np.arange(int(np.ceil(src["frame_host_time_s"][-1]*1000))+1)*.001
    out.mkdir()
    save(out/"source.json", receipt)
    kin, diagnostics = nominal(CB3, src, t)
    print(trial, "IK", diagnostics, flush=True)
    q = kin["nominal_q_rad"]
    xyz = kin["nominal_tcp_m_rotvec_rad"][:, :3]
    core.aconfig.Y_NOMINAL = float(xyz[0, 1])
    c = dict(t=t, q=q, along=(xyz[:,0]-xyz[0,0])*1000,
             phase=np.zeros(len(t),dtype=int))
    for i, (lo, hi) in enumerate(receipt["phase_host_bounds_s"].values(), 1):
        c["phase"][(t >= lo) & (t <= hi)] = i
    fields = ("dq","dqd","remaining","remaining_dot")
    n_old = len(frozen["scaled"]["remaining"])
    if len(t) > n_old:
        raise ValueError("This run exceeds the frozen C background length")
    scaled = c | {k:frozen["scaled"][k][:len(t)].copy() for k in fields}
    pressure = core.pressure(PLAN["encoder_pressure"])
    if len(t[::8]) > len(pressure):
        raise ValueError("Use the original source B repeat/join construction for a longer pressure record")
    env = model()
    env.reset(q[0])
    J = env.jacp()
    direction = J.T @ np.linalg.solve(J @ J.T, [0.,1.,0.])
    mapping = compiled_mapping(env.model)
    mapping["gravity_m_s2"] = env.model.opt.gravity.copy()
    # The numerical probe runs on its own MjData and does not set the live state.
    original_q = core.aconfig.INIT_Q
    dc = disturbance.measure_dc_compliance
    try:
        core.aconfig.INIT_Q = q[len(t)//2].copy()
        disturbance.measure_dc_compliance = lambda: dc(model())
        plant = disturbance.measure_plant(model())
    finally:
        disturbance.measure_dc_compliance = dc
        core.aconfig.INIT_Q = original_q
    disturbance.save_plant(plant, out/"plant.npz")
    sweep.SimEnv = model
    windows = [receipt["windows_s"][k] for k in receipt["platform_names"]]
    ft = src["frame_host_time_s"]
    valid = src["frame_valid"]
    a = src["A_centered_um"]
    picks = [(lo <= ft) & (ft <= hi) & valid for lo, hi in windows]
    a_levels = [float(np.median(a[mask])) for mask in picks]
    trend = smooth_platforms(t, windows, a_levels)[0]
    ctx = dict(c=c, scaled=scaled, pressure=pressure, nominal_xyz=xyz, compiled=mapping,
               parameters=frozen["parameters"], input_id=sha256(q.tobytes()).hexdigest(),
               corr=correction(t, windows, np.zeros(4), direction))
    n_case = core.read(PROJECT/"evidence/D_N_o1_fraw_r1_gain1_seed1001_d0.json")
    force = np.zeros(len(t))
    pre, check = run_case(sweep, out/"N_without_common", ctx, n_case, force)
    if not check["full"]:
        raise RuntimeError("N with frozen background and zero common correction stopped")
    total = pre["physical_e_um"] + pre["remaining"]*1e6
    levels = [float(np.median(np.interp(ft[mask], t, total)-a[mask])) for mask in picks]
    ctx["corr"] = correction(t, windows, levels, direction)
    baseline, check = run_case(sweep, out/"N_common_only", ctx, n_case, force)
    if not check["full"]:
        raise RuntimeError("N with new public table and zero external force stopped")
    force_best = force.copy()
    trace_best = baseline
    best = -np.inf
    attempts = []
    timing = axial_timing(baseline, src, t, kin["nominal_tcp_m_rotvec_rad"], receipt)
    for number, relaxation in enumerate([0.] + PLAN["iterations"]):
        if number:
            total = baseline["physical_e_um"] + baseline["remaining"]*1e6
            residual = np.interp(t, ft[valid], a[valid]) - total
            force = force + relaxation*inverse(residual, plant)
            baseline, check = run_case(sweep, out/f"N_fit_{number}", ctx, n_case, force)
        else:
            check = {"full": True}
        scores = evaluate(t, baseline, src, receipt, trend, PLAN["flat_reference_um"])
        required = [scores[k][layer].get("fit") for k in ("full","start","turn") for layer in ("raw","oscillation")]
        complete = check["full"] and all(scores[k]["complete"] for k in ("full","start","turn"))
        finite = [x for x in required if x is not None]
        minimum = min(finite, default=-np.inf)
        fairness = [{"layer_fit":scores[k]["physical_oscillation"].get("fit"), "label":k}
                    for k in ("full","start","turn")]
        physical_good = all(x["layer_fit"] is not None and x["layer_fit"] >= PLAN["fit_min"] for x in fairness)
        attempts.append(dict(attempt=number, complete=complete, scores=scores,
                             assessment_min=float(minimum) if np.isfinite(minimum) else None,
                             physical_80=physical_good,
                             pressure_sha256=sha256(pressure.tobytes()).hexdigest(),
                             force_max_N=float(np.max(abs(force))), force_sha256=sha256(force.tobytes()).hexdigest()))
        save(out/"attempts.json", attempts)
        if complete and minimum > best:
            best, best_attempt, force_best, trace_best = minimum, number, force.copy(), baseline
            timing = axial_timing(baseline, src, t, kin["nominal_tcp_m_rotvec_rad"], receipt)
        if not complete or (minimum >= PLAN["fit_target"] and physical_good and timing["pass"]):
            break
    if not np.isfinite(best):
        raise RuntimeError("All candidate observations lacked comparable complete windows")
    fit = attempts[best_attempt]
    status = dict(best_attempt=best_attempt, best_min_fit=best, source_timing=timing,
                  reconstruction_pass=bool(best >= PLAN["fit_min"] and fit["physical_80"] and timing["pass"]),
                  ik=diagnostics, common_plateau_residual_um=levels, A_drift_levels_um=a_levels,
                  common_effect_rms_um=float(np.sqrt(np.mean((pre["physical_y"] -
                                   core.data_arrays(out/"N_common_only/trace.npz")["physical_y"])**2))*1e6),
                  remaining_rms_um=float(np.sqrt(np.mean(scaled["remaining"]**2))*1e6),
                  initial_nominal_q_rad=q[0].tolist(), initial_qdot_rad_s=[0.]*6,
                  initial_source_q_rad=src["actual_q_rad"][int(np.argmin(abs(src["host_time_s"])))].tolist(),
                  compiled=mapping, parameters=ctx["parameters"])
    np.savez_compressed(out/"locked_inputs.npz", **(scaled | ctx["corr"]),
                        pressure=pressure, Fd=force_best, nominal_xyz=xyz, A_trend_um=trend)
    np.savez_compressed(out/"original_A.npz", **src)
    save(out/"status.json", status)
    print(trial, "best FIT", best, "timing", timing, "status", status["reconstruction_pass"], flush=True)
    return status


def run_controls(sweep, folder, status, frozen):
    src = core.data_arrays(folder/"original_A.npz")
    source = core.read(folder/"source.json")
    locked = core.data_arrays(folder/"locked_inputs.npz")
    current = {name: locked[name] for name in ("t","q","along","phase")}
    corr = {k:v for k,v in locked.items() if k.startswith("common_")}
    core.aconfig.Y_NOMINAL = float(locked["nominal_xyz"][0,1])
    ctx = dict(c=current, scaled=current | {k:locked[k] for k in ("dq","dqd","remaining","remaining_dot")},
               corr=corr, pressure=locked["pressure"], compiled=status["compiled"],
               nominal_xyz=locked["nominal_xyz"], parameters=frozen["parameters"],
               input_id=sha256(current["q"].tobytes()).hexdigest())
    index = status["best_attempt"]
    n_path = folder/("N_common_only" if index == 0 else f"N_fit_{index}")
    n_trace = core.data_arrays(n_path/"trace.npz")
    n_case = core.read(PROJECT/"evidence/D_N_o1_fraw_r1_gain1_seed1001_d0.json")
    probe = {}
    for label, pulse in (("positive", 1.), ("negative", -1.)):
        a, check = run_case(sweep, folder/f"pulse_{label}", ctx, n_case, locked["Fd"], pulse)
        if not check["full"]:
            save(folder/"pulse_gate.json", dict(passed=False, reason=f"{label} pulse stopped"))
            return {"ran_controls":False, "reason":"pulse stopped"}
        probe[label] = a["physical_e_um"]-n_trace["physical_e_um"]
    positive, negative = probe["positive"], probe["negative"]
    causal = bool(np.array_equal(positive[:801], np.zeros(801)) and
                  np.array_equal(negative[:801], np.zeros(801)))
    passed = bool(causal and positive[801] > 0 > negative[801])
    save(folder/"pulse_gate.json", dict(passed=passed, causal=causal,
                                       positive_peak_um=float(positive.max()),
                                       negative_peak_um=float(negative.min())))
    if not passed:
        return {"ran_controls":False, "reason":"micro-command pulse gate failed"}
    summary = {}
    for cutoff in PLAN["cutoffs_hz"]:
        for law, gain in (("L",1), ("L",4), ("S",1)):
            key = f"L{gain}_{cutoff}Hz" if law == "L" else f"SFC_{cutoff}Hz"
            template = ("D_L_o1_f5_r0_gain4_seed1001_d0" if law=="L" else "D_S_o1_f5_r0_gain1_seed1001_d0")
            case = core.read(PROJECT/f"evidence/{template}.json")
            case.update(id=key, fc=cutoff, linear_multiplier=gain)
            trace, check = run_case(sweep,folder/key,ctx,case,locked["Fd"])
            comparison = {}
            for name,(lo,hi) in source["windows_s"].items():
                span = (current["t"] >= lo) & (current["t"] <= hi)
                if not np.all(np.flatnonzero(span) < len(trace["t"])):
                    comparison[name] = {"complete":False}
                    continue
                y = trace["physical_e_um"][span[:len(trace["t"])]] - locked["A_trend_um"][span]
                n = n_trace["physical_e_um"][span] - locked["A_trend_um"][span]
                comparison[name] = dict(complete=True, physical_rms_um=float(np.sqrt(np.mean(y*y))),
                                        N_physical_rms_um=float(np.sqrt(np.mean(n*n))),
                                        rms_ratio=float(np.linalg.norm(y)/np.linalg.norm(n)) if np.linalg.norm(n) else None)
            summary[key] = {"status":check, "relative_to_N":comparison,
                            "observation":evaluate(current["t"],trace,src,source,locked["A_trend_um"],PLAN["flat_reference_um"])}
            save(folder/"control_results.json",summary)
    return {"ran_controls":True, "case_ids":list(summary)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--prepared-root", type=Path, default=PROJECT/"prepared_A")
    parser.add_argument("--out", type=Path, required=True)
    import csv
    with (PROJECT/"instructions/04_A_registered_candidates.csv").open(encoding="utf-8-sig") as source:
        registered = [c["segment_base"].split("_B05_",1)[1] for c in csv.DictReader(source)]
    parser.add_argument("--trials", nargs="+", choices=registered, default=TRIALS)
    parser.add_argument("--with-controls", action="store_true")
    parser.add_argument("--historical-regression-receipt", type=Path,
                        default=PROJECT/"evidence/regression_receipt.json")
    options = parser.parse_args()
    if len(set(options.trials)) != len(options.trials):
        parser.error("Each trial must be independent")
    reg = core.read(options.historical_regression_receipt)
    if len(reg) != 3 or not all(r["all_arrays_equal"] and r["pinned_by_history"] and
                                 r["inputs_equal"] and r["old_guard"]==r["new_guard"] for r in reg.values()):
        raise RuntimeError("V02 regression receipt has not passed")
    options.out.mkdir(parents=True, exist_ok=False)
    frozen = core.context()
    print("V02 effective controller", frozen["parameters"], flush=True)
    expected = dict(K=5000., B0=400., n=1.5, mu=70879.56438014314)
    assert all(frozen["parameters"][k] == v for k,v in expected.items())
    integration_folder = options.out/"source"
    integration_folder.mkdir()
    sweep = core.integration_module(integration_folder)
    receipt = {}
    for trial in options.trials:
        try:
            status = reconstruct_trial(sweep, options.out/trial, options.source_root, options.prepared_root, trial, frozen)
            receipt[trial] = status
        except (AssertionError, ValueError, RuntimeError, FileNotFoundError, StopIteration) as ex:
            receipt[trial] = dict(reconstruction_pass=False, error=str(ex))
            save(options.out/"receipt.json", receipt)
            import traceback
            traceback.print_exc()
    eligible = [trial for trial, status in receipt.items() if status["reconstruction_pass"]]
    coverage = len(set(map(speed, eligible))) >= 2
    gate = dict(eligible=eligible, multi_speed_event_coverage=coverage, control_requested=options.with_controls,
                controls={})
    if options.with_controls and coverage:
        for trial in eligible:
            gate["controls"][trial] = run_controls(sweep, options.out/trial, receipt[trial], frozen)
    save(options.out/"control_gate.json", gate)
    save(options.out/"receipt.json", receipt)


if __name__ == "__main__":
    main()
