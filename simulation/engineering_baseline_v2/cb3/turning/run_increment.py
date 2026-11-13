"""Run the paired CB3 plant with causal A-background and model-increment feedback."""

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

from increment_model import (ALIGNED_MODEL, aligned_model, checked_actuation, flange, pose_error,
                             reconstruct, recorded_pose, stationary)
from diagnose import MODEL, ExecutionManager, POSITION_EQUIV, sha
from controller import SFCController
from config import ABORT_E_ABS_MAX, ABORT_QVEL_MAX

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "increment_config.json"
sys.path.insert(0, str(HERE / "feedback"))
from filter_sensing import Channel, coefficients


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


class IncrementFeedback:
    def __init__(self, cfg, obs, time):
        self.cfg, self.obs, self.t = cfg, obs, time
        self.c = Channel(cfg["camera_hz"], cfg["camera_availability_delay_ms"], None, 0., False, False)
        self.q = Channel(1000/cfg["encoder_sample_ms"], cfg["encoder_availability_delay_ms"],
                         cfg["velocity_low_pass_hz"], 0., False, True)
        self.fk = aligned_model(cfg)
        self.camera_sources, self.encoder_sources = [], []
        self.start_ms = time[0]*1000
        self.previous = 0.

    @staticmethod
    def prior(source_t, t, max_age):
        idx = int(np.searchsorted(source_t, t+1e-12, side="right")-1)
        if idx < 0 or (t-source_t[idx])*1000 > max_age:
            raise RuntimeError(f"Missing current measurement support at host t={t:.6f}")
        return idx

    def tick(self, i, env, y_delta, q_delta, y_nom):
        cfg, obs = self.cfg, self.obs
        n = len(self.c.frames)
        exposure_ms = n*1000/cfg["camera_hz"]
        if i == (n*1000+cfg["camera_hz"]-1)//cfg["camera_hz"]:
            exposure = (self.start_ms+exposure_ms)*.001
            k = self.prior(obs["frame_host_time_s"], exposure, cfg["max_camera_source_age_ms"])
            if not obs["checker_valid"][k]:
                raise RuntimeError(f"Invalid checker measurement at source frame {k}")
            # This interpolation is between physics knots already computed when
            # the fractional camera exposure is simulated, before publication.
            delta = y_delta + (i-exposure_ms)*(self.previous-y_delta)
            self.c.capture(i, exposure_ms, delta, obs["mapped_visual_y_centered_um"][k]*1e-6)
            self.camera_sources.append(k)
        if i % cfg["encoder_sample_ms"] == 0:
            k = self.prior(obs["host_time_s"], self.t[i], cfg["max_encoder_source_age_ms"])
            q_observed = obs["actual_q_rad"][k]+q_delta
            y_q = flange(self.fk, q_observed)[1, 3]
            self.q.capture(i, float(i), y_q-y_nom, 0.)
            self.encoder_sources.append(k)
        self.previous = y_delta
        self.c.release(i)
        self.q.release(i)

    def measure(self, env, y_d, y_d_dot=0.):
        c, q = self.c.latest, self.q.latest
        return (c.e if c else 0., q.vf if q and q.valid and c else 0.)

    def ages(self, i):
        c, q = self.c.latest, self.q.latest
        result = []
        for sample, indices, source in ((c, self.camera_sources, self.obs["frame_host_time_s"]),
                                         (q, self.encoder_sources, self.obs["host_time_s"])):
            if sample is None:
                result.extend([-1]*5)
                continue
            k = indices[sample.id]
            source_ms = source[k]*1000
            assert source_ms <= self.start_ms+sample.sample+1e-8 and sample.available <= i
            result.extend([self.start_ms+sample.sample, self.start_ms+sample.available, source_ms,
                           i+self.start_ms-source_ms, k])
        return result


def simulate(cfg, obs, trajectory, mode, multiplier, reference=None):
    env = aligned_model(cfg)
    qnom = trajectory["nominal_q_rad"]
    qbase = qnom+trajectory["gravity_offset_rad"]
    t = trajectory["time_s"]
    env.reset(qnom[0])
    actuation = ExecutionManager(env, POSITION_EQUIV)
    feedback = IncrementFeedback(cfg, obs, t) if reference is not None else None
    params = {key: cfg[key] for key in ("K", "B0", "mu", "n", "gamma")}
    params["B0"] *= multiplier
    controller = SFCController(env, params, mode, measurement=feedback, actuation=actuation) if feedback else None
    if controller is None and mode != "A0":
        raise ValueError("Active control requires a paired N trajectory")
    q, v, xyz, ctrl, base, torque, tau, delta, feedback_rows, cost = ([] for _ in range(10))
    result = {"F_sfc": 0., "F_K": 0., "F_B": 0., "F_shear": 0., "tau": np.zeros(6)}
    active = common = qbase[0]
    failure = None
    for i in range(len(t)):
        env.forward()
        y_nom = trajectory["nominal_tcp_m_rotvec_rad"][i, 1]
        if feedback:
            feedback.tick(i, env, env.tcp_y()-reference["xyz_m"][i, 1],
                          env.qpos()-reference["q_rad"][i], y_nom)
        update = i % cfg["control_update_ms"] == 0
        if update:
            if controller:
                result = controller.step(y_nom)
            common = qbase[i]
            active = actuation.command(common)
        actuation.assert_applied(env)
        env.set_ctrl(active)
        try:
            realized = checked_actuation(env)
            if not np.isfinite(np.r_[env.data.qpos, env.data.qvel, realized]).all():
                raise RuntimeError("Non-finite state")
            if abs(env.tcp_y()-y_nom) > ABORT_E_ABS_MAX or max(abs(env.data.qvel)) > ABORT_QVEL_MAX:
                raise RuntimeError("Existing position or speed divergence guard")
        except RuntimeError as exc:
            failure = dict(host_time_s=float(t[i]), reason=str(exc), command_rad=active.tolist(),
                           q_rad=env.qpos().tolist(), qd_rad_s=env.data.qvel.tolist())
            break
        if feedback:
            c, enc = feedback.c.latest, feedback.q.latest
            e, vel = feedback.measure(env, y_nom)
            feedback_rows.append([i, int(update), e, vel, enc.e if enc else 0., enc.v if enc else 0.,
                                  c.additive if c else 0., c.physical_e if c else 0., *feedback.ages(i)])
        q.append(env.qpos())
        v.append(env.data.qvel.copy())
        xyz.append(env.tcp_pos())
        ctrl.append(active)
        base.append(common)
        delta.append(actuation.delta.copy())
        torque.append(realized)
        tau.append(result["tau"])
        cost.append([result[key] for key in ("F_sfc", "F_K", "F_B", "F_shear")])
        if i+1 < len(t):
            env.step()
    arrays = {key: np.asarray(value) for key, value in (
        ("q_rad", q), ("qd_rad_s", v), ("xyz_m", xyz), ("command_rad", ctrl), ("common_command_rad", base),
        ("torque_Nm", torque), ("requested_increment_Nm", tau), ("delta_q_command_rad", delta), ("force_N", cost))}
    arrays["time_s"] = t[:len(q)]
    arrays["feedback"] = np.asarray(feedback_rows)
    if feedback:
        arrays["camera_source_index"] = np.array(feedback.camera_sources, dtype=np.int64)
        arrays["encoder_source_index"] = np.array(feedback.encoder_sources, dtype=np.int64)
        assert np.allclose((arrays["command_rad"]-arrays["common_command_rad"])*actuation.kp,
                           arrays["requested_increment_Nm"], atol=1e-11, rtol=0)
    return arrays, failure


FEEDBACK_COLUMNS = ["physics_index", "compute_and_apply", "published_c_e_m", "published_q_velocity_filtered_m_s",
    "published_q_e_m", "published_q_velocity_raw_m_s", "published_c_A_background_m", "published_c_increment_m",
    "c_sim_sample_host_ms", "c_available_host_ms", "c_raw_source_host_ms", "c_raw_age_ms", "c_source_index",
    "q_sim_sample_host_ms", "q_available_host_ms", "q_raw_source_host_ms", "q_raw_age_ms", "q_source_index"]


def review_metrics(arrays, ref, obs, windows):
    t = arrays["time_s"]
    visual_time = obs["frame_host_time_s"]
    valid = (visual_time >= t[0]) & (visual_time <= t[-1]) & obs["checker_valid"]
    ft = visual_time[valid]
    e_a = obs["mapped_visual_y_centered_um"][valid]
    change = (arrays["xyz_m"][:, 1]-ref["xyz_m"][:len(t), 1])*1e6
    delta = np.interp(ft, t, change)
    error = e_a+delta
    fb = arrays["feedback"]
    arrays.update(metric_frame_time_s=ft, metric_frame_id=obs["frame_id"][valid],
                  background_um=e_a, increment_um=delta, estimated_error_um=error)
    metrics = {}
    for name, (a, b) in windows.items():
        idx = (ft >= a) & (ft <= b)
        q = (t >= a) & (t <= b)
        e = error[idx]
        f = arrays["force_N"][q, 0]
        if len(e) and len(f):
            metrics[name] = dict(frames=int(idx.sum()), rms_um=float(np.sqrt(np.mean(e**2))),
                peak_abs_um=float(np.max(abs(e))), mean_um=float(np.mean(e)),
                increment_rms_um=float(np.sqrt(np.mean(delta[idx]**2))),
                force_rms_N=float(np.sqrt(np.mean(f**2))), force_peak_N=float(np.max(abs(f))),
                torque_increment_rms_vector_Nm=float(np.sqrt(np.mean(np.sum(
                    arrays["requested_increment_Nm"][q]**2,axis=1)))),
                torque_increment_peak_axis_Nm=float(np.max(abs(arrays["requested_increment_Nm"][q]))),
                micro_command_peak_axis_urad=float(np.max(abs(arrays["delta_q_command_rad"][q]))*1e6))
    if len(fb):
        applied = (fb[:, 1]==1) & (fb[:, 11]>=0) & (fb[:, 16]>=0)
        ages = fb[applied][:, [11, 16]]
        age_summary = dict(zip(("camera", "encoder"), (dict(min=float(x.min()),
            median=float(np.median(x)), max=float(x.max())) for x in ages.T)))
    else:
        age_summary = None
    return dict(windows=metrics, measurement_age_ms=age_summary,
                completed_interval_s=[float(t[0]), float(t[-1])],
                physics_steps=len(t)-1, saturation_samples_executed=0,
                actuator_peak_axis_Nm=np.max(abs(arrays["torque_Nm"]), axis=0).tolist())


def common_motion_summary(baseline, obs, trajectory, window):
    time = baseline["time_s"]
    original = obs["host_time_s"]
    in_run = (original >= time[0]) & (original <= time[-1])
    target_x = trajectory["nominal_tcp_m_rotvec_rad"][:, 0]
    model_x = baseline["xyz_m"][:, 0]
    actual = obs["actual_tcp_pose"][in_run, 0]
    model_at_samples = np.interp(original[in_run], time, model_x)
    in_turn = np.flatnonzero((time >= window[0]) & (time <= window[1]))
    actual_turn = np.flatnonzero((original >= window[0]) & (original <= window[1]))
    return dict(model_x_peak_time_s=float(time[in_turn[np.argmax(model_x[in_turn])]]),
        nominal_x_peak_time_s=float(time[in_turn[np.argmax(target_x[in_turn])]]),
        actual_x_peak_time_s=float(original[actual_turn[np.argmax(obs["actual_tcp_pose"][actual_turn, 0])]]),
        model_x_span_mm=float(np.ptp(model_x)*1000), actual_x_span_mm=float(np.ptp(actual)*1000),
        model_actual_x_rms_um=float(np.sqrt(np.mean((model_at_samples-actual)**2))*1e6),
        model_nominal_y_rms_um=float(np.sqrt(np.mean((baseline["xyz_m"][:, 1]-
            trajectory["nominal_tcp_m_rotvec_rad"][:, 1])**2))*1e6))


def run_trial(args, cfg, key, cases):
    prefix = f"X_P05_{key}"
    source = args.observed / f"{prefix}_observed.npz"
    with np.load(source, allow_pickle=False) as f:
        obs = dict(f)
    t0, t1 = (round(v*1000) for v in cfg["interval_host_s"])
    t = np.arange(t0, t1+1)*.001
    trajectory, reconstruction = reconstruct(cfg, obs, t)
    args.output.joinpath(key).mkdir()
    folder = args.output / key
    np.savez_compressed(folder/"nominal.npz", **trajectory)
    windows = args.input_audit["trials"][prefix]["events"]["host_windows_s"].copy()
    windows["effective"] = cfg["interval_host_s"]
    entry = dict(observed_sha256=sha(source), windows=windows, reconstruction=reconstruction,
                 nominal_sha256=sha(folder/"nominal.npz"), cases={})
    baseline, failure = simulate(cfg, obs, trajectory, "A0", 1)
    if failure:
        np.savez_compressed(folder/"N_prefix.npz", **baseline)
        entry["cases"]["N"] = dict(failure=failure, samples=len(baseline["time_s"]))
        write(folder/"summary.json", entry)
        raise RuntimeError(f"Common no-control model stopped: {failure}")
    entry["common_motion"] = common_motion_summary(baseline, obs, trajectory, windows["review_turn"])
    all_cases = [("N", "A0", 1), *cases]
    for label, mode, multiplier in all_cases:
        values, failure = simulate(cfg, obs, trajectory, mode, multiplier, baseline)
        metrics = review_metrics(values, baseline, obs, windows)
        if label == "N":
            for field in ("q_rad", "xyz_m", "command_rad"):
                assert np.array_equal(values[field], baseline[field]), field
            assert np.array_equal(values["estimated_error_um"], values["background_um"])
            assert not np.any(values["feedback"][:, 7])
        entry["cases"][label] = dict(samples=len(values["time_s"]), failure=failure,
            params=dict(K=cfg["K"], B0=cfg["B0"]*multiplier, n=cfg["n"], mu=cfg["mu"], gamma=cfg["gamma"],
                        law=mode), **metrics)
        np.savez_compressed(folder/f"{label}.npz", **values)
        if key == "R01" and label == "S" and failure is None:
            perturbed = dict(obs)
            k = np.searchsorted(obs["frame_host_time_s"], 4.0)
            changed = obs["mapped_visual_y_centered_um"].copy()
            changed[k:] += 80.
            perturbed["mapped_visual_y_centered_um"] = changed
            proof, proof_failure = simulate(cfg, perturbed, trajectory, mode, multiplier, baseline)
            comparable = min(len(proof["time_s"]), len(values["time_s"]))
            mask = values["time_s"][:comparable] < obs["frame_host_time_s"][k]
            assert np.array_equal(proof["command_rad"][:comparable][mask], values["command_rad"][:comparable][mask])
            first = int(np.flatnonzero(np.any(proof["command_rad"][:comparable] != values["command_rad"][:comparable],
                                            axis=1))[0])
            assert proof["feedback"][first, 10] >= obs["frame_host_time_s"][k]*1000-1e-8
            assert np.array_equal(proof["q_rad"][first], values["q_rad"][first])
            assert np.any(proof["q_rad"][first+1] != values["q_rad"][first+1])
            assert np.any(proof["feedback"][first+1:, 4] != values["feedback"][first+1:comparable, 4])
            entry["causality_check"] = dict(source_changed_at_s=float(obs["frame_host_time_s"][k]),
                first_changed_command_s=float(values["time_s"][first]), first_changed_state_s=float(values["time_s"][first+1]),
                future_input_prefix_unchanged=True, encoder_background_responds=True,
                synthetic_addition_um=80., validation_only_failure=proof_failure)
    write(folder/"summary.json", entry)
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observed", type=Path)
    parser.add_argument("previous_probe", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stationary-only", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if cfg["command_hold_ms"] != cfg["control_update_ms"] or cfg["command_execution_extra_delay_ms"] != 0:
        raise ValueError("This entry implements same-tick application with full-command zero-order hold")
    args.output.mkdir(parents=True, exist_ok=False)
    args.input_audit = json.loads((args.observed/"input_audit.json").read_text(encoding="utf-8"))
    anchor = json.loads(args.previous_probe.read_text(encoding="utf-8"))
    assert anchor["pose_candidates"]["one_r01_pose_alignment"]["flange_to_comparison_pose"] == cfg["local_flange_to_site"]
    assert anchor["model_sha256"] == sha(MODEL)
    q = np.array(anchor["fixed_pose_smoke"]["initial_q_rad"])
    env = aligned_model(cfg)
    with np.load(args.observed/"X_P05_R01_observed.npz", allow_pickle=False) as obs:
        assert np.linalg.norm(pose_error(flange(env, q), recorded_pose(obs["actual_tcp_pose"][0]))) < 1e-12
    env.forward()
    force_check = env.check_tcp_force_generalized()
    assert force_check["passed"]
    traces, probe = stationary(cfg, q)
    probe["aligned_tcp_force_check"] = force_check
    np.savez_compressed(args.output/"stationary.npz", **traces)
    write(args.output/"stationary.json", probe)
    if args.stationary_only:
        return
    candidates = [(f"L{m}", "Alinear", m) for m in cfg["linear_candidates"]]
    r01 = run_trial(args, cfg, "R01", [*candidates, ("S", "ASFC", 1)])
    eligible = [case for case in candidates if r01["cases"][case[0]]["failure"] is None]
    if r01["cases"]["N"]["failure"] or not eligible or r01["cases"]["S"]["failure"] or not r01.get("causality_check"):
        raise RuntimeError("R01 structural comparison failed; R02 remains held out")
    chosen = min(eligible, key=lambda case: r01["cases"][case[0]]["windows"]["review_turn"]["rms_um"])
    write(args.output/"selection.json", dict(selected_linear=chosen[0],
        criterion="minimum R01 fixed-turn RMS among completed historical linear candidates",
        other_params="historical S; all shared settings unchanged"))
    r02 = run_trial(args, cfg, "R02", [chosen, ("S", "ASFC", 1)])
    b, a, _ = coefficients(1000/cfg["encoder_sample_ms"], cfg["velocity_low_pass_hz"])
    dll = Path(mujoco.__file__).parent/"mujoco.dll"
    write(args.output/"summary.json", dict(configuration=cfg, config_sha256=sha(CONFIG), model_sha256=sha(MODEL),
        aligned_model_sha256=sha(ALIGNED_MODEL), previous_probe_sha256=sha(args.previous_probe),
        python=sys.version, numpy=np.__version__,
        mujoco=mujoco.__version__, mujoco_dll_sha256=sha(dll) if dll.is_file() else None,
        encoder_filter=dict(b=b.tolist(), a=a.tolist()), feedback_columns=FEEDBACK_COLUMNS,
        force_columns=["total", "position", "linear_damping", "shear"], stationary=probe,
        selected_linear=chosen[0], trials=dict(R01=r01, R02=r02)))
    if any(case["failure"] for case in r02["cases"].values()):
        raise RuntimeError("R02 has a protected prefix; inspect the saved run receipt")


if __name__ == "__main__":
    main()
