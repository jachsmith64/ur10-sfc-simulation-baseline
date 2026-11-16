"""Offline reproduction of the E132 paired whole-command hold experiment."""
import argparse
import platform
from importlib.metadata import version
from pathlib import Path

from b132_common import B, P, arrays, mujoco, np, read, save, sha, sys, time
from b132_adapters import Coordinator132, Logger132
from b132_measurement import SampledMeasurement
from b132_run import context
from position_execution import POSITION_EQUIV
import config
import sweep


class GuardTrace(Coordinator132):
    """Record the rejected request while preserving the original guard."""

    rejection = None

    def command(self, i, env, sfc_command, out, actuation):
        try:
            return super().command(i, env, sfc_command, out, actuation)
        except RuntimeError:
            command = sfc_command + self.Tp * self.state[0]
            raw = self.kp * (command - env.data.qpos) - self.kd * env.data.qvel
            self.rejection = dict(
                step=i, t_s=i * config.GRID_DT_S, F_requested_N=out["F_sfc"],
                position_request_rad=command, total_torque_request_Nm=raw,
                position_joints=np.flatnonzero((command < self.ctrl_lo) | (command > self.ctrl_hi)) + 1,
                torque_joints=np.flatnonzero((raw < self.tau_lo) | (raw > self.tau_hi)) + 1,
                torque_lower_Nm=self.tau_lo, torque_upper_Nm=self.tau_hi,
            )
            raise


def validate(trace, measurement, ctx, period, row, extension):
    n = len(trace["t"])
    assert n == row["n_steps_done"] and n > 1, row
    assert row["status"] == "OK" or extension.rejection is not None, row
    assert (n == len(ctx["c"]["t"])) == (row["status"] == "OK"), row
    assert np.array_equal(trace["t"], ctx["c"]["t"][:n])
    np.testing.assert_allclose(trace["model_time_s"], trace["t"], atol=1e-9, rtol=0)
    assert np.array_equal(trace["q_nominal"], ctx["c"]["q"][:n])
    assert np.array_equal(trace["q"][0], ctx["c"]["q"][0]) and not trace["qdot"][0].any()
    assert np.array_equal(trace["Fd"], ctx["fd_values"][:n])
    assert np.array_equal(trace["remaining"], ctx["scaled"]["remaining"][:n])
    assert np.array_equal(trace["common_delta_q"], ctx["corr"]["common_delta_q"][:n])
    assert np.array_equal(trace["qfrc_applied"], trace["qfrc_disturbance"])
    assert np.all(trace["gamma"] == 1) and np.all(trace["p"] == 0) and not extension.decisions
    step = np.arange(n)
    assert np.array_equal(trace["updated"], step % period == 0)
    held = step - step % period
    assert np.array_equal(trace["final_ctrl"], trace["final_ctrl"][held])
    assert np.array_equal(trace["F_requested"], trace["F_requested"][held])
    np.testing.assert_allclose(
        trace["final_ctrl"], trace["base_ctrl_submitted"] + trace["delta_q_sfc_actual"],
        atol=1e-13, rtol=0,
    )
    frames = measurement.frames
    sample = np.array([f["sample_step"] for f in frames])
    available = np.array([f["available_step"] for f in frames])
    assert np.array_equal(sample, (1000 * np.arange(len(frames)) + 131) // 132)
    assert np.all(available - sample == measurement.delay_steps)
    frame = np.searchsorted(available, step, side="right") - 1
    assert np.array_equal(trace["measurement_frame_id"], frame)
    assert np.array_equal(trace["measurement_valid"] > 0, frame >= 0)
    error = np.array([f["error_m"] for f in frames])
    velocity = np.array([f["velocity_m_s"] for f in frames])
    np.testing.assert_allclose(velocity[1:], np.diff(error) / np.diff(sample * .001), atol=1e-13)
    usable = held[frame[held] >= 0]
    np.testing.assert_array_equal(trace["sample_e_m"][frame[held] < 0], 0)
    np.testing.assert_array_equal(trace["sample_v_m_s"][frame[held] < 0], 0)
    np.testing.assert_array_equal(trace["sample_e_m"][frame[held] >= 0], error[frame[usable]])
    np.testing.assert_array_equal(trace["sample_v_m_s"][frame[held] >= 0], velocity[frame[usable]])
    update_steps = np.flatnonzero(trace["updated"] & (frame >= 0))
    ages_ms = update_steps - sample[frame[update_steps]]
    return dict(
        passed=True, physics_steps=n, command_updates=int(trace["updated"].sum()),
        frames_sampled=int((sample < n).sum()),
        unique_frames_used=len(np.unique(frame[update_steps])),
        command_age_p50_ms=float(np.median(ages_ms)), command_age_max_ms=int(ages_ms.max()),
    )


def run_case(output, ctx, design, name, mode, period, seed):
    folder = output / f"{name}_{period}ms"
    folder.mkdir()
    params = ctx["parameters"]
    c = ctx["c"]
    measurement = SampledMeasurement(
        "difference", design["frame_noise_std_m"], design["visual_delay_steps"], seed,
        nominal=lambda _t: config.Y_NOMINAL,
    )
    extension = GuardTrace("sfc" if mode != "A0" else "baseline", c["q"], params)
    logger = Logger132(
        folder / "trace.npz", len(c["t"]), "C_ORDINARY", POSITION_EQUIV, period,
        ctx["input_id"], input_data=ctx["scaled"], common_correction=ctx["corr"],
        extension=extension, adapter=measurement, hold=True,
    )
    logger.arr["dq_C_dot"][:] = ctx["scaled"]["dqd"]
    logger.arr["nominal_qdot"][:] = np.gradient(c["q"], config.GRID_DT_S, axis=0)
    logger.meta.update(name=name, mode=mode, visual_delay_steps=measurement.delay_steps, noise_seed=seed)
    start = time.perf_counter()
    row = sweep.run_case(
        folder.name, mode, params["K"], params["B0"], params["n"], params["mu"],
        params["r"], params["v_ref"], ctx["fd"], c["along"], c["q"], phase_code=c["phase"],
        c_input=ctx["scaled"], execution_mode=POSITION_EQUIV, control_update_steps=period,
        precise_logger=logger, common_position_correction=ctx["corr"]["common_delta_q"],
        e_extension=extension, measurement_adapter=measurement, final_command_hold=True,
    )
    row["wall_s"] = time.perf_counter() - start
    save(folder / "row.json", row)
    save(folder / "frames.json", measurement.frames)
    save(folder / "guard.json", extension.rejection)
    trace = arrays(folder / "trace.npz")
    save(folder / "validation.json", validate(trace, measurement, ctx, period, row, extension))
    if period == 8 and name == "sfc" and seed == design["frame_noise_seed"]:
        historical = arrays(P / "evidence" / "R5_sfc_reference.npz")
        exact = {k: np.array_equal(trace[k], old, equal_nan=True) for k, old in historical.items()}
        save(folder / "historical_regression.json", exact)
        assert all(exact.values()), exact
    print(folder.name, row["status"], row["n_steps_done"], row["note"], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--replicate", choices=["sfc", "linear"])
    args = parser.parse_args()
    design = read(P / "experiment.json")
    ctx = context()
    assert ctx["input_id"] == design["expected_input_id"]
    assert design["measurement_hz"] == 132 and design["physics_dt_s"] == config.GRID_DT_S
    args.out.mkdir()
    manifest = {
        p.relative_to(P).as_posix(): sha(p)
        for folder in ["inputs", "assets_ur10e"] for p in (B / folder).rglob("*") if p.is_file()
    }
    save(args.out / "identity.json", dict(
        python=sys.version, platform=platform.platform(), model_files=manifest,
        packages={k: version(k) for k in ["numpy", "mujoco", "scipy", "matplotlib", "ruckig"]},
        design=design, input_id=ctx["input_id"], parameters=ctx["parameters"],
        g=ctx["g"], physical_dt_s=config.physics_dt(), nominal_y_m=config.Y_NOMINAL,
        source_commit=read(P / "evidence" / "version_record.json")["source_commit"],
        sweep_sha256=sha(B / "sweep.py"), hardware_calls=0,
    ))
    methods = [args.replicate] if args.replicate else ["sfc", "linear", "baseline"]
    seed = design["review_seed"] if args.replicate else design["frame_noise_seed"]
    for name in methods:
        for period in design["hold_steps"]:
            run_case(args.out, ctx, design, name, design["methods"][name], period, seed)


if __name__ == "__main__":
    main()
