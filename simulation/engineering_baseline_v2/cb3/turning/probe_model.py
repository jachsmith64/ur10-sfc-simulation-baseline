"""Test CB3 model response and the feasibility of two offline pose mappings."""

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnose import MODEL, ExecutionManager, POSITION_EQUIV, SimEnv, sha


def transform(xyz, rotation):
    result = np.eye(4)
    result[:3, 3] = xyz
    result[:3, :3] = rotation
    return result


def flange(env, q):
    env.data.qpos[:] = q
    mujoco.mj_kinematics(env.model, env.data)
    return transform(env.tcp_pos(), env.data.site_xmat[env.site_id].reshape(3, 3))


def recorded_pose(pose):
    return transform(pose[:3], Rotation.from_rotvec(pose[3:]).as_matrix())


def pose_error(got, want):
    return np.r_[got[:3, 3]-want[:3, 3],
                 Rotation.from_matrix(want[:3, :3].T @ got[:3, :3]).as_rotvec()]


def probe_poses(env, observations, seed, tool):
    """Reuse the C offline audit's six-component least-squares formulation."""
    observed_q = observations["actual_q_rad"]
    target = observations["target_tcp_pose"]
    actual = observations["actual_tcp_pose"]
    t = observations["host_time_s"]
    gap = np.flatnonzero(np.diff(observations["cabinet_time_s"]) > .01601)
    fit, tracking = [], []
    previous = seed
    lower, upper = env.model.actuator_ctrlrange.T
    step_max = 0.
    for i, pose in enumerate(target):
        want = recorded_pose(pose)
        solve = least_squares(lambda q: pose_error(flange(env, q) @ tool, want), previous,
                              bounds=(lower, upper), xtol=1e-12, ftol=1e-12, gtol=1e-12)
        if not solve.success or np.max(abs(solve.fun)) > 1e-8:
            raise RuntimeError(f"Six-dimensional back-substitution failed at row {i}: {solve.message}")
        step_max = max(step_max, float(np.max(abs(solve.x-previous)))) if i else step_max
        previous = solve.x
        fit.append(solve.fun)
        tracking.append(pose_error(flange(env, observed_q[i]) @ tool, recorded_pose(actual[i])))
    fit, tracking = np.array(fit), np.array(tracking)
    return dict(rows=len(target), cabinet_gaps_over_16ms=len(gap),
                observed_time_s=[float(t[0]), float(t[-1])],
                position_back_substitution_peak_m=float(np.linalg.norm(fit[:, :3], axis=1).max()),
                rotation_back_substitution_peak_rad=float(np.linalg.norm(fit[:, 3:], axis=1).max()),
                max_adjacent_solution_joint_delta_rad=step_max,
                actual_fk_position_rms_m=float(np.sqrt(np.mean(np.sum(tracking[:, :3]**2, axis=1)))),
                actual_fk_rotation_rms_rad=float(np.sqrt(np.mean(np.sum(tracking[:, 3:]**2, axis=1)))))


def fixed_pose_response(q):
    """Exercise the compiled plant at a fixed pose; no measured waveform is used."""
    rows = {}
    for mode in ("fixed_command", "external_y", "position_increment"):
        env = SimEnv(MODEL)
        env.reset(q)
        manager = ExecutionManager(env, POSITION_EQUIV)
        bias = env.data.qfrc_bias.copy()
        origin = env.tcp_pos()
        path = []
        for i in range(500):
            env.forward()
            active = 200 <= i < 250
            force = 1.0 if active and mode == "external_y" else 0.
            env.apply_disturbance_force_y(force)
            tau = env.jacp()[1] if active and mode == "position_increment" else np.zeros(6)
            manager.apply_requested_torque(env, tau)
            manager.assert_applied(env)
            env.set_ctrl(manager.command(q))
            path.append(env.tcp_pos()-origin)
            env.step()
        rows[mode] = np.array(path)
    base = rows["fixed_command"]
    errors = {key: float(np.max(np.abs(path[:, 1]-base[:, 1]))*1e6)
              for key, path in rows.items() if key != "fixed_command"}
    assert min(errors.values()) > 1.
    assert np.max(abs(rows["external_y"]-rows["position_increment"])) < 1e-12
    return rows, dict(steps_per_case=len(base), cases=len(rows), physics_step_ms=1,
                      pulse_N=1., pulse_window_ms=[200, 250], initial_q_rad=q.tolist(),
                      initial_gravity_bias_Nm=bias.tolist(), final_tcp_shift_mm=(base[-1]*1000).tolist(),
                      peak_relative_y_um=errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observed", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    trials = {key: dict(np.load(args.observed / f"X_P05_{key}_observed.npz", allow_pickle=False))
              for key in ("R01", "R02")}
    r01 = trials["R01"]
    seed = r01["actual_q_rad"][0]
    env = SimEnv(MODEL)
    anchor_tool = np.linalg.solve(flange(env, seed), recorded_pose(r01["actual_tcp_pose"][0]))
    result = dict(model_sha256=sha(MODEL), source="C offline audit pose least squares",
                  status="mapping feasibility only; no commanded joint trajectory selected or exported",
                  pose_candidates={})
    for label, tool in (("zero_offset", np.eye(4)), ("one_r01_pose_alignment", anchor_tool)):
        result["pose_candidates"][label] = dict(flange_to_comparison_pose=tool.tolist(),
            trials={key: probe_poses(env, obs, seed, tool) for key, obs in trials.items()})
    response, result["fixed_pose_smoke"] = fixed_pose_response(seed)
    np.savez_compressed(args.output / "fixed_pose_smoke.npz", time_s=np.arange(500)*.001, **response)
    (args.output / "probe.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
