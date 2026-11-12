"""Reproduce recorded A event features and check nominal CB3 FK offline."""
from __future__ import annotations

import argparse
import csv
from hashlib import file_digest
import json
from pathlib import Path
import platform
import sys
from xml.etree import ElementTree

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from position_execution import ExecutionManager, POSITION_EQUIV, compiled_mapping
from sim_env import SimEnv

MODEL = ROOT / "assets" / "ur10_cb3.xml"
CONFIG = ROOT / "validation.json"


def sha(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def columns(rows, names):
    data = np.array([[float(row[name]) for name in names] for row in rows])
    if not np.all(np.isfinite(data)):
        raise ValueError(f"Non-finite data in {names}")
    return data


def vector_stats(values):
    magnitude = np.linalg.norm(values, axis=1)
    return {
        "rms": float(np.sqrt(np.mean(magnitude**2))),
        "peak": float(np.max(magnitude)),
        "median_vector": np.median(values, axis=0).tolist(),
    }


def sample_time(t):
    dt = np.diff(t)
    if len(dt) == 0 or np.any(dt <= 0):
        raise ValueError("Expected original, strictly increasing timestamps")
    return {
        "rows": len(t), "first_s": float(t[0]), "last_s": float(t[-1]),
        "median_dt_ms": float(1e3*np.median(dt)), "max_dt_ms": float(1e3*dt.max()),
        "gaps_over_1p5_median": int(np.sum(dt > 1.5*np.median(dt))),
    }


def a_event(path, output, cfg):
    fields, rows = read_csv(path)
    t, frame, cabinet_x, vision_x, cabinet_speed, vision_speed, cross = columns(
        rows, ["time_s", "frame_id", "ur_along_mm", "vision_along_mm",
               "ur_speed_mm_s", "vision_speed_mm_s", "cross_track_um"],
    ).T
    phase = np.array([row["phase"] for row in rows])
    bounds = {}
    for name in ("pre_motion", "X_outbound", "far_end_turn", "X_return"):
        span = t[phase == name]
        if not len(span):
            raise ValueError(f"{path.name} has no {name} phase")
        bounds[name] = [float(span[0]), float(span[-1])]
    turn = np.flatnonzero(phase == "far_end_turn")
    marker = turn[np.argmax(cabinet_x[turn])]
    speed_sign_change = turn[:-1][(cabinet_speed[turn[:-1]] > 0) & (cabinet_speed[turn[1:]] < 0)]
    reversal_brackets = [[float(t[i]), float(t[i+1])] for i in speed_sign_change]
    trim = cfg["ordinary_trim_s"]
    ordinary = [(bounds[name][0]+trim, bounds[name][1]-trim)
                for name in ("X_outbound", "X_return")]
    lo = bounds["X_outbound"][1]-cfg["turn_lead_s"]
    hi = bounds["X_return"][0]+cfg["turn_tail_s"]
    baseline_um = float(np.median(cross[phase == "pre_motion"]))
    cross_centered = cross - baseline_um
    windows = {}
    for name, (start, end) in zip(("outbound", "return", "reversal"), [*ordinary, (lo, hi)]):
        idx = (t >= start) & (t <= end)
        windows[name] = {
            "interval_s": [start, end], "n": int(idx.sum()),
            "median_ur_speed_mm_s": float(np.median(cabinet_speed[idx])),
            "median_vision_speed_mm_s": float(np.median(vision_speed[idx])),
            "cross_raw_um": vector_stats(cross[idx, None]),
            "cross_pre_centered_um": vector_stats(cross_centered[idx, None]),
        }
    np.savez_compressed(output / f"{path.stem}.npz", time_s=t, frame_id=frame.astype(np.int64),
                        phase=phase, ur_along_mm=cabinet_x, vision_along_mm=vision_x,
                        ur_speed_mm_s=cabinet_speed, vision_speed_mm_s=vision_speed,
                        cross_raw_um=cross, cross_pre_centered_um=cross_centered)
    return {
        "file": path.name, "sha256": sha(path), "columns": fields, **sample_time(t),
        "phase_bounds_s": bounds, "reversal_time_s": float(t[marker]),
        "reversal_frame_id": int(frame[marker]), "reversal_ur_x_mm": float(cabinet_x[marker]),
        "reversal_ur_speed_mm_s": float(cabinet_speed[marker]),
        "reversal_adjacent_source_times_s": t[marker-1:marker+2].tolist(),
        "positive_to_negative_speed_brackets_s": reversal_brackets,
        "pre_motion_center_um": baseline_um, "windows": windows,
    }


def dh_position(q, cfg):
    matrix = np.eye(4)
    for angle, a, d, turns in zip(q, cfg["reference_dh_a_m"], cfg["reference_dh_d_m"],
                                  cfg["reference_dh_alpha_quarter_turn"], strict=True):
        c, s = np.cos(angle), np.sin(angle)
        ca, sa = np.cos(turns*np.pi/2), np.sin(turns*np.pi/2)
        matrix = matrix @ np.array([[c, -s*ca, s*sa, a*c], [s, c*ca, -c*sa, a*s],
                                    [0, sa, ca, d], [0, 0, 0, 1]])
    return matrix[:3, 3] + matrix[:3, :3] @ cfg["tcp_offset_flange_m"]


def fk(env, q):
    env.data.qpos[:] = q
    mujoco.mj_kinematics(env.model, env.data)
    return env.tcp_pos()


def compiled_identity(path, env):
    xml = ElementTree.parse(path)
    mapping = compiled_mapping(env.model)
    mapping["conditions"] = {key: bool(value) for key, value in mapping["conditions"].items()}
    return {
        "model_name": xml.getroot().attrib["model"], "xml_sha256": sha(path),
        "included_xml_sha256": {inc.attrib["file"]: sha(path.parent/inc.attrib["file"])
                                for inc in xml.findall("include")},
        "local_site_m": env.model.site_pos[env.site_id].tolist(),
        "moving_body_mass_kg": env.model.body_mass[env.model.body_jntnum > 0].tolist(),
        "moving_body_principal_inertia_kg_m2": env.model.body_inertia[env.model.body_jntnum > 0].tolist(),
        "mapping": mapping,
    }


def model_checks(env, poses, cfg):
    fk_error = []
    jac_error = []
    for q in poses:
        fk_error.append(np.linalg.norm(fk(env, q) - dh_position(q, cfg)))
        env.data.qpos[:] = q
        env.forward()
        basis = np.eye(6)*1e-6
        jac_fd = np.column_stack([(dh_position(q+dq, cfg)-dh_position(q-dq, cfg))/2e-6
                                  for dq in basis])
        jac_error.append(np.max(np.abs(jac_fd-env.jacp())))
    env.reset(poses[0])
    force = env.check_tcp_force_generalized(f_test=1.0)
    env.clear_external()
    manager = ExecutionManager(env, POSITION_EQUIV)
    desired_tau = env.jacp().T @ np.array([0., 1., 0.])
    tau0 = env.data.qfrc_actuator.copy()
    manager.apply_requested_torque(env, desired_tau)
    manager.assert_applied(env)
    env.set_ctrl(manager.command(poses[0]))
    env.forward()
    torque_error = float(np.max(np.abs(env.data.qfrc_actuator-tau0-desired_tau)))
    if max(fk_error) > 1e-12 or max(jac_error) > 1e-8 or torque_error > 1e-8:
        raise RuntimeError("Nominal model identity or position-equivalent torque check failed")
    return {
        "poses": len(poses), "fk_vs_independent_dh_max_m": float(max(fk_error)),
        "jacobian_vs_dh_central_diff_max_m_per_rad": float(max(jac_error)),
        "mj_applyft_vs_jacobian": force,
        "one_newton_y_requested_tau_Nm": desired_tau.tolist(),
        "one_newton_y_command_increment_rad": manager.delta.tolist(),
        "same_state_torque_increment_max_error_Nm": torque_error,
    }


def geometry(path, output, cfg, legacy_path):
    fields, rows = read_csv(path)
    t = columns(rows, ["robot_timestamp_s"])[:, 0]
    if "synthetic" in fields and np.any(columns(rows, ["synthetic"]) != 0):
        raise ValueError("Synthetic rows cannot support a measured TCP comparison")
    q_rad = [f"actual_q_J{j}_rad" for j in range(1, 7)]
    q_deg = [f"actual_q_J{j}_deg" for j in range(1, 7)]
    q = columns(rows, q_rad) if all(name in fields for name in q_rad) else np.deg2rad(columns(rows, q_deg))
    tcp = columns(rows, ["actual_tcp_x", "actual_tcp_y", "actual_tcp_z"])
    stages = np.array([row.get("stage", "") for row in rows])
    stationary = stages == "path_pre_motion"
    models = {"cb3": MODEL}
    if legacy_path:
        models["historical_ur10e_attachment"] = legacy_path
    arrays = {"robot_timestamp_s": t, "actual_q_rad": q, "cabinet_tcp_m": tcp,
              "stage": stages}
    summary = {}
    for name, model in models.items():
        env = SimEnv(model)
        model_id = compiled_identity(model, env)
        if name == "cb3" and (model_id["local_site_m"] != cfg["tcp_offset_flange_m"]
                              or env.model.opt.timestep*1e3 != cfg["physics_step_ms"]):
            raise ValueError("CB3 compiled model differs from the validation configuration")
        pred = np.array([fk(env, joints) for joints in q])
        err_mm = (pred-tcp)*1e3
        arrays[f"{name}_tcp_m"] = pred
        arrays[f"{name}_error_mm"] = err_mm
        arrays[f"{name}_relative_error_mm"] = err_mm-err_mm[0]
        entry = {
            "compiled": model_id, "absolute_mm": vector_stats(err_mm),
            "first_offset_mm": err_mm[0].tolist(),
            "first_offset_mm_norm": float(np.linalg.norm(err_mm[0])),
            "relative_to_first_mm": vector_stats(err_mm-err_mm[0]),
            "cabinet_travel_from_first_mm": vector_stats((tcp-tcp[0])*1e3),
            "stage_absolute_mm": {stage: vector_stats(err_mm[stages == stage])
                                  for stage in sorted(set(stages)) if stage},
        }
        if stationary.any():
            entry["pre_motion_static_mm"] = vector_stats(err_mm[stationary])
            entry["motion_residual_after_static_median_mm"] = vector_stats(
                err_mm[~stationary]-np.median(err_mm[stationary], axis=0))
        if name == "cb3":
            entry["checks"] = model_checks(env, q[np.linspace(0, len(q)-1, 5, dtype=int)], cfg)
        summary[name] = entry
    np.savez_compressed(output / f"{path.stem}_geometry.npz", **arrays)
    return {"file": path.name, "sha256": sha(path), "columns": fields,
            **sample_time(t), "models": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, nargs="+", required=True)
    parser.add_argument("--tcp", type=Path, required=True,
                        help="Explicit same-row measured q/TCP CSV. XYZ in metres.")
    parser.add_argument("--legacy-scene", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    results = {
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "mujoco": mujoco.__version__, "os": platform.platform()},
        "config_sha256": sha(CONFIG),
        "a_event": [a_event(path, args.out, cfg["a_event"]) for path in args.a],
        "geometry": geometry(args.tcp, args.out, cfg["geometry"], args.legacy_scene),
    }
    (args.out/"summary.json").write_text(json.dumps(results, indent=2, allow_nan=False)+"\n",
                                        encoding="utf-8")


if __name__ == "__main__":
    main()
