"""Reconcile original B05 records and prepare timestamped review arrays offline."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnose import CONFIG, MODEL, SimEnv, a_event, compiled_identity, geometry, sha

TRIALS = ("ST05", "X_P05_R01", "X_P05_R02")
POSE_FIELDS = ("actual_q_rad", "actual_tcp_pose", "actual_tcp_speed", "target_tcp_pose")
REVIEW_WINDOW = (3.60, 4.56)
MAX_HOST_BRACKET_S = 0.020
MAX_CABINET_BRACKET_S = 0.01601


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def json_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_csv(path, names, rows):
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(names)
        writer.writerows(rows)


def stats(values):
    values = np.asarray(values)
    return dict(n=len(values), rms=float(np.sqrt(np.mean(values**2))),
                min=float(values.min()), max=float(values.max()),
                peak_to_peak=float(np.ptp(values)), median=float(np.median(values)))


def read_trial(root, trial):
    original = root / "data/original" / trial
    paths = {label: next(original.glob(f"*_{label}.{ext}"))
             for label, ext in (("UR10", "txt"), ("RUNLOG", "txt"), ("VISION", "txt"),
                                ("META", "json"), ("FRAME_TIMESTAMPS", "csv"))}
    ur, log, vision = (json_rows(paths[key]) for key in ("UR10", "RUNLOG", "VISION"))
    robot = [row for row in ur if row["kind"] == "ROBOT"]
    log_robot = [{key: row[key] for key in robot[0]} for row in log if row["kind"] == "ROBOT"]
    assert robot == log_robot
    assert all(not any("target_q" in key.lower() for key in row) for row in robot)
    unique = {}
    for row in robot:
        first = unique.setdefault(row["robot_timestamp_s"], row)
        assert all(first[key] == row[key] for key in POSE_FIELDS)
    unique = list(unique.values())

    frames = [row for row in vision if row["kind"] == "VISION"]
    log_frames = [row for row in log if row["kind"] == "CAMERA_FRAME"]
    sidecar = csv_rows(paths["FRAME_TIMESTAMPS"])
    keys = ("frame_id", "host_ns", "camera_timestamp_raw")
    frame_keys = [tuple(int(row[key]) for key in keys) for row in sidecar]
    assert frame_keys == [tuple(row[key] for key in keys) for row in frames]
    assert frame_keys == [tuple(row[key] for key in keys) for row in log_frames]
    assert [row["capture_index"] for row in frames] == [int(row["segment_frame_index"]) for row in sidecar]
    assert [row["segment_frame_index"] for row in log_frames] == list(range(len(frames)))
    frame_keys = np.array(frame_keys, dtype=np.int64)
    assert np.all(np.diff(frame_keys, axis=0) > 0)
    origin = int(frame_keys[0, 1])
    robot_host_ns = np.array([row["host_ns"] for row in unique], dtype=np.int64)
    t = (robot_host_ns - origin) * 1e-9
    cabinet = np.array([row["robot_timestamp_s"] for row in unique])
    assert np.all(np.diff(t) > 0) and np.all(np.diff(cabinet) > 0)
    ft = (frame_keys[:, 1] - origin) * 1e-9
    arrays = {key: np.array([row[key] for row in unique]) for key in POSE_FIELDS}
    assert all(x.shape == (len(unique), 6) and np.isfinite(x).all() for x in arrays.values())
    meta = json.loads(paths["META"].read_text(encoding="utf-8-sig"))
    trajectory = next((row for row in ur if row["kind"] == "TRAJECTORY"), None)
    arrays.update(host_time_s=t, host_ns=robot_host_ns, cabinet_time_s=cabinet,
                  frame_id=frame_keys[:, 0], frame_host_time_s=ft, frame_host_ns=frame_keys[:, 1],
                  camera_raw_tick=frame_keys[:, 2],
                  checker_image_dxdy_mm=np.array([[row["checker_dx_mm"], row["checker_dy_mm"]] for row in frames]),
                  checker_valid=np.array([row["checker_is_valid"] for row in frames]))
    gaps = np.diff(cabinet)
    frame_diff = np.diff(frame_keys[:, 0])
    rw = (t >= REVIEW_WINDOW[0]) & (t <= REVIEW_WINDOW[1])
    fw = (ft >= REVIEW_WINDOW[0]) & (ft <= REVIEW_WINDOW[1])
    clock = dict(robot_rows=len(robot), unique_cabinet_rows=len(unique), duplicates=len(robot)-len(unique),
                 target_tcp_pose_rows=len(robot), target_q_rows=0, vision_sidecar_runlog_rows=len(frames),
                 camera_missing_frames=int(np.sum(frame_diff - 1)), unique_cabinet_intervals_ms=dict(
                     sorted(Counter(int(round(dt*1000)) for dt in gaps).items())),
                 cabinet_long_gaps_host_s=[[float(t[i]), float(t[i+1])] for i in np.flatnonzero(gaps > .03201)],
                 median_host_minus_cabinet_s=float(np.median(robot_host_ns*1e-9-cabinet)),
                 median_hardware_tick_per_frame_id=float(np.median(np.diff(frame_keys[:, 2])/frame_diff)),
                 host_review_window=dict(robot_rows=int(rw.sum()), camera_rows=int(fw.sum()),
                     max_cabinet_interval_ms=float(np.max(np.diff(cabinet[rw]))*1000),
                     missing_frames=int(np.sum(np.diff(frame_keys[fw, 0])-1))))
    identity = {key: dict(path=path.relative_to(root).as_posix(), sha256=sha(path))
                for key, path in paths.items()}
    return arrays, dict(clock=clock, identity=identity, meta=meta, trajectory=trajectory,
                        runlog_events=[row for row in log if row["kind"] == "EVENT"])


def align_short(t, cabinet, y, frame_t):
    """Interpolate only where both the host and cabinet brackets are short."""
    right = np.searchsorted(t, frame_t, side="right")
    right = np.clip(right, 1, len(t)-1)
    left = right - 1
    valid = ((frame_t >= t[0]) & (frame_t <= t[-1]) &
             ((t[right]-t[left]) <= MAX_HOST_BRACKET_S) &
             ((cabinet[right]-cabinet[left]) <= MAX_CABINET_BRACKET_S))
    value = np.full(len(frame_t), np.nan)
    value[valid] = np.interp(frame_t[valid], t, y)
    return value, valid


def analyze_motion(root, out, trial, arrays, record, cfg, legacy_scene):
    mapping_path = root / "data/batch/axis_mapping.json"
    inv = np.array(json.loads(mapping_path.read_text(encoding="utf-8"))["J_inverse_mm_per_px"])
    derived = csv_rows(next((root / "data/derived").glob(f"*_{trial}_derived.csv")))
    legacy = root / "data/B_reference" / f"A_{trial}.csv"
    visual = csv_rows(legacy)
    arrays["phase"] = np.array([row["phase"] for row in visual])
    assert np.array_equal(arrays["frame_id"], [int(row["frame_id"]) for row in visual])
    assert np.array_equal(arrays["frame_id"], [int(row["frame_id"]) for row in derived])
    uv = np.array([[float(row[f"image_{axis}_px_robust_reference"]) for axis in ("u", "v")] for row in derived])
    mapped = (uv @ inv.T)[:, 1]*1000
    base = float(np.median(mapped[arrays["phase"] == "pre_motion"]))
    centered = mapped-base
    assert np.allclose(centered, [float(row["cross_track_um"]) for row in visual], rtol=0, atol=1e-8)
    t, cabinet = arrays["host_time_s"], arrays["cabinet_time_s"]
    actual, target = arrays["actual_tcp_pose"], arrays["target_tcp_pose"]
    phases = csv_rows(root / "data/batch/phase_summary.csv")
    by_phase = {row["phase"]: [(int(row[key])-int(arrays["frame_host_ns"][0]))*1e-9
                               for key in ("start_host_ns", "end_host_ns")]
                for row in phases if row["segment_base"].endswith(trial)}
    stage = np.full(len(t), "", dtype="<U20")
    for name, (a, b) in by_phase.items():
        stage[(t >= a) & (t <= b)] = f"path_{name}"
    fk_input = out / f"{trial}_cabinet_fk_input.csv"
    write_csv(fk_input, ["robot_timestamp_s", "host_ns",
                         *[f"actual_q_J{j}_rad" for j in range(1, 7)],
                         "actual_tcp_x", "actual_tcp_y", "actual_tcp_z", "stage"],
              zip(cabinet, arrays["host_ns"], *arrays["actual_q_rad"].T, *actual[:, :3].T, stage, strict=True))
    record["geometry"] = geometry(fk_input, out, cfg["geometry"], legacy_scene)
    record["historical_event"] = a_event(legacy, out, cfg["a_event"])
    ft = arrays["frame_host_time_s"]
    pre = stage == "path_pre_motion"
    actual_y_ref = float(np.median(actual[pre, 1]))
    cabinet_y, valid = align_short(t, cabinet, (actual[:, 1]-actual_y_ref)*1e6, ft)
    arrays.update(mapped_visual_y_raw_um=mapped, mapped_visual_y_centered_um=centered,
                  cabinet_y_aligned_um=cabinet_y, short_bracket_valid=valid)
    record["alignment"] = dict(visual_center_um=base, actual_tcp_y_pre_center_m=actual_y_ref,
                               matched_frames=int(valid.sum()), unmatched_frames=int((~valid).sum()),
                               max_host_bracket_ms=MAX_HOST_BRACKET_S*1000,
                               max_cabinet_bracket_ms=round(MAX_CABINET_BRACKET_S*1000, 2))

    vx = arrays["actual_tcp_speed"][:, 0] * 1000
    bounds = {}
    for name, direction in (("X_outbound", 1), ("X_return", -1)):
        moving = t[(stage == f"path_{name}") & (direction*vx >= cfg["a_event"]["ordinary_min_motion_speed_mm_s"])]
        trim = cfg["a_event"]["ordinary_trim_s"]
        bounds[name] = [float(moving[0]+trim), float(moving[-1]-trim)]
    turn = np.flatnonzero(stage == "path_far_end_turn")
    marker = turn[np.argmax(actual[turn, 0])]
    lo, hi = REVIEW_WINDOW
    review = (t >= lo) & (t <= hi)
    max_target = np.max(target[review, 0])
    plateau = t[review & (abs(target[:, 0]-max_target) < 1e-9)]
    bounds["review_turn"] = [lo, hi]
    bounds["early_return"] = [float(t[marker]), float(t[marker]+.2)]
    bounds["late_return"] = [float(t[marker]+.2), float(t[marker]+.5)]
    record["events"] = dict(host_phase_s=by_phase, host_windows_s=bounds,
        actual_far_xmax_host_s=float(t[marker]), actual_far_xmax_mm=float(actual[marker, 0]*1000),
        first_outbound_speed_crossing_host_s=float(t[np.flatnonzero((t >= 0)&(vx > 2.5))[0]]),
        first_return_speed_crossing_host_s=float(t[np.flatnonzero((t > t[marker])&(vx < -2.5))[0]]),
        target_xmax_mm=float(max_target*1000), target_peak_plateau_host_s=[float(plateau[0]), float(plateau[-1])],
        target_travel_x_mm=float(np.ptp(target[:, 0])*1000),
        target_orientation_variation_mrad=float(np.max((Rotation.from_rotvec(target[0, 3:]).inv() *
            Rotation.from_rotvec(target[:, 3:])).magnitude())*1000))
    record["windows"] = {}
    for name, (lo, hi) in bounds.items():
        v = (ft >= lo) & (ft <= hi)
        r = (t >= lo) & (t <= hi)
        matched = v & valid
        record["windows"][name] = dict(visual_centered_um=stats(centered[v]),
            actual_minus_target_y_um=stats((actual[r, 1]-target[r, 1])*1e6),
            visual_minus_aligned_cabinet_um=stats(centered[matched]-cabinet_y[matched]),
            cabinet_y_relative_pre_um=stats((actual[r, 1]-actual_y_ref)*1e6),
            cabinet_speed_x_mm_s=stats(vx[r]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--legacy-scene", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    results = {}
    for trial in TRIALS:
        arrays, record = read_trial(args.input, trial)
        if trial == "ST05":
            record["static"] = dict(cabinet_y_relative_median_um=stats(
                (arrays["actual_tcp_pose"][:, 1]-np.median(arrays["actual_tcp_pose"][:, 1]))*1e6),
                checker_image_relative_um=[stats((col-np.median(col))*1000)
                    for col in arrays["checker_image_dxdy_mm"].T])
        else:
            analyze_motion(args.input, args.output, trial, arrays, record, cfg, args.legacy_scene)
        np.savez_compressed(args.output / f"{trial}_observed.npz", **arrays)
        results[trial] = record
    result = dict(compiled_model=compiled_identity(MODEL, SimEnv(MODEL)),
                  model_config_sha256=sha(CONFIG), trials=results, event_integrations=0)
    (args.output / "input_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False,
        allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
