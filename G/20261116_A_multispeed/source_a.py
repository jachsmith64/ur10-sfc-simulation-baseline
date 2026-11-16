"""Bind A source rows to one CB3 trial on the original host-time clock."""
import json
import numpy as np
from audit_inputs import read_trial, csv_rows, json_rows, POSE_FIELDS
from increment_model import reconstruct
from run_campaign import sha

TRIALS = ("X_P05_R01", "X_P05_R02", "X_P10_R01")


def speed(trial):
    return float(int(trial.split("_")[1][1:]))


def archived_source(root, trial):
    """Join the archived per-frame export to independent original sidecar and cabinet rows."""
    original = root/"data/original"/trial
    archive = root/"data/analysis_20260902"
    stem = f"20260901_184044_B05_{trial}"
    cabinet = original/(stem+"_UR10.txt")
    sidecar = original/(stem+"_FRAME_TIMESTAMPS.csv")
    meta = original/(stem+"_META.json")
    inventory = archive/"inventory.csv"
    derived = archive/(stem+"_derived.csv")
    phase_path = archive/"phase_summary.csv"
    mapping = archive/"axis_mapping.json"
    registration = {row["file_role"]:row for row in csv_rows(inventory) if row["detected_segment_base"] == stem}
    if sha(cabinet) != registration["UR10"]["sha256"]:
        raise ValueError("Original cabinet SHA-256 differs from the archived analysis input")
    seen = {}
    conflicts = []
    robot = []
    duplicate_rows = 0
    for row in json_rows(cabinet):
        if row["kind"] != "ROBOT":
            continue
        assert not any("target_q" in k.lower() for k in row)
        stamp = row["robot_timestamp_s"]
        payload = tuple(tuple(row[k]) for k in POSE_FIELDS)
        old = seen.setdefault(stamp, [])
        if payload in old:
            duplicate_rows += 1
            continue
        if old:
            conflicts.append({"cabinet_s":stamp, "host_ns":row["host_ns"],
                              "changed_fields":[k for k,a,b in zip(POSE_FIELDS,old[-1],payload,strict=True) if a!=b]})
        old.append(payload)
        robot.append(row)
    t_raw, mapped = csv_rows(sidecar), csv_rows(derived)
    fields = ("frame_id","host_ns","camera_timestamp_raw")
    keys = np.array([[int(r[k]) for k in fields] for r in t_raw],dtype=np.int64)
    np.testing.assert_array_equal(keys, [[int(r[k]) for k in fields] for r in mapped])
    np.testing.assert_array_equal([int(r["segment_frame_index"]) for r in t_raw],
                                  [int(r["capture_index"]) for r in mapped])
    origin = keys[0,1]
    host_ns = np.array([r["host_ns"] for r in robot],dtype=np.int64)
    cabinet_time = np.array([r["robot_timestamp_s"] for r in robot])
    assert np.all(np.diff(keys,axis=0)>0) and np.all(np.diff(host_ns)>0) and np.all(np.diff(cabinet_time)>=0)
    data = {key:np.array([r[key] for r in robot]) for key in POSE_FIELDS}
    assert all(x.shape==(len(robot),6) and np.isfinite(x).all() for x in data.values())
    data.update(host_time_s=(host_ns-origin)*1e-9, host_ns=host_ns, cabinet_time_s=cabinet_time,
                frame_id=keys[:,0], frame_host_time_s=(keys[:,1]-origin)*1e-9, frame_host_ns=keys[:,1],
                camera_raw_tick=keys[:,2],
                checker_valid=np.array([r["checker_is_valid_original"] == "1" for r in mapped]))
    phase = np.full(len(keys),"",dtype="<U12")
    for row in csv_rows(phase_path):
        if row["segment_base"] != stem:
            continue
        lo,hi = int(row["start_index"]),int(row["end_index"])
        assert keys[lo,1] == int(row["start_host_ns"]) and keys[hi,1] == int(row["end_host_ns"])
        assert np.all(phase[lo:hi+1]=="")
        phase[lo:hi+1] = row["phase"]
    assert np.all(phase!="")
    record = dict(source_kind="archived_per_frame_with_original_cabinet_and_sidecar",
                  identity={p.name:dict(path=p.relative_to(root).as_posix(),sha256=sha(p))
                            for p in (cabinet,sidecar,meta,inventory,derived,phase_path,mapping,archive/"run_manifest.json")},
                  recorded_originals={k:{"filename":v["filename"],"source_path":v["absolute_path"],"sha256":v["sha256"]}
                                      for k,v in registration.items()},
                  original_vision_txt_and_runlog_locally_available=False,
                  exact_frame_identity_rows=len(keys), frame_identity_fields=list(fields),
                  clock={"distinct_cabinet_times":len(seen),"retained_host_rows":len(robot),"target_q_rows":0,
                         "identical_repeated_payload_rows_removed":duplicate_rows,
                         "equal_cabinet_distinct_payload_rows_retained":conflicts})
    return data,record,mapped,phase,mapping


def load_source(root, trial):
    inputs = root/"data/original"/trial
    if list(inputs.glob("*_VISION.txt")) and (root/"data/B_reference"/f"A_{trial}.csv").is_file():
        data, record, mapped, phase, mapping, visual = handoff_source(root,trial)
    else:
        data, record, mapped, phase, mapping = archived_source(root,trial)
        visual = None
    return align_source(data,record,mapped,phase,mapping,visual,trial)


def handoff_source(root,trial):
    data, record = read_trial(root, trial)
    record["source_kind"] = "original_vision_cabinet_runlog_and_sidecar"
    batch = root / "data/batch"
    mapping = batch / "axis_mapping.json"
    derived = next((root / "data/derived").glob(f"*_{trial}_derived.csv"))
    reference = root / "data/B_reference" / f"A_{trial}.csv"
    phases = batch / "phase_summary.csv"
    for path in (mapping, derived, reference, phases):
        record["identity"][path.stem] = dict(path=path.relative_to(root).as_posix(), sha256=sha(path))
    visual = csv_rows(reference)
    mapped = csv_rows(derived)
    np.testing.assert_array_equal(data["frame_id"], [int(r["frame_id"]) for r in visual])
    return data,record,mapped,np.array([r["phase"] for r in visual]),mapping,visual


def align_source(data,record,mapped,phase,mapping,visual,trial):
    inverse = np.array(json.loads(mapping.read_text(encoding="utf-8"))["J_inverse_mm_per_px"])
    frame_t = data["frame_host_time_s"]
    np.testing.assert_array_equal(data["frame_id"], [int(r["frame_id"]) for r in mapped])
    np.testing.assert_array_equal(data["frame_host_ns"], [int(r["host_ns"]) for r in mapped])
    uv = np.array([[float(r[f"image_{axis}_px_robust_reference"]) for axis in ("u", "v")] for r in mapped])
    raw = (uv @ inverse.T)[:, 1] * 1000
    valid = data["checker_valid"] & np.isfinite(raw)
    zero = float(np.median(raw[valid & (phase == "pre_motion")]))
    if visual is not None:
        np.testing.assert_allclose(raw - zero, [float(r["cross_track_um"]) for r in visual], rtol=0, atol=1e-8)
    host = data["host_time_s"]
    tcp = data["target_tcp_pose"]
    vx = data["actual_tcp_speed"][:, 0] * 1000
    assert np.max(abs(tcp[:, 1:] - tcp[0, 1:])) < 1e-8
    assert np.linalg.norm(data["actual_tcp_speed"][np.argmin(abs(host)), :3]) < .00005
    grid = np.arange(int(np.ceil(frame_t[-1] * 1000)) + 1) * .001
    gaps = np.flatnonzero(np.diff(host) > .020)
    gap_dx = abs(np.diff(tcp[:, 0])[gaps]) * 1000
    if np.any(gap_dx > .002):
        raise ValueError("A long host bracket spans material change in the nominal X")
    events = {name: [float(frame_t[phase == name][0]), float(frame_t[phase == name][-1])]
              for name in ("pre_motion", "X_outbound", "far_end_turn", "X_return", "post_stop")}
    setting = speed(trial)
    onset = float(host[np.flatnonzero((host > .75) & (vx > setting/2))[0]])
    turn0, turn1 = events["far_end_turn"]
    turn_samples = (host >= turn0) & (host <= turn1)
    turn = float(host[turn_samples][np.argmax(data["actual_tcp_pose"][turn_samples, 0])])
    stop = float(host[(host > turn) & (vx < -.9*setting)][-1])
    start_window = [events["X_outbound"][0] - .035, onset + .27]
    turn_window = [turn0 - .15, turn1 + .23]
    post = [7.15, min(7.55,float(frame_t[-1]))] if setting==5. else [frame_t[-1]-.5, frame_t[-1]-.12]
    windows = dict(full=[0., float(frame_t[-1])], start=start_window, turn=turn_window,
                   outbound=[start_window[1] + .12, turn_window[0] - .08],
                   return_straight=[turn_window[1] + .08, stop - .12],
                   pre_static=[.70, .96], post_static=post)
    record.update(speed_mm_s=setting, raw_visual_zero_um=zero, phase_host_bounds_s=events, windows_s=windows,
                  platform_names=["pre_static", "outbound", "return_straight", "post_static"],
                  first_half_speed_s=onset, turn_actual_xmax_s=turn,
                  last_return_90_percent_speed_s=stop,
                  host_gap_evidence=[[float(host[i]), float(host[i+1]), float(dx)]
                                     for i, dx in zip(gaps, gap_dx, strict=True)],
                  missing_frame_count=int(np.sum(np.diff(data["frame_id"])-1)),
                  source_clock_origin_ns=int(data["frame_host_ns"][0]),
                  pre_stationary_axial_speed_mm_s=float(vx[np.argmin(abs(host))]),
                  source_median_vx_mm_s={k: float(np.median(vx[(host >= windows[k][0]) & (host <= windows[k][1])]))
                                        for k in ("outbound", "return_straight")})
    for label in ("start", "turn", "outbound", "return_straight"):
        lo, hi = windows[label]
        if any(host[i] < hi and host[i + 1] > lo for i in gaps if host[i+1]-host[i] > .05):
            raise ValueError(f"Long host gap intersects the scored {label} window")
    data.update(frame_phase=phase, frame_valid=valid, A_raw_um=raw, A_centered_um=raw-zero)
    return data, record, grid


def nominal(cfg, data, grid):
    host = data["host_time_s"]
    cabinet = data["cabinet_time_s"]
    large = (np.diff(cabinet) > .03201) & (np.diff(host) > .020)
    if np.any(abs(np.diff(data["target_tcp_pose"][:, 0])[large]) > .002e-3):
        raise ValueError("Large cabinet gap intersects a material nominal displacement")
    aligned = dict(cfg, max_source_host_bracket_ms=float(np.max(np.diff(host))*1000 + 1e-6),
                   max_source_cabinet_gap_ms=float(np.max(np.diff(cabinet))*1000 + 1e-6))
    return reconstruct(aligned, data, grid, gravity_support=False)
