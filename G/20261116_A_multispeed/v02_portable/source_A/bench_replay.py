"""Time a serial replay pipeline with an explicit simulated-input handoff."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter_ns

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parent
VISION = ROOT / "vision"
paths = sys.path.copy()
sys.path.insert(0, str(VISION))
from algorithms import TemplateIC, cv2

# The vision module retains its own vendor config. Restore the search order
# before loading the simulator, which has a separate module named config.
del sys.modules["config"]
sys.path[:] = paths
from run_trial import GuardTrace
from b132_common import arrays, mujoco, np, read, save, sha
from controller import SFCController
from position_execution import ExecutionManager, POSITION_EQUIV
from sim_env import SimEnv
import sfc


class FrozenMeasurement:
    def measure(self, env, y_d, y_d_dot=0.0):
        return self.error, self.velocity


def profile(clip, references, frozen, params, env, out):
    cap = cv2.VideoCapture(str(clip["video"]))
    assert cap.isOpened()
    cap.set(cv2.CAP_PROP_POS_FRAMES, clip["start_index"])
    with Path(clip["timestamps"]).open(encoding="utf-8-sig", newline="") as stream:
        timestamps = list(csv.DictReader(stream))[clip["start_index"]:clip["end_index_exclusive"]]
    assert len(timestamps) == clip["requested_frames"]
    with references.open(encoding="utf-8-sig", newline="") as stream:
        expected = list(csv.DictReader(stream))
    tracker = TemplateIC()
    measure = FrozenMeasurement()
    manager = ExecutionManager(env, POSITION_EQUIV)
    control = SFCController(env, params, "ASFC", measurement=measure, actuation=manager)
    guard = GuardTrace("sfc", frozen["q_nominal"], params)
    guard.start(env, measure, control)
    rows = []
    previous = None
    scalar_ns = 0
    original = sfc.force_terms

    def timed_scalar(*args, **kwargs):
        nonlocal scalar_ns
        start = perf_counter_ns()
        terms = original(*args, **kwargs)
        scalar_ns = perf_counter_ns() - start
        return terms

    sfc.force_terms = timed_scalar
    try:
        for index, ts in enumerate(timestamps):
            # Independent frozen model inputs are used only for computational
            # cost. Real pixel displacements have no validated world-Y map.
            step = (1285 if clip["id"] == "R2" else 4580) + index
            h = int(ts["host_ns"])
            start = perf_counter_ns()
            ok, bgr = cap.read()
            decoded = perf_counter_ns()
            if not ok:
                raise RuntimeError(f"Decode failed at {clip['id']} frame {index}")
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            result = tracker.process(gray, h / 1e9)
            recognized = perf_counter_ns()
            shift = result["shift_mm"]
            velocity = None
            if result["valid"] and previous is not None:
                previous_h, previous_shift = previous
                assert h > previous_h
                velocity = ((shift - previous_shift) / ((h - previous_h) / 1e9)).tolist()
            previous = (h, shift.copy()) if result["valid"] else None
            aligned = perf_counter_ns()

            env.data.qpos[:] = frozen["q"][step]
            env.data.qvel[:] = frozen["qdot"][step]
            env.data.ctrl[:] = frozen["final_ctrl"][step]
            env.forward()
            measure.error = frozen["sample_e_m"][step]
            measure.velocity = frozen["sample_v_m_s"][step]
            base = frozen["base_ctrl_submitted"][step]
            ready = perf_counter_ns()
            guard.before_step(index, env, measure, control, base)
            force = control.step(frozen["nominal_y"][step])
            command = guard.command(index, env, manager.command(base), force, manager)
            env.set_ctrl(command)
            manager.assert_applied(env)
            assembled = perf_counter_ns()
            json.dumps(dict(
                timestamp_ns=h, valid=result["valid"],
                shift_mm=shift.tolist() if result["valid"] else None, velocity_mm_s=velocity,
                synthetic_command_rad=command.tolist(),
            ), allow_nan=False)
            published = perf_counter_ns()
            assert list(gray.shape[::-1]) == clip["size"]
            assert int(expected[index]["host_ns"]) == h
            assert hashlib.sha256(gray.tobytes()).hexdigest() == expected[index]["gray_sha256"]
            rows.append(dict(
                clip=clip["id"], index=index, host_ns=h, valid=result["valid"],
                initialization=index == 0, warmup=index < 6,
                read_decode_ms=(decoded - start) / 1e6,
                recognition_position_ms=(recognized - decoded) / 1e6,
                alignment_velocity_ms=(aligned - recognized) / 1e6,
                frozen_state_handoff_FK_ms=(ready - aligned) / 1e6,
                force_law_ms=scalar_ns / 1e6,
                mapping_assembly_guard_ms=(assembled - ready - scalar_ns) / 1e6,
                measurement_to_command_ms=(assembled - ready) / 1e6,
                payload_serialization_ms=(published - assembled) / 1e6,
                direct_replay_to_publishable_payload_ms=(published - start) / 1e6,
            ))
    finally:
        sfc.force_terms = original
        cap.release()
    with out.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(clip["id"], len(rows), sum(r["valid"] for r in rows), "frame hashes verified", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir()
    reuse = ROOT / "evidence" / "reused_timing"
    clips = read(reuse / "clip_selection_frozen.json")["clips"]
    cv2.setNumThreads(1)
    frozen = arrays(args.trial / "linear_2ms" / "trace.npz")
    params = read(args.trial / "identity.json")["parameters"]
    env = SimEnv()

    def forbid(*args, **kwargs):
        raise RuntimeError("Physical integration forbidden during timing")

    mujoco.mj_step = mujoco.mj_step1 = mujoco.mj_step2 = forbid
    inputs = []
    for clip in clips:
        name = clip["id"]
        if name in ("R2", "R3"):
            assert sha(clip["timestamps"]) == clip["timestamps_sha256"]
            assert sha(clip["metadata"]) == clip["metadata_sha256"]
            assert Path(clip["video"]).stat().st_size == clip["video_bytes"]
            profile(clip, reuse / f"{name}_frame_hashes.csv", frozen, params, env, args.out / f"{name}.csv")
            inputs.append(dict(clip=name, video=clip["video"], sha256=sha(clip["video"])))
    save(args.out / "identity.json", dict(
        inputs=inputs, algorithm_sha256=sha(VISION / "algorithms.py"),
        software_handoff="Sequential replay + independent frozen simulator measurement, no calibrated pixel-to-world transform",
        python=sys.version, opencv=cv2.__version__, numpy=np.__version__, mujoco=mujoco.__version__,
        cv2_threads=cv2.getNumThreads(), blas_threads=1, physical_steps=0, hardware_calls=0,
    ))


if __name__ == "__main__":
    main()
