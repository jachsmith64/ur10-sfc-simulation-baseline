"""Summarize comparable windows, preserve failure prefixes, and plot world-Y error."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pair(directory, method, group, metrics):
    folders = [directory / f"{method}_{p}ms" for p in (8, 2)]
    traces = []
    for folder in folders:
        with np.load(folder / "trace.npz", allow_pickle=False) as archive:
            traces.append({key: archive[key] for key in archive.files})
    a, b = traces
    shared = min(len(a["t"]), len(b["t"]))
    for key in ["t", "Fd", "q_nominal", "remaining", "remaining_dot", "common_delta_q"]:
        np.testing.assert_array_equal(a[key][:shared], b[key][:shared])
    frame_logs = [read_json(folder / "frames.json") for folder in folders]
    for left, right in zip(*frame_logs):
        for field in ["frame_id", "sample_step", "available_step", "noise_m"]:
            assert left[field] == right[field], field
    assert (method != "baseline"
            or not np.array_equal(a["final_ctrl"], b["final_ctrl"]))

    for period, data, folder in zip((8, 2), traces, folders):
        row = read_json(folder / "row.json")
        reject = read_json(folder / "guard.json")
        verified = read_json(folder / "validation.json")
        assert verified["passed"]
        valid = len(data["t"])
        e = data["physical_e_um"][:shared]
        equivalent = e + data["remaining"][:shared] * 1e6
        ticks = np.arange(valid)
        updates = np.flatnonzero(data["updated"] & (data["measurement_frame_id"] >= 0))
        first = updates[np.r_[True, np.diff(data["measurement_frame_id"][updates]) != 0]]
        first_age = first - data["sample_step_used"][first]
        held = ticks - ticks % period
        sample = data["sample_step_used"][held]
        physical_tick_age = (ticks - sample)[sample >= 0]
        metrics.append(dict(
            group=group, method=method, hold_ms=period, status=row["status"],
            valid_steps=valid, shared_window_samples=shared,
            shared_window_end_s=data["t"][shared - 1],
            physical_Y_rms_um=np.sqrt(np.mean(e**2)),
            physical_Y_peak_abs_um=np.max(np.abs(e)),
            equivalent_Y_rms_um=np.sqrt(np.mean(equivalent**2)),
            equivalent_Y_peak_abs_um=np.max(np.abs(equivalent)),
            requested_force_rms_N=np.sqrt(np.mean(data["F_requested"][:shared]**2)),
            requested_force_peak_abs_N=np.max(np.abs(data["F_requested"][:shared])),
            largest_executed_request_abs_N=np.max(np.abs(data["F_requested"])),
            saturation_ticks=np.count_nonzero(np.any(
                data["ctrl_saturation"] | data["force_saturation"] | data["joint_saturation"], axis=1)),
            hard_guard_step=reject["step"] if reject else None,
            rejected_torque_joints=",".join(map(str, reject["torque_joints"])) if reject else None,
            rejected_force_N=reject["F_requested_N"] if reject else None,
            command_ticks=verified["command_updates"], camera_frames=verified["frames_sampled"],
            camera_frames_used=verified["unique_frames_used"],
            first_command_age_p50_ms=np.median(first_age),
            first_command_age_max_ms=np.max(first_age),
            applied_command_sample_age_p50_ms=np.median(physical_tick_age),
            applied_command_sample_age_max_ms=np.max(physical_tick_age),
        ))
    return traces


def timing(fresh, reuse):
    rows = []

    def collect(module, source, samples):
        data = np.asarray(samples, dtype=float)
        rows.append(dict(
            module=module, source=source, n=len(data), p50_ms=np.median(data),
            p95_ms=np.percentile(data, 95), max_ms=np.max(data),
            over_2ms=np.count_nonzero(data > 2), over_8ms=np.count_nonzero(data > 8),
            over_32ms=np.count_nonzero(data > 32),
        ))

    for clip in ("R2", "R3"):
        source = read_csv(fresh / f"{clip}.csv")
        steady = [r for r in source if r["warmup"] == "False"]
        assert all(r["valid"] == "True" for r in source)
        for field in [key for key in source[0] if key.endswith("_ms")]:
            collect(field, f"new serial {clip}", [float(r[field]) for r in steady])
        collect("recognition_initialization_ms", f"new serial {clip}", [float(source[0]["recognition_position_ms"])])
        with np.load(reuse / f"{clip}_C_replay.npz", allow_pickle=False) as z:
            arr = z["records"]
            collect("vision_software_publish_to_result", f"historical paced {clip}", (arr["end_ns"] - arr["publish_ns"]) / 1e6)
            collect("vision_software_publish_to_consumer_start", f"historical paced {clip}", (arr["begin_ns"] - arr["publish_ns"]) / 1e6)

    sync = read_csv(reuse / "完整控制链逐次.csv")
    for flag, label in [("True", "decision"), ("False", "ordinary")]:
        subset = [r for r in sync if r["decision"] == flag]
        for column in ("e2e_s", "controller_s", "logger_s"):
            collect(f"{label}_{column}", "historical six saved E states", [float(r[column]) * 1000 for r in subset])
    parts = read_csv(reuse / "协调原始计时.csv")
    for column in ["coordination_s", "plan_advance_s", "prediction_scoring_s", "json_s"]:
        collect(column, "historical six saved E states", [float(r[column]) * 1000 for r in parts])
    return rows


def plot_pairs(pairs, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.8), layout="constrained")
    for index, (ax, (title, traces)) in enumerate(zip(axes.flat, pairs)):
        for period, color, data in zip((8, 2), ("#b14618", "#005a98"), traces):
            full = len(data["t"]) == 10000
            t = data["t"] if full else data["t"] * 1000
            ax.plot(t, data["physical_e_um"], label=f"{period} ms hold",
                    color=color, alpha=.88, lw=1)
            if not full:
                ax.axvline(len(t), color=color, ls=":", lw=1)
        ax.set(title=title, ylabel="World Y error (µm)",
               xlabel="Time (s)" if index < 2 else "Time (ms)")
        ax.grid(alpha=.2)
        ax.legend(loc="upper right" if index < 2 else "lower left")
    limit = max(abs(y) for ax in axes[0] for y in ax.get_ylim())
    for ax in axes[0]:
        ax.set(xlim=(0, 10), ylim=(-limit, limit))
    limit = max(abs(y) for ax in axes[1] for y in ax.get_ylim())
    for ax in axes[1]:
        ax.set(xlim=(0, 83), ylim=(-limit, limit))
    fig.suptitle("E132 whole-command hold | 132 Hz samples, 4 ms availability lag")
    fig.savefig(path, dpi=175)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    for name in ("trial", "replicate", "timing", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir()
    metrics = []
    pairs = [(title, pair(args.trial, method, "first", metrics)) for title, method in
             [("No fast compensation", "baseline"), ("Matched linear feedback", "linear"),
              ("SFC only | original noise seed", "sfc")]]
    pairs.append(("SFC only | second noise seed", pair(args.replicate, "sfc", "replicate", metrics)))
    write_csv(args.out / "effects.csv", metrics)
    write_csv(args.out / "timing.csv", timing(args.timing, ROOT / "evidence" / "reused_timing"))
    plot_pairs(pairs, args.out / "error_world_y.png")


if __name__ == "__main__":
    main()
