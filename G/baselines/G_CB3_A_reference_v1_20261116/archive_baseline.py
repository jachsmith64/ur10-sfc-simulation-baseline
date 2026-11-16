"""Archive already reviewed runs and bind their source from its exact Git identity."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = "57f6aa7dc498cdcd2398e8a33fbb8dc00720444c"
PARENT = "0fd746f487cb9e5382b7e4249a3824c5c0c8bfe3"
TAG = "g-cb3-a-reference-v1-20261116"
DIRECT_PARENT = "4daecbd0ec3fd9a7c6e89fbc6f7eb30b2c885dff"
TRIALS = {"X_P05_R01": 3, "X_P05_R02": 4, "X_P10_R01": 2}
CORRECTION = ("common_delta_q", "common_delta_q_dot", "common_delta_q_ddot",
              "common_baseline_um", "common_delta_y_m")
REVIEW_ZIP_SHA = "9deb5699e9cbf4378317770ff03a1353b2169635bd04b4b9ed71dc89f53f0fb4"
V02 = "G/20261116_A_multispeed"
PREPROCESS = "G/20261116_preprocess_timing"
SERVO = "G/20261116_servo_minimal"
CLOCK = "G/20261116_clock_event"
ENGINEERING = "simulation/engineering_baseline_v2"
SOURCE_PATHS = [
    *(f"{V02}/{p}" for p in ("v02_portable", "prepared_A", "evidence",
                            "prepare_v02.py", "run_g.py", "score_g.py", "source_a.py", "g_config.json")),
    *(f"{PREPROCESS}/{p}" for p in ("run_preprocess.py", "axial.py", "reference.py",
                                   "windows.py", "preprocess_config.json")),
    f"{SERVO}/velocity_reference.py", f"{SERVO}/run_servo.py", f"{SERVO}/x_metrics.py",
    f"{SERVO}/servo_config.json",
    *(f"{CLOCK}/{p}" for p in ("run_clock_event.py", "clock_map.py", "robust_events.py",
                              "refit_y.py", "clock_event_config.json", "windows_frozen.json")),
    f"{ENGINEERING}/cb3", f"{ENGINEERING}/assets_ur10e",
]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
                    encoding="utf-8", newline="\n")


def file_sha(path):
    with path.open("rb") as f:
        return __import__("hashlib").file_digest(f, "sha256").hexdigest()


def array_identity(a):
    return {"shape": list(a.shape), "dtype": a.dtype.str, "sha256": sha256(a.tobytes()).hexdigest()}


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def binding(repo, dest, create_manifest=False):
    manifest = HERE/"pinned_source_sha256.json"
    dest.mkdir(parents=True)
    archive = dest/"pinned_source_git.zip"
    subprocess.run(["git", "-C", str(repo), "archive", "--format=zip", "-o", str(archive),
                    SOURCE, "--", *SOURCE_PATHS], check=True)
    with zipfile.ZipFile(archive) as z:
        z.extractall(dest/"source")
    archive.unlink()
    root = dest/"source"
    files = {p.relative_to(root).as_posix(): file_sha(p)
             for p in sorted(root.rglob("*")) if p.is_file()}
    if create_manifest:
        write(manifest, {"source_commit": SOURCE, "files": files})
    elif read(manifest) != {"source_commit": SOURCE, "files": files}:
        raise ValueError("Pinned source closure differs from archive manifest")
    subprocess.run([sys.executable, str(root/V02/"prepare_v02.py"), str(dest/"v02_runtime")],
                   check=True)
    print(f"Pinned source files: {len(files)}", flush=True)


def archive_one(source, output, trial, index, window, locks):
    traces = HERE/"local"/"archived_full"/trial
    traces.mkdir(parents=True)
    target = output/trial
    target.mkdir()
    prepared = source.parent/"20261116_A_multispeed"/"prepared_A"/trial
    verified = source/"runs/input_verified"/trial
    results = source/"results"/trial
    y_root = source/"runs/y_physical"/trial
    tag = f"N_fit_{index}"
    best = y_root/tag
    status = read(results/"y_status.json")
    receipt = read(best/"case.json")
    assert status["best_index"] == index and status["physical_Y_major_windows_gate"]
    assert status["attempts"][index]["name"] == tag and status["pulse_pass"]
    assert file_sha(verified/"input.npz") == locks["input_hashes"][trial]
    assert file_sha(prepared/"A_source.npz") == window["source_npz_sha256"]
    assert file_sha(prepared/"source.json") == window["source_record_sha256"]
    assert read(best/"guard.json") is None and read(best/"checks.json")["complete"]
    for name in ("A_source.npz", "source.json"):
        shutil.copyfile(prepared/name, target/name)
    for name in ("input.npz", "input.json"):
        shutil.copyfile(verified/name, target/name)
    for name in ("case.json", "row.json", "guard.json", "checks.json"):
        shutil.copyfile(best/name, target/("N_"+name))
    shutil.copyfile(results/"y_status.json", target/"y_status.json")
    for name in ("trace.npz", "trace.json", "command_events.json", "sample_receipt.json"):
        shutil.copyfile(best/name, traces/name)
    with (np.load(target/"input.npz") as inputs, np.load(best/"trace.npz") as historical,
          np.load(y_root/"refit_review.npz") as refit):
        for a, b in (("t", "t"), ("q_nom", "q_nominal"), ("dq", "dq_C")):
            np.testing.assert_array_equal(inputs[a], historical[b])
        for a, b in (("force_N", "Fd"), ("common_delta_q", "common_delta_q"),
                     ("common_command_y_m", "common_delta_y_m"), ("fit_y_um", "physical_e_um")):
            np.testing.assert_array_equal(refit[a], historical[b])
        np.testing.assert_array_equal(inputs["initial_q_rad"], historical["q"][0])
        np.testing.assert_array_equal(inputs["feedback_reference_um"], refit["score_reference_um"])
        np.testing.assert_array_equal(refit["feedback_reference_um"], refit["score_reference_um"])
        assert array_identity(historical["Fd"])["sha256"] == receipt["Fd_sha256"] == status["attempts"][index]["force_sha256"]
        assert array_identity(historical["common_delta_q"])["sha256"] == receipt["physical_command_correction_sha256"]
        assert not historical["remaining"].any() and not historical["remaining_dot"].any()
        shared = {k: historical[k] for k in ("phase", "Fd", *CORRECTION)}
        shared["score_reference_um"] = refit["score_reference_um"]
        np.savez_compressed(target/"shared.npz", **shared)
        pulses = {k: refit[k] for k in ("plus_delta_y_um", "plus_micro_q_rad",
                                       "minus_delta_y_um", "minus_micro_q_rad")}
        np.savez_compressed(target/"inherited_pulse.npz", **pulses)
        trace_summary = {k: array_identity(historical[k]) for k in historical.files}
        input_ids = {k: array_identity(inputs[k]) for k in inputs.files}
    full_files = {p.name: file_sha(p) for p in sorted(traces.iterdir())}
    assert full_files["trace.npz"] == file_sha(best/"trace.npz")
    write(target/"identity.json", {
        "trial": trial, "source_commit": SOURCE, "best_N": tag,
        "source_receipt_paths": {"target": f"runs/input_verified/{trial}",
                                 "N": f"runs/y_physical/{trial}/{tag}"},
        "local_full_output_relative_to_archive": f"local/archived_full/{trial}",
        "full_files_sha256": full_files, "trace_arrays": trace_summary, "input_arrays": input_ids,
        "shared_arrays": {k: array_identity(v) for k, v in shared.items()},
        "pulse_inherited": {"rerun": False, "first_response_s": status["pulse_first_response_s"],
                            "positive_um": status["positive_micro_peak_um"],
                            "negative_um": status["negative_micro_peak_um"]},
    })
    print(f"{trial}: {tag}, accepted minimum FIT {status['best_min_fit']:.6f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=HERE.parents[2])
    cmd = parser.add_subparsers(dest="action", required=True)
    capture = cmd.add_parser("capture")
    capture.add_argument("--reviewed", required=True, type=Path)
    bind = cmd.add_parser("bind")
    bind.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    if args.action == "bind":
        binding(args.repo.resolve(), args.destination.resolve())
        return
    source = args.reviewed.resolve()
    assert git(source, "rev-parse", "HEAD") == SOURCE
    assert git(source, "rev-parse", "HEAD^") == DIRECT_PARENT
    assert git(source, "rev-parse", "HEAD~2") == PARENT
    assert file_sha(source/"G_clock_event_review_20261116.zip") == REVIEW_ZIP_SHA
    assert not git(source, "status", "--porcelain")
    locks = read(source/"results/lock.json")
    windows_path = source/"windows_frozen.json"
    assert file_sha(windows_path) == locks["windows_sha256"]
    windows = read(windows_path)["trials"]
    output = HERE/"trials"
    output.mkdir()
    for trial, index in TRIALS.items():
        archive_one(source, output, trial, index, windows[trial], locks)
    shutil.copyfile(windows_path, HERE/"windows_frozen.json")
    shutil.copyfile(source/"results/lock.json", HERE/"reviewed_input_lock.json")
    shutil.copyfile(source/"delivery.json", HERE/"reviewed_delivery.json")
    shutil.copyfile(source/"x_reference_for_y.json", HERE/"reviewed_x_scope.json")
    shutil.copyfile(source/"G_clock_event_review.html", HERE/"reviewed_core_figures.html")
    binding(args.repo.resolve(), HERE/"local"/"bound", create_manifest=True)


if __name__ == "__main__":
    main()
