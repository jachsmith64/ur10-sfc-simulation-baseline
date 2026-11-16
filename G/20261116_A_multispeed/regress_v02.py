import argparse
import json
from pathlib import Path
import numpy as np
import run_campaign as core

parser = argparse.ArgumentParser()
parser.add_argument("--historical", required=True, type=Path)
parser.add_argument("--out", required=True, type=Path)
args = parser.parse_args()
project = Path(__file__).resolve().parent
args.out.mkdir(parents=True, exist_ok=True)
anchors = json.loads((project / "evidence/v02_historical_hashes.json").read_text(encoding="utf-8"))
sweep, context = core.initialize(args.out)
receipt = {}
for case_path in sorted((project / "evidence").glob("D_*.json"), key=lambda p: "NLS".index(p.name[2])):
    case = core.read(case_path)
    name = case["id"]
    result = core.run_case(args.out, sweep, context, case)
    old = args.historical / name
    new = args.out / "runs" / name
    with np.load(old / "trace.npz", allow_pickle=False) as a, np.load(new / "trace.npz", allow_pickle=False) as b:
        deltas = {}
        equal = set(a.files) == set(b.files)
        for key in sorted(set(a.files) & set(b.files)):
            same_shape = a[key].shape == b[key].shape
            same = same_shape and np.array_equal(a[key], b[key], equal_nan=True)
            equal &= same
            if a[key].dtype.kind in "fiub" and same_shape:
                x, y = a[key].astype(float), b[key].astype(float)
                valid = np.isfinite(x) & np.isfinite(y)
                delta = float(np.max(abs(x[valid] - y[valid]), initial=0))
            else:
                delta = None
            deltas[key] = dict(equal=bool(same), max_abs_finite=delta)
    old_guard, new_guard = core.read(old / "guard.json"), core.read(new / "guard.json")
    receipt[name] = dict(completed=result["completed"], old_sha=core.sha(old / "trace.npz"),
                         new_sha=core.sha(new / "trace.npz"), all_arrays_equal=bool(equal),
                         old_guard=old_guard, new_guard=new_guard, deltas=deltas,
                         inputs_equal=core.read(old / "inputs.json") == core.read(new / "inputs.json"),
                         pinned_by_history=core.sha(old / "trace.npz") in json.dumps(anchors))
    core.write(args.out / "regression_receipt.json", receipt)
    print(name, "arrays equal", equal, "guard equal", old_guard == new_guard, flush=True)
