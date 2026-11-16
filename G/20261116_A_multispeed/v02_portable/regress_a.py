"""Three historical 2 ms cases through the unchanged A entry point."""
import sys
import argparse
from pathlib import Path
from setup_case import ROOT, SOURCE, DEPS, read, sha, write

sys.path[:0] = [str(SOURCE), str(DEPS)]

import numpy as np
import run_trial as archived


def main(out):
    files = read(SOURCE/"manifest.sha256.json")
    receipt = read(ROOT/"source_receipt.json")["a_extracted"]
    assert all(receipt[k] == v for k,v in files.items())
    assert all(sha(SOURCE/k) == v for k,v in files.items())
    design = read(SOURCE/"experiment.json")
    ctx = archived.context()
    assert ctx["input_id"] == design["expected_input_id"]
    dependency = read(SOURCE/"evidence/dependencies.json")
    import mujoco, ruckig
    assert sys.version == dependency["python"] and np.__version__ == dependency["numpy"]
    assert mujoco.__version__ == dependency["mujoco"] and sha(Path(ruckig.__file__)) == dependency["ruckig_module_sha256"]
    assert sha(Path(mujoco.__file__).parent/"mujoco.dll") == "127144b04e9dce183b2a3935d21be30bafb7a2965fb652ca7a8749bed128f32b"
    out.mkdir(exist_ok=True)
    checks = {}
    for name in ("baseline","linear","sfc"):
        case = f"{name}_2ms"
        if not (out/case/"validation.json").exists():
            archived.run_case(out,ctx,design,name,design["methods"][name],2,design["frame_noise_seed"])
        new = archived.arrays(out/case/"trace.npz")
        ref = archived.arrays(SOURCE/"trials"/case/"trace.npz")
        fields = ["t","physical_e_um","F_requested","final_ctrl","measurement_frame_id","sample_e_m","sample_v_m_s"]
        exact = {k:np.array_equal(ref[k],new[k],equal_nan=True) for k in fields}
        diff = {k:float(np.max(abs(ref[k]-new[k]))) for k in fields}
        row = read(out/case/"row.json")
        old = read(SOURCE/"trials"/case/"row.json")
        assert (row["status"],row["n_steps_done"]) == (old["status"],old["n_steps_done"])
        assert all(diff[k] <= (1e-6 if k == "physical_e_um" else 1e-9) for k in fields)
        checks[name] = dict(exact=exact,max_abs_delta=diff,status=row["status"],steps=row["n_steps_done"],
                            physical_rms_um=float(np.sqrt(np.mean(new["physical_e_um"]**2))),
                            archived_trace_sha256=sha(SOURCE/"trials"/case/"trace.npz"))
    write(out/"regression.json",dict(checks=checks, input_id=ctx["input_id"],
          model_files=read(SOURCE/"trials/identity.json")["model_files"],
          parameters=ctx["parameters"],archive_head=(SOURCE/"evidence/archive_head.txt").read_text().strip(),
          source_commit=read(SOURCE/"evidence/version_record.json")["source_commit"],
          numpy_archived=dependency["numpy"],numpy_executed=np.__version__,hardware_calls=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT/"regression_A")
    main(parser.parse_args().out)
