"""Verify extracted ZIP identities and independently recompute the scored arrays."""
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verify(root):
    members = read(root/"SHA256SUMS.json")
    present = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    assert present == set(members) | {"SHA256SUMS.json"}
    for name, expected in members.items():
        b = (root/name).read_bytes()
        assert len(b)==expected["bytes"] and sha256(b).hexdigest()==expected["sha256"], name
    trials = sorted((root/"prepared_A").glob("*/source.json"))
    assert len(trials) == 3
    rows = []
    for r in trials:
        trial = r.parent.name
        result = root/"results"/trial
        source = read(r)
        status = read(result/"status.json")
        cases = read(result/"attempts.json")
        with (np.load(r.parent/"A_source.npz",allow_pickle=False) as source_array,
              np.load(result/"locked_inputs.npz",allow_pickle=False) as locked,
              np.load(result/"N_components_and_attempts.npz",allow_pickle=False) as components,
              np.load(result/"best_N.npz",allow_pickle=False) as trace):
            t = trace["t"]
            ft = source_array["frame_host_time_s"]
            a = source_array["A_raw_um"]
            center = source["raw_visual_zero_um"]
            np.testing.assert_array_equal(source_array["A_centered_um"],a-center)
            np.testing.assert_array_equal(t,locked["t"])
            remaining = trace["remaining"]*1e6
            selected = cases[status["best_attempt"]]
            np.testing.assert_array_equal(locked["Fd"],trace["Fd"])
            np.testing.assert_array_equal(trace["physical_e_um"],components["fit_physical_um"][status["best_attempt"]])
            np.testing.assert_array_equal(trace["q_nominal"],locked["q"])
            np.testing.assert_array_equal(trace["dq_C"],locked["dq"])
            np.testing.assert_array_equal(trace["qfrc_applied"],trace["qfrc_disturbance"])
            assert np.any(trace["dq_C"]!=0) and np.any(remaining!=0)
            assert np.all(trace["F_requested"]==0) and np.all(trace["actual_micro_q"]==0)
            j=np.flatnonzero(~trace["updated"])
            np.testing.assert_array_equal(trace["final_ctrl"][j],trace["final_ctrl"][j-1])
            assert sha256(trace["Fd"].tobytes()).hexdigest()==selected["force_sha256"]
            for c in cases:
                obs = components["fit_physical_um"][c["attempt"]] + remaining
                phy = components["fit_physical_um"][c["attempt"]]
                for label,(lo,hi) in source["windows_s"].items():
                    m=source_array["frame_valid"] & (ft>=lo) & (ft<=hi)
                    tt=ft[m]
                    trend=np.interp(tt,t,locked["A_trend_um"])
                    a_frame=a[m]
                    model=np.interp(tt,t,obs)+center
                    for layer,target,got in (("raw",a_frame,model),
                                             ("oscillation",a_frame-center-trend,model-center-trend),
                                             ("physical_oscillation",a_frame-center-trend,np.interp(tt,t,phy)-trend)):
                        reference=np.linalg.norm(target-np.mean(target))
                        residual=np.linalg.norm(got-target)
                        fit = 100*(1-residual/reference) if reference/np.sqrt(len(tt))>1.0 else None
                        expected=c["scores"][label][layer]
                        assert len(tt)==expected["samples"]
                        np.testing.assert_allclose(residual/np.sqrt(len(tt)),expected["rmse_um"],atol=1e-10,rtol=0)
                        if fit is None:
                            assert expected["fit"] is None
                        else:
                            np.testing.assert_allclose(fit,expected["fit"],atol=1e-9,rtol=0)
                        rows.append((trial,c["attempt"],label,layer))
    gate=read(root/"results/control_gate.json")
    assert not gate["eligible"] and not gate["multi_speed_event_coverage"] and not gate["controls"]
    print(f"{len(members)} verified hashes, {len(rows)} independently recomputed fit metrics, 3 N cases, 0 eligible control cases.")


if __name__=="__main__":
    verify(Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent)
