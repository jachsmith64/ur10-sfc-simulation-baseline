"""Verify migrated native sampling and the conservative candidate's feedback path."""

import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "feedback"), str(HERE.parents[1])]
from filter_sensing import Measurement, checks, coefficients
from sfc import force_terms


def main():
    candidate = json.loads((HERE.parent / "validation.json").read_text(encoding="utf-8"))[
        "future_control_candidate"]
    period = candidate["encoder_sample_ms_assumed"]
    hz = candidate["camera_hz_assumed"]
    config = dict(encoder_ms=period, camera_hz=hz, encoder_delay_ms=0)
    case = dict(estimator="Q", branch="V", fc=candidate["velocity_low_pass_hz"],
                rho=candidate["velocity_high_frequency_residual_fraction"],
                camera_delay_ms=candidate["camera_availability_delay_ms_assumed"])
    proof = checks(config)
    sensor = Measurement(config, case, 0., np.zeros(32), np.zeros(34))
    for step in range(250):
        sensor.tick(step, 1e-6*step, 0.)
        if step % period == 0:
            _, v = sensor.measure(None, 0.)
            if sensor.last_command["ready"]:
                assert np.isclose(v, .001, rtol=0, atol=1e-14)
            assert all(sensor.last_command[f"{name}_available"] <= step for name in ("c", "q"))
    gains = tuple(candidate[key] for key in ("K_N_per_m", "B0_N_s_per_m", "sfc_mu", "sfc_n"))
    args = (2e-5, 5e-4, *gains, "ASFC")
    assert force_terms(*args) == force_terms(*args, gamma=1.)
    b, a, _ = coefficients(1000/period, case["fc"])
    proof.update(candidate_encoder_frames=len(sensor.q.frames), candidate_camera_frames=len(sensor.c.frames),
                 candidate_filter_b=b.tolist(), candidate_filter_a=a.tolist(), default_gamma=1.)
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
