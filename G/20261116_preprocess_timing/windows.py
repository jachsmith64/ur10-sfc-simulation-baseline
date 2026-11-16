"""Freeze the source-only review windows before any MuJoCo cases are run."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
TRIALS = ("X_P05_R01", "X_P05_R02", "X_P10_R01")
NAMES = ("start", "start_transition", "outbound", "turn", "turn_transition",
         "return_straight", "stop", "stop_transition")
PLATFORMS = ("pre_static", "outbound", "return_straight", "post_static")
EVENT_TO_TRANSITION = {"start":"start_transition", "turn":"turn_transition", "stop":"stop_transition"}
# These boundaries were chosen from the original RTDE and valid camera frames on
# the shared monotonic host clock, with no new model output available.
EDGES = {
    "X_P05_R01": (1.255, 1.96, 2.24, 3.84, 4.63, 5.02, 6.42, 7.20, 7.40),
    "X_P05_R02": (1.275, 1.98, 2.27, 3.83, 4.58, 5.02, 6.31, 7.22, 7.42),
    "X_P10_R01": (1.29, 1.98, 2.18, 3.62, 4.68, 4.97, 5.94, 6.73, 6.94),
}
BASIS = {
    "X_P05_R01": [
        "X leaves rest near 1.29 s. Y reaches a second and larger peak around 1.75 s after X is already near 5 mm/s.",
        "Cabinet deceleration follows about 3.84 s. The negative Y trough near 4.25 s and the next recovery lobe extend past reversal in X.",
        "The Y tail grows before final deceleration. Cabinet rows are sparse across 6.603–6.754 s. Camera resolves the tail and post-stop platform."
    ],
    "X_P05_R02": [
        "The larger Y peak repeats near 1.75 s, after the first X rise. Stable fitting starts after the Y recovery and shoulder.",
        "The return Y trough near 4.26 s is much later than the X zero crossing. The return platform is calibrated from 5.02 s.",
        "Vision Y changes sharply near 6.5 s. Cabinet speed is near zero after the 6.601–6.670 s gap, although the archived return label persists to 7.584 s."
    ],
    "X_P10_R01": [
        "X leaves rest near 1.33 s. Y peaks around 1.58 s and keeps returning towards the outbound platform through about 2 s.",
        "X decelerates near 3.62 s. Y stays depressed after reverse speed is established. Protect the trough and recovery through 4.68 s.",
        "The terminal Y excursion and post-stop recovery bracket the 6.120–6.314 s cabinet gap. The camera establishes the later static plateau."
    ]
}


def mask(t, bounds, end):
    lo, hi = bounds
    return (t >= lo) & ((t < hi) | ((hi == end) & (t == hi)))


def describe(t, y, valid, pair, end):
    hits = valid & mask(t, pair, end)
    z = y[hits]
    return dict(valid_frames=int(len(z)), invalid_saved_frames=int(np.count_nonzero(mask(t,pair,end) & ~valid)),
                median_um=float(np.median(z)), p10_p90_um=np.percentile(z,[10,90]).tolist(),
                third_medians_um=[float(np.median(a)) for a in np.array_split(z,3)])


def freeze(prepared, output):
    out = {"membership":"left inclusive, right exclusive; final original frame included",
           "source_only":True, "transition_policy":"C2 between old and new phase medians, after protected events",
           "slow_trend_policy":"Separate optional additive within-stage drift. Keep zero unless repeatable, monotonic change exceeds within-stage scatter. No shared cross-stage slope.",
           "boundary_sensitivity":"Individual trial only, move each event-to-transition boundary -60 or +60 ms; move following stable boundary with it, all other boundaries retained.",
           "trials":{}}
    for trial in TRIALS:
        path = prepared/trial/"A_source.npz"
        receipt = json.loads((path.parent/"source.json").read_text(encoding="utf-8"))
        with np.load(path,allow_pickle=False) as a:
            ft, y, valid, ht, v = (a[k] for k in ("frame_host_time_s","A_centered_um","frame_valid","host_time_s","actual_tcp_speed"))
        end = float(ft[-1])
        edges = EDGES[trial]
        names = ("pre_static",) + NAMES + ("post_static",)
        limits = (0.,)+edges+(end,)
        wi = dict(zip(names,zip(limits[:-1],limits[1:],strict=True),strict=True))
        stats = {name:describe(ft,y,valid,span,end) for name,span in wi.items()}
        for name,span in wi.items():
            keep = mask(ht,span,end)
            stats[name]["cabinet_samples"] = int(keep.sum())
            stats[name]["median_x_speed_mm_s"] = float(np.median(v[keep,0])*1000)
        assert sum(s["valid_frames"] for s in stats.values()) == int(valid.sum())
        modes = {}
        for event, transition in EVENT_TO_TRANSITION.items():
            following = ("outbound","return_straight","post_static")[list(EVENT_TO_TRANSITION).index(event)]
            modes[event] = []
            for delta in (-.06,.06):
                alternate = [wi[following][0]+delta,wi[following][1]]
                if alternate[1]-alternate[0] > .1:
                    modes[event].append(dict(delta_s=delta, transition=[wi[transition][0]+delta,wi[transition][1]+delta],
                                             stable_window=alternate,
                                             stable_source=describe(ft,y,valid,alternate,end)))
        gaps = np.diff(ht)
        out["trials"][trial] = dict(source_npz_sha256=sha256(path.read_bytes()).hexdigest(),
            source_record_sha256=sha256((path.parent/"source.json").read_bytes()).hexdigest(),
            source_clock_origin_ns=receipt["source_clock_origin_ns"], speed_mm_s=receipt["speed_mm_s"],
            cabinet_gaps_s=[[float(ht[i]),float(ht[i+1])] for i in np.flatnonzero(gaps>.05)],
            original_stage_labels_s=receipt["phase_host_bounds_s"], windows_s=wi, source_statistics=stats,
            bases_um={name:stats[name]["median_um"] for name in PLATFORMS}, optional_drift_um=0.,
            rationale=BASIS[trial], alternatives=modes)
    output.write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepared",type=Path,default=ROOT.parent/"20261116_A_multispeed/prepared_A")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    freeze(args.prepared,args.output)
