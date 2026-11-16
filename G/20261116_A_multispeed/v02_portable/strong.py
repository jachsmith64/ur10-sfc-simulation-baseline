"""Bounded strong-filter campaign using the archived position-equivalent loop."""
import argparse
import csv
from hashlib import sha256
from itertools import product
from pathlib import Path

import numpy as np
from scipy import signal

from filter_sensing import Mix, coefficients
import run_campaign as core
from setup_case import ROOT, SOURCE, read, sha, write

P = read(ROOT / "protocol.json")
CAL = P["calibration_seeds"]
HOLD = P["holdout_seeds"]
OLD = ROOT.parent / "filter_sfc_20261110" / "results_valid"
OLD_ID = "f701623822204d3cf89f07758607c01dc217a76c5c9d7e8e923292e83dd4a699"
REFS = {
    "P1_N-N_V_fraw_r1_g1_b1_s2": (28.50, None),
    "P1_Q-S_V_f10_r0.25_g1_b1_s2": (16.77, 11.40),
    "P5_Q-L_V_f10_r0.25_g1_b4_s2": (17.01, 10.03),
    "P1_Q-S_V_f10_r0_g1_b1_s2": (None, None),
}


def freeze(path, data):
    if path.exists():
        assert read(path) == data, path
    else:
        write(path, data)


def table(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def events():
    path = SOURCE / "evidence/predeclared_matrix.json"
    source = read(path)
    assert source["source_input_id"] == read(SOURCE / "experiment.json")["expected_input_id"]
    labels = {"start": "启动", "turn": "直线路径换向", "deceleration_stop": "减速与停止"}
    return dict(source="source_A/evidence/predeclared_matrix.json", source_sha256=sha(path),
                source_input_id=source["source_input_id"], corner_tested=False,
                recovery_time="N/A", windows=[
                    dict(event=key, label=label, start_s=source["windows"][key][0],
                         end_s=source["windows"][key][1])
                    for key, label in labels.items()])


def case(stage, law, seed, filt=None, gain=1, delay=0):
    order, hz = filt or (1, None)
    rho = 1 if hz is None else .25 if stage == "history" else 0
    c = core.settings(stage, "N" if law == "N" else "Q", law, seed, hz, rho,
                      execution_delay_ms=delay, gamma=gain if law == "S" else 1.,
                      linear_multiplier=gain if law == "L" else 1.)
    c["id"] = f"{stage}_{law}_o{order}_f{hz or 'raw'}_r{rho:g}_gain{gain:g}_seed{seed}_d{delay}"
    if order != 1:
        c["filter_order"] = order
    assert c["command_ms"] == P["command_ms"] and c["camera_delay_ms"] == P["camera_delay_ms"]
    return c


def condition(c):
    return {k: v for k, v in c.items() if k not in ("id", "group")}


def rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


def score(a, row, windows):
    result = {}
    event_rows = []
    if not row["completed"]:
        for key in list(row):
            if key.startswith(("full_", "ordinary_")) and key.endswith("_um"):
                row["prefix_" + key] = row.pop(key)
    t, y, tau = a["t"], a["physical_e_um"], a["tau_realized_increment"]
    spans = [tuple(round(x * 1000) for x in span) for span in P["ordinary_windows_s"]]
    ordinary = np.logical_or.reduce([(t >= lo / 1000) & (t < hi / 1000) for lo, hi in spans])
    if row["completed"]:
        for scope, mask in (("full", np.ones(len(t), bool)), ("ordinary", ordinary)):
            rate = np.diff(tau, axis=0)[mask[1:] & mask[:-1]] / .001
            norm = np.linalg.norm(tau[mask], axis=1)
            rate = np.linalg.norm(rate, axis=1)
            result.update({f"{scope}_tau_rms_Nm": rms(norm), f"{scope}_tau_peak_Nm": float(norm.max()),
                           f"{scope}_tau_rate_rms_Nm_s": rms(rate),
                           f"{scope}_tau_rate_peak_Nm_s": float(rate.max())})
            spec = core.spectrum(y, [(0, len(t))] if scope == "full" else spans)
            f, density, count = spec
            result[f"{scope}_psd_segments"] = count
            for lo, hi in ((.5, 5), (5, 15), (15, 40), (40, 60), (15, 60), (60, 500)):
                mask = (f >= lo) & ((f <= hi) if hi == 500 else (f < hi))
                result[f"{scope}_y_band_{lo:g}_{hi:g}_um"] = float(np.sqrt(.5 * density[mask].sum()))
    for window in windows:
        lo, hi = window["start_s"], window["end_s"]
        mask = (t >= lo - 1e-12) & (t <= hi + 1e-12)
        done = round(t[-1] * 1000) >= round(hi * 1000)
        event_rows.append(dict(id=row["id"], seed=row["seed"], delay_ms=row["execution_delay_ms"],
                               **window, duration_s=hi - lo, samples=int(mask.sum()), complete=done,
                               peak_um=float(np.max(abs(y[mask]))) if done else None,
                               iae_um_s=float(np.trapezoid(abs(y[mask]), t[mask])) if done else None,
                               tau_rms_Nm=rms(np.linalg.norm(tau[mask], axis=1)) if done else None,
                               tau_peak_Nm=float(np.linalg.norm(tau[mask], axis=1).max()) if done else None,
                               recovery_s="N/A"))
    return result, event_rows


class Batch:
    def __init__(self, out, archive):
        self.out, self.archive = out, archive
        self.windows = events()["windows"]
        out.mkdir(parents=True, exist_ok=True)
        freeze(out / "protocol.json", P)
        freeze(out / "event_windows.json", events())
        self.sweep, self.ctx = core.initialize(out)
        self.identity = read(out / "identity.json")
        assert P["ordinary_windows_s"] == core.CONFIG["ordinary_windows_s"]
        assert self.identity["duration_samples"] == P["duration_s"] * 1000
        self.old_manifest = {}
        if archive.exists():
            assert sha(archive / "identity.json") == OLD_ID
            for key, value in read(archive / "identity.json").items():
                if key != "implementation":
                    assert self.identity[key] == value, key
            assert sha(ROOT / "run_campaign.py") == read(archive / "identity.json")["implementation"]["run_campaign.py"]
            assert sha(out / "command_channel.patch") == sha(archive / "command_channel.patch")
            self.old_manifest = read(archive / "run_manifest.json")
            assert not (set(HOLD) & {v["case"]["seed"] for v in self.old_manifest.values()})
        self.rows, self.events, self.manifest = {}, {}, {}
        assert set(CAL).isdisjoint(HOLD)
        self.filter_checks()

    def filter_checks(self):
        rows = []
        frequencies = np.array([1, 3, 5, 10, 20, 40])
        fs = P["encoder_hz"]
        filters = [(1, hz, 0) for hz in P["first_order_hz"]] + [(1, 10, .25)]
        filters += [(2, hz, 0) for hz in P["second_order_hz"]]
        for order, hz, rho in filters:
            b, a, zi = coefficients(fs, hz, order)
            assert np.all(abs(np.roots(a)) < 1)
            x = np.r_[np.ones(50) * .001, np.sin(np.arange(200) / fs * 2 * np.pi * 15)]
            expected, _ = signal.lfilter(b, a, x, zi=zi * x[0])
            filt = Mix(fs, hz, rho, order)
            np.testing.assert_allclose([filt.push(v) for v in x], expected + rho * (x - expected),
                                       rtol=0, atol=1e-13)
            mixed_b = rho * a + (1 - rho) * b
            _, h = signal.freqz(mixed_b, a, worN=frequencies, fs=fs)
            _, delay = signal.group_delay((mixed_b, a), w=frequencies, fs=fs)
            for f, response, gd in zip(frequencies, h, delay, strict=True):
                rows.append(dict(order=order, cutoff_hz=hz, rho=rho, fs_hz=fs, frequency_hz=int(f),
                                 gain=float(abs(response)), attenuation_dB=float(-20 * np.log10(abs(response))),
                                 phase_deg=float(np.angle(response, deg=True)), group_delay_ms=float(gd / fs * 1000),
                                 b=b.tolist(), a=a.tolist()))
        table(self.out / "filter_response.csv", rows)

    def run(self, c, old_id=None):
        cid = c["id"]
        if cid in self.rows:
            return self.rows[cid]
        old_id = old_id or next((key for key, entry in self.old_manifest.items()
                                 if entry["status"] == "COMPLETE" and condition(c) == condition(entry["case"])), None)
        cached = old_id is not None and old_id in self.old_manifest
        folder = self.archive / "runs" / old_id if cached else self.out / "runs" / cid
        if cached:
            old_case = read(folder / "case.json")
            assert old_case == self.old_manifest[old_id]["case"] and condition(c) == condition(old_case)
            assert read(folder / "validation.json")["passed"]
            row = read(folder / "metrics.json")
            assert all(row[k] == v for k, v in old_case.items())
        else:
            assert folder.exists() or len(list((self.out / "runs").iterdir())) < P["budget"]
            row = core.run_case(self.out, self.sweep, self.ctx, c)
            if row is None:
                raise RuntimeError(f"Technical failure or interrupted run at {folder}")
        arrays = core.data_arrays(folder / "trace.npz")
        inputs = read(folder / "inputs.json")
        noise = np.random.default_rng(c["seed"]).normal(0, core.CONFIG["camera_noise_m"], 1320)
        assert inputs == {k: sha256(v.tobytes()).hexdigest() for k, v in
                          dict(camera_additive_noise=noise, encoder_pressure=core.pressure(c["pressure"]),
                               Fd=self.ctx["fd_values"]).items()}
        np.testing.assert_array_equal(arrays["Fd"], self.ctx["fd_values"][:len(arrays["t"])])
        np.testing.assert_array_equal(arrays["q_nominal"], self.ctx["c"]["q"][:len(arrays["t"])])
        np.testing.assert_array_equal(arrays["common_delta_q"], self.ctx["corr"]["common_delta_q"][:len(arrays["t"])])
        assert rms(arrays["physical_e_um"]) == row["full_rms_um"]
        row = dict(row, **c, cache_id=old_id if cached else None)
        additional, event_rows = score(arrays, row, self.windows)
        self.rows[cid] = dict(row, **additional)
        self.events[cid] = event_rows
        self.manifest[cid] = dict(case=c, cached=cached, source_id=old_id if cached else cid,
                                  inputs=inputs, trace_sha256=sha(folder / "trace.npz"),
                                  validation=read(folder / "validation.json"), guard=read(folder / "guard.json"))
        self.save()
        return self.rows[cid]

    def save(self):
        write(self.out / "run_manifest.json", self.manifest)
        write(self.out / "metrics.json", list(self.rows.values()))
        table(self.out / "metrics.csv", list(self.rows.values()))
        table(self.out / "events.csv", [row for rows in self.events.values() for row in rows])

    def gate(self, row, base, old):
        result = dict(id=row["id"], n=base["id"], history=old["id"], seed=row["seed"],
                      delay_ms=row["execution_delay_ms"])
        assert self.manifest[row["id"]]["inputs"] == self.manifest[base["id"]]["inputs"]
        assert self.manifest[row["id"]]["inputs"] == self.manifest[old["id"]]["inputs"]
        if not (row["completed"] and base["completed"] and old["completed"]):
            return dict(result, gate1=False, gate2=False, gate3=False, gate4=False, gate5=False,
                        passed_count=0, passed=False, worst_excess=None, summed_event_iae_ratio=None)
        limit = {
            "full_rms": row["full_rms_um"] / (base["full_rms_um"] * P["rms_ratio_max"]),
            "ordinary_rms": row["ordinary_rms_um"] / (base["ordinary_rms_um"] * P["rms_ratio_max"]),
            "ordinary_peak": row["ordinary_peak_um"] / (base["ordinary_peak_um"] * P["peak_ratio_max"]),
            "ordinary_H": row["ordinary_y_band_15_60_um"] / (base["ordinary_y_band_15_60_um"] + P["h_margin_um"]),
            "ordinary_tau": row["ordinary_tau_rms_Nm"] / (old["ordinary_tau_rms_Nm"] * P["torque_ratio_max"]),
        }
        pairs = list(zip(self.events[row["id"]], self.events[base["id"]], strict=True))
        event_limit = {f"{a['event']}_{key}": a[key] / (b[key] * P["event_ratio_max"])
                       for a, b in pairs for key in ("peak_um", "iae_um_s")}
        passes = [self.manifest[row["id"]]["guard"] is None,
                  max(limit[k] for k in ("full_rms", "ordinary_rms", "ordinary_peak")) <= 1,
                  limit["ordinary_H"] <= 1, limit["ordinary_tau"] <= 1, max(event_limit.values()) <= 1]
        return dict(result, **{f"gate{i}": flag for i, flag in enumerate(passes, 1)},
                    **{f"fraction_of_limit_{k}": v for k, v in (limit | event_limit).items()},
                    passed_count=sum(passes), passed=all(passes),
                    worst_excess=max(0., max((limit | event_limit).values()) - 1),
                    summed_event_iae_ratio=sum(a["iae_um_s"] for a, _ in pairs) /
                                           sum(b["iae_um_s"] for _, b in pairs))

    def representative_checks(self):
        verified = []
        for name, (full, ordinary) in REFS.items():
            seed = 11012
            if name.startswith("P1_N"):
                c = case("reference", "N", seed)
            elif name.startswith("P5"):
                c = core.settings("reference", "Q", "L", seed, 10, .25, linear_multiplier=4.)
            else:
                c = core.settings("reference", "Q", "S", seed, 10, .25 if "_r0.25_" in name else 0.)
            row = self.run(c, old_id=name)
            assert row["completed"]
            for key, target in (("full_rms_um", full), ("ordinary_rms_um", ordinary)):
                if target is not None:
                    assert round(row[key], 2) == target, (name, key, row[key], target)
            verified.append(dict(id=row["id"], prior_case=name, full_rms_um=row["full_rms_um"],
                                 ordinary_rms_um=row["ordinary_rms_um"], cached=bool(row["cache_id"])))
        freeze(self.out / "old_representatives.json", verified)


def execute(batch):
    batch.representative_checks()
    baselines = {s: batch.run(case("baseline", "N", s)) for s in CAL}
    histories = {s: batch.run(case("history", "S", s, (1, 10))) for s in CAL}
    ranking = []
    gates = []

    def test(stage, law, filt, gain):
        rows = [batch.run(case(stage, law, s, filt, gain)) for s in CAL]
        pair = [batch.gate(row, baselines[row["seed"]], histories[row["seed"]]) for row in rows]
        gates.extend(pair)
        result = dict(stage=stage, law=law, filter=list(filt), gain=gain, ids=[r["id"] for r in rows],
                      complete=sum(g["gate1"] for g in pair), gates_passed=sum(g["passed_count"] for g in pair),
                      passed=all(g["passed"] for g in pair),
                      worst_excess=max(g["worst_excess"] if g["worst_excess"] is not None else float("inf") for g in pair),
                      worst_iae=max(g["summed_event_iae_ratio"] if g["summed_event_iae_ratio"] is not None else float("inf") for g in pair),
                      torque=max(r.get("ordinary_tau_rms_Nm", float("inf")) for r in rows))
        ranking.append(result)
        table(batch.out / "gates.csv", gates)
        return result

    def rank(r):
        return (-r["complete"], -r["gates_passed"], r["worst_excess"], r["worst_iae"], r["torque"],
                *r["filter"], r["gain"])

    def find(law, filt):
        return sorted((r for r in ranking if r["law"] == law and r["filter"] == list(filt)), key=rank)

    first = [(1, hz) for hz in P["first_order_hz"]]
    a_sfc = []
    for filt in first:
        a_sfc.append(test("A", "S", filt, 1))
        for gain in P["linear_multipliers"]:
            test("A", "L", filt, gain)
    ordered = sorted(a_sfc, key=rank)
    b_filters = [tuple(r["filter"]) for r in ordered[:2]] if any(r["complete"] for r in ordered) else [(1, 3), (1, 5)]
    b_sfc = [test("B", "S", filt, gain) for filt, gain in product(b_filters, P["sfc_gammas_B"])]
    do_c = not any(r["passed"] for r in a_sfc + b_sfc)
    best_b = min(b_sfc, key=rank)
    gamma_c = best_b["gain"] if best_b["complete"] == len(CAL) else .25
    if do_c:
        for hz in P["second_order_hz"]:
            test("C", "S", (2, hz), gamma_c)
            for gain in P["linear_multipliers"]:
                test("C", "L", (2, hz), gain)
    selected = []
    boundary = []
    sfc_order = sorted((r for r in ranking if r["law"] == "S"), key=rank)
    passing = any(r["passed"] for r in sfc_order)
    for sfc in sfc_order:
        filt = sfc["filter"]
        if (passing and not sfc["passed"]) or any(s["filter"] == filt for s in selected):
            continue
        linear = find("L", filt)[0]
        checks = []
        if linear["gain"] == 4 and linear["complete"] == len(CAL):
            x2 = next(r for r in find("L", filt) if r["gain"] == 2)
            if x2["complete"] == len(CAL):
                for a, b in zip(linear["ids"], x2["ids"], strict=True):
                    ratios = {key: batch.rows[a][key] / batch.rows[b][key]
                              for key in ("full_rms_um", "ordinary_rms_um")}
                    ratios["summed_event_iae"] = (sum(e["iae_um_s"] for e in batch.events[a]) /
                                                   sum(e["iae_um_s"] for e in batch.events[b]))
                    checks.append(ratios)
                if (any(all(x[k] <= P["effect_gain_ratio"] for x in checks) for k in checks[0]) and
                        all(max(x.values()) <= P["similar_effect_ratio"] for x in checks)):
                    test("X", "L", tuple(filt), 8)
                    linear = find("L", filt)[0]
        boundary.append(dict(filter=filt, x4_over_x2=checks,
                             x8_tested=any(r["gain"] == 8 for r in find("L", filt))))
        selected.append(dict(filter=filt, sfc_gamma=sfc["gain"], linear_multiplier=linear["gain"],
                             sfc_calibration=sfc["ids"], linear_calibration=linear["ids"],
                             sfc_calibration_pass=sfc["passed"]))
        if len(selected) == 2:
            break
    finite_ranking = [{k: None if v == float("inf") else v for k, v in r.items()}
                      for r in sorted(ranking, key=rank)]
    freeze(batch.out / "selection.json", dict(b_filters=[list(f) for f in b_filters],
           ran_C=do_c, C_gamma=gamma_c if do_c else None, boundary=boundary,
           ranking=finite_ranking, selected=selected, holdout_seeds=HOLD,
           protocol_sha256=sha(batch.out / "protocol.json"), event_sha256=sha(batch.out / "event_windows.json")))
    print("FROZEN_HOLDOUT", selected, flush=True)
    for delay in (0, P["stress_execution_delay_ms"]):
        for seed in HOLD:
            stage = "D" if delay == 0 else "E"
            n = batch.run(case(stage, "N", seed, delay=delay))
            old = batch.run(case("history", "S", seed, (1, 10)))
            for setup in selected if delay == 0 else selected[:1]:
                for law, gain in (("S", setup["sfc_gamma"]), ("L", setup["linear_multiplier"])):
                    row = batch.run(case(stage, law, seed, tuple(setup["filter"]), gain, delay))
                    gates.append(batch.gate(row, n, old))
            table(batch.out / "gates.csv", gates)
    write(batch.out / "complete.json", dict(total=len(batch.manifest),
          new=sum(not m["cached"] for m in batch.manifest.values()),
          reused=sum(m["cached"] for m in batch.manifest.values()), budget=P["budget"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--archive", type=Path, default=OLD)
    parser.add_argument("--case-json", type=Path, help="Run one exact case from the manifest or an individual case JSON")
    args = parser.parse_args()
    batch = Batch(args.out.resolve(), args.archive.resolve())
    if args.case_json:
        c = read(args.case_json)
        batch.run(c.get("case", c))
    else:
        execute(batch)


if __name__ == "__main__":
    main()
