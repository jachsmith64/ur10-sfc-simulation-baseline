"""Frozen staged campaign with paired inputs and calibration-only selection."""
import argparse
from collections import Counter
from hashlib import sha256
from itertools import product
from pathlib import Path
import time

import numpy as np

from run_campaign import ROOT, SECONDARY, data_arrays, initialize, read, run_case, settings, write

SEEDS = (11010,11011,11012,11013)
GRID = [(None,1.)] + list(product((10,20,35),(0.,.25,.5)))
LOW = "ordinary_e_band_0.5_15"
FULL = "full_rms_um"
PEAK = "full_peak_um"
COST = "torque_increment_rms"


class Batch:
    def __init__(self,out,deadline):
        self.out,self.deadline = out,deadline
        self.sweep,self.ctx = initialize(out)
        self.manifest = read(out/"run_manifest.json") if (out/"run_manifest.json").exists() else {}

    def run(self,cases):
        results = []
        for case in cases:
            label = case["id"]
            folder = self.out/"runs"/label
            entry = self.manifest.setdefault(label,{"case":case})
            assert entry["case"] == case
            if not folder.exists() and time.time() >= self.deadline:
                entry["status"] = "UNRUN_WALL_CLOCK"
                continue
            entry["status"] = "IN_PROGRESS"
            write(self.out/"run_manifest.json",self.manifest)
            result = run_case(self.out,self.sweep,self.ctx,case)
            entry["status"] = "COMPLETE" if result and result["completed"] else "FAILED" if result else "TECHNICAL"
            if result:
                results.append(result)
        write(self.out/"run_manifest.json",self.manifest)
        return results

    def checkpoint(self,stage,**extra):
        payload = dict(stage=stage,counts=dict(Counter(v["status"] for v in self.manifest.values())),
                       compute_minutes_left=round((self.deadline-time.time())/60,1),**extra)
        write(self.out/f"checkpoint_{stage}.json",payload)
        print("CHECKPOINT",payload,flush=True)

    def trace(self,result):
        return data_arrays(self.out/"runs"/result["id"]/"trace.npz")


def p1_cases(seed,grid,group):
    yield settings(group,"N","N",seed)
    for source, (fc,rho), law in product(("Q","C"),grid,("L","S")):
        yield settings(group,source,law,seed,fc,rho)


def calibrate(rows,grid):
    index = {(r["seed"],r["estimator"],r["law"],r["fc"],r["rho"]):r for r in rows}
    baselines = [index[(s,"N","N",None,1.)] for s in SEEDS[:2]]
    assert all(x["completed"] for x in baselines)
    records, reps, choices = [], {}, {}
    for source in ("Q","C"):
        entries = []
        for (fc,rho), law in product(grid,("S","L")):
            if not fc:
                continue
            runs = [index.get((s,source,law,fc,rho)) for s in SEEDS[:2]]
            pairs = [(r,n) for r,n in zip(runs,baselines) if r and r["completed"]]
            ratios = [{k:r[k]/n[k] for k in (FULL,LOW,PEAK)} for r,n in pairs]
            passed = len(ratios)==2 and all(x[FULL]<1 and x[LOW]<1 and x[PEAK]<=1.1 for x in ratios)
            rank = [max(x[FULL] for x in ratios),max(x[LOW] for x in ratios),
                    max(r[COST] for r,_ in pairs)] if len(ratios)==2 else [1e9,1e9,1e9]
            item = dict(source=source,fc=fc,rho=rho,law=law,passed=passed,rank=rank,
                        complete_calibration_seeds=len(pairs),ratios=ratios,
                        calibration_ids=[r["id"] for r in runs if r])
            entries.append(item)
        records.extend(entries)
        def rank_key(x):
            # Among passing rows, score worst paired RMS, then low-band RMS, then applied torque.
            return (not x["passed"], -x["complete_calibration_seeds"], *x["rank"],
                    x["law"]!="S",x["fc"],x["rho"])
        choices[source] = [min((x for x in entries if (x["rho"]==0)==pure),key=rank_key)
                           for pure in (True,False)]
        reps[source] = min(choices[source],key=rank_key)
    s_pool = [x for x in records if x["law"]=="S" and x["passed"]
              and (x["fc"],x["rho"]) in {(r["fc"],r["rho"]) for r in choices[x["source"]]}]
    primary = min(s_pool or list(reps.values()),key=rank_key)
    return dict(uses_seeds=list(SEEDS[:2]),gate="both complete; full and ordinary 0.5–15 Hz RMS < paired N; peak <=1.10*N",
                rank="gate, completion, worst paired full RMS, worst paired low RMS, max incremental torque RMS",
                quotas="one pure LP and one residual-retaining representative per source",
                candidates=choices,representatives=reps,primary=primary,calibration=records)


def p2(batch,selection):
    for source,x in selection["representatives"].items():
        fc,rho=x["fc"],x["rho"]
        batch.run([settings("P2",source,law,seed,fc,rho,branch=branch)
                   for seed in SEEDS[:3] for branch,law in (("EV","L"),("EV","S"),("NL","S"))])
    rows = batch.run([settings("P2weak",source,"S",seed,gamma=gamma)
                      for source,gamma,seed in product(("Q","C"),(.25,.5),SEEDS[:2])])
    picks = {}
    for source in ("Q","C"):
        def rank(gamma):
            runs = [x for x in rows if x["estimator"]==source and x["gamma"]==gamma and x["completed"]]
            return (-len(runs),max((x[FULL] for x in runs),default=1e9),gamma)
        picks[source] = min((.25,.5),key=rank)
    freeze = batch.out/"selection_weak.json"
    if freeze.exists():
        assert read(freeze)==picks
    else:
        write(freeze,picks)
    batch.run([settings("P2weak",source,"S",SEEDS[2],gamma=gamma) for source,gamma in picks.items()])
    batch.checkpoint("P2",weak_gamma=picks)


def matched(group, primary, **changes):
    seed = SEEDS[3]
    source,fc,rho = (primary[k] for k in ("source","fc","rho"))
    return [settings(group,"N","N",seed,**changes),
            *[settings(group,source,law,seed,**changes) for law in ("L","S")],
            *[settings(group,source,law,seed,fc,rho,**changes) for law in ("L","S")]]


def p3(batch,primary):
    for group, changes in (("P3_2ms",dict(command_ms=2)),
                           ("P3_D8",dict(camera_delay_ms=8)),
                           ("P3_exec3",dict(execution_delay_ms=3)),
                           ("P3_record",dict(pressure=SECONDARY))):
        batch.run(matched(group,primary,**changes))
    batch.checkpoint("P3",primary=primary)


def p4(batch,primary,base):
    n = next((x for x in base if x["law"]=="N" and x["completed"]),None)
    batch.run(matched("P4_M",primary,measurement_injection=True,scenario="M"))
    if n is None:
        batch.checkpoint("P4",blocked="S3 no paired N baseline for force calibration")
        return
    reference = batch.trace(n)
    calibration = {}
    for label, hz in (("P4_P8",[8,45]),("P4_P12",[12,55])):
        cases = batch.run([settings(label+"_probe","N","N",SEEDS[3],scenario=label,
                                     force_hz=hz,force_amplitude_N=[.1,.1])])
        if not cases or not cases[0]["completed"]:
            batch.checkpoint(label,blocked="N calibration unavailable")
            continue
        trial = batch.trace(cases[0])
        mask = (trial["t"]>=3)&(trial["t"]<4)
        x = trial["t"][mask]
        delta = trial["physical_e_um"][mask]-reference["physical_e_um"][mask]
        basis = np.column_stack([fun(2*np.pi*f*x) for f in hz for fun in (np.sin,np.cos)])
        amp = np.linalg.norm(np.linalg.lstsq(basis,delta,rcond=None)[0].reshape(2,2),axis=1)
        assert np.all(amp>0)
        target = n["ordinary_rms_um"] * np.array([.6,.12])
        fixed = (.1*target/amp).tolist()
        calibration[label] = dict(probe=cases[0]["id"],hz=hz,probe_response_um=amp.tolist(),
                                   target_incremental_displacement_um=target.tolist(),frozen_force_N=fixed,
                                   fit_window_s=[3,4],reference=n["id"])
        write(batch.out/"force_calibration.json",calibration)
        batch.run(matched(label,primary,scenario=label,force_hz=hz,force_amplitude_N=fixed))
    if "P4_P8" in calibration:
        p8 = calibration["P4_P8"]
        n_m = next((r for r in batch.manifest.values() if r["case"]["group"]=="P4_M" and r["case"]["law"]=="N"),None)
        n_p = next((r for r in batch.manifest.values() if r["case"]["group"]=="P4_P8" and r["case"]["law"]=="N"),None)
        if n_m and n_p and n_m["status"]==n_p["status"]=="COMPLETE":
            batch.run(matched("P4_MP8",primary,measurement_injection=True,scenario="MP8",
                              force_hz=p8["hz"],force_amplitude_N=p8["frozen_force_N"]))
    batch.checkpoint("P4",calibration=calibration)


def p5(batch,primary,p1):
    source,fc,rho = (primary[k] for k in ("source","fc","rho"))
    def paired(law):
        return [next((r for r in p1 if r["seed"]==seed and r["estimator"]==source and r["law"]==law
                      and (r["fc"],r["rho"])==(fc,rho) and r["completed"]),None) for seed in SEEDS[:2]]
    linear,sfc = paired("L"),paired("S")
    if not all(l and s and s[FULL] <= .95*l[FULL] for l,s in zip(linear,sfc)):
        batch.checkpoint("P5",skipped="Filtered S did not beat identical-filter L by >=5% on both calibration seeds")
        return
    rows = batch.run([settings("P5",source,"L",seed,fc,rho,linear_multiplier=multiplier)
                      for seed,multiplier in product(SEEDS[:2],(1.5,2.,4.))])
    def rank(multiplier):
        pairs = [(r,s) for s in sfc for r in rows if r["seed"]==s["seed"] and
                 r["linear_multiplier"]==multiplier and r["completed"]]
        return -len(pairs),max((abs(r[COST]/s[COST]-1) for r,s in pairs),default=1e9),multiplier
    selected = min((1.5,2.,4.),key=rank)
    write(batch.out/"selection_strong_linear.json",dict(multiplier=selected,rank=rank(selected),calibration_only=True))
    batch.run([settings("P5",source,"L",SEEDS[2],fc,rho,linear_multiplier=selected)])
    batch.checkpoint("P5",linear_multiplier=selected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=ROOT/"results")
    parser.add_argument("--compute-until",type=float,default=0)
    args = parser.parse_args()
    batch = Batch(args.out,args.compute_until or time.time()+6000)
    pilot_cases = [settings("P1","N","N",SEEDS[0]),
                   settings("P1","C","L",SEEDS[0]),
                   settings("P1","C","S",SEEDS[0]),
                   settings("P1","Q","S",SEEDS[0],10,0.)]
    pilot = batch.run(pilot_cases)
    batch.checkpoint("P0",pilot=[(r["id"],r["status"],r["wall_s"]) for r in pilot])
    assert len(pilot)==4 and pilot[0]["completed"] and pilot[1]["completed"], "P0 common setup failed"
    n, l = [batch.trace(x) for x in pilot[:2]]
    assert not np.array_equal(n["physical_y"],l["physical_y"])
    assert not np.array_equal(n["cam_e"],l["cam_e"])
    assert read(batch.out/"runs"/pilot[0]["id"]/"inputs.json")==read(batch.out/"runs"/pilot[1]["id"]/"inputs.json")
    plan_path = batch.out/"grid.json"
    if plan_path.exists():
        grid = [tuple(x) for x in read(plan_path)["grid"]]
    else:
        forecast = max(x["wall_s"] for x in pilot)*95*1.5
        grid = GRID if time.time()+forecast < batch.deadline else [x for x in GRID if x[1] != .25]
        write(plan_path,dict(grid=grid,wall_forecast_s=forecast))
    results = []
    for seed in SEEDS[:2]:
        results.extend(batch.run(p1_cases(seed,grid,"P1")))
        batch.checkpoint(f"P1_S{seed-11010}")
    if len(results)<2*(1+4*len(grid)):
        batch.checkpoint("P1_partial",reason="calibration incomplete or technical failure; holdout selection withheld")
        return
    selection = calibrate(results,grid)
    freeze = batch.out/"selection.json"
    if freeze.exists():
        assert read(freeze)==selection
    else:
        write(freeze,selection)
    holdout = [settings("P1","N","N",SEEDS[2])]
    for source, reps in selection["candidates"].items():
        for fc,rho in [(None,1.)]+[(x["fc"],x["rho"]) for x in reps]:
            holdout += [settings("P1",source,law,SEEDS[2],fc,rho) for law in ("L","S")]
    results.extend(batch.run(holdout))
    batch.checkpoint("P1_S2",primary=selection["primary"])
    p2(batch,selection)
    primary = selection["primary"]
    p3(batch,primary)
    reference = batch.run(matched("P4_ref",primary))
    p4(batch,primary,reference)
    p5(batch,primary,results)


if __name__ == "__main__":
    main()
