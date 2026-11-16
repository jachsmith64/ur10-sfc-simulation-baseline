"""Independently verify the extracted payload using Python and NumPy only."""
import argparse
from hashlib import sha256, file_digest
import json
import math
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_bytes())


def load(path):
    with np.load(path,allow_pickle=False) as data:
        return dict(data)


def digest(p):
    with p.open("rb") as f:
        return file_digest(f,"sha256").hexdigest()


def bytes_identity(a):
    return {"shape":list(a.shape),"dtype":a.dtype.str,"sha256":sha256(a.tobytes()).hexdigest()}


def scalar_check(a,b):
    assert (a is None and b is None) or (
        a is not None and b is not None and math.isclose(a,b,rel_tol=2e-12,abs_tol=1e-12)),(a,b)


def vector_stats(a):
    values = [math.fsum(float(v)**2 for v in x) for x in a] if a.ndim == 2 else [float(x)**2 for x in a]
    if not values:
        return None,None
    return math.sqrt(math.fsum(values)/len(values)),math.sqrt(max(values))


def low_pass_5_hz(v):
    gain = math.tan(math.pi*5/125)
    b = gain/(1+gain)
    a = (gain-1)/(gain+1)
    result = np.zeros_like(v)
    if len(v)>1:
        result[1] = v[1]
    for i in range(2,len(v)):
        result[i] = b*(v[i]+v[i-1])-a*result[i-1]
    return result


def verify_feedback(folder, p, u, data, baseline):
    c,q = (load(folder/f"{name}_native.npz") for name in ("camera","encoder"))
    publication = load(folder/"command_events_native.npz")
    tick = u["step"]
    # Frames captured by the station refer to each run's own pre-step physical Y.
    t = data["t"][:len(p["physical_e_um"])]
    y = p["physical_e_um"]*1e-6
    for sensor,fs,delay in ((c,132,4),(q,125,0)):
        n = len(sensor["sample"])
        sample = np.arange(n)*1000/fs
        np.testing.assert_allclose(sample,sensor["sample"],atol=1e-10,rtol=0)
        np.testing.assert_array_equal(sensor["available"],sensor["model_step"]+delay)
        np.testing.assert_allclose(sensor["available_exact_ms"],sample+delay,atol=1e-10,rtol=0)
        valid = sample <= (len(t)-1)
        np.testing.assert_allclose(sensor["physical_e"][valid],np.interp(sample[valid]/1000,t,y),
                                   rtol=0,atol=2e-13)
        np.testing.assert_allclose(sensor["e"],sensor["physical_e"]+sensor["additive"],atol=2e-18,rtol=0)
        np.testing.assert_allclose(sensor["v"][1:],np.diff(sensor["e"])/np.diff(sensor["sample"]/1000),
                                   atol=1e-14,rtol=0)
        lo,hi = baseline["sample_span"][0][0],baseline["sample_span"][-1][1]
        # Compute the archived C2 table independently at each asynchronous exposure.
        times = sensor["sample"]/1000
        level = tuple(baseline["level_um"])
        ref = np.full(n,level[0])
        for j,(start,end) in enumerate(baseline["sample_span"]):
            x = np.clip((times-start)/(end-start),0.,1.)
            ref += (level[j+1]-level[j])*x**3*(10-15*x+6*x*x)
        np.testing.assert_allclose(sensor["reference_m"],ref*1e-6,atol=5e-19,rtol=0)
        assert lo < hi
    np.testing.assert_allclose(c["additive"],-c["reference_m"],atol=2e-18,rtol=0)
    np.testing.assert_allclose(q["additive"],data["pressure"][:len(q["sample"])]-q["reference_m"],atol=2e-18,rtol=0)
    np.testing.assert_allclose(q["vf"],low_pass_5_hz(q["v"]),atol=2e-14,rtol=0)
    np.testing.assert_array_equal(c["ef"],c["e"])
    np.testing.assert_array_equal(publication["compute"],tick)
    np.testing.assert_array_equal(publication["submit"],tick)
    np.testing.assert_array_equal(publication["execute"],tick)
    np.testing.assert_array_equal(publication["hold_until_exclusive"],tick+8)
    cq = np.searchsorted(c["available"],tick,side="right")-1
    qq = np.searchsorted(q["available"],tick,side="right")-1
    np.testing.assert_array_equal(publication["c_id"],cq)
    np.testing.assert_array_equal(publication["q_id"],qq)
    ready = (cq>=1)&(qq>=1)
    np.testing.assert_array_equal(publication["ready"],ready)
    np.testing.assert_array_equal(u["sample_e_m"],np.where(cq>=0,c["e"][cq],0.0))
    np.testing.assert_array_equal(u["sample_v_m_s"],np.where(ready,q["vf"][qq],0.0))
    for name,frame,index in (("c",c,cq),("q",q,qq)):
        np.testing.assert_array_equal(publication[f"{name}_available"],np.where(index>=0,frame["available"][index],-1))
        np.testing.assert_array_equal(publication[f"{name}_sample"],np.where(index>=0,frame["sample"][index],-1.))
        assert np.all(publication[f"{name}_available"]<=tick)
    return len(c["id"])+len(q["id"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path,nargs="?",default=Path(__file__).resolve().parent)
    p.add_argument("--without-manifest",action="store_true")
    args = p.parse_args()
    root = args.root
    files = 0
    if not args.without_manifest:
        listed = read(root/"SHA256SUMS.json")
        actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
        assert actual == set(listed)|{"SHA256SUMS.json"}
        for name,d in listed.items():
            assert digest(root/name)==d,name
        files=len(listed)
    config = read(root/"comparison.json")
    output = read(root/"metrics.json")
    windows = read(root/"windows_frozen.json")["trials"]
    execution = read(root/"execution.json")
    lock = read(root/"reviewed_input_lock.json")
    assert digest(root/"windows_frozen.json")==lock["windows_sha256"]
    assert execution["archive_commit"] == config["archive_commit"]
    keys = { (r["trial"],r["mode"],r["window"]):r for r in output["windows"]}
    rebuilt = {}
    n_frames = 0
    n_rows = 0
    n_values = 0
    previous_end = None
    run_items = {(r["trial"],r["law"]):r for r in execution["runs"]}
    assert len(run_items) == len(execution["runs"]) == 9
    background = load(root/"data/shared_background.npz")
    for trial,win in windows.items():
        d = root/"data"/trial
        a, data, common = (load(d/f"{name}.npz") for name in ("A_source","locked_input","shared"))
        ids = read(d/"identity.json")
        data["pressure"] = background["pressure"]
        data["dq"] = background["dq"][:len(data["t"])]
        data["dqd"] = background["dqd"][:len(data["t"])]
        data["feedback_reference_um"] = common["score_reference_um"]
        assert digest(d/"A_source.npz")==win["source_npz_sha256"]
        for name,arr in data.items():
            assert bytes_identity(arr)==ids["input_arrays"][name],name
        for name,arr in common.items():
            assert bytes_identity(arr)==ids["shared_arrays"][name],name
        np.testing.assert_array_equal(data["feedback_reference_um"],common["score_reference_um"])
        mt = load(d/"display_mapping.npz")
        assert mt["trial_id"].item()==trial
        np.testing.assert_array_equal(mt["original_time_s"],data["t"])
        np.testing.assert_array_equal(mt["display_time_s"],data["t"]+mt["display_time_s"][0])
        if previous_end is not None:
            scalar_check(mt["display_time_s"][0]-previous_end,config["display_gap_s"])
        previous_end = mt["display_time_s"][-1]
        windows_s = win["windows_s"]
        baseline = {"sample_span":[windows_s[k] for k in ("start_transition","turn_transition","stop_transition")],
                    "level_um":[win["bases_um"][k] for k in ("pre_static","outbound","return_straight","post_static")]}
        vel = load(d/"velocity_nominal_native.npz")
        for mode in "NLS":
            folder = d/mode
            phy,u = (load(folder/f"{name}.npz") for name in ("physical_1ms","command_native"))
            receipt=read(folder/"case.json")
            row=read(folder/"row.json")
            prov=read(folder/"trace_provenance.json")
            actual=run_items[trial,mode]
            assert actual["input_file_sha256"]==lock["input_hashes"][trial]
            assert actual["shared_file_sha256"]==digest(d/"shared.npz")
            assert row["mode"]==actual["physical_mode"]==config["mode_check"][mode]
            assert receipt["case"]["law"] == mode and receipt["parameters"] == read(d/"N_case.json")["parameters"]
            assert all(receipt["case"][k]==v for k,v in config["case_check"].items())
            assert not receipt["remaining_active"] and receipt["camera_noise_m"]==0 and not actual["planning"]
            assert all(prov[k] for k in ("all_step_Fd_exact","all_step_external_applied_equals_only_disturbance","all_step_remaining_and_derivative_zero"))
            t = data["t"][:len(phy["physical_e_um"])]
            cmds = u["step"]
            np.testing.assert_array_equal(cmds,np.arange(0,len(t),8))
            if not len(t):
                assert row["status"]!="OK"
                continue
            n_frames += verify_feedback(folder,phy,u,data,baseline)
            ready = row["status"] == "OK"
            np.testing.assert_array_equal(phy["q"][0],data["initial_q_rad"])
            np.testing.assert_array_equal(phy["qdot"][0],np.zeros(6))
            np.testing.assert_array_equal(vel["qdot_nom_command_rad_s"][:len(cmds)],data["qdot_nom_rad_s"][cmds])
            np.testing.assert_array_equal(vel["q_velocity_ff_rad"][:len(cmds)],0.1*data["qdot_nom_rad_s"][cmds])
            kp, kd = np.array(actual["compiled"]["Kp_Nm_per_rad"]),np.array(actual["compiled"]["Kd_Nms_per_rad"])
            base = data["q_nom"][cmds]+data["dq"][cmds]+common["common_delta_q"][cmds]+vel["q_velocity_ff_rad"][:len(cmds)]
            np.testing.assert_array_equal(u["base_ctrl"],base)
            np.testing.assert_array_equal(u["actual_micro_q"],u["final_ctrl"]-u["base_ctrl"])
            pconf=receipt["parameters"]
            e,v=u["sample_e_m"],u["sample_v_m_s"]
            zero=np.zeros_like(e)
            forces = (-pconf["K"]*e if mode!="N" else zero,
                      -pconf["B0"]*v if mode!="N" else zero,
                      -pconf["mu"]*np.abs(v)**(pconf["n"]-1)*v if mode=="S" else zero)
            for name,x in zip(("F_spring","F_linear","F_shear"),forces):
                np.testing.assert_allclose(u[name],x,rtol=0,atol=1e-12)
            assert bool(np.any(u["F_shear"]))==(mode=="S")
            np.testing.assert_allclose(u["F_requested"],sum(forces),rtol=0,atol=1e-12)
            np.testing.assert_allclose(u["tau_requested"],u["F_requested"][:,None]*u["J_y_m_rad"],rtol=0,atol=1e-12)
            np.testing.assert_allclose(u["actual_micro_q"],u["tau_requested"]/kp,rtol=0,atol=1e-15)
            np.testing.assert_allclose(u["qfrc_applied"],common["Fd"][cmds,None]*u["J_y_m_rad"],rtol=0,atol=2e-12)
            hold = np.arange(len(t))//8
            sat = np.logical_or.reduce([np.any(phy[k],axis=1) for k in ("ctrl_saturation","force_saturation","joint_saturation")])
            modeled_tau = kp*(u["final_ctrl"][hold]-phy["q"])-kd*phy["qdot"]
            np.testing.assert_allclose(phy["qfrc_actuator"][~sat],modeled_tau[~sat],atol=1e-8,rtol=1e-10)
            e = phy["physical_e_um"]-common["score_reference_um"][:len(t)]
            if mode == "N":
                assert actual["regression"]["all_arrays_byte_identical"]
                for key in ("physical_e_um","physical_v","vx_m_s","q","qdot","qfrc_actuator","tau_realized_increment"):
                    assert bytes_identity(phy[key])==ids["trace_arrays"][key]
            for name,(lo,hi) in {"full":(0.,float("inf")),**windows_s}.items():
                select = np.array([i for i,time in enumerate(t) if time>=lo and
                                  (time<hi or time==hi==a["frame_host_time_s"][-1])],dtype=np.intp)
                record = keys[trial,mode,name]
                assert record["n"]==len(select) and record["complete_run"]==ready
                n_rows += 1
                pairs=np.array([i for j,i in zip(select,select[1:]) if i==j+1],dtype=np.intp)
                inc = phy["tau_realized_increment"]
                channels={"score_um":e[select],"requested_N":u["F_requested"][hold[select]],
                          "spring_N":u["F_spring"][hold[select]],"linear_N":u["F_linear"][hold[select]],
                          "nonlinear_N":u["F_shear"][hold[select]],"micro_q_rad":u["actual_micro_q"][hold[select]],
                          "request_tau_Nm":u["tau_requested"][hold[select]],"actuator_total_tau_Nm":phy["qfrc_actuator"][select],
                          "realized_increment_tau_Nm":inc[select],"realized_increment_tau_rate_Nm_s":1000*(inc[pairs]-inc[pairs-1])}
                for label,arr in channels.items():
                    x,y=vector_stats(arr)
                    scalar_check(record[label]["rms"],x)
                    scalar_check(record[label]["peak"],y)
                    n_values+=2
                    if label=="score_um":
                        rebuilt[trial,mode,name]=(x,y,ready)
                peak = float(t[max(select,key=lambda i:abs(e[i]))]) if len(select) else None
                scalar_check(record["peak_time_s"],peak)
                for flag,count in record["saturation_steps"].items():
                    assert int(phy[flag][select].any(axis=1).sum())==count
    for row in output["reductions_percent"]:
        for comparison in ("LN","SN","SL"):
            c,b=[rebuilt[row["trial"],law,row["window"]] for law in comparison]
            for i,key in enumerate(("rms","peak")):
                val=100*(1-c[i]/b[i]) if c[2] and b[2] and b[i] and b[i]>config["reduction_denominator_min_um"] else None
                scalar_check(row[comparison][key],val)
                n_values+=1
    print(f"Verified {files} file hashes, {n_rows} run/window rows, {n_values} metric values, "
          f"{n_frames} native observations and all 9 control / timing records.")


if __name__ == "__main__":
    main()
