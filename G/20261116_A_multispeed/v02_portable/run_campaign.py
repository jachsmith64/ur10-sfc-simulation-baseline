"""Run the fixed feedback campaign through the archived A MuJoCo integrator."""
from collections import deque
from dataclasses import asdict
import difflib
from hashlib import sha256
import importlib.util
import sys
import time
import traceback
from pathlib import Path

from setup_case import ROOT, SOURCE, DEPS, read, sha, write

sys.path[:0] = [str(SOURCE),str(DEPS)]

import numpy as np
import mujoco
import scipy
from scipy.signal import welch
from b132_run import context
from b132_common import B, clean, save
from run_trial import GuardTrace
from position_execution import POSITION_EQUIV, compiled_mapping
from position_logging import PreciseLogger
from disturbance import FrozenDisturbance
from sim_env import SimEnv
import config as aconfig
import sfc

from filter_sensing import Measurement, checks

CONFIG = read(ROOT/"campaign.json")
CFG_SHA = "ba1b97624f9e8841776e7893ecf42b94a1d8988734746382a1230aa4716796d9"
PRIMARY = CONFIG["primary_pressure"]
SECONDARY = CONFIG["secondary_pressure"]
DATA = ROOT/"frozen"
MODES = dict(N="A0", L="Alinear", S="ASFC")
BANDS = ((0,.5),(.5,5),(5,15),(15,40),(40,60),(60,500),(.5,15))
REPLACE_SIGNATURE = "measurement_adapter=None, final_command_hold=False)"
REPLACE_HOLD = """            if final_command_hold:
                if updated or held_final_command is None:held_final_command=command.copy()
                command=held_final_command
"""
INSERT_HOLD = """            if command_channel is not None:
                command = command_channel.tick(i, env, command, base_ctrl, out, updated)
            elif final_command_hold:
                if updated or held_final_command is None:held_final_command=command.copy()
                command=held_final_command
"""


def integration_module(out):
    source = B/"sweep.py"
    original = source.read_bytes().decode().replace("\r\n","\n")
    assert original.count(REPLACE_SIGNATURE) == 1 and original.count(REPLACE_HOLD) == 1
    code = original.replace(REPLACE_SIGNATURE, "measurement_adapter=None, final_command_hold=False, command_channel=None)").replace(REPLACE_HOLD,INSERT_HOLD)
    path = out/"sweep_campaign.py"
    if not path.exists():
        path.write_text(code,encoding="utf-8",newline="\n")
        patch = difflib.unified_diff(original.splitlines(keepends=True),code.splitlines(keepends=True),
                                    "a/01_工程/sweep.py","b/01_工程/sweep.py")
        (out/"command_channel.patch").write_text("".join(patch),encoding="utf-8",newline="\n")
    assert path.read_text(encoding="utf-8") == code
    spec = importlib.util.spec_from_file_location("sweep_campaign",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Coordinator(GuardTrace):
    def __init__(self, ctx, case, sensor):
        super().__init__("baseline" if case["law"] == "N" else "sfc",ctx["c"]["q"],ctx["parameters"])
        assert not self.planning
        self.case,self.sensor = case,sensor
        self.rejection = None

    def before_step(self, i, env, measurement, ctrl, base):
        super().before_step(i,env,measurement,ctrl,base)
        ctrl.params["gamma"] = self.case["gamma"]

    def command(self, i, env, sfc_command, out, actuation):
        try:
            return super().command(i,env,sfc_command,out,actuation)
        except RuntimeError:
            self.rejection.update(stage="original_pre_hold",executed=False, q=env.data.qpos.copy(),
              qd=env.data.qvel.copy(), force_components_N=[out["F_K"],out["F_B"],out["F_shear"]],
              command_measurement=self.sensor.last_command.copy())
            raise


class CommandChannel:
    """Deterministic enqueue, optional execution delay and full-vector ZOH."""
    def __init__(self, delay, period, ext):
        self.delay,self.period,self.ext = delay,period,ext
        self.pending = deque()
        self.active = None
        self.events = []
        self.applied_event = False

    def tick(self, step, env, command, base, out, updated):
        self.applied_event = False
        if self.active is None:
            self.active = dict(compute=-1,submit=-1,execute=-1,command=env.data.ctrl.copy(),
                               base=env.data.ctrl.copy(),force=0.,c_sample=-1,q_sample=-1)
        if updated:
            observation = self.ext.sensor.last_command.copy()
            assert observation["compute"] == step and all(observation[x] <= step for x in ("c_available","q_available"))
            self.pending.append(dict(compute=step,submit=step,execute=step+self.delay,command=command.copy(),
                                     base=base.copy(),force=out["F_sfc"],c_sample=observation["c_sample"],
                                     q_sample=observation["q_sample"], observation=observation))
        if self.pending and self.pending[0]["execute"] <= step:
            packet = self.pending.popleft()
            assert packet["execute"] == step
            if self.delay:
                x = self.ext
                request = packet["command"]
                tau = x.kp*(request-env.data.qpos)-x.kd*env.data.qvel
                bad_q = np.flatnonzero((request<x.ctrl_lo)|(request>x.ctrl_hi))+1
                bad_tau = np.flatnonzero((tau<x.tau_lo)|(tau>x.tau_hi))+1
                if len(bad_q) or len(bad_tau) or not np.all(np.isfinite(tau)):
                    x.rejection = dict(step=step,t_s=step*.001,stage="delayed_dispatch",executed=False,
                        position_request_rad=request,total_torque_request_Nm=tau,position_joints=bad_q,
                        torque_joints=bad_tau,torque_lower_Nm=x.tau_lo,torque_upper_Nm=x.tau_hi,
                        F_requested_N=packet["force"],q=env.data.qpos.copy(),qd=env.data.qvel.copy(),
                        command_measurement=packet["observation"])
                    raise RuntimeError("Delayed command violates compiled hard boundary; stop valid prefix")
            self.active = packet
            self.applied_event = True
            self.events.append(dict(packet["observation"], submit=packet["submit"],execute=step,
                                    hold_until_exclusive=step+self.period))
        return self.active["command"]


class Logger(PreciseLogger):
    def __init__(self, path, ctx, case, sensor, channel):
        super().__init__(path,len(ctx["c"]["t"]),"C_ORDINARY",POSITION_EQUIV,case["command_ms"],ctx["input_id"],
                         input_data=ctx["scaled"],common_correction=ctx["corr"])
        self.sensor,self.channel,self.case = sensor,channel,case
        for key in list(sensor.values()) + ["compute_cam_sample","compute_cam_available","compute_cam_age_ms",
                "compute_enc_sample","compute_enc_available","compute_enc_age_ms","applied_source_compute",
                "applied_cam_age_ms","applied_enc_age_ms","applied_force","exec_event",
                "compute_ready","compute_raw_v","compute_filtered_v","hold_until_exclusive"]:
            self.arr[key] = np.empty(self.n)
        for key in ("actual_micro_q","prehold_tau"):
            self.arr[key] = np.empty((self.n,6))
        self.arr["physical_xyz_m"] = np.empty((self.n,3))
        self.arr["physical_xyz_error_um"] = np.empty((self.n,3))
        self.nominal_xyz = ctx["nominal_xyz"]
        self.sat_run = 0

    def sample(self,i,env,measurement,y_d,out,fd,base_ctrl,manager,updated,remaining,remaining_dot):
        p = self.channel.active
        super().sample(i,env,measurement,y_d,out,fd,p["base"],manager,updated,remaining,remaining_dot)
        c = self.sensor.last_command
        for k,v in self.sensor.values().items():
            self.arr[k][i] = v
        for key, v in dict(compute_cam_sample=c["c_sample"],compute_cam_available=c["c_available"],
                compute_cam_age_ms=c["c_age_ms"],compute_enc_sample=c["q_sample"],
                compute_enc_available=c["q_available"],compute_enc_age_ms=c["q_age_ms"],compute_ready=c["ready"],
                applied_source_compute=p["compute"],applied_cam_age_ms=i-p["c_sample"] if p["c_sample"]>=0 else -1,
                applied_enc_age_ms=i-p["q_sample"] if p["q_sample"]>=0 else -1,applied_force=p["force"],
                exec_event=self.channel.applied_event,compute_raw_v=c["v_raw_m_s"],compute_filtered_v=c["v_filtered_m_s"],
                hold_until_exclusive=p["execute"]+self.case["command_ms"] if p["execute"]>=0 else self.case["execution_delay_ms"]).items():
            self.arr[key][i] = v
        self.arr["actual_micro_q"][i] = p["command"]-p["base"]
        ext = self.channel.ext
        self.arr["prehold_tau"][i] = ext.records[i]["tau_total_requested"]
        self.arr["physical_xyz_m"][i] = env.tcp_pos()
        self.arr["physical_xyz_error_um"][i] = (self.arr["physical_xyz_m"][i]-self.nominal_xyz[i])*1e6
        anysat = any(self.arr[k][i].any() for k in ("ctrl_saturation","force_saturation","joint_saturation"))
        self.sat_run = self.sat_run+1 if anysat else 0
        if self.sat_run >= 20:
            ext.rejection = dict(step=i,t_s=i*.001,stage="persistent_saturation",executed=False,
                                 position_request_rad=p["command"],F_requested_N=p["force"])
            raise RuntimeError("Persistent compiled output saturation20ms;stopvalidprefix")


def pressure(name):
    if name == "none":
        return np.zeros(1250)
    file = DATA/f"{name}.npz"
    assert sha(file) == read(DATA/"pressure_identity.json")["scenarios"][name]["sha256"]
    with np.load(file,allow_pickle=False) as z:
        assert np.array_equal(z["encoder_step"],np.arange(0,10000,8))
        return z["delta_m"].copy()


def case_id(case):
    return case["id"]


def settings(group, estimator, law, seed, fc=None, rho=1., **changes):
    case = dict(group=group,estimator=estimator,law=law,seed=seed,camera_delay_ms=4,
                command_ms=8,execution_delay_ms=0,pressure=PRIMARY,branch="V",
                fc=fc,rho=rho,gamma=1.,linear_multiplier=1.,scenario="main",
                measurement_injection=False,force_hz=[],force_amplitude_N=[])
    case.update(changes)
    case["id"] = (f"{group}_{estimator}-{law}_{case['branch']}_f{fc or 'raw'}_r{rho:g}"
                  f"_g{case['gamma']:g}_b{case['linear_multiplier']:g}_s{seed-11010}")
    return case


def as_frames(sensor):
    return dict(camera=[asdict(f) for f in sensor.c.frames],encoder=[asdict(f) for f in sensor.q.frames])


def validate(a, ctx, case, sensor, channel, row, physical_steps, force_values):
    n = row["n_steps_done"]
    step = np.arange(n)
    assert n == len(a["t"]) == physical_steps and n > 0
    assert (row["status"]=="OK") == (n==10000)
    assert np.array_equal(a["t"],ctx["c"]["t"][:n])
    assert np.array_equal(a["q_nominal"],ctx["c"]["q"][:n])
    assert np.array_equal(a["q"][0],ctx["c"]["q"][0]) and not a["qdot"][0].any()
    assert np.array_equal(a["Fd"],force_values[:n])
    assert np.array_equal(a["remaining"],ctx["scaled"]["remaining"][:n])
    assert np.array_equal(a["common_delta_q"],ctx["corr"]["common_delta_q"][:n])
    assert np.array_equal(a["qfrc_applied"],a["qfrc_disturbance"]) and not channel.ext.decisions
    assert np.array_equal(a["updated"],step%case["command_ms"]==0)
    assert np.array_equal(a["exec_event"]>0,(step>=case["execution_delay_ms"])&((step-case["execution_delay_ms"])%case["command_ms"]==0))
    held = np.maximum.accumulate(np.where((a["exec_event"]>0)|(step==0),step,0))
    assert np.array_equal(a["final_ctrl"],a["final_ctrl"][held])
    assert np.allclose(a["base_ctrl"]+a["actual_micro_q"],a["final_ctrl"],rtol=0,atol=1e-13)
    for sensor_name, ch in (("camera",sensor.c),("encoder",sensor.q)):
        frames = ch.frames
        stamps = np.array([f.sample for f in frames])
        avail = np.array([f.available for f in frames])
        grid = np.array([f.model_step for f in frames])
        assert np.allclose(stamps,np.arange(len(stamps))*1000/ch.fs,rtol=0,atol=1e-10)
        assert np.array_equal(grid,np.ceil(stamps)) and np.all(avail-grid==ch.delay)
        valid = grid<n
        if valid.any():
            observed = np.array([f.physical_e for f in frames])[valid]
            assert np.allclose(observed,np.interp(stamps[valid],step,a["physical_y"]-aconfig.Y_NOMINAL),rtol=0,atol=1e-16)
            expected_additive = (np.interp(stamps[valid],step,a["remaining"])+sensor.noise[:len(stamps)][valid]) if sensor_name=="camera" else sensor.pressure[:len(stamps)][valid]
            assert np.allclose(np.array([f.additive for f in frames])[valid],expected_additive,rtol=0,atol=1e-16)
        if len(frames)>1:
            e = np.array([f.e for f in frames])
            assert np.allclose(np.diff(e)/(np.diff(stamps)*.001),[f.v for f in frames[1:]],rtol=0,atol=1e-13)
    assert np.all(a["compute_cam_available"]<=step) and np.all(a["compute_enc_available"]<=step)
    commands = a["updated"]
    raw = a["compute_raw_v"][commands]
    filtered = a["compute_filtered_v"][commands]
    p = ctx["parameters"]
    v_lin = raw if case["branch"]=="NL" else filtered
    if case["law"] != "N":
        assert np.allclose(a["F_linear"][commands],-p["B0"]*case["linear_multiplier"]*v_lin,rtol=0,atol=1e-12)
    if case["law"] == "S":
        assert np.allclose(a["F_shear"][commands],-case["gamma"]*p["mu"]*abs(filtered)**(p["n"]-1)*filtered,rtol=0,atol=1e-11)
    return dict(passed=True, physical_steps=physical_steps, camera_frames=len(sensor.c.frames),
                encoder_frames=len(sensor.q.frames), executed_commands=len(channel.events),live_observation_verified=True,
                causal_samples=True,frozen_input_identity=True,normal_planner_disabled=True,whole_command_hold=True)


def data_arrays(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def rms(x):
    return float(np.sqrt(np.mean(x*x)))


def magnitude(x):
    return abs(x) if x.ndim==1 else np.linalg.norm(x,axis=1)


def spectrum(a, spans):
    nperseg = 2000
    powers = []
    for lo, hi in spans:
        x = np.asarray(a[lo:hi])
        if len(x) < nperseg:
            continue
        f, p = welch(x, fs=1000, window="hann", nperseg=nperseg, noverlap=1000,
                     detrend="constant", axis=0)
        count = (len(x)-1000)//1000
        powers.append((count, p))
    if not powers:
        return None
    count = sum(w for w,_ in powers)
    return f,sum(w*p for w,p in powers)/count,count


def frequency_bands(a, spans):
    spec = spectrum(a,spans)
    if spec is None:
        return {}
    f,p,count = spec
    result = {"segments":count,"resolution_Hz":.5}
    for lo,hi in BANDS:
        mask = (f>=lo) & ((f<=hi) if hi==500 else (f<hi))
        result[f"{lo:g}_{hi:g}"] = float(np.sqrt(.5 * p[mask].sum()))
    return result


def metrics(a, row, case, guard):
    t = a["t"]
    error = a["physical_e_um"]
    full = row["status"]=="OK" and len(t)==10000
    scopes = {"full":np.ones(len(t),bool),"ordinary":np.zeros(len(t),bool)}
    for lo,hi in CONFIG["ordinary_windows_s"]:
        scopes["ordinary"] |= (t>=lo)&(t<hi)
    result = dict(case, status=row["status"], valid_steps=len(t), note=row["note"],
                  stop_s=guard.get("t_s",float(len(t)/1000)) if guard else (None if full else float(len(t)/1000)),
                  completed=full)
    spans = [(round(lo*1000),round(hi*1000)) for lo,hi in CONFIG["ordinary_windows_s"]]
    for name, mask in scopes.items():
        result[f"{name}_valid_n"] = int(mask.sum())
        if not mask.any():
            continue
        y = abs(error[mask])
        for key,value in dict(rms=rms(y),peak=float(y.max()),p95=float(np.quantile(y,.95)),
                              p99=float(np.quantile(y,.99))).items():
            result[f"{name}_{key}_um"] = value
        result[f"{name}_mean_um"] = float(np.mean(error[mask]))
        for axis,index in (("X",0),("Z",2)):
            values = a["physical_xyz_error_um"][mask,index]
            result[f"{name}_{axis}_rms_um"] = rms(values)
            result[f"{name}_{axis}_peak_um"] = float(np.max(abs(values)))
    if full:
        for name, windows in (("full",[(0,len(t))]),("ordinary",spans)):
            for tag,key,scale in (("e","physical_e_um",1),("v","physical_v",1e6),
                                   ("cmd","applied_force",1),("v_used","sample_v_m_s",1e6)):
                for band,value in frequency_bands(a[key]*scale,windows).items():
                    result[f"{name}_{tag}_band_{band}"] = value
    for label,key in (("F","F_requested"),("applied_F","applied_force"),("micro","actual_micro_q"),
                      ("torque_increment","tau_realized_increment"),("torque_total","qfrc_actuator"),
                      ("shear","F_shear"),("linear","F_linear")):
        x = a[key]
        m = magnitude(x)
        result[f"{label}_rms"] = rms(m)
        result[f"{label}_peak"] = float(m.max())
        result[f"{label}_rate_rms_per_s"] = rms(magnitude(np.diff(x,axis=0)/.001)) if len(t)>1 else None
        result[f"{label}_rate_peak_per_s"] = float(magnitude(np.diff(x,axis=0)/.001).max()) if len(t)>1 else None
    result["linear_p99_abs_N"] = float(np.quantile(abs(a["F_linear"]),.99))
    for key in ("compute_cam_age_ms","compute_enc_age_ms","applied_cam_age_ms","applied_enc_age_ms"):
        samples = a[key][a[key]>=0]
        result[key+"_p50"] = float(np.median(samples)) if len(samples) else None
        result[key+"_max"] = float(samples.max()) if len(samples) else None
    result["saturation_steps"] = int(np.any(a["ctrl_saturation"]|a["force_saturation"]|a["joint_saturation"],axis=1).sum())
    result["peak_time_s"] = float(t[np.argmax(abs(error))])
    result["scope"] = "full" if full else "valid_prefix_only"
    result["wall_s"] = row["wall_s"]
    return result


def run_case(out, sweep, ctx, case):
    folder = out/"runs"/case_id(case)
    if folder.exists():
        if (folder/"metrics.json").is_file():
            assert read(folder/"case.json")==case
            return read(folder/"metrics.json")
        print("PRESERVED_INCOMPLETE",folder.name,flush=True)
        return None
    assert len(list((out/"runs").iterdir()))+6 < 240
    folder.mkdir()
    write(folder/"case.json",case)
    noise = np.random.default_rng(case["seed"]).normal(0,CONFIG["camera_noise_m"],1320)
    if case["measurement_injection"]:
        t = np.arange(1320)/132
        inside = (t>=3) & (t<3.4)
        noise[inside] += 4e-6*np.sin(2*np.pi*45*(t[inside]-3))*np.sin(np.pi*(t[inside]-3)/.4)**2
    encoder_pressure = pressure(case["pressure"])
    sensor = Measurement(CONFIG,case,aconfig.Y_NOMINAL,encoder_pressure,noise)
    ext = Coordinator(ctx,case,sensor)
    channel = CommandChannel(case["execution_delay_ms"],case["command_ms"],ext)
    logger = Logger(folder/"trace.npz",ctx,case,sensor,channel)
    c,p = ctx["c"],ctx["parameters"]
    fd = ctx["fd"]
    force_values = ctx["fd_values"]
    if case["force_hz"]:
        force_values = force_values.copy()
        for hz,amplitude in zip(case["force_hz"],case["force_amplitude_N"],strict=True):
            force_values += amplitude*np.sin(2*np.pi*hz*c["t"])
        force_sha = sha256(force_values.tobytes()).hexdigest()
        fd = FrozenDisturbance(Path("mixed_force"),c["t"],force_values,{"frozen":{"sha256":force_sha}})
    save(folder/"inputs.json",{k:sha256(x.tobytes()).hexdigest()
                             for k,x in dict(camera_additive_noise=noise,encoder_pressure=encoder_pressure,
                                             Fd=force_values).items()})
    physical_steps = 0
    step = mujoco.mj_step
    original_terms = sfc.force_terms
    def counted(*args,**kwargs):
        nonlocal physical_steps
        step(*args,**kwargs)
        physical_steps += 1
    def split_velocity(e, v, K, B0, mu, n, mode, gamma):
        terms = original_terms(e,v,K,B0,mu,n,mode,gamma)
        nonlinear = original_terms(e,sensor.last_command["v_filtered_m_s"],K,B0,mu,n,mode,gamma)
        terms["F_shear"] = nonlinear["F_shear"]
        terms["F_total"] = terms["F_K"]+terms["F_B"]+terms["F_shear"]
        return terms
    start = time.perf_counter()
    try:
        mujoco.mj_step = counted
        if case["branch"] == "NL":
            sfc.force_terms = split_velocity
        row = sweep.run_case(folder.name,MODES[case["law"]],p["K"],p["B0"]*case["linear_multiplier"],
            p["n"],p["mu"],p["r"],p["v_ref"],fd,c["along"],c["q"],phase_code=c["phase"],
            c_input=ctx["scaled"],execution_mode=POSITION_EQUIV,control_update_steps=case["command_ms"],
            precise_logger=logger,common_position_correction=ctx["corr"]["common_delta_q"],
            e_extension=ext,measurement_adapter=sensor,final_command_hold=True,command_channel=channel,
            log=lambda line:print(line,flush=True))
        row["wall_s"] = time.perf_counter()-start
        save(folder/"row.json",row)
        save(folder/"guard.json",ext.rejection)
        if row["status"]!="OK" and ext.rejection is None:
            # Distinguish inherited post-step stop and implementation failure.
            assert row["status"]=="INVALID",row.get("traceback",row["note"])
        arrays = data_arrays(folder/"trace.npz")
        save(folder/"validation.json",validate(arrays,ctx,case,sensor,channel,row,physical_steps,force_values))
        save(folder/"frames.json",as_frames(sensor))
        save(folder/"command_events.json",channel.events)
        result = metrics(arrays,row,case,ext.rejection)
        save(folder/"metrics.json",result)
        print(folder.name,row["status"],physical_steps,f"RMS {result['full_rms_um']:.3f} um",flush=True)
        return result
    except Exception as ex:
        save(folder/"technical_failure.json",dict(error=str(ex),traceback=traceback.format_exc(),physics_steps=physical_steps))
        print("TECHNICAL_FAILURE",folder.name,str(ex),flush=True)
        return None
    finally:
        mujoco.mj_step = step
        sfc.force_terms = original_terms


def initialize(out):
    assert sha(ROOT/"campaign.json")==CFG_SHA and read(DATA/"pressure_identity.json")["config_sha256"]==CFG_SHA
    regression = ROOT/"regression_exact"
    assert (regression/"regression.json").exists()
    out.mkdir(exist_ok=True)
    (out/"runs").mkdir(exist_ok=True)
    receipt = read(ROOT/"source_receipt.json")
    for name, expected in read(SOURCE/"manifest.sha256.json").items():
        assert receipt["a_extracted"][name]==expected and sha(SOURCE/name)==expected
    ctx = context()
    assert ctx["input_id"] == read(SOURCE/"experiment.json")["expected_input_id"]
    env = SimEnv()
    env.reset(ctx["c"]["q"][0])
    model = clean(compiled_mapping(env.model))
    assert model["dt"] == .001 and model["simple_mapping_applicable"]
    assert model == read(regression/"baseline_2ms/trace.json")["model_mapping"]
    ctx["nominal_xyz"] = np.empty((len(ctx["c"]["t"]),3))
    for i,q in enumerate(ctx["c"]["q"]):
        env.data.qpos[:] = q
        mujoco.mj_kinematics(env.model,env.data)
        ctx["nominal_xyz"][i] = env.tcp_pos()
    ctx["nominal_xyz"][:,1] = aconfig.Y_NOMINAL
    env.reset(ctx["c"]["q"][0])
    binding = dict(input_id=ctx["input_id"],parameters=ctx["parameters"],model=model,
        initial_q_rad=env.data.qpos.tolist(),duration_samples=len(ctx["c"]["t"]),
        source_commit=read(SOURCE/"evidence/version_record.json")["source_commit"],config_sha256=CFG_SHA,
        sweep_sha256=sha(B/"sweep.py"),python=sys.version,mujoco=mujoco.__version__,numpy=np.__version__,scipy=scipy.__version__,
        implementation={name:sha(ROOT/name) for name in ("run_campaign.py","filter_sensing.py","execute.py")},
        mujoco_dll_sha256=sha(Path(mujoco.__file__).parent/"mujoco.dll"),
        pressure_identity=read(DATA/"pressure_identity.json"),sensor_unit_checks=checks(CONFIG),
        software_offline_guard=True, hardware_calls=0)
    if (out/"identity.json").exists():
        assert read(out/"identity.json")==binding
    else:
        write(out/"identity.json",binding)
    return integration_module(out),ctx
