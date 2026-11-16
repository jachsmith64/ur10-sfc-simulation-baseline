"""Aligned CB3 site, independent gravity support, and the stationary gate."""

import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from probe_model import MODEL, SimEnv, flange, pose_error, recorded_pose
from probe_model import ExecutionManager, POSITION_EQUIV
from position_execution import actuator_limits

ALIGNED_MODEL = MODEL.with_name("ur10_cb3_b05_aligned.xml")


def aligned_model(cfg):
    env = SimEnv(ALIGNED_MODEL)
    tool = np.asarray(cfg["local_flange_to_site"])
    quat_xyzw = Rotation.from_matrix(tool[:3, :3]).as_quat()
    if (not np.allclose(env.model.site_pos[env.site_id], tool[:3, 3], atol=1e-14, rtol=0) or
            not np.allclose(env.model.site_quat[env.site_id], np.roll(quat_xyzw, 1), atol=1e-14, rtol=0)):
        raise ValueError("Compiled site differs from the selected local alignment")
    if abs(env.model.opt.timestep*1000-cfg["physics_step_ms"]) > 1e-12:
        raise ValueError("Physics clock differs from the compiled model")
    return env


class GravitySupport:
    def __init__(self, env):
        # ExecutionManager checks fixed positive Kp, unit gear and one-to-one transmission.
        self.kp = ExecutionManager(env, POSITION_EQUIV).kp
        self.model = env.model
        self.state = mujoco.MjData(self.model)

    def offset(self, q_nom):
        self.state.qpos[:] = q_nom
        self.state.qvel[:] = 0.
        mujoco.mj_forward(self.model, self.state)
        return self.state.qfrc_bias.copy()/self.kp


def checked_actuation(env):
    limits = actuator_limits(env.model, env.data)
    if any(limits[name].any() for name in ("ctrl_saturation", "force_saturation", "joint_saturation")):
        raise RuntimeError("Compiled position or torque limit exceeded")
    return limits["qfrc_actuator"]


def stationary(cfg, q):
    traces = {}
    for name, direction, command_channel in (("hold", 0, True), ("positive", 1, True),
                                              ("negative", -1, True), ("external", 1, False)):
        env = aligned_model(cfg)
        env.reset(q)
        support = GravitySupport(env)
        off = support.offset(q)
        manager = ExecutionManager(env, POSITION_EQUIV)
        origin = env.tcp_pos()
        p, qpos, torque = [], [], []
        for i in range(501):
            env.forward()
            pulse = direction if 200 <= i < 250 else 0.
            requested = pulse*env.jacp()[1]
            env.apply_disturbance_force_y(0. if command_channel else pulse)
            manager.apply_requested_torque(env, requested if command_channel else np.zeros(6))
            manager.assert_applied(env)
            env.set_ctrl(manager.command(q+off))
            p.append(env.tcp_pos()-origin)
            qpos.append(env.qpos())
            torque.append(checked_actuation(env))
            if i < 500:
                env.step()
        traces[name] = np.array(p)
        traces[f"{name}_q"] = np.array(qpos)
        traces[f"{name}_servo_torque"] = np.array(torque)
    hold = traces["hold"]
    plus, minus = (traces[key][:, 1]-hold[:, 1] for key in ("positive", "negative"))
    drift = float(np.max(np.linalg.norm(hold, axis=1))*1e6)
    equivalence = float(np.max(abs(traces["positive"]-traces["external"])))
    if drift > cfg["stationary_gate_um"] or equivalence > 1e-12:
        raise RuntimeError(f"Stationary hold gate failed: {drift=}, {equivalence=}")
    if not (plus[201] > 0 > minus[201] and np.array_equal(plus[:201], np.zeros(201))
            and np.array_equal(minus[:201], np.zeros(201)) and abs(plus[-1]) < plus.max()):
        raise RuntimeError("Pulse did not show the expected causal and signed response")
    traces["time_s"] = np.arange(501)*.001
    return traces, dict(drift_um=drift, positive_peak_um=float(plus.max()*1e6),
                        negative_peak_um=float(minus.min()*1e6), applied_offset_rad=off.tolist(),
                        gravity_Nm=(off*support.kp).tolist(), first_response_ms=201,
                        positive_at_500ms_um=float(plus[-1]*1e6), position_external_max_diff_m=equivalence)


def reconstruct(cfg, obs, t, *, gravity_support=True):
    env = aligned_model(cfg)
    host = obs["host_time_s"]
    i0 = np.searchsorted(host, t[0], side="right") - 1
    i1 = np.searchsorted(host, t[-1], side="left")
    if i0 < 0 or i1 >= len(host):
        raise ValueError("Requested nominal interval exceeds cabinet record")
    idx = slice(i0, i1+1)
    source_t = host[idx]
    if (np.max(np.diff(source_t))*1000 > cfg["max_source_host_bracket_ms"] or
            np.max(np.diff(obs["cabinet_time_s"][idx]))*1000 > cfg["max_source_cabinet_gap_ms"]):
        raise ValueError("A cabinet gap crosses the requested interval")
    target = obs["target_tcp_pose"][idx]
    q = obs["actual_q_rad"][i0]
    lower, upper = env.model.actuator_ctrlrange.T
    knots = []
    for pose in target:
        want = recorded_pose(pose)
        solve = least_squares(lambda x: pose_error(flange(env, x), want), q,
                              bounds=(lower, upper), xtol=1e-12, ftol=1e-12, gtol=1e-12)
        if not solve.success or np.linalg.norm(solve.fun) > 1e-8:
            raise RuntimeError(f"Six-dimensional IK failed at target row {i0+len(knots)}")
        q = solve.x
        knots.append(q)
    knots = np.array(knots)
    nominal_q = np.stack([np.interp(t, source_t, col) for col in knots.T], axis=1)
    if np.max(abs(target[:, 3:]-target[0, 3:])) > 1e-8:
        raise ValueError("This B05 entry requires the recorded constant target orientation")
    nominal_tcp = np.stack([np.interp(t, source_t, col) for col in target.T], axis=1)
    offset = np.zeros_like(nominal_q)
    if gravity_support:
        support = GravitySupport(env)
        offset = np.stack([support.offset(q) for q in nominal_q])
    back = np.array([pose_error(flange(env, q), recorded_pose(p)) for q, p in zip(nominal_q, nominal_tcp)])
    max_pos = float(np.linalg.norm(back[:, :3], axis=1).max())
    max_angle = float(np.linalg.norm(back[:, 3:], axis=1).max())
    if max_pos > 1e-7 or max_angle > 1e-6:
        raise ValueError(f"Interpolated target back-substitution failed: {max_pos=}, {max_angle=}")
    result = dict(time_s=t, nominal_q_rad=nominal_q, nominal_tcp_m_rotvec_rad=nominal_tcp,
                  gravity_offset_rad=offset, target_knot_time_s=source_t, target_knot_q_rad=knots,
                  target_knot_tcp=target, target_knot_index=np.arange(i0, i1+1),
                  back_substitution_error=back)
    summary = dict(knot_count=len(knots), host_bracket_ms=float(np.diff(source_t).max()*1000),
                   max_position_back_error_m=max_pos, max_rotation_back_error_rad=max_angle,
                   max_adjacent_joint_step_rad=float(np.max(abs(np.diff(knots, axis=0)))),
                   max_gravity_offset_rad=np.max(abs(offset), axis=0).tolist())
    return result, summary
