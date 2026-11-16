"""Native-sample Butterworth mixing and delayed immutable observations."""
from collections import deque
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import signal


@lru_cache
def coefficients(fs, cutoff, order=1):
    b, a = signal.butter(order, cutoff, fs=fs)
    return b, a, signal.lfilter_zi(b, a)


class Mix:
    def __init__(self, fs, cutoff, rho, order=1):
        self.coeff = coefficients(fs, cutoff, order) if cutoff and rho != 1 else None
        self.rho = rho
        self.state = None

    def push(self, value):
        if self.coeff is None:
            return value
        b, a, zi = self.coeff
        if self.state is None:
            self.state = zi * value
        low, self.state = signal.lfilter(b, a, [value], zi=self.state)
        return float(low[0] + self.rho * (value - low[0]))


@dataclass(frozen=True, slots=True)
class Frame:
    id: int
    sample: float
    available: int
    e: float
    v: float
    ef: float
    vf: float
    valid: bool
    physical_e: float
    additive: float
    model_step: int
    available_exact_ms: float


class Channel:
    def __init__(self, fs, delay, cutoff, rho, filter_e, filter_v, order=1):
        self.fs, self.delay = fs, delay
        self.position = Mix(fs, cutoff if filter_e else None, rho)
        self.velocity = Mix(fs, cutoff if filter_v else None, rho, order)
        self.frames, self.pending, self.latest = [], deque(), None

    def capture(self, step, sample, physical_e, additive):
        value = physical_e + additive
        old = self.frames[-1] if self.frames else None
        v = (value - old.e) / ((sample - old.sample) / 1000) if old else 0.
        frame = Frame(len(self.frames), sample, step + self.delay, value, v,
                      self.position.push(value), self.velocity.push(v) if old else 0.,
                      old is not None, physical_e, additive, step, sample + self.delay)
        self.frames.append(frame)
        self.pending.append(frame)

    def release(self, step):
        while self.pending and self.pending[0].available <= step:
            self.latest = self.pending.popleft()


class Measurement:
    name = "native_filter"

    def __init__(self, config, case, nominal, pressure, noise):
        self.case, self.nominal = case, nominal
        self.period, self.hz = config["encoder_ms"], config["camera_hz"]
        fc, rho = case["fc"], case["rho"]
        self.q = Channel(1000 / self.period, config["encoder_delay_ms"], fc, rho,
                         False, case["estimator"] == "Q", case.get("filter_order", 1))
        self.c = Channel(self.hz, case["camera_delay_ms"], fc, rho,
                         case["branch"] == "EV", case["estimator"] == "C")
        self.pressure, self.noise = pressure, noise
        self.step, self.previous = -1, None
        self.cam_event = self.q_event = False
        self.y_effective, self.v_effective = nominal, 0.
        self.last_command = {}

    def tick(self, step, physical_e, remaining):
        assert step == self.step + 1
        self.step = step
        self.q_event = step % self.period == 0
        frame = len(self.c.frames)
        sample_ms = 1000 * frame / self.hz
        self.cam_event = step == (1000 * frame + self.hz - 1) // self.hz
        if self.q_event:
            self.q.capture(step, float(step), physical_e, float(self.pressure[step // self.period]))
        if self.cam_event:
            # Reconstruct the past fractional exposure from its two physics knots.
            # Both knots are known by the frame's first possible publication.
            w = step - sample_ms
            old_e, old_r = self.previous if w else (physical_e, remaining)
            camera_e = physical_e + w * (old_e - physical_e)
            camera_r = remaining + w * (old_r - remaining)
            self.c.capture(step, sample_ms, camera_e, camera_r + float(self.noise[frame]))
        self.previous = (physical_e, remaining)
        self.q.release(step)
        self.c.release(step)
        if self.c.latest:
            self.y_effective = self.nominal + self.c.latest.e
            self.v_effective = self.c.latest.v

    def update(self, env, remaining, remaining_dot):
        self.tick(round(env.data.time * 1000), env.tcp_y() - self.nominal, remaining)

    def measure(self, env, y_d, y_d_dot=0):
        assert y_d == self.nominal and y_d_dot == 0
        q, c = self.q.latest, self.c.latest
        ready = bool(q and c and q.valid and c.valid)
        source = {"N": None, "Q": q, "C": c}[self.case["estimator"]]
        raw = source.v if ready and source else 0.
        filtered = source.vf if ready and source else 0.
        v = raw if self.case["branch"] == "NL" else filtered
        e = c.ef if c else 0.
        self.last_command = dict(compute=self.step, e_m=e, v_m_s=v, v_raw_m_s=raw,
                                 v_filtered_m_s=filtered, ready=ready)
        for label, frame in (("c", c), ("q", q)):
            self.last_command.update({
                f"{label}_id": frame.id if frame else -1,
                f"{label}_sample": frame.sample if frame else -1,
                f"{label}_available": frame.available if frame else -1,
                f"{label}_age_ms": self.step - frame.sample if frame else -1,
            })
        return e, v

    def values(self):
        result = {}
        for label, channel, event in (("cam", self.c, self.cam_event), ("enc", self.q, self.q_event)):
            frame = channel.latest
            result.update({f"{label}_{key}": getattr(frame, key) if frame else default
                           for key, default in (("e", 0.), ("v", 0.), ("ef", 0.), ("vf", 0.),
                                                ("id", -1), ("sample", -1), ("available", -1))})
            result[f"{label}_event"] = event
        return result


def checks(config):
    for fs in (125, 132):
        for fc in (10, 20, 35):
            x = np.r_[np.ones(40) * 3, np.ones(40) * 7, np.sin(np.arange(2000) * 2*np.pi * 15/fs)]
            b, a, zi = coefficients(fs, fc)
            reference, _ = signal.lfilter(b, a, x, zi=zi*x[0])
            for rho in (0., .25, .5, 1.):
                filt = Mix(fs, fc, rho)
                actual = np.array([filt.push(v) for v in x])
                if rho == 1:
                    assert np.array_equal(actual, x)
                assert np.allclose(actual, reference + rho*(x-reference), atol=1e-14, rtol=0)
                assert actual[0] == 3 or abs(actual[0]-3) < 1e-14
                _, h = signal.freqz(b, a, worN=[15], fs=fs)
                time = np.arange(500, len(x)-80) / fs
                basis = np.array([np.sin(2*np.pi*15*time), np.cos(2*np.pi*15*time)]).T
                gain = np.linalg.lstsq(basis, actual[580:], rcond=None)[0]
                assert abs(complex(*gain) - (rho + (1-rho)*h[0])) < 1e-11
    for branch in ("V", "EV", "NL"):
        for source in ("Q", "C"):
            case = dict(estimator=source, branch=branch, fc=20, rho=.25, camera_delay_ms=4)
            left, right = [Measurement(config, case, 0., np.zeros(125), np.zeros(132)) for _ in range(2)]
            for step in range(250):
                left.tick(step, step*1e-6, 0.)
                right.tick(step, step*1e-6 + (1. if step >= 180 else 0.), 0.)
                if step % 8 != 0:
                    continue
                result = left.measure(None, 0.)
                if step < 180:
                    assert result == right.measure(None, 0.)
                obs = left.last_command
                assert all(obs[f"{key}_available"] <= step for key in ("c", "q"))
                if obs["ready"]:
                    assert abs(result[1] - .001) < 1e-14
                else:
                    assert result[1] == 0.
            assert len(left.q.frames) == 32 and len(left.c.frames) == 33
            assert left.c.latest == max((f for f in left.c.frames if f.available <= 249), key=lambda f:f.id)
    return dict(butterworth_lfilter_steady_state=True, identity_rho1=True, rho0_reference=True,
                step_ramp_sine_and_complex_response=True, native_frame_counts=True,
                snapshot_publication=True, no_future_prefix=True)
