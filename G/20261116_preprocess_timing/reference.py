"""Independent physical and observer references with protected events."""
import numpy as np
from filter_sensing import Measurement
from windows import PLATFORMS, EVENT_TO_TRANSITION


class Schedule:
    def __init__(self, window_record, levels_um):
        self.levels = [float(levels_um[name]) for name in PLATFORMS]
        self.ramps = [window_record["windows_s"][name] for name in EVENT_TO_TRANSITION.values()]

    def at(self, t):
        t = np.asarray(t)
        y = np.full_like(t, self.levels[0], dtype=float)
        rate = np.zeros_like(y)
        accel = np.zeros_like(y)
        for (lo, hi), before, after in zip(self.ramps,self.levels[:-1],self.levels[1:],strict=True):
            u = np.clip((t-lo)/(hi-lo),0.,1.)
            delta = after-before
            y += delta*u**3*(10+u*(-15+6*u))
            rate += delta*30*u*u*(1-u)**2/(hi-lo)
            accel += delta*60*u*(1-u)*(1-2*u)/(hi-lo)**2
        return y,rate,accel

    def value(self, t):
        # Scalar sample-time path. It uses fixed schedule parameters and no live lookahead.
        for (lo,hi),before,after in zip(self.ramps,self.levels[:-1],self.levels[1:],strict=True):
            if t <= lo:
                return before
            if t < hi:
                u = (t-lo)/(hi-lo)
                return before + (after-before)*u**3*(10+u*(-15+6*u))
        return self.levels[-1]


class ReferencedMeasurement(Measurement):
    """Reuse V02 transport and difference/LPF channels with a per-sample reference."""
    def __init__(self, config, case, nominal, pressure, noise, reference):
        super().__init__(config,case,nominal,pressure,noise)
        if np.any(noise):
            raise ValueError("Independent camera noise is disabled")
        self.reference = reference
        self.camera_reference_m, self.encoder_reference_m = [], []

    def update(self, env, remaining, remaining_dot):
        if remaining != 0 or remaining_dot != 0:
            raise ValueError("Legacy observation term reached the new sensor")
        self.tick(round(env.data.time*1000),env.tcp_y()-self.nominal,remaining)

    def tick(self, step, physical_e, remaining):
        assert step==self.step+1 and remaining==0
        self.step=step
        self.q_event=step%self.period==0
        frame=len(self.c.frames)
        sample_ms=1000*frame/self.hz
        self.cam_event=step==(1000*frame+self.hz-1)//self.hz
        if self.q_event:
            ref=self.reference.value(step/1000)*1e-6
            self.encoder_reference_m.append(ref)
            self.q.capture(step,float(step),physical_e,float(self.pressure[step//self.period])-ref)
        if self.cam_event:
            w=step-sample_ms
            old=self.previous if w else physical_e
            exposure=physical_e+w*(old-physical_e)
            ref=self.reference.value(sample_ms/1000)*1e-6
            self.camera_reference_m.append(ref)
            self.c.capture(step,sample_ms,exposure,-ref)
        self.previous=physical_e
        self.q.release(step)
        self.c.release(step)
        if self.c.latest:
            self.y_effective=self.nominal+self.c.latest.e
            self.v_effective=self.c.latest.v
