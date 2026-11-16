"""Single-DOF local Ruckig; analytic library position extrema, V/A/J proof."""
import e_bootstrap
import numpy as np
from ruckig import Ruckig,InputParameter,Trajectory
D_MAX=100e-6;V_MAX=500e-6;A_MAX=.01;J_MAX=1.;DT=.001
class PlanRejected(ValueError):pass
def profile_bounds(traj):
    """Velocity extrema at a=0; acceleration extrema at segment ends; exact jerk.
    Ruckig validation prevents a braking pre-profile for our legal current states.
    If a future library returns one, reject rather than omit its extrema proof.
    """
    velocities=[];accelerations=[];jerks=[]
    for section in traj.profiles:
        for pr in section:
            if getattr(pr.brake,'duration',0)>1e-14:
                raise PlanRejected('braking pre-profile needs explicit extrema support')
            for dur,p,v,a,j in zip(pr.t,pr.p,pr.v,pr.a,pr.j):
                velocities.extend([v,v+a*dur+.5*j*dur*dur]);accelerations.extend([a,a+j*dur]);jerks.append(j)
                if abs(j)>1e-15 and 0 < -a/j < dur:
                    s=-a/j;velocities.append(v+a*s+.5*j*s*s)
    return dict(velocity_peak=max(map(abs,velocities),default=0),acceleration_peak=max(map(abs,accelerations),default=0),jerk_peak=max(map(abs,jerks),default=0))
class Plan:
    def __init__(self,state,target):
        state=np.asarray(state,float)
        if state.shape!=(3,) or not np.isfinite(state).all() or abs(target)>D_MAX+1e-15:raise PlanRejected('state/target invalid')
        inp=InputParameter(1)
        inp.current_position=[float(state[0])];inp.current_velocity=[float(state[1])];inp.current_acceleration=[float(state[2])]
        inp.target_position=[float(target)];inp.target_velocity=[0.];inp.target_acceleration=[0.]
        inp.max_velocity=[V_MAX];inp.max_acceleration=[A_MAX];inp.max_jerk=[J_MAX]
        solver=Ruckig(1,DT);self.trajectory=Trajectory(1)
        try:
            solver.validate_input(inp,True,True)
            result=solver.calculate(inp,self.trajectory)
            if int(result)<0:raise PlanRejected(str(result))
            bounds=self.trajectory.position_extrema[0]
            if bounds.min < -D_MAX-1e-14 or bounds.max > D_MAX+1e-14:raise PlanRejected('analytic position extremum violates Dmax')
            peaks=profile_bounds(self.trajectory)
            if peaks['velocity_peak']>V_MAX+1e-12 or peaks['acceleration_peak']>A_MAX+1e-12 or peaks['jerk_peak']>J_MAX+1e-12:raise PlanRejected('analytic V/A/J violation')
        except Exception as ex:raise PlanRejected(str(ex)) from ex
        self.evidence=dict(position_min=bounds.min,position_max=bounds.max,duration_s=self.trajectory.duration,**peaks)
        self.elapsed=0.;self.target=float(target);self.initial=state.copy()
    def at(self,t):
        return np.asarray(self.trajectory.at_time(min(max(float(t),0.),self.trajectory.duration)),float)[:,0]
    def preview(self,n=160,h=DT):return np.array([self.at(i*h) for i in range(n+1)])
    def advance(self):self.elapsed+=DT;return self.at(self.elapsed)
    def jerk_at(self,t):
        if t>=self.trajectory.duration:return 0.
        left=float(t)
        for section in self.trajectory.profiles:
            for pr in section:
                for dur,j in zip(pr.t,pr.j):
                    if left<dur:return float(j)
                    left-=dur
        return 0.
