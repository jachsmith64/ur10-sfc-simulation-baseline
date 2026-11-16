"""Nominal IK velocity, held on the existing complete six-axis command clock."""
import numpy as np


def nominal_derivative(t, knots_t, knots_q):
    if t[0]<knots_t[0] or t[-1]>knots_t[-1] or np.any(np.diff(knots_t)<=0):
        raise ValueError("The existing host-axis nominal spline does not cover the run")
    slopes=np.diff(knots_q,axis=0)/np.diff(knots_t)[:,None]
    index=np.minimum(np.searchsorted(knots_t,t,side="right")-1,len(slopes)-1)
    return slopes[index], slopes, index


class VelocityReference:
    def __init__(self,core,logger_class,qdot_nom,tick_steps):
        self.core=core
        self.logger_class=logger_class
        self.qdot_nom=qdot_nom
        self.samples=np.arange(len(qdot_nom))//tick_steps*tick_steps

    def __call__(self,path,ctx,case,sensor,coordinator):
        ff=self
        model=ctx["compiled"]
        kp=np.array(model["Kp_Nm_per_rad"])
        kd=np.array(model["Kd_Nms_per_rad"])
        offset_at_tick=self.qdot_nom*(kd/kp)
        held_offset=offset_at_tick[self.samples]
        held_speed=self.qdot_nom[self.samples]
        original_command=coordinator.command

        def command(i,env,q_cmd,out,actuation):
            # Preserve the inherited pre-hold guard and pulse route.
            return original_command(i,env,q_cmd+held_offset[i],out,actuation)

        coordinator.command=command

        class VelocityCommandChannel(ff.core.CommandChannel):
            def tick(self,step,env,q_cmd,base,out,updated):
                # The base counterfactual includes feedforward, so it is not logged as micro-compensation.
                return super().tick(step,env,q_cmd,base+held_offset[step],out,updated)

        channel=VelocityCommandChannel(case["execution_delay_ms"],case["command_ms"],coordinator)
        logger=self.logger_class(path/"trace.npz",ctx,case,sensor,channel)
        logger.arr["qdot_nom_command_rad_s"]=held_speed.copy()
        logger.arr["q_velocity_ff_rad"]=held_offset.copy()
        return channel,logger
