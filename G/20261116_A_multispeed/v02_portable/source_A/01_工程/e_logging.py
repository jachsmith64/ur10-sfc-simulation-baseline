"""Extend inherited precise logger without changing its same-state actuation checks."""
import numpy as np
from position_logging import PreciseLogger
class ELogger(PreciseLogger):
    def __init__(self,*args,extension,**kwargs):
        super().__init__(*args,**kwargs);self.extension=extension
        for k in ['p','w','a','p_star','gamma','gamma_star','jerk_segment','jerk_interval_average','decision','coordination_s','degraded','nominal_v','nominal_y_dot']:
            self.arr[k]=np.zeros(self.n)
        for k in ['delta_q_p','delta_q_sfc','tau_total_requested','T_p','J_y_actual','dq_C_dot','nominal_qdot']:
            self.arr[k]=np.zeros((self.n,6))
        self.meta.update(jerk_evidence='jerk_segment = exact current Ruckig profile segment; jerk_interval_average = delta acceleration / dt, not peak proof',
            planner_pose_constraint='First-order world TCP position/orientation at current nominal q; T_p held within decision interval',nominal_y_dot=0.,no_future_environment_in_predictor=True)
    def sample(self,i,*args):
        super().sample(i,*args)
        for k,v in self.extension.records[i].items():
            if k in self.arr:self.arr[k][i]=v
    def finish(self,row):
        self.meta.update(method=self.extension.method,model_mapping=self.extension.mapping if hasattr(self.extension,'mapping') else None)
        super().finish(row)
