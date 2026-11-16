from b132_common import *
from gamma_timing import TimedCoordinator
from e_logging import ELogger
class JointFeedbackView:
 """Existing fastq/qd model assumption; no ideal endpointvelocity accessor."""
 def __init__(self,env):
  self.env=env;self.fk=mujoco.MjData(env.model);self.fk.qpos[:]=env.data.qpos;self.fk.qvel[:]=env.data.qvel;mujoco.mj_forward(env.model,self.fk)
 def __getattr__(self,k):return getattr(self.env,k)
 def tcp_y(self):return float(self.fk.site_xpos[self.env.site_id,1])
 def tcp_y_velocity(self):raise RuntimeError('Ideal endpoint velocity bypass prohibited in coordinator')
class Coordinator132(TimedCoordinator):
 def current(self,i,env,measurement,base):return super().current(i,JointFeedbackView(env),measurement,base)
class Logger132(ELogger):
 def __init__(self,*a,adapter,hold,**kw):
  super().__init__(*a,**kw);self.adapter=adapter;self.hold=hold;self.sat_run=0
  for k in ['measurement_frame_id','sample_step_used','available_step_used','measurement_age_s','measurement_valid','velocity_estimate_valid','sample_event','noise_used_m','gamma_sfc_used']:
   self.arr[k]=np.empty(self.n)
  for k in ['internal_final_ctrl','base_ctrl_internal','base_ctrl_submitted','delta_q_p_actual','delta_q_sfc_actual']:self.arr[k]=np.empty((self.n,6))
  self.held_base=None
 def sample(self,i,env,measurement,y_d,out,fd,base,manager,updated,rem,remd):
  if updated or self.held_base is None:
   self.held_base=base.copy();self.held_p=self.extension.Tp*self.extension.state[0];self.held_sfc=manager.delta.copy();self.held_gamma=self.extension.gamma
  # Fair same-state baseline uses nominal/background/common command submittedatthesame executionperiod.
  actual_base=self.held_base if self.hold else base
  super().sample(i,env,measurement,y_d,out,fd,actual_base,manager,updated,rem,remd)
  self.arr['internal_final_ctrl'][i]=base+manager.delta+self.extension.Tp*self.extension.state[0]
  self.arr['base_ctrl_internal'][i]=base;self.arr['base_ctrl_submitted'][i]=actual_base;self.arr['delta_q_p_actual'][i]=self.held_p if self.hold else self.extension.Tp*self.extension.state[0];self.arr['delta_q_sfc_actual'][i]=self.held_sfc if self.hold else manager.delta
  self.arr['gamma_sfc_used'][i]=self.held_gamma
  if self.adapter:
   for k,v in self.adapter.record(i).items():self.arr[k][i]=v
  else:
   for k in ['measurement_frame_id','sample_step_used','available_step_used']:self.arr[k][i]=i
   self.arr['measurement_age_s'][i]=0.;self.arr['measurement_valid'][i]=1.;self.arr['velocity_estimate_valid'][i]=1.;self.arr['sample_event'][i]=1.;self.arr['noise_used_m'][i]=0.
  anysat=any(np.any(self.arr[k][i]) for k in ['ctrl_saturation','force_saturation','joint_saturation'])
  self.sat_run=self.sat_run+1 if anysat else 0
  if self.sat_run>=20:raise RuntimeError('Persistent compiled output saturation20ms;stopvalidprefix')
