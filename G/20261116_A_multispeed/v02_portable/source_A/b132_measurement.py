"""Explicit synthetic position-sampling baseline; no images and no hardware."""
import numpy as np
class SampledMeasurement:
 name='synthetic132Hz_timestamped_ZOH'
 def __init__(self,velocity='difference',noise_std=0.,delay_steps=0,seed=13220261015,nominal=None):
  self.velocity_mode=velocity;self.noise_std=noise_std;self.delay_steps=delay_steps;self.rng=np.random.default_rng(seed);self.nominal=nominal or (lambda t:-.174)
  self.k=0;self.frame=-1;self.queue=[];self.frames=[];self.previous_sample=None
  self.y_effective=self.nominal(0);self.v_effective=0.;self.used_frame=-1;self.used_sample_step=-1;self.used_available_step=-1;self.velocity_valid=False;self.sample_event=False
 def update(self,env,remaining,remaining_dot):
  step=int(round(env.data.time/.001));self.sample_event=False
  # Integer ceil(1000*k/132), no future interpolation and no8ms masqueradingas132Hz.
  if step==(1000*self.k+131)//132:
   sample_t=step*.001;nominal_t=self.k/132;noise=float(self.rng.normal(0,self.noise_std))
   y=float(env.tcp_y())+float(remaining)+noise;e=y-self.nominal(sample_t)
   velocity=0.;vvalid=False
   if self.velocity_mode=='sampled_ideal':velocity=float(env.tcp_y_velocity())+float(remaining_dot);vvalid=True
   elif self.previous_sample is not None:
    pt,pe=self.previous_sample;velocity=(e-pe)/(sample_t-pt);vvalid=True
   self.previous_sample=(sample_t,e)
   frame=dict(frame_id=self.k,nominal_sample_s=nominal_t,sample_step=step,sample_s=sample_t,quantization_s=sample_t-nominal_t,available_step=step+self.delay_steps,available_s=(step+self.delay_steps)*.001,noise_m=noise,y_m=y,nominal_y_m=self.nominal(sample_t),error_m=e,velocity_m_s=velocity,velocity_valid=vvalid,first_use_step=-1)
   self.frames.append(frame);self.queue.append(frame);self.k+=1;self.sample_event=True
  while self.queue and self.queue[0]['available_step']<=step:
   f=self.queue.pop(0);self.used_frame=f['frame_id'];self.used_sample_step=f['sample_step'];self.used_available_step=f['available_step'];self.y_effective=f['error_m']+self.nominal(f['sample_s']);self.v_effective=f['velocity_m_s'];self.velocity_valid=f['velocity_valid'];f['first_use_step']=step
 def measure(self,env,y_d,y_d_dot=0):
  # Use sampled reference, never current moving nominal. ProductionnominalYisconstant.
  if self.used_frame<0:return 0.,0.
  f=self.frames[self.used_frame];return f['error_m'],self.v_effective
 def record(self,step):
  return dict(measurement_frame_id=self.used_frame,sample_step_used=self.used_sample_step,available_step_used=self.used_available_step,measurement_age_s=(step-self.used_sample_step)*.001 if self.used_frame>=0 else np.nan,measurement_valid=self.used_frame>=0,velocity_estimate_valid=self.velocity_valid,sample_event=self.sample_event,noise_used_m=self.frames[self.used_frame]['noise_m'] if self.used_frame>=0 else 0.)
