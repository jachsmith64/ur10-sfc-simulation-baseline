"""Full 1ms float64 logging on independent MjData; primary dynamics untouched."""
from pathlib import Path
import numpy as np
import mujoco
from position_execution import TORQUE_DIRECT,actuator_limits

class PreciseLogger:
    def __init__(self,path,n,background,execution,update_steps,input_sha,input_data=None,common_correction=None):
        self.path=Path(path);self.n=n
        self.meta=dict(background_mode=background,execution_mode=execution,update_steps=update_steps,input_sha256=input_sha,
             sample_position='pre-mj_step, after correct state refresh and current command/disturbance assignment',
             counterfactual='independent MjData copied from same actual state, base ctrl; mj_fwdActuation only, never integrated',
             shared_inputs='inputs/frozen_common.npz: t/phase/along/q_nominal/dq_C/remaining/remaining_dot; B dq/remaining/dot zero; base ctrl reconstructed exactly',
             units='t:s; q/q_nominal/dq/base_ctrl/delta_u/final_ctrl:rad; qdot:rad/s; y:m; e:um; v:m/s; F:N; tau/qfrc:Nm; saturation/updated:bool')
        self.arr={}
        common=np.load(Path(__file__).resolve().parent/'inputs/frozen_common.npz') if input_data is None else input_data
        try:
            self.arr['phase']=common['phase'].copy()
            self.arr['q_nominal']=common['q'].copy()
            self.arr['dq_C']=common['dq'].copy() if background=='C_ORDINARY' else np.zeros_like(common['dq'])
            self.arr['original_base_ctrl']=self.arr['q_nominal']+self.arr['dq_C']
        finally:
            if input_data is None:common.close()
        correction=common_correction or {}
        for key in ['common_delta_q','common_delta_q_dot','common_delta_q_ddot']:
            self.arr[key]=np.asarray(correction.get(key,np.zeros((n,6))),dtype=float).copy()
        for key in ['common_baseline_um','common_delta_y_m','common_delta_y_dot_m_s','common_delta_y_ddot_m_s2']:
            self.arr[key]=np.asarray(correction.get(key,np.zeros(n)),dtype=float).copy()
        self.arr['updated']=np.zeros(n,bool)
        for k in ['q','qdot','base_ctrl','delta_u','final_ctrl','tau_requested','qfrc_actuator','qfrc_applied','qfrc_disturbance','tau_realized_increment','base_qfrc_actuator','actuator_force','effective_ctrl']:
            self.arr[k]=np.empty((n,6),np.float64)
        for k in ['ctrl_saturation','force_saturation','joint_saturation','base_ctrl_saturation','base_force_saturation','base_joint_saturation']:
            self.arr[k]=np.zeros((n,6),bool)
        for k in ['t','model_time_s','physical_y','effective_y','nominal_y','physical_v','effective_v','physical_e_um','effective_e_um','remaining','remaining_dot','sample_e_m','sample_v_m_s','F_spring','F_linear','F_shear','F_requested','Fd']:
            self.arr[k]=np.empty(n,np.float64)
        self.actual=None;self.base=None;self.done=0

    def sample(self,i,env,measurement,y_d,out,fd,base_ctrl,manager,updated,remaining,remaining_dot):
        m=env.model
        if self.actual is None:self.actual=mujoco.MjData(m);self.base=mujoco.MjData(m)
        mujoco.mj_copyData(self.actual,m,env.data);mujoco.mj_fwdActuation(m,self.actual)
        mujoco.mj_copyData(self.base,m,env.data);self.base.ctrl[:]=base_ctrl;mujoco.mj_fwdActuation(m,self.base)
        limits=actuator_limits(m,self.actual);bl=actuator_limits(m,self.base)
        # Recorded actual MuJoCo force and affine diagnostics must agree; never used for compensation.
        assert np.allclose(limits['qfrc_actuator'],self.actual.qfrc_actuator,rtol=1e-10,atol=1e-8)
        assert np.allclose(bl['qfrc_actuator'],self.base.qfrc_actuator,rtol=1e-10,atol=1e-8)
        inc=self.actual.qfrc_actuator-self.base.qfrc_actuator
        if manager.mode==TORQUE_DIRECT:inc=inc+env._tau_sfc
        matrices=dict(q=env.data.qpos,qdot=env.data.qvel,base_ctrl=base_ctrl,delta_u=manager.delta,final_ctrl=env.data.ctrl,
                tau_requested=out['tau'],qfrc_actuator=self.actual.qfrc_actuator,qfrc_applied=env.data.qfrc_applied,
                qfrc_disturbance=env._qfrc_dist,tau_realized_increment=inc,base_qfrc_actuator=self.base.qfrc_actuator,
                actuator_force=self.actual.actuator_force,effective_ctrl=limits['ctrl_effective'])
        for k,x in matrices.items():self.arr[k][i]=x
        for k in ['ctrl_saturation','force_saturation','joint_saturation']:
            self.arr[k][i]=limits[k];self.arr['base_'+k][i]=bl[k]
        values=dict(t=i*m.opt.timestep,model_time_s=env.data.time,physical_y=measurement.y_physical,effective_y=measurement.y_effective,nominal_y=y_d,
                physical_v=measurement.v_physical,effective_v=measurement.v_effective,physical_e_um=(measurement.y_physical-y_d)*1e6,
                effective_e_um=(measurement.y_effective-y_d)*1e6,remaining=remaining,
                remaining_dot=remaining_dot,sample_e_m=out['e'],sample_v_m_s=out['v'],
                F_spring=out['F_K'],F_linear=out['F_B'],F_shear=out['F_shear'],F_requested=out['F_sfc'],Fd=fd)
        for k,x in values.items():self.arr[k][i]=x
        self.arr['updated'][i]=updated;self.done=i+1

    def finish(self,row):
        import json
        self.path.parent.mkdir(parents=True,exist_ok=True)
        valid=min(self.done,int(row['n_steps_done']))
        status=row['status']
        if status=='OK' and valid!=self.n:status='INCOMPLETE'
        self.meta.update(status=status,n_steps_done=row['n_steps_done'],logger_samples=valid,logger_samples_before_prefix_trim=self.done,n_steps_requested=self.n,note=row['note'])
        np.savez_compressed(self.path,**{k:v[:valid] for k,v in self.arr.items()})
        self.path.with_suffix('.json').write_text(json.dumps(self.meta,ensure_ascii=False,indent=2),encoding='utf-8')
