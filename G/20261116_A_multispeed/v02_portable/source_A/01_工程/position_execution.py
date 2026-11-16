"""Compiled-model-specific torque to target offset; no controller or new dynamics."""
import numpy as np
import mujoco

TORQUE_DIRECT='TORQUE_DIRECT'
POSITION_EQUIV='POSITION_EQUIV'

class UnsupportedMapping(RuntimeError):pass

def compiled_mapping(model):
    m=model
    keys=['actuator_trntype','actuator_trnid','actuator_gear','actuator_gaintype','actuator_biastype','actuator_dyntype','actuator_actadr','actuator_actnum',
          'actuator_gainprm','actuator_biasprm','actuator_ctrllimited','actuator_ctrlrange','actuator_forcelimited','actuator_forcerange',
          'actuator_actlimited','actuator_actrange','jnt_type','jnt_qposadr','jnt_dofadr','jnt_actfrclimited','jnt_actfrcrange','dof_damping','dof_armature']
    result={k:getattr(m,k).tolist() for k in keys}
    result.update(nq=m.nq,nv=m.nv,nu=m.nu,na=m.na,dt=m.opt.timestep,integrator=int(m.opt.integrator),
            integrator_name=mujoco.mjtIntegrator(m.opt.integrator).name,disableflags=int(m.opt.disableflags),enableflags=int(m.opt.enableflags),gravity=m.opt.gravity.tolist(),
            actuator_names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(m.nu)],
            joint_names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(m.njnt)],
            meaning='actuator_force是执行器标量力；经transmission moment映射为qfrc_actuator广义关节力矩；本模型unit gear一一映射才可数值等同')
    conditions={
        'six_independent_hinges':m.nq==m.nv==m.nu==m.njnt==6 and np.all(m.jnt_type==mujoco.mjtJoint.mjJNT_HINGE),
        'joint_transmission':np.all(m.actuator_trntype==mujoco.mjtTrn.mjTRN_JOINT),
        'identity_dof_order':np.array_equal(m.actuator_trnid[:,0],np.arange(6)) and np.array_equal(m.jnt_qposadr,np.arange(6)) and np.array_equal(m.jnt_dofadr,np.arange(6)),
        'unit_gear':np.array_equal(m.actuator_gear,np.tile([1.,0.,0.,0.,0.,0.],(6,1))),
        'fixed_positive_gain':np.all(m.actuator_gaintype==mujoco.mjtGain.mjGAIN_FIXED) and np.all(m.actuator_gainprm[:,0]>0) and np.all(m.actuator_gainprm[:,1:]==0),
        'affine_servo_bias':np.all(m.actuator_biastype==mujoco.mjtBias.mjBIAS_AFFINE) and np.all(m.actuator_biasprm[:,0]==0) and np.array_equal(m.actuator_biasprm[:,1],-m.actuator_gainprm[:,0]) and np.all(m.actuator_biasprm[:,2]<=0) and np.all(m.actuator_biasprm[:,3:]==0),
        'no_activation':m.na==0 and np.all(m.actuator_dyntype==mujoco.mjtDyn.mjDYN_NONE),
        'actuation_enabled':not bool(m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_ACTUATION),
    }
    result['conditions']=conditions;result['simple_mapping_applicable']=all(conditions.values())
    result['Kp_Nm_per_rad']=m.actuator_gainprm[:,0].tolist();result['Kd_Nms_per_rad']=(-m.actuator_biasprm[:,2]).tolist()
    return result

class ExecutionManager:
    def __init__(self,env,mode):
        if mode not in [TORQUE_DIRECT,POSITION_EQUIV]:raise ValueError(mode)
        report=compiled_mapping(env.model)
        if not report['simple_mapping_applicable']:raise UnsupportedMapping(str(report['conditions']))
        self.mode=mode;self.kp=env.model.actuator_gainprm[:,0].copy()
        self.requested_tau=np.zeros(env.nv);self.delta=np.zeros(env.model.nu)
        env.apply_joint_torque(np.zeros(env.nv)) # clear previous execution branch

    def apply_requested_torque(self,env,tau):
        self.requested_tau=np.asarray(tau,dtype=float).copy()
        if self.mode==TORQUE_DIRECT:
            self.delta[:]=0.;env.apply_joint_torque(self.requested_tau)
        else:
            self.delta[:]=self.requested_tau/self.kp
            env.apply_joint_torque(np.zeros(env.nv)) # no residual torque compensation

    def command(self,base):return np.asarray(base,dtype=float)+self.delta

    def assert_applied(self,env):
        if self.mode==TORQUE_DIRECT:expected=env._qfrc_dist+self.requested_tau
        else:
            assert np.all(env._tau_sfc==0),'SFC torque leaked into position path'
            expected=env._qfrc_dist
        assert np.array_equal(env.data.qfrc_applied,expected),'External force composition differs'

def actuator_limits(model,data):
    """Limit diagnostics from this validated affine mapping, verified against MuJoCo."""
    m=model;ctrl=data.ctrl.copy()
    clamp=not bool(m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL)
    ctrl_flag=np.zeros(m.nu,dtype=bool)
    if clamp:
        ctrl_flag=m.actuator_ctrllimited & ((ctrl<m.actuator_ctrlrange[:,0]) | (ctrl>m.actuator_ctrlrange[:,1]))
        effective=np.where(m.actuator_ctrllimited,np.clip(ctrl,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]),ctrl)
    else:effective=ctrl
    raw=m.actuator_gainprm[:,0]*effective+m.actuator_biasprm[:,1]*data.qpos+m.actuator_biasprm[:,2]*data.qvel
    force_flag=m.actuator_forcelimited & ((raw<m.actuator_forcerange[:,0]) | (raw>m.actuator_forcerange[:,1]))
    actuator=np.where(m.actuator_forcelimited,np.clip(raw,m.actuator_forcerange[:,0],m.actuator_forcerange[:,1]),raw)
    joint_flag=m.jnt_actfrclimited & ((actuator<m.jnt_actfrcrange[:,0]) | (actuator>m.jnt_actfrcrange[:,1]))
    generalized=np.where(m.jnt_actfrclimited,np.clip(actuator,m.jnt_actfrcrange[:,0],m.jnt_actfrcrange[:,1]),actuator)
    return dict(ctrl_effective=effective,raw_actuator_force=raw,actuator_force=actuator,qfrc_actuator=generalized,
                ctrl_saturation=ctrl_flag,force_saturation=force_flag,joint_saturation=joint_flag)
