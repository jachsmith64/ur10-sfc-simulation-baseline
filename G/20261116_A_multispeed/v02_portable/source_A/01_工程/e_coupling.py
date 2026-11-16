"""E extension to the inherited F loop; actual plan state advances once per step."""
import time
import numpy as np,mujoco
from position_execution import compiled_mapping
from e_plan import Plan,PlanRejected,D_MAX,DT
from e_predict import predict,select,N,DP
class Coordinator:
    def __init__(self,method,nominal,params,callback=None):
        self.method=method;self.nominal=np.asarray(nominal).copy();self.params=params.copy();self.callback=callback
        self.state=np.zeros(3);self.target=0.;self.gamma=1.;self.gamma_target=1.;self.plan=Plan(self.state,0.)
        self.decisions=[];self.records={};self.reason='';self.Tp=np.zeros(6)
        self.planning=method in ['planner','additive','sp_only','ps_only','coupled']
    def start(self,env,measurement,ctrl):
        m=env.model;info=compiled_mapping(m)
        if not info['simple_mapping_applicable']:raise RuntimeError('Unsupported compiled mapping')
        info['conditions']={k:bool(v) for k,v in info['conditions'].items()}
        info['simple_mapping_applicable']=bool(info['simple_mapping_applicable'])
        self.mapping=info;self.kp=m.actuator_gainprm[:,0].copy();self.kd=-m.actuator_biasprm[:,2].copy()
        self.ctrl_lo=np.where(m.actuator_ctrllimited,m.actuator_ctrlrange[:,0],-np.inf)
        self.ctrl_hi=np.where(m.actuator_ctrllimited,m.actuator_ctrlrange[:,1],np.inf)
        self.tau_lo=np.maximum(np.where(m.actuator_forcelimited,m.actuator_forcerange[:,0],-np.inf),np.where(m.jnt_actfrclimited,m.jnt_actfrcrange[:,0],-np.inf))
        self.tau_hi=np.minimum(np.where(m.actuator_forcelimited,m.actuator_forcerange[:,1],np.inf),np.where(m.jnt_actfrclimited,m.jnt_actfrcrange[:,1],np.inf))
        self.tau_scale=np.minimum(abs(self.tau_lo),abs(self.tau_hi))
        if not np.isfinite(self.tau_scale).all() or np.any(self.tau_scale<=0):raise RuntimeError('Finite compiled force scale required')
        self.temp=mujoco.MjData(m)
    def current(self,i,env,measurement,base):
        # independent nominal data: no reset/forward of the physical simulation
        m=env.model;tmp=self.temp;tmp.qpos[:]=self.nominal[i];tmp.qvel[:]=0;mujoco.mj_forward(m,tmp)
        Jp=np.zeros((3,6));Jr=np.zeros((3,6));mujoco.mj_jacSite(m,tmp,Jp,Jr,env.site_id)
        pose=np.vstack([Jp,Jr]);cond=float(np.linalg.cond(pose));rank=int(np.linalg.matrix_rank(pose))
        if rank!=6 or not np.isfinite(cond) or cond>1e8:raise RuntimeError(f'Nominal pose singular rank={rank} cond={cond}')
        self.Tp=np.linalg.solve(pose,[0,1,0,0,0,0]);mapping_error=float(np.max(abs(pose@self.Tp-[0,1,0,0,0,0])))
        M=np.empty((6,6));mujoco.mj_fullM(m,env.data,M)
        return dict(q=env.data.qpos.copy(),qd=env.data.qvel.copy(),y_meas=measurement.y_effective,v_meas=measurement.v_effective,
            y_fk=env.tcp_y(),y_nominal=-.174,base=base.copy(),J=env.jacp()[1],Tp=self.Tp.copy(),M=M,bias=env.data.qfrc_bias.copy(),
            damping=m.dof_damping.copy(),kp=self.kp,kd=self.kd,ctrl_lo=self.ctrl_lo,ctrl_hi=self.ctrl_hi,tau_lo=self.tau_lo,tau_hi=self.tau_hi,
            tau_scale=self.tau_scale,gamma=self.gamma,pose_condition=cond,pose_rank=rank,mapping_error=mapping_error)
    def before_step(self,i,env,measurement,ctrl,base):
        self.decision=False;elapsed=0.
        if self.planning and i%32==0:
            self.decision=True;start=time.perf_counter();self.reason=''
            saved=[env.data.qpos.copy(),env.data.qvel.copy(),env.data.ctrl.copy(),float(env.data.time)]
            current=self.current(i,env,measurement,base)
            known=self.nominal[np.minimum(np.arange(i,i+N+1),len(self.nominal)-1)].copy()
            candidates=np.unique(np.clip(self.target+np.array([-DP,0,DP]),-D_MAX,D_MAX))
            plans=[];pt=[];rejected=[]
            for target in candidates:
                try:plans.append(Plan(self.state,target));pt.append(target)
                except PlanRejected as ex:rejected.append(dict(target=float(target),reason=str(ex),stage='ruckig'))
            scores=[];selected=None;result=None;chosen_plan=None;chosen_gamma=1.
            if plans:
                pt=np.asarray(pt);preview=np.asarray([p.preview() for p in plans]);count=len(plans)
                if self.method=='coupled':
                    targets=np.repeat(pt,3);gammas=np.tile([.9,1.,1.1],count);paths=np.repeat(preview,3,axis=0)
                    result=predict(current,known,paths,targets,gammas,self.target,True,self.params);selected=select(result,targets,gammas,self.target)
                    if selected is not None:chosen_plan=plans[selected//3];chosen_gamma=float(gammas[selected])
                    stage='joint'
                else:
                    targets=pt;gammas=np.ones(count)
                    result=predict(current,known,preview,targets,gammas,self.target,self.method=='sp_only',self.params);selected=select(result,targets,gammas,self.target);stage='position'
                    if selected is not None:chosen_plan=plans[selected]
                def rows(res,ts,gs,stage):
                    return [dict(target=float(ts[j]),gamma_target=float(gs[j]),score=float(res['scores'][j]) if np.isfinite(res['scores'][j]) else None,feasible=bool(np.isfinite(res['scores'][j])),reason=res['reasons'][j],stage=stage) for j in range(len(ts))]
                scores+=rows(result,targets,gammas,stage)
                if self.method=='ps_only' and chosen_plan is not None:
                    targets=np.full(3,chosen_plan.target);gammas=np.array([.9,1.,1.1]);paths=np.repeat(chosen_plan.preview()[None],3,axis=0)
                    result=predict(current,known,paths,targets,gammas,self.target,True,self.params);selected=select(result,targets,gammas,self.target)
                    scores+=rows(result,targets,gammas,'gamma_after_position')
                    if selected is None:chosen_plan=None
                    else:chosen_gamma=float(gammas[selected])
            old=self.target
            if chosen_plan is None:
                # preserve accepted trajectory, including current nonzero velocity/acceleration
                self.gamma_target=1.;self.reason='no feasible candidate: preserve accepted plan, gamma -> 1'
                predicted=None
            else:
                self.plan=chosen_plan;self.target=chosen_plan.target;self.gamma_target=chosen_gamma
                predicted=dict(e=result['e'][selected].tolist(),v=result['v'][selected].tolist(),tau=result['tau'][selected].tolist(),
                    meaning='0..160ms held candidate counterfactual; only [0,32) compared against realized policy',dt=DT)
            unchanged=all(np.array_equal(x,y) for x,y in zip(saved[:3],[env.data.qpos,env.data.qvel,env.data.ctrl])) and saved[3]==env.data.time
            if not unchanged:raise RuntimeError('Preview changed physical state')
            elapsed=time.perf_counter()-start
            self.decisions.append(dict(step=i,t=i*DT,old_target=old,target=self.target,gamma=self.gamma,gamma_target=self.gamma_target,state=self.state.tolist(),physical_q=current['q'].tolist(),physical_qdot=current['qd'].tolist(),
                current_measurement=[current['y_meas'],current['v_meas']],current_base=base.tolist(),candidates=scores,rejected=rejected,degradation=self.reason,selected_prediction=predicted,
                pose_condition=current['pose_condition'],pose_rank=current['pose_rank'],mapping_error=current['mapping_error'],preview_unchanged=unchanged,seconds=elapsed,plan_evidence=self.plan.evidence))
        ctrl.params['gamma']=self.gamma
        jerk=self.plan.jerk_at(self.plan.elapsed)
        self.records[i]=dict(p=self.state[0],w=self.state[1],a=self.state[2],p_star=self.target,gamma=self.gamma,gamma_star=self.gamma_target,
            jerk_segment=jerk,jerk_interval_average=(self.plan.at(self.plan.elapsed+DT)[2]-self.state[2])/DT,decision=int(self.decision),coordination_s=elapsed,degraded=int(bool(self.reason)))
    def command(self,i,env,sfc_command,out,actuation):
        delta=self.Tp*self.state[0];command=sfc_command+delta
        raw=self.kp*(command-env.data.qpos)-self.kd*env.data.qvel
        if not np.isfinite(command).all() or np.any(command<self.ctrl_lo) or np.any(command>self.ctrl_hi) or np.any(raw<self.tau_lo) or np.any(raw>self.tau_hi):
            raise RuntimeError('Current accepted command violates compiled hard boundary; stop valid prefix')
        self.records[i].update(delta_q_p=delta.copy(),delta_q_sfc=actuation.delta.copy(),tau_total_requested=raw.copy(),T_p=self.Tp.copy(),J_y_actual=env.jacp()[1])
        return command
    def after_step(self,i,env):
        if self.callback:self.callback(i+1)
        self.state=self.plan.advance();self.gamma+=float(np.clip(self.gamma_target-self.gamma,-DT,DT))
        if not np.isfinite(env.data.qpos).all() or abs(self.state[0])>D_MAX+1e-14:raise RuntimeError('nonfinite/plan position violation')
    def finish(self,done,row):self.completed_steps=done
