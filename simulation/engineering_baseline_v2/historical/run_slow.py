"""Offline public slow position correction, then three fixed anti-jitter cases. Max6 runs."""
from pathlib import Path
import sys,json,hashlib,shutil,argparse
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'F组首次接入_20261010/.runtime'));sys.path.insert(0,str(ROOT))
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import config,sweep,disturbance
from short_prepare import write,sha,smooth,smooth_dot
from run_short import read,common,array,PARAMS,input_id,fd_object
from input_acceptance import require_calibration,require_event
from position_logging import PreciseLogger
from position_execution import POSITION_EQUIV,compiled_mapping
from sim_env import SimEnv

VERSION='F10-PRESCHEDULED-SLOW-v1'
CODE=['run_slow.py','run_short.py','input_acceptance.py','sweep.py','position_logging.py','position_execution.py','controller.py','sfc.py','sim_env.py','config.py','background.py','f_measurement.py','f_config.json','inputs/fixed_position_test_parameters.json','inputs/仿真误差参数.json']
def budget():return read(ROOT/'review/slow_run_budget.json') if (ROOT/'review/slow_run_budget.json').exists() else []
def upstream():
    fr=read(ROOT/'baseline_review/frozen_version.json');c=common();g=fr['g']
    checks=dict(calibration=require_calibration(read(ROOT/'review/ordinary_calibration.json')),event=require_event(read(ROOT/'review/event_reconstruction.json')))
    assert sha(ROOT/'inputs/frozen_common.npz')==fr['common_sha256']
    assert sha(ROOT/'inputs/Fd_event.csv')==fr['event_sha256']
    assert checks['calibration']['g']==g and PARAMS==fr['parameters']
    f=np.loadtxt(ROOT/'inputs/Fd_event.csv',delimiter=',',skiprows=1)[:,1];fd=fd_object(c,f,ROOT/'inputs/Fd_event.csv')
    assert input_id(c,g,fd.sha256)==fr['scenarios']['S3']['input_id']
    assert read(ROOT/'review/R026_failure_checks.json')['passed']
    write(ROOT/'review/upstream_acceptance.json',dict(passed=True,checks=checks,source_S3_id=fr['scenarios']['S3']['input_id'],no_new_calibration_or_event_generation=True))
    return c,g,fd,fr
def baseline():
    a=array(ROOT/'reference_old_S3/S3__A0.npz');source=read(ROOT/'review/source_version.json')
    assert sha(ROOT/'reference_old_S3/S3__A0.npz')==source['source_case_logs']['S3__A0.npz']
    windows=[('pre_rest',.85,1.075),('outbound',2.226,4.38),('return',5.943,8.667),('post_tail_short',9.95,9.999)]
    rows=[];levels=[]
    for name,lo,hi in windows:
        m=(a['t']>=lo)&(a['t']<=hi);x=a['effective_e_um'][m];level=float(np.median(x));levels.append(level)
        rows.append(dict(platform=name,t0_s=lo,t1_s=hi,samples=int(m.sum()),effective_median_um=level,physical_median_um=float(np.median(a['physical_e_um'][m])),std_um=float(np.std(x)),
            qualification='Only49ms/50 samples near end; event-only old A0 max1.20um here, separately checked in prior results. Short tail representative, not independent long settled state.' if name=='post_tail_short' else 'Outside cold<0.8s and chosen event supports; median of inherited A0 only.'))
    plan=dict(source='reference_old_S3/S3__A0.npz',source_sha256=sha(ROOT/'reference_old_S3/S3__A0.npz'),source_channel='effective_e_um',units='um',platforms=rows,
        levels_um=levels,connections_s=[[1.075,2.226],[4.38,5.943],[8.667,9.95]],connection='quintic smoothstep, analytic C2, no fitting within an event',
        noncausal=True,ordinary_or_event_waveform_fitted=False,tail_limitation='Final representative is short and not proof of long-term drift stability. Its level nearly equals return; stop correction changes by only~0.094um.',
        event_windows=read(ROOT/'review/event_targets.json'))
    write(ROOT/'review/baseline_plan.json',plan);return plan
def slow_center(t,plan):
    levels=plan['levels_um'];b=np.full_like(t,levels[0]);bd=np.zeros_like(t);bdd=np.zeros_like(t)
    for j,(lo,hi) in enumerate(plan['connections_s']):
        T=hi-lo;m=(t>=lo)&(t<=hi);u=(t[m]-lo)/T;diff=levels[j+1]-levels[j]
        b[m]=levels[j]+diff*smooth(u);bd[m]=diff*smooth_dot(u,T)
        bdd[m]=diff*(60*u-180*u*u+120*u*u*u)/(T*T)
        b[t>hi]=levels[j+1]
    return b,bd,bdd
def static_map(c):
    env=SimEnv();env.reset(c['q'][0]);env.forward();J=env.jacp().copy();sv=np.linalg.svd(J,compute_uv=False);condition=float(sv[0]/sv[-1])
    assert sv[-1]>1e-5 and condition<1e5,'Static translation mapping is ill-conditioned; stop branch'
    M=J.T@np.linalg.solve(J@J.T,np.array([0.,1.,0.]));assert np.max(abs(J@M-np.array([0.,1.,0.])))<1e-12
    # Nominal initial FK can have tiny existing IK residual; compare actual states, not config rounding.
    env.reset(c['q'][0]);env.forward();p0=env.tcp_pos().copy();env.reset(c['q'][0]+M*1e-6);env.forward();shift=env.tcp_pos()-p0
    assert shift[1]>0 and abs(shift[1]/1e-6-1)<.01
    mapping=dict(J0_translation_m_per_rad=J,M_rad_per_m=M,singular_values=sv,condition=condition,first_order_JM=[0.,1.,0.],positive1um_FK_shift_m=shift,
        method='Constant minimum-norm translation right inverse at original initial nominal posture; X/Z first-order correction zero. Static only, no dynamic Jacobian/observer.',
        full_compiled_model=compiled_mapping(env.model),original_initial_q=c['q'][0])
    write(ROOT/'review/static_mapping.json',mapping);return M
def make_correction(c,plan,M,gain):
    b,bd,bdd=slow_center(c['t'],plan);dy=-gain*b*1e-6;dyd=-gain*bd*1e-6;dydd=-gain*bdd*1e-6
    return dict(common_baseline_um=b,common_delta_y_m=dy,common_delta_y_dot_m_s=dyd,common_delta_y_ddot_m_s2=dydd,
        common_delta_q=dy[:,None]*M,common_delta_q_dot=dyd[:,None]*M,common_delta_q_ddot=dydd[:,None]*M)
def correction_id(corr,source_id):
    h=hashlib.sha256(source_id.encode())
    for k in sorted(corr):h.update(k.encode());h.update(np.ascontiguousarray(corr[k]).tobytes())
    return h.hexdigest()
def run(name,controller,c,g,fd,corr,source_id,role):
    dest=ROOT/'outputs/full_1ms'/f'{name}.npz';identity=correction_id(corr,source_id);p=PARAMS[controller]
    if dest.exists() and dest.with_suffix('.row.json').exists():
        meta=read(dest.with_suffix('.json'));row=read(dest.with_suffix('.row.json'))
        assert meta['input_sha256']==identity and meta['controller_parameters']==p and row['status']=='OK' and row['n_steps_done']==len(c['t'])
        print('REUSE',name,flush=True);return dest
    runs=budget();assert len(runs)<6,'6-run budget exhausted'
    assert not any(x['case_id']==name for x in runs),'No silent re-run of failed/incomplete case'
    if role=='construct':assert controller=='A0' and sum(x['role']=='construct' for x in runs)<3
    if role=='formal':assert controller in ['Alinear','ASFC_strong'] and sum(x['role']=='formal' for x in runs)<2
    runs.append(dict(index=len(runs)+1,case_id=name,controller=controller,role=role,input_id=identity,status='RUNNING'));write(ROOT/'review/slow_run_budget.json',runs)
    scaled={k:(v*g if k in ['dq','dqd','remaining','remaining_dot'] else v) for k,v in c.items()}
    logger=PreciseLogger(dest,len(c['t']),'C_ORDINARY',POSITION_EQUIV,1,identity,input_data=scaled,common_correction=corr)
    logger.meta.update(module_version=VERSION,g=g,controller_parameters=p,source_S3_input_id=source_id,fd_sha256=fd.sha256,
        code_snapshot_sha256=sha(ROOT/'review/slow_implementation_freeze.json'),reference_y_unchanged_m=config.Y_NOMINAL,
        counterfactual='Independent actuator evaluation at same actual state. Base includes public slow position correction. Recorded realized increment is anti-jitter only; public correction is reported as position/speed/acceleration, not invented force.',
        shared_inputs='Original frozen q/dq/dqd/remaining/remdot/Fd unchanged; public correction same every case; no output trend subtraction.')
    print('RUN',len(runs),name,flush=True)
    row=sweep.run_case(name,p['mode'],p['K'],p['B0'],p['n'],p['mu'],p['r'],p['v_ref'] or 0.,fd,c['along'],c['q'],phase_code=c['phase'],c_input=scaled,
        execution_mode=POSITION_EQUIV,control_update_steps=1,precise_logger=logger,common_position_correction=corr['common_delta_q'],keep_trace=False)
    write(dest.with_suffix('.row.json'),row);runs=budget();runs[-1].update(status=row['status'],n_steps_done=row['n_steps_done'],note=row['note'],log_sha256=sha(dest));write(ROOT/'review/slow_run_budget.json',runs)
    print('DONE',name,row['status'],row['n_steps_done'],flush=True);return dest
def centers(a,plan):
    return [float(np.median(a['effective_e_um'][(a['t']>=p['t0_s'])&(a['t']<=p['t1_s'])])) for p in plan['platforms']]
def construction_check(p,c,plan,corr):
    a=array(p);row=read(p.with_suffix('.row.json'));vals=centers(a,plan)
    m=(a['t']>=.85)&(a['t']<=1.075);cold=a['t']<.8;ic=np.flatnonzero(cold)[np.argmax(abs(a['effective_e_um'][cold]))]
    sat=np.any(a['ctrl_saturation']|a['force_saturation']|a['joint_saturation'],axis=1)
    passed=row['status']=='OK' and len(a['t'])==len(c['t']) and not sat.any() and max(abs(np.array(vals)))<=5.
    return dict(passed=bool(passed),status=row['status'],platform_centers_um=vals,original_platform_centers_um=plan['levels_um'],center_threshold_um=5.,saturated_samples=int(sat.sum()),
        correction_initial_nonzero_rad=corr['common_delta_q'][0],cold_peak_t_s=float(a['t'][ic]),cold_peak_um=float(a['effective_e_um'][ic]),
        public_delta_q_peak_abs_rad=float(np.max(abs(corr['common_delta_q']))),public_qdot_peak_abs_rad_s=float(np.max(abs(corr['common_delta_q_dot']))),public_qddot_peak_abs_rad_s2=float(np.max(abs(corr['common_delta_q_ddot']))),
        interpretation='A0 only. Accept approximate platform centering within5um, including separately qualified short tail. No parameter/g/event adjustment. Cold state not changed.')
def validate_derivatives(c,plan,M,corr,gain):
    x=np.array([1.20,1.9,4.7,5.3,8.9,9.7]);eps=1e-5;b,bd,bdd=slow_center(x,plan);bp,dp,_=slow_center(x+eps,plan);bm,dm,_=slow_center(x-eps,plan)
    err1=float(np.max(abs((bp-bm)/(2*eps)-bd)));err2=float(np.max(abs((dp-dm)/(2*eps)-bdd)));assert max(err1,err2)<1e-7
    joint_step=float(np.max(abs(np.diff(corr['common_delta_q'],axis=0))))
    assert joint_step<=.001*np.max(abs(corr['common_delta_q_dot']))*1.001+1e-15
    write(ROOT/'review/slow_derivative_checks.json',dict(passed=True,baseline_derivative_error_um_s=err1,baseline_second_derivative_error_um_s2=err2,
        C2_endpoint_continuity=True,static_map_only=True,max_adjacent_common_command_step_rad=joint_step,
        initial_jump_is_separate_from_analytic_derivative=True,initial_common_position_step_rad=corr['common_delta_q'][0]))
def main(stage='all'):
    c,g,fd,fr=upstream();snapshot={p:sha(ROOT/p) for p in CODE}
    freeze=ROOT/'review/slow_implementation_freeze.json'
    if freeze.exists():assert read(freeze)==snapshot,'Do not silently change implementation after construction validation'
    else:write(freeze,snapshot)
    source_id=fr['scenarios']['S3']['input_id'];plan=baseline();M=static_map(c)
    path=ROOT/'review/public_correction_acceptance.json'
    if path.exists():
        accepted=read(path);assert accepted['passed']
        with np.load(ROOT/'inputs/public_slow_correction.npz') as z:corr={k:z[k] for k in z.files}
        assert correction_id(corr,source_id)==accepted['input_id']
        gain=accepted['gain'];p=ROOT/accepted['A0_log']
        assert construction_check(p,c,plan,corr)['passed']
    else:
        gain=1.;history=[]
        for attempt in range(3):
            corr=make_correction(c,plan,M,gain);validate_derivatives(c,plan,M,corr,gain)
            p=run(f'construct_{attempt+1}__A0','A0',c,g,fd,corr,source_id,'construct');check=construction_check(p,c,plan,corr)
            check.update(attempt=attempt+1,gain=gain,log=str(p.relative_to(ROOT)));history.append(check);write(ROOT/'review/public_correction_attempts.json',history)
            if check['passed']:break
            if check['status']!='OK' or check['saturated_samples']:break
            # A0 ordinary centers only, no event/controller performance used to set gain.
            old=np.array(plan['levels_um'][1:3]);new=np.array(check['platform_centers_um'][1:3]);response=old-new
            proposed=gain*float(np.median(old/response)) if np.all(abs(response)>1.) else float('nan')
            if not np.isfinite(proposed) or not .5<=proposed<=2.:
                check['rejection']='Simple gain would leave permitted static correction range; no complex fit';break
            if attempt<2:gain=proposed
        if not check['passed']:
            write(path,dict(passed=False,attempts=len(history),reason='Approximate common-center premise not reached within budget; no formal dependent cases.'))
            raise RuntimeError('Public correction failed acceptance; preserve attempted logs and stop formal dependencies')
        np.savez_compressed(ROOT/'inputs/public_slow_correction.npz',**corr)
        accepted=dict(passed=True,gain=gain,input_id=correction_id(corr,source_id),A0_log=str(p.relative_to(ROOT)),check=check,attempts=len(history))
        write(path,accepted)
    final=dict(version=VERSION,source_version_sha256=sha(ROOT/'baseline_review/frozen_version.json'),source_S3_input_id=source_id,shared_input_id=accepted['input_id'],g=g,
        public_correction_sha256=sha(ROOT/'inputs/public_slow_correction.npz'),baseline_plan_sha256=sha(ROOT/'review/baseline_plan.json'),mapping_sha256=sha(ROOT/'review/static_mapping.json'),
        parameters=PARAMS,dt_s=.001,duration_s=10.,fixed_y_target_m=config.Y_NOMINAL,implementation_sha256=sha(freeze),initial_condition_unchanged=True,
        feedback_velocity='Measured physicalYdot + original remaining_dot; original nominalYdot=0. Public analytic velocity is diagnostic, not subtracted as a changed target.',
        execution='original q_nom + original g*dq + frozen public delta_q_slow + existing tau_anti/Kp; only Fd_event in qfrc_applied',noncausal_offline=True)
    file=ROOT/'review/slow_frozen_version.json'
    if file.exists():assert read(file)==final
    else:write(file,final)
    dst=ROOT/'outputs/full_1ms/PC__A0.npz'
    if not dst.exists():
        shutil.copy2(p,dst)
        for suffix in ['.json','.row.json']:
            meta=read(p.with_suffix(suffix));meta.update(reused_from=str(p.relative_to(ROOT)))
            if suffix=='.row.json':meta['case_id']='PC__A0'
            write(dst.with_suffix(suffix),meta)
    if stage=='construct':return
    for controller in ['Alinear','ASFC_strong']:run('PC__'+controller,controller,c,g,fd,corr,source_id,'formal')
    write(ROOT/'review/slow_complete.json',dict(completed=True,new_integrations=len(budget()),formal_cases=3,A0_reused=True,max_integrations=6))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['construct','all'],nargs='?',default='all');main(ap.parse_args().stage)
