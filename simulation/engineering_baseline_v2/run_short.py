"""Bounded 1 ms study. Reuses the accepted F loop; never recalibrates controllers."""
from pathlib import Path
import sys,json,hashlib,shutil,argparse
ROOT=Path(__file__).resolve().parent
RUNTIME=ROOT.parent/'F组首次接入_20261010/.runtime'
if RUNTIME.is_dir():sys.path.insert(0,str(RUNTIME))
sys.path.insert(0,str(ROOT))
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import config,disturbance,sweep
from short_prepare import prepare,write,sha
from position_logging import PreciseLogger
from position_execution import POSITION_EQUIV
from input_acceptance import require_calibration,require_event,calibration_verdict

PARAMS={p['case_id']:p for p in json.loads((ROOT/'inputs/fixed_position_test_parameters.json').read_text(encoding='utf-8'))['C_ORDINARY'] if p['case_id'] in ['A0','Alinear','ASFC_strong']}
CODE_NAMES=['config.py','sim_env.py','sfc.py','controller.py','sweep.py','background.py','f_measurement.py','position_execution.py','position_logging.py','disturbance.py','short_prepare.py','run_short.py','f_config.json','inputs/fixed_position_test_parameters.json','inputs/仿真误差参数.json']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def common():
    with np.load(ROOT/'inputs/frozen_common.npz') as z:return {k:z[k].copy() for k in z.files}
def ledger():return read(ROOT/'review/run_budget.json') if (ROOT/'review/run_budget.json').exists() else []
def input_id(c,g,fd_sha):
    h=hashlib.sha256()
    for k in ['t','q','phase','dq','dqd','remaining','remaining_dot']:
        a=c[k]*g if k in ['dq','dqd','remaining','remaining_dot'] else c[k]
        h.update(k.encode());h.update(np.ascontiguousarray(a).tobytes())
    h.update(fd_sha.encode());return h.hexdigest()
def fd_object(c,f,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():np.savetxt(path,np.column_stack([c['t'],f]),delimiter=',',fmt='%.17g',header='t_s,Fd_event_N',comments='')
    saved=np.loadtxt(path,delimiter=',',skiprows=1)
    assert np.array_equal(saved[:,0],c['t']) and np.array_equal(saved[:,1],f)
    return disturbance.FrozenDisturbance(path,c['t'],f,{'frozen':{'sha256':sha(path)}})
def run(name,controller,c,g,fd,role):
    out=ROOT/'outputs/full_1ms'/name;out=out.with_suffix('.npz')
    identity=input_id(c,g,fd.sha256);p=PARAMS[controller]
    if out.exists() and out.with_suffix('.row.json').exists():
        meta=read(out.with_suffix('.json'));assert meta['input_sha256']==identity and meta['controller_parameters']==p
        print('REUSE',name,flush=True);return out
    runs=ledger();assert len(runs)<22,'22-run budget exhausted'
    assert not any(x['name']==name for x in runs),'A partially failed run needs explicit accounting; do not silently repeat it'
    entry=dict(index=len(runs)+1,name=name,role=role,controller=controller,g=g,input_id=identity,status='RUNNING')
    runs.append(entry);write(ROOT/'review/run_budget.json',runs)
    scaled={k:(v*g if k in ['dq','dqd','remaining','remaining_dot'] else v) for k,v in c.items()}
    logger=PreciseLogger(out,len(c['t']),'C_ORDINARY',POSITION_EQUIV,1,identity,input_data=scaled)
    logger.meta.update(scenario=name.split('__')[0],g=g,controller_parameters=p,fd_sha256=fd.sha256,
        common_sha256=sha(ROOT/'inputs/frozen_common.npz'),model_sha256=sha(ROOT/'assets_ur10e/ur10e.xml'),
        shared_inputs='Same frozen10s q/phase and ordinary arrays, unified g including fixed bias; same event table. Scaled dq enters base once; remaining enters measured feedback once.',
        code_snapshot_sha256=sha(ROOT/'review/implementation_freeze.json'))
    print('RUN',entry['index'],name,'g=',g,flush=True)
    row=sweep.run_case(name,p['mode'],p['K'],p['B0'],p['n'],p['mu'],p['r'],p['v_ref'] or 0.,fd,c['along'],c['q'],
        phase_code=c['phase'],c_input=scaled,execution_mode=POSITION_EQUIV,control_update_steps=1,precise_logger=logger,keep_trace=False)
    write(out.with_suffix('.row.json'),row)
    runs=ledger();runs[-1].update(status=row['status'],n_steps_done=row['n_steps_done'],note=row['note'],log_sha256=sha(out));write(ROOT/'review/run_budget.json',runs)
    print('DONE',name,row['status'],row['n_steps_done'],flush=True)
    return out
def array(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}
def ordinary_p95(p):
    a=array(p);return float(np.mean([np.percentile(abs(a['effective_e_um'][a['phase']==code]),95) for code in [1,3]]))
def calibrate(c,zero):
    dst=ROOT/'review/ordinary_calibration.json'
    if dst.exists():
        result=read(dst);require_calibration(result);return result
    base=run('S0__A0','A0',c,1.,zero,'formal reused as calibration')
    assert read(base.with_suffix('.row.json'))['status']=='OK'
    target=read(ROOT/'review/A_ordinary_reference.json')['target_P95_um'];baseline=ordinary_p95(base)
    g=min(1.,target/baseline);history=[]
    # Initial estimate plus at most two corrections, all A0 only.
    for attempt in range(3):
        name='calibration_'+str(attempt+1)+'__A0'
        p=run(name,'A0',c,g,zero,'A0 ordinary scale validation')
        assert read(p.with_suffix('.row.json'))['status']=='OK'
        value=ordinary_p95(p);history.append(dict(g=g,output_P95_um=value,log=str(p.relative_to(ROOT))))
        if abs(value/target-1)<=.10 or g>=1:break
        if attempt<2:g=min(1.,g*target/value)
    result=dict(g=g,A_target_P95_um=target,S0_P95_um=baseline,S1_P95_um=value,history=history,
        definition='Equal average of P95(abs raw effective Y error) in outbound and return steady phases; no output recentering.',
        affects_fixed_bias=True,controller_parameters_unchanged=True,check_tolerance_relative=.10)
    result['accepted']=calibration_verdict(result)['passed']
    write(dst,result);require_calibration(result)
    formalize(p,'S1__A0',g,zero,c)
    return result
def formalize(src,name,g,fd,c):
    dest=ROOT/'outputs/full_1ms'/f'{name}.npz'
    if src!=dest:shutil.copy2(src,dest)
    meta=read(src.with_suffix('.json'));meta.update(scenario=name.split('__')[0],reused_from=str(src.relative_to(ROOT)),input_sha256=input_id(c,g,fd.sha256))
    write(dest.with_suffix('.json'),meta)
    row=read(src.with_suffix('.row.json'));row['case_id']=name;write(dest.with_suffix('.row.json'),row)
def inverse_padded(e,plant):
    pad=2000;return disturbance.invert_series(np.pad(e,(pad,pad)),plant)[pad:pad+len(e)]
def event_check(target,y,c):
    ix=c['t']>=.8;r=target-y
    out=dict(relative_rms_error=float(np.linalg.norm(r[ix])/np.linalg.norm(target[ix])),
        rmse_um=float(np.sqrt(np.mean(r[ix]**2))),cold_initial_max_um=float(np.max(abs(y[c['t']<.8]))),
        target_pre_event_max_um=float(np.max(abs(target[c['t']<.8]))),last100ms_max_um=float(np.max(abs(y[-100:]))),events=[])
    for ev in read(ROOT/'review/event_targets.json'):
        lo,hi=ev['core_s'];m=(c['t']>=lo)&(c['t']<=hi)
        out['events'].append(dict(event=ev['event'],target_min_um=float(np.min(target[m])),target_max_um=float(np.max(target[m])),
            response_min_um=float(np.min(y[m])),response_max_um=float(np.max(y[m])),core_rmse_um=float(np.sqrt(np.mean(r[m]**2)))))
    return out
def reconstruct(c):
    dst=ROOT/'review/event_reconstruction.json'
    if dst.exists():
        result=read(dst);require_event(result)
        assert sha(ROOT/'inputs/Fd_event.csv')==result['final_fd_sha256'],'Cached accepted event file identity changed'
        f=np.loadtxt(ROOT/'inputs/Fd_event.csv',delimiter=',',skiprows=1)[:,1]
        return fd_object(c,f,ROOT/'inputs/Fd_event.csv'),result
    plant=disturbance.load_plant(ROOT/'reference_B_packet/plant_frf.npz');assert plant is not None
    target=c['event_target_um'];f=inverse_padded(target,plant);history=[]
    for attempt in range(3):
        fd=fd_object(c,f,ROOT/'inputs/event_candidates'/f'Fd_event_{attempt+1}.csv')
        p=run(f'event_validation_{attempt+1}__A0','A0',c,0.,fd,'A0 event reconstruction validation')
        assert read(p.with_suffix('.row.json'))['status']=='OK'
        y=array(p)['physical_e_um'];check=event_check(target,y,c);check.update(attempt=attempt+1,Fd_abs_peak_N=float(np.max(abs(f))),log=str(p.relative_to(ROOT)))
        history.append(check);print('EVENT relative RMS residual',check['relative_rms_error'],flush=True)
        if check['relative_rms_error']<=.20:break
        if attempt<2:f=f+.7*inverse_padded(target-y,plant)
    result=dict(history=history,final_check=history[-1],plant_sha256=sha(ROOT/'reference_B_packet/plant_frf.npz'),
        final_fd_sha256=fd.sha256,settings=dict(pass_hz=config.INV_F_PASS_HZ,stop_hz=config.INV_F_STOP_HZ,zero_padding_each_side_s=2.,residual_learning_rate=.7,max_A0_validations=3),
        finite_memory_note='Zero-padded inverse suppresses periodic wrap; trimmed tails and cold-start causal response are explicitly checked. Source core e_vib amplitudes are unchanged; output is an approximate plant response, not fresh physical measurement.',
        sufficient_for_approximate_comparison=history[-1]['relative_rms_error']<=.20)
    write(dst,result);require_event(result)
    shutil.copy2(fd.path,ROOT/'inputs/Fd_event.csv');final=fd_object(c,f,ROOT/'inputs/Fd_event.csv')
    formalize(p,'S2__A0',0.,final,c)
    return final,result
def freeze(c,g,event,zero):
    scenarios={'S0':(1.,False),'S1':(g,False),'S2':(0.,True),'S3':(g,True),'S4':(1.,True)}
    info=dict(version='F10-1ms-Acal-event-v1',duration_s=len(c['t'])*.001,dt_s=.001,parameters=PARAMS,
        feedback='Current mj_forward state, effective Y = physicalY + remaining, analytic remaining derivative; no Fd read by SFC.',
        execution='ctrl=q_nom+dq*g+tau_sfc/5000; external qfrc_applied=Fd event only; no duplicated SFC torque.',
        common_sha256=sha(ROOT/'inputs/frozen_common.npz'),event_sha256=event.sha256,zero_event_sha256=zero.sha256,
        implementation_snapshot_sha256=sha(ROOT/'review/implementation_freeze.json'),g=g,
        scenarios={s:dict(g=gg,event_on=on,input_id=input_id(c,gg,(event if on else zero).sha256)) for s,(gg,on) in scenarios.items()})
    dest=ROOT/'review/frozen_version.json'
    if dest.exists():assert read(dest)==info,'Frozen version cannot silently change'
    else:write(dest,info)
    return scenarios
def main(stage):
    if stage=='prepare':prepare();return
    assert (ROOT/'inputs/frozen_common.npz').exists()
    snapshot={p:sha(ROOT/p) for p in CODE_NAMES}
    path=ROOT/'review/implementation_freeze.json'
    if path.exists():assert read(path)==snapshot,'Simulation implementation changed after A0 validations'
    else:write(path,snapshot)
    c=common();assert config.GRID_DT_S==.001
    zero=fd_object(c,np.zeros_like(c['t']),ROOT/'inputs/Fd_zero.csv')
    if stage=='calibrate':calibrate(c,zero);return
    if stage=='events':reconstruct(c);return
    cal=calibrate(c,zero);event,recon=reconstruct(c)
    require_calibration(cal);require_event(recon)
    scenarios=freeze(c,cal['g'],event,zero)
    for scenario,(g,on) in scenarios.items():
        for controller in PARAMS:run(scenario+'__'+controller,controller,c,g,event if on else zero,'formal')
    write(ROOT/'review/run_complete.json',dict(formal_cases=15,executed_runs=len(ledger()),budget=22,simulation_finished=True))
if __name__=='__main__':
    raise SystemExit('This independent copy uses run_slow.py. Legacy15-case matrix is retained only for audit; it is not an entry point here.')
