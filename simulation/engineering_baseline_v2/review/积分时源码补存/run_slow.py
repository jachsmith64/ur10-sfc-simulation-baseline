"""Every new simulation integrates three fixed cases in a unique run_id. No result cache."""
from pathlib import Path
import sys,hashlib,argparse,traceback,time,shutil
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'F组首次接入_20261010/.runtime'));sys.path.insert(0,str(ROOT))
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import config,sweep,disturbance
from input_acceptance import require_calibration,require_event
from position_execution import POSITION_EQUIV
from position_logging import PreciseLogger
from batch_identity import read,sha,atomic_json,new_directory,safe_id,files_snapshot,provenance
VERSION='F10-IDEAL-EXECUTION-ENGINEERING-CANDIDATE-v2'
CONTROLLERS=('A0','Alinear','ASFC_strong')
class RunRejected(RuntimeError):pass
def require(ok,message):
    if not ok:raise RunRejected(message)
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def correction_id(corr,source_id):
    h=hashlib.sha256(source_id.encode())
    for k in sorted(corr):h.update(k.encode());h.update(np.ascontiguousarray(corr[k]).tobytes())
    return h.hexdigest()
def source_id(c,g,fd_sha):
    h=hashlib.sha256()
    for k in ['t','q','phase','dq','dqd','remaining','remaining_dot']:
        a=c[k]*g if k in ['dq','dqd','remaining','remaining_dot'] else c[k];h.update(k.encode());h.update(np.ascontiguousarray(a).tobytes())
    h.update(fd_sha.encode());return h.hexdigest()
def load_inputs(root=ROOT):
    root=Path(root);hist=root/'historical/public_slow_review';fr=read(hist/'slow_frozen_version.json');old=read(root/'baseline_review/frozen_version.json')
    require(fr['version']=='F10-PRESCHEDULED-SLOW-v1','Unexpected inherited correction version')
    require_calibration(read(hist/'ordinary_calibration.json'));require_event(read(hist/'event_reconstruction.json'))
    c=arrays(root/'inputs/frozen_common.npz');corr=arrays(root/'inputs/public_slow_correction.npz');g=fr['g']
    require(g==old['g'],'Frozen g changed');require(sha(root/'inputs/frozen_common.npz')==old['common_sha256'],'Source arrays changed')
    require(sha(root/'inputs/public_slow_correction.npz')==fr['public_correction_sha256'],'Public correction changed')
    fd_sha=sha(root/'inputs/Fd_event.csv');require(fd_sha==old['event_sha256'],'Event force changed')
    require(source_id(c,g,fd_sha)==fr['source_S3_input_id'],'Source input mismatch')
    require(correction_id(corr,fr['source_S3_input_id'])==fr['shared_input_id'],'Public logical input mismatch')
    accepted=read(hist/'public_correction_acceptance.json');require(accepted['passed'] is True and accepted['input_id']==fr['shared_input_id'],'Public input not accepted')
    pars={p['case_id']:p for p in read(root/'inputs/fixed_position_test_parameters.json')['C_ORDINARY'] if p['case_id'] in CONTROLLERS}
    require(pars==fr['parameters'],'Frozen controller parameters changed')
    n=len(c['t']);require(n==10000 and np.array_equal(c['t'],np.arange(n)*.001),'Expected exact10000 pre-step samples')
    for key,a in c.items():require(a.shape[0]==n and np.isfinite(a).all(),'Invalid source '+key)
    for key,a in corr.items():
        require(a.shape==((n,6) if key.startswith('common_delta_q') else (n,)),'Public shape '+key);require(np.isfinite(a).all(),'Public values '+key)
    saved=np.loadtxt(root/'inputs/Fd_event.csv',delimiter=',',skiprows=1)
    require(saved.shape==(n,2) and np.array_equal(saved[:,0],c['t']) and np.isfinite(saved).all(),'Force time axis')
    fd=disturbance.FrozenDisturbance(root/'inputs/Fd_event.csv',c['t'],saved[:,1],{'frozen':{'sha256':fd_sha}})
    return dict(c=c,corr=corr,g=g,fd=fd,params=pars,freeze=fr,input_id=fr['shared_input_id'],fd_values=saved[:,1],fd_sha=fd_sha,nominal_Y_m=config.Y_NOMINAL)
def inputs_from_run(d):
    d=Path(d);identity=read(d/'input_identity.json')
    for n,h in identity['run_input_snapshots'].items():require((d/'frozen_inputs'/n).is_file() and sha(d/'frozen_inputs'/n)==h,'Historical input missing/corrupt: '+n)
    c=arrays(d/'frozen_inputs/frozen_common.npz');corr=arrays(d/'frozen_inputs/public_slow_correction.npz')
    fd=np.loadtxt(d/'frozen_inputs/Fd_event.csv',delimiter=',',skiprows=1)
    require(source_id(c,identity['g'],sha(d/'frozen_inputs/Fd_event.csv'))==identity['source_S3_input_id'],'Historical source ID mismatch')
    require(correction_id(corr,identity['source_S3_input_id'])==identity['input_id'],'Historical public ID mismatch')
    return dict(c=c,corr=corr,g=identity['g'],params=identity['parameters'],input_id=identity['input_id'],fd_values=fd[:,1],nominal_Y_m=identity['nominal_Y_m'])
SCALARS=['t','model_time_s','physical_y','effective_y','nominal_y','physical_v','effective_v','physical_e_um','effective_e_um','remaining','remaining_dot',
    'sample_e_m','sample_v_m_s','F_spring','F_linear','F_shear','F_requested','Fd','phase','updated',
    'common_baseline_um','common_delta_y_m','common_delta_y_dot_m_s','common_delta_y_ddot_m_s2']
MATRICES=['q_nominal','dq_C','original_base_ctrl','common_delta_q','common_delta_q_dot','common_delta_q_ddot','q','qdot','base_ctrl','delta_u','final_ctrl',
    'tau_requested','qfrc_actuator','qfrc_applied','qfrc_disturbance','tau_realized_increment','base_qfrc_actuator','actuator_force','effective_ctrl',
    'ctrl_saturation','force_saturation','joint_saturation','base_ctrl_saturation','base_force_saturation','base_joint_saturation']
def validate_case(path,row,ctx,run_id,generation_id,controller):
    require(row.get('status')=='OK','Returned '+str(row.get('status'))+': '+str(row.get('note')))
    n=len(ctx['c']['t']);require(row.get('n_steps_done')==n and row.get('n_steps_total')==n,'Step count mismatch')
    require(Path(path).is_file() and Path(path).with_suffix('.json').is_file(),'Missing data/metadata')
    meta=read(Path(path).with_suffix('.json'));a=arrays(path)
    require(meta.get('status')=='OK' and meta.get('n_steps_done')==n and meta.get('logger_samples')==n and meta.get('n_steps_requested')==n,'Metadata status/count mismatch')
    require(meta.get('run_id')==run_id and meta.get('generation_id')==generation_id and meta.get('case_id')==controller,'Generation/case mismatch')
    require(meta.get('input_sha256')==ctx['input_id'] and meta.get('controller_parameters')==ctx['params'][controller],'Input/parameter metadata mismatch')
    for k in SCALARS+MATRICES:
        require(k in a,'Missing field '+k);require(a[k].shape==((n,6) if k in MATRICES else (n,)),'Shape mismatch '+k)
    for k,v in a.items():require(v.ndim in (1,2) and v.shape[0]==n and np.isfinite(v).all(),'Nonfinite/short '+k)
    require(np.array_equal(a['t'],ctx['c']['t']) and np.max(abs(a['model_time_s']-a['t']))<1e-9,'Time/sample position mismatch')
    require(np.array_equal(a['q_nominal'],ctx['c']['q']) and np.array_equal(a['dq_C'],ctx['c']['dq']*ctx['g']),'Source positions differ')
    require(np.array_equal(a['remaining'],ctx['c']['remaining']*ctx['g']) and np.array_equal(a['remaining_dot'],ctx['c']['remaining_dot']*ctx['g']),'Remaining differs')
    for k,v in ctx['corr'].items():require(np.array_equal(a[k],v),'Public differs: '+k)
    require(np.array_equal(a['Fd'],ctx['fd_values']) and np.array_equal(a['phase'],ctx['c']['phase']),'Force/phase differs')
    require(np.all(a['nominal_y']==ctx['nominal_Y_m']) and np.array_equal(a['q'][0],ctx['c']['q'][0]) and np.all(a['qdot'][0]==0),'Target/initial state changed')
    require(np.array_equal(a['base_ctrl'],a['q_nominal']+a['dq_C']+a['common_delta_q']),'Base command mismatch')
    require(np.array_equal(a['final_ctrl'],a['base_ctrl']+a['delta_u']),'Final command mismatch')
    require(np.array_equal(a['qfrc_applied'],a['qfrc_disturbance']),'Duplicate anti torque')
    require(np.max(abs(a['effective_y']-a['physical_y']-a['remaining']))<1e-15 and np.max(abs(a['effective_v']-a['physical_v']-a['remaining_dot']))<1e-15,'Measurement mismatch')
    sums={k:int(np.count_nonzero(a[k])) for k in ['ctrl_saturation','force_saturation','joint_saturation']}
    logical=hashlib.sha256()
    for k in sorted(a):logical.update(k.encode());logical.update(str(a[k].dtype).encode());logical.update(np.ascontiguousarray(a[k]).tobytes())
    return dict(passed=True,samples=n,case_id=controller,run_id=run_id,data_sha256=sha(path),logical_arrays_sha256=logical.hexdigest(),saturation_entries=sums)
def json_safe(x):
    if isinstance(x,dict):return {k:json_safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [json_safe(v) for v in x]
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def run_case(d,run_id,controller,ctx,gen):
    path=Path(d)/'full_1ms'/f'{controller}.npz';require(not path.exists() and not path.with_suffix('.json').exists(),'Case collision; never reuse')
    p=ctx['params'][controller];c=ctx['c'];scaled={k:(v*ctx['g'] if k in ['dq','dqd','remaining','remaining_dot'] else v) for k,v in c.items()}
    logger=PreciseLogger(path,len(c['t']),'C_ORDINARY',POSITION_EQUIV,1,ctx['input_id'],input_data=scaled,common_correction=ctx['corr'])
    logger.meta.update(run_id=run_id,case_id=controller,generation_id=gen['generation_id'],module_version=VERSION,controller_parameters=p,g=ctx['g'],fd_sha256=ctx['fd_sha'],
        reference_y_unchanged_m=config.Y_NOMINAL,public_correction_source='Explicit accepted input; no historical result used',
        anti_increment_meaning='Same-state instantaneous actuator torque increment, NOT compensation position accuracy')
    print('INTEGRATE',run_id,controller,flush=True)
    row=sweep.run_case(controller,p['mode'],p['K'],p['B0'],p['n'],p['mu'],p['r'],p['v_ref'] or 0.,ctx['fd'],c['along'],c['q'],phase_code=c['phase'],
        c_input=scaled,execution_mode=POSITION_EQUIV,control_update_steps=1,precise_logger=logger,common_position_correction=ctx['corr']['common_delta_q'],keep_trace=False)
    atomic_json(path.with_suffix('.row.json'),json_safe(row));verdict=validate_case(path,row,ctx,run_id,gen['generation_id'],controller)
    atomic_json(path.with_suffix('.validation.json'),verdict);print('VALIDATED',controller,verdict['samples'],flush=True);return verdict
def new_run(root=ROOT,run_id=None,retry_of=None,make_report=True):
    root=Path(root)
    if retry_of:
        prev=root/'runs'/safe_id(retry_of);require((prev/'state.json').is_file() and read(prev/'state.json')['status'] in ['FAILED','INCOMPLETE'],'Retry needs specified failed batch')
    rid,d=new_directory(root,run_id);start=time.monotonic();integrations=0;checks=[]
    atomic_json(d/'state.json',dict(run_id=rid,status='RUNNING',completed=False,integrations_started=0,retry_of=retry_of))
    try:
        ctx=load_inputs(root);gen=provenance(root,VERSION,ctx['input_id']);gen.update(run_id=rid,retry_of=retry_of);atomic_json(d/'generation.json',gen)
        (d/'frozen_inputs').mkdir()
        input_copies={}
        for n in ['frozen_common.npz','public_slow_correction.npz','Fd_event.csv']:
            shutil.copy2(root/'inputs'/n,d/'frozen_inputs'/n);input_copies[n]=sha(d/'frozen_inputs'/n)
        for n in ['metric_windows.json','event_targets.json','trajectory.json']:
            shutil.copy2(root/'historical/public_slow_review'/n,d/'frozen_inputs'/n);input_copies[n]=sha(d/'frozen_inputs'/n)
        atomic_json(d/'input_identity.json',dict(input_id=ctx['input_id'],source_S3_input_id=ctx['freeze']['source_S3_input_id'],g=ctx['g'],parameters=ctx['params'],dt_s=.001,control_update_steps=1,
            public_source_version='F10-PRESCHEDULED-SLOW-v1',inherited_input_files={p:v for p,v in gen['files'].items() if p.startswith('inputs/')},run_input_snapshots=input_copies,not_inherited_results=True,
            initial_q=ctx['c']['q'][0].tolist(),initial_qdot=[0.]*6,nominal_Y_m=config.Y_NOMINAL))
        for ctrl in CONTROLLERS:
            integrations+=1;atomic_json(d/'state.json',dict(run_id=rid,status='RUNNING',completed=False,current_case=ctrl,integrations_started=integrations,validated_cases=checks,retry_of=retry_of))
            checks.append(run_case(d,rid,ctrl,ctx,gen))
        require(files_snapshot(root)==gen['files'],'Implementation/assets/inputs changed during run')
        atomic_json(d/'validation.json',dict(passed=True,run_id=rid,generation_id=gen['generation_id'],integrations_started=integrations,cases=checks,accepted_before_report=True,
            success_rule='All3: status/count/finite fields/shapes/sample timing/identities/common conditions checked'))
        if make_report:
            from run_report import render
            render(d,root=root,validated_before_completion=True)
        complete=dict(run_id=rid,status='SUCCESS',completed=True,integrations_started=integrations,cases=checks,generation_id=gen['generation_id'],elapsed_s=time.monotonic()-start,report_generated=make_report,retry_of=retry_of)
        atomic_json(d/'completed.json',complete);atomic_json(d/'state.json',complete);print('SUCCESS',rid,'3 new integrations',flush=True);return rid
    except BaseException as e:
        failure=dict(run_id=rid,status='FAILED',completed=False,integrations_started=integrations,validated_cases=checks,retry_of=retry_of,reason=f'{type(e).__name__}: {e}',traceback=traceback.format_exc(),elapsed_s=time.monotonic()-start)
        atomic_json(d/'state.json',failure);atomic_json(d/'failure.json',failure)
        if (d/'completed.json').exists():(d/'completed.json').unlink()
        raise
def open_history(run_id,root=ROOT,redraw=False):
    d=Path(root)/'runs'/safe_id(run_id);require((d/'completed.json').is_file() and read(d/'completed.json').get('completed') is True,'Specified batch missing/failed/incomplete')
    ctx=inputs_from_run(d);gen=read(d/'generation.json');done=read(d/'completed.json')
    for ctrl in CONTROLLERS:
        p=d/'full_1ms'/f'{ctrl}.npz';verdict=next(x for x in done['cases'] if x['case_id']==ctrl)
        require(p.is_file() and sha(p)==verdict['data_sha256'],'Specified data missing/corrupt: '+ctrl)
        validate_case(p,read(p.with_suffix('.row.json')),ctx,run_id,gen['generation_id'],ctrl)
    if redraw:
        from run_report import render
        render(d,root=Path(root),validated_before_completion=False)
    require((d/'results.html').is_file(),'Specified batch has no report');return d/'results.html'
def cli(argv=None):
    ap=argparse.ArgumentParser(description='new integrates; show/redraw require exact run_id');sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('new');p.add_argument('--run-id');p.add_argument('--retry-of')
    for cmd in ['show','redraw']:p=sub.add_parser(cmd);p.add_argument('--run-id',required=True)
    args=ap.parse_args(argv)
    try:
        if args.command=='new':rid=new_run(run_id=args.run_id,retry_of=args.retry_of);print('VIEW: python run_slow.py show --run-id '+rid,flush=True)
        else:
            path=open_history(args.run_id,redraw=args.command=='redraw');print('HISTORICAL DATA ONLY:',path,flush=True)
            import os
            os.startfile(str(path))
        return 0
    except Exception as e:print('FAILED (no success/fallback):',type(e).__name__,str(e),file=sys.stderr,flush=True);return 2
if __name__=='__main__':raise SystemExit(cli())
