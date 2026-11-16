from b132_common import *
import run_slow,sweep,config,e_predict
from gamma_timing import TimedCoordinator
from e_logging import ELogger
from b132_adapters import Coordinator132,Logger132
from b132_measurement import SampledMeasurement
from position_execution import POSITION_EQUIV
def context():
 c=run_slow.load_inputs(B);c['scaled']={k:v*c['g'] if k in ['dq','dqd','remaining','remaining_dot'] else v for k,v in c['c'].items()}
 c['parameters']=dict(c['params']['ASFC_strong'],w_gamma=.01);return c
def adapter_tests():
 class Dummy:
  class Data:time=0.
  data=Data()
  def tcp_y(self):return .1*self.data.time+5e-6*np.sin(2*np.pi*7*self.data.time)
  def tcp_y_velocity(self):raise AssertionError('R2idealvelocitybypass')
 d=Dummy();m=SampledMeasurement(noise_std=3e-6,delay_steps=4,nominal=lambda t:.1*t);records=[]
 class Forbidden:
  def __float__(self):raise AssertionError('remaining_dot bypass')
 for i in range(1000):
  d.data.time=i*.001;m.update(d,0,Forbidden());e,v=m.measure(d,99);records.append((m.used_frame,e,v,m.record(i)['measurement_age_s']))
 fs=m.frames;ss=np.array([x['sample_step'] for x in fs]);assert len(fs)==132 and set(np.diff(ss))=={7,8}
 assert all(0<=x['quantization_s']<.001+1e-14 for x in fs)
 assert all(x['first_use_step']==x['available_step']>=x['sample_step'] for x in fs)
 for i,x in enumerate(fs):
  expect=5e-6*np.sin(2*np.pi*7*x['sample_s'])+x['noise_m'];assert abs(x['error_m']-expect)<1e-16
  if i:assert np.isclose(x['velocity_m_s'],(x['error_m']-fs[i-1]['error_m'])/(x['sample_s']-fs[i-1]['sample_s']))
 for i in range(1,len(records)):
  if records[i][0]==records[i-1][0]:assert records[i][1:3]==records[i-1][1:3]
 save(E/'adapter_nohardware_tests.json',dict(passed=True,frames=len(fs),interval_steps=sorted(set(np.diff(ss))),max_quantization_s=max(x['quantization_s'] for x in fs),delayed_first_use_exact=True,noise_and_velocity_held=True,moving_nominal_sample_time_error_exact=True,ideal_velocity_and_remaining_dot_tripwires_passed=True,physical_steps=0))
def historical():
 H=P.parent/'E组_双向预测微补偿_V1_20261014';ledger=read(H/'03_新运行/attempt_ledger.json')
 refs={}
 for method in ['baseline','sfc','additive','coupled']:
  row=next(x for x in ledger['attempts'] if x.get('method')==method and x.get('scene')=='ON')
  folder=H/row['path'];trace=folder/'trace.npz'
  refs[method]=dict(run_id=row['run_id'],path=str(folder),trace_sha256=sha(trace),arrays=trace)
 save(E/'R0_historical_sources.json',{k:{a:str(b) if isinstance(b,Path) else b for a,b in v.items()} for k,v in refs.items()})
 return refs
def integrate(ctx,budget,condition,method,settings,regression=False):
 path=RUN/(condition+'_'+method)
 if path.exists():raise FileExistsError('No automatic reruns '+str(path))
 path.mkdir();params=ctx['parameters'];period=settings['output_steps'];adapter=None if settings['velocity']=='original' else SampledMeasurement(settings['velocity'],settings['noise_m'],settings['delay_steps'])
 cls=TimedCoordinator if regression else Coordinator132
 ext=cls(method,ctx['c']['q'],params)
 loggercls=ELogger if regression else Logger132
 kwargs={} if regression else dict(adapter=adapter,hold=True)
 logger=loggercls(path/'trace.npz',10000,'C_ORDINARY',POSITION_EQUIV,period,ctx['input_id'],input_data=ctx['scaled'],common_correction=ctx['corr'],extension=ext,**kwargs)
 for k in ['coordination_full_s','model_s','ruckig_generation_s','trajectory_sampling_s','prediction_scoring_s']:logger.arr[k]=np.zeros(10000)
 logger.arr['dq_C_dot'][:]=ctx['scaled']['dqd'];logger.arr['nominal_qdot'][:]=np.gradient(ctx['c']['q'],.001,axis=0)
 identity=dict(run_id=new_id(),condition=condition,method=method,settings=settings,params=params,scene='ON',source_input_id=ctx['input_id'],sources=source_id(),online_predictor='e_predict.predict original160ms;Td32ms;coord32ms;w_gamma.01',offline_failed_r15_enabled=False,measurement_model='syntheticadapter,notcamera rendering',fast_joint_feedback='q/qd/J/M/bias at1ms available;modelFKfromq,notdirectmeasuredendpointvelocity',initial_velocity='0 until2 distinct samplingtimes;zeroerroruntilfirstavailableframe',final_command_hold='whole nominal+background+common+planning+SFC sixjointcommand')
 logger.meta.update(**{k:v for k,v in identity.items() if k!='sources'});save(path/'generation.json',identity)
 budget.start(condition,method,path);t=time.perf_counter()
 row=sweep.run_case(identity['run_id'],'A0' if method=='baseline' else params['mode'],params['K'],params['B0'],params['n'],params['mu'],params['r'],params['v_ref'],ctx['fd'],ctx['c']['along'],ctx['c']['q'],phase_code=ctx['c']['phase'],c_input=ctx['scaled'],execution_mode=POSITION_EQUIV,control_update_steps=period,precise_logger=logger,common_position_correction=ctx['corr']['common_delta_q'],e_extension=ext,measurement_adapter=adapter,final_command_hold=not regression)
 budget.finish(row);row['wall_s']=time.perf_counter()-t;save(path/'row.json',row)
 write=time.perf_counter();save(path/'decisions.json',ext.decisions);save(path/'sample_frames.json',adapter.frames if adapter else []);save(path/'write_timing.json',dict(decisions_and_frames_write_s=time.perf_counter()-write))
 a=arrays(path/'trace.npz');n=len(a['t'])
 checks=dict(samples=n,status=row['status'],completed=n==10000,actual_physical_steps=budget.data['attempts'][-1]['physical_steps'],no_duplicate_torque=np.array_equal(a['qfrc_applied'],a['qfrc_disturbance']) if n else True)
 if not regression and n:
  nonupdate=np.flatnonzero(~a['updated']);nonupdate=nonupdate[nonupdate>0]
  checks.update(final_command_exactly_held=np.array_equal(a['final_ctrl'][nonupdate],a['final_ctrl'][nonupdate-1]),command_composition_max_rad=float(np.max(abs(a['final_ctrl']-a['base_ctrl_submitted']-a['delta_q_sfc_actual']-a['delta_q_p_actual']))),causal=bool(np.all(a['available_step_used'][a['measurement_valid']>0]<=np.flatnonzero(a['measurement_valid']>0))),main_velocity_mode=settings['velocity'])
  assert checks['final_command_exactly_held'] and checks['command_composition_max_rad']<1e-12 and checks['causal']
 save(path/'validation.json',checks)
 print('RESULT',condition,method,row['status'],n,'physics',budget.data['used_physical_steps'],row['note'],flush=True)
 return path,row
def main():
 adapter_tests();ctx=context();refs=historical()
 design=dict(physics_dt_s=.001,measurement_hz=132,coordination_steps=32,prediction_steps=160,params=ctx['parameters'],source_input_id=ctx['input_id'],windows=read(B/'historical/public_slow_review/metric_windows.json')['windows'],conditions=[
  dict(id='R1',velocity='sampled_ideal',output_steps=1,noise_m=0.,delay_steps=0),
  dict(id='R2',velocity='difference',output_steps=1,noise_m=0.,delay_steps=0),
  dict(id='R3',velocity='difference',output_steps=8,noise_m=0.,delay_steps=0),
  dict(id='R4',velocity='difference',output_steps=8,noise_m=3e-6,delay_steps=0),
  dict(id='R5',velocity='difference',output_steps=8,noise_m=3e-6,delay_steps=4)],methods=['sfc','additive','coupled'],noise_seed=13220261015,stop_guards=dict(inherited_ABS_MAX_m=config.ABORT_E_ABS_MAX,inherited_QVEL_MAX=config.ABORT_QVEL_MAX,inherited_compiled_current_request_boundary=True,persistent_compiled_saturation_ms=20),remaining_ideal_velocity='R1samplingeventdiagnostic only;R2toR5neverused',delay='0/4ms artificial sensitivity;not exposure or hardware delay',old_predictor_rank_defect='failedholdout candidate ranking remains;offline trend/residual modelnotimported')
 save(E/'predeclared_matrix.json',design);save(E/'source_execution_freeze.json',source_id())
 with Budget() as budget:
  path,row=integrate(ctx,budget,'R0_regression','coupled',dict(velocity='original',output_steps=1,noise_m=0.,delay_steps=0),True)
  a=arrays(path/'trace.npz');old=arrays(Path(refs['coupled']['arrays']));fields=['physical_y','effective_y','physical_e_um','effective_e_um','physical_v','sample_v_m_s','F_requested','final_ctrl','p','p_star','gamma','gamma_star']
  checks={k:dict(exact=np.array_equal(a[k],old[k]),max_abs=float(np.max(abs(a[k]-old[k])))) for k in fields}
  passed=row['status']=='OK' and all(x['exact'] for x in checks.values());save(E/'ideal_disabled_regression.json',dict(passed=passed,fields=checks,historical=refs['coupled']['path'],new_path=str(path)))
  if not passed:raise RuntimeError('Ideal disabled regression mismatch;stopbeforematrix')
  # Fair8ms uncompensatedbaseline, ON commoncorrectionretained.
  integrate(ctx,budget,'B8','baseline',dict(velocity='difference',output_steps=8,noise_m=0.,delay_steps=0))
  for condition in design['conditions']:
   for method in design['methods']:integrate(ctx,budget,condition['id'],method,condition)
 save(E/'campaign_completed.json',dict(matrix_attempts=15,used_physical_steps=read(E/'physical_attempt_ledger.json')['used_physical_steps'],regression_passed=True,optional8msdelay_stress_notrun=True))
if __name__=='__main__':main()
