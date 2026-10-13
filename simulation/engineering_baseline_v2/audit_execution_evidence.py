"""R017 directed offline audit of already recorded C data. No robot imports/commands or fitting."""
from pathlib import Path
import sys,csv,json,bisect,math
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'F组首次接入_20261010/.runtime'));sys.path.insert(0,str(ROOT))
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from batch_identity import read,sha,atomic_json
from run_report import table
C=Path('D:/UR10_Data/20261008_190952');P=Path('D:/UR10_Data/20261010_141616/path_speed/path-six')
Q=ROOT.parent/'C组量化_V4.1_20261010';CODE=Path('C:/Users/kinli/Documents/Codex/2026-10-02/UR10_CB3_single_joint_pretest_local_acceptance/src/sj_pretest')
OUT=ROOT/'review/R017';OUT.mkdir(exist_ok=True)
def csvread(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def stat(x):
    x=np.array(x,float)
    if not len(x):return dict(n=0,mean=None,std=None,peak=None)
    return dict(n=len(x),mean=float(np.mean(x)),std=float(np.std(x)),peak=float(np.max(abs(x))))
def arrays(p):
    with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def main():
    ids=read(ROOT/'review/accepted_run_ids.json')['run_ids'];run=ROOT/'runs'/ids[0]
    source=read(Q/'逐次明细/C_cache_source.json');native=C/'robot_states.csv';cache=Q/'逐次明细/C_capture_robot.csv'
    if native.stat().st_size!=source['size'] or native.stat().st_mtime_ns!=source['mtime_ns']:raise RuntimeError('C source signature changed; do not infer from stale extraction')
    rows=csvread(cache)
    if any(r['synthetic']!='0' for r in rows):raise RuntimeError('Synthetic rows in measured C data')
    rows.sort(key=lambda x:int(x['host_ns']));stamps=[int(r['host_ns']) for r in rows]
    metadata=[];native_trials=[];unique_periods=[];all_windows=[];source_files={str(cache):sha(cache),str(Q/'逐次明细/C_cache_source.json'):sha(Q/'逐次明细/C_cache_source.json')}
    source_files[str(C/'config.json')]=sha(C/'config.json');source_files[str(C/'run_manifest.json')]=sha(C/'run_manifest.json')
    expected={sid:(a,b) for sid,a,b in source['windows']}
    for directory in sorted((C/'segments').glob('pretest-*')):
        p=directory/'capture_metadata.json';m=read(p);metadata.append(m);source_files[str(p)]=sha(p)
        if m['synthetic'] or expected.get(m['segment_id'])!=(m['first_host_ns'],m['last_host_ns']):raise RuntimeError('C extraction window provenance mismatch')
        rr=rows[bisect.bisect_left(stamps,m['first_host_ns']):bisect.bisect_right(stamps,m['last_host_ns'])]
        rr=list({float(r['robot_timestamp_s']):r for r in rr}.values());rr.sort(key=lambda r:float(r['robot_timestamp_s']))
        clock=np.array([float(r['robot_timestamp_s']) for r in rr]);dt=np.diff(clock);unique_periods.extend(dt[dt>0].tolist())
        j=m['joint'];key='actual_q_'+j+'_deg';target='command_q_'+j+'_deg'
        if not rr or any(not math.isfinite(float(r[key])) or not math.isfinite(float(r[target])) for r in rr):raise RuntimeError('Missing/nonfinite native paired joint samples')
        for phase in m['phases']:
            lo=phase['start_s'];hi=phase['end_s'];lo=(lo+hi)/2 if phase['label']=='hold' else lo
            selected=[r for r in rr if lo<=(int(r['host_ns'])-m['first_host_ns'])/1e9<=hi]
            e=[float(r[key])-float(r[target]) for r in selected];s=stat(e)
            tr=np.array([float(r[target]) for r in selected]);tt=np.array([float(r['robot_timestamp_s']) for r in selected]);diff=np.diff(tt)
            valid=(diff>0)&(diff<=.016001);v=np.diff(tr)[valid]/diff[valid] if len(tr)>1 else np.array([])
            native_trials.append(dict(segment=m['segment_id'],joint=j,amplitude_deg=m['amplitude_deg'],direction=m['direction'],repeat=m['repeat'],phase=phase['label'],
                window_t0_s=lo,window_t1_s=hi,metadata_path=str(p),native_source=str(native),extracted_source=str(cache),
                **{'error_'+k:val for k,val in s.items()},observed_internal_target_speed_peak_deg_s=float(np.max(abs(v))) if len(v) else None,
                target_kind='RTDE getTargetQ, not host final moveJ goal',deduplicated_robot_timestamps=True))
            if m['amplitude_deg']==.01 and m['repeat']==1 and phase['label'] in ['move','return']:
                for r in selected:all_windows.append(dict(segment=m['segment_id'],phase=phase['label'],joint=j,host_ns=r['host_ns'],robot_timestamp_s=r['robot_timestamp_s'],
                    time_s=(int(r['host_ns'])-m['first_host_ns'])/1e9,internal_target_q_deg=r[target],actual_q_deg=r[key],actual_minus_internal_target_deg=float(r[key])-float(r[target])))
    if len(metadata)!=72:raise RuntimeError('Expected72 measured micro trials')
    table(OUT/'C微动逐试次与分窗.csv',native_trials);table(OUT/'C最小幅值正反向原始窗口.csv',all_windows)
    params=read(ROOT/'inputs/仿真误差参数.json');source_files[str(ROOT/'inputs/仿真误差参数.json')]=sha(ROOT/'inputs/仿真误差参数.json')
    summaries=[]
    for j in ['J'+str(i) for i in range(1,7)]:
        ms=[m for m in metadata if m['joint']==j];hold=[r for r in native_trials if r['joint']==j and r['phase']=='hold'];moving=[r for r in native_trials if r['joint']==j and r['phase'] in ['move','return']]
        meanhold=float(np.mean([r['error_mean'] for r in hold]));stdhold=float(np.mean([r['error_std'] for r in hold]));stdmotion=float(np.mean([r['error_std'] for r in moving]))
        summaries.append(dict(joint=j,trials=len(ms),amplitudes_deg=','.join(str(x) for x in sorted({m['amplitude_deg'] for m in ms})),both_directions=True,repeats_each=2,
            mean_hold_error_deg=meanhold,mean_hold_std_deg=stdhold,mean_motion_std_deg=stdmotion,
            already_background_bias_deg=params['joints'][j]['bias_center']['deg']['value'],already_background_static_std_deg=params['joints'][j]['static_demeaned_std']['deg']['value'],
            already_background_dynamic_std_deg=params['joints'][j]['dynamic_demeaned_std']['deg']['value'],
            dynamic_background_source='Five path-six measured move straight-inner windows, not this micro move/return mean',not_independent_new_noise=True))
    table(OUT/'C覆盖与已入背景.csv',summaries)
    coverage=[]
    logs={k:arrays(run/'full_1ms'/f'{k}.npz') for k in ['A0','Alinear','ASFC_strong']}
    for j in range(6):
        result=dict(joint='J'+str(j+1),C_min_single_joint_move_deg=.01,C_max_single_joint_move_deg=.2,C_native_update_median_ms=float(np.median(unique_periods))*1000,F_update_ms=1.)
        for ctrl,a in logs.items():
            delta=a['common_delta_q'][:,j]+a['delta_u'][:,j];dqstep=np.diff(delta)
            result[ctrl+'_total_added_peak_deg']=float(np.max(abs(delta))*180/np.pi);result[ctrl+'_added_adjacent_speed_peak_deg_s']=float(np.max(abs(dqstep))/.001*180/np.pi)
        result['public_peak_deg']=float(np.max(abs(logs['A0']['common_delta_q'][:,j]))*180/np.pi)
        result['public_smooth_speed_peak_deg_s']=float(np.max(abs(logs['A0']['common_delta_q_dot'][:,j]))*180/np.pi)
        result['public_initial_step_deg']=float(logs['A0']['common_delta_q'][0,j]*180/np.pi)
        result['coverage_note']='Single-joint moveJ versus simultaneous6joint1ms additive position commands: not equivalent excitation/feedback/pose/command cadence.'
        coverage.append(result)
    table(OUT/'当前补偿与C覆盖关系.csv',coverage)
    path_rows=[];path_windows=[]
    for folder in sorted(P.glob('path-six-*')):
        p=folder/'matched_samples.csv';report=folder/'report.json'
        if not p.exists() or not report.exists():continue
        rep=read(report)
        if rep.get('status')!='analyzed' or rep.get('synthetic') is not False:continue
        source_files[str(p)]=sha(p);source_files[str(report)]=sha(report)
        data=csvread(p)
        # Matched rows duplicate RTDE samples at camera cadence; deduplicate for robot-only statistics.
        selected=[r for r in data if int(r['pair_valid'])==1 and int(r['visual_valid'])==1 and float(r['pair_gap_ms'])<=20]
        for phase in ['pre_motion','move','post_motion','return','post_return']:
            rows_phase=[r for r in selected if r['phase']==phase]
            if not rows_phase:continue
            rr=list({float(r['robot_timestamp_s']):r for r in rows_phase}.values());rr.sort(key=lambda r:float(r['robot_timestamp_s']))
            dt=np.diff([float(r['robot_timestamp_s']) for r in rr]);dt=dt[dt>0]
            y=[(float(r['actual_tcp_y'])-float(r['target_tcp_y']))*1e6 for r in rr]
            native_y=stat(y);vision=[float(r['visual_minus_actual_projection_norm_px']) for r in rows_phase]
            path_rows.append(dict(segment=rep['segment_id'],speed_mm_s=rep['speed_mm_s'],repeat=rep['repeat'],phase=phase,source=str(p),window_t0_s=min(float(r['time_s']) for r in rows_phase),window_t1_s=max(float(r['time_s']) for r in rows_phase),
                camera_paired_samples=len(rows_phase),unique_robot_samples=len(rr),median_unique_robot_dt_ms=float(np.median(dt))*1000 if len(dt) else None,max_pair_gap_ms=max(float(r['pair_gap_ms']) for r in rows_phase),
                **{'encoder_Y_error_'+k+'_um':v for k,v in native_y.items()},visual_minus_actual_norm_P95_px=float(np.percentile(vision,95)),
                observed_qd_peak_deg_s=max(abs(float(r[f'actual_qd_J{i}_deg_s'])) for r in rr for i in range(1,7)),
                scope='Encoder Y uses same RTDE row target/actual; vision uses host nearest-pair <=20ms, not independently calibrated exposure latency/3D ground truth.'))
            if rep['segment_id']=='path-six-v3-r1' and phase in ['move','return']:
                for r in rr: path_windows.append({k:r[k] for k in ['time_s','robot_timestamp_s','rtde_host_ns','camera_host_ns','pair_gap_ms','phase','actual_tcp_y','target_tcp_y','actual_tcp_z','target_tcp_z','actual_qd_J1_deg_s','command_q_J1_deg','actual_q_J1_deg','visual_minus_actual_projection_norm_px']})
    table(OUT/'C补测同窗执行观测.csv',path_rows);table(OUT/'C补测v3r1原始同窗.csv',path_windows)
    for n in ['robot_joint.py','path_report.py']:
        source_files[str(CODE/n)]=sha(CODE/n)
        # Preserve code evidence as text, never import a hardware module.
        text=(CODE/n).read_text(encoding='utf-8');(OUT/(n+'.txt')).write_text(text,encoding='utf-8')
    manifest=dict(passed=True,C_micro_trials=72,path_trials=len({r['segment'] for r in path_rows}),C_native_unique_robot_dt_median_ms=float(np.median(unique_periods))*1000,
        C_extraction_source_signature_matches=True,C_native_source=dict(path=str(native),bytes=native.stat().st_size,mtime_ns=native.stat().st_mtime_ns,
            full_3_7GB_file_not_rehashed=True,scope='Existing capture extraction checked against recorded size/mtime and all72 metadata windows; no claim of a fresh whole-file content hash.'),
        files_sha256=source_files,run_id_for_F_amplitude_coverage=ids[0],already_background_summary=summaries,coverage=coverage,
        inference='No compensation-specific execution model or parameter identified; no new noise/algorithm/data collection.')
    atomic_json(OUT/'evidence_manifest.json',manifest)
    evidence=[
        dict(phenomenon='单关节微动的内部目标—实际角差、保持偏差与动作波动',source='D:/UR10_Data/20261008_190952/segments/pretest-*/capture_metadata.json；robot_states.csv既有capture提取',window='72试次，0.01/0.05/0.2deg，正负各2次；pre/move/hold后半/return/post',current_coverage='保持偏置/静态统计已入冻结六关节背景；本次微动动作差另列，当前动态std来自补测直线内窗；MuJoCo也有伺服跟随',missing_relation='总误差无法独立拆出补偿专属执行误差；目标是RTDE内部目标，不是每1ms外部补偿指令',constraints='可约束当前姿态/接口下观测量级与重复性；不能辨识1ms小补偿Kp/Kd、死区或回差'),
        dict(phenomenon='弯曲往返的关节/TCP目标—实际及实际速度',source='D:/UR10_Data/20261010_141616/path_speed/path-six/*/matched_samples.csv',window='5个analyzed实测，1/3mm/s各2次、5mm/s1次；按有效同窗且RTDE时间去重',current_coverage='路径动作角差参与原动态背景量级；当前伺服含惯性/阻尼/耦合',missing_relation='moveL整路径不是总目标上每1ms叠加6关节微纠偏；姿态/轨迹不同，控制器内部目标≠外部发送流',constraints='可约束该路径下内部目标跟随量级/速度，不能移植为小补偿传递函数或精确执行增益'),
        dict(phenomenon='视觉—编码器局部投影剩余',source='C组量化_V4.1_20261010/逐次明细/C逐次统计.json；补测matched_samples及path_report.py',window='主试次固定局部映射；补测host最近邻≤20ms有效对',current_coverage='已压缩进remaining等效输出项并冻结，频率/相关生成有假设',missing_relation='含相机、时间与二维投影影响，非纯机械柔性或补偿误差；A时间问题不推广成全部C无效',constraints='可描述投影观测，不能独立标定三维TCP或另加同样末端噪声'),
        dict(phenomenon='tau/Kp同状态执行器增量几乎精确',source='position_execution.py；本轮full_1ms有符号tau_requested/qfrc_actuator/base_qfrc_actuator/tau_realized_increment',window='两批10秒1ms无饱和基线',current_coverage='已知同一仿射执行器在同一状态下增量相符；真实q需通过动力学推进',missing_relation='不能证明真实CB3小纠偏准确；实际q−final_ctrl是总跟踪差，不是补偿独有差',constraints='只验证模型内代数适配；未由现有C资料辨识真实小补偿执行关系'),
        dict(phenomenon='公共慢纠偏幅值、强SFC快速换向与更新周期',source='review/R017/当前补偿与C覆盖关系.csv',window='本轮公共最大单关节约0.001845deg；全部实际值逐关节列出，另列初始阶跃',current_coverage='总位置目标共享名义伺服与机械动力学；原背景保持',missing_relation='多数公共量小于C最小0.01deg；部分抗抖峰落在C范围也不等于频率/同时多轴/更新周期覆盖',constraints='1ms外部命令下执行误差不能从8ms内部目标采样、moveJ/moveL重复动作直接辨识')]
    table(OUT/'执行误差证据表.csv',evidence)
    print('R017 existing data audit:',manifest['C_micro_trials'],'micro trials;',manifest['path_trials'],'path trials; no new model or robot operation',flush=True)
if __name__=='__main__':main()
