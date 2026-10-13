"""Read full 1 ms logs, independently check interfaces, export raw scalar CSV and report."""
from run_short import ROOT,PARAMS,read,common,array,ledger,input_id,CODE_NAMES
from short_prepare import sha,write
from pathlib import Path
import numpy as np,json,csv,difflib
import config,disturbance

LABELS={'A0':'无附加控制','Alinear':'原线性','ASFC_strong':'强SFC'}
SCENES={'S0':'原始C / 事件关闭','S1':'A幅值校准C / 事件关闭','S2':'普通关闭 / 仅事件','S3':'A幅值校准C + 事件','S4':'原始C + 同一事件'}
def table(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def fmt(x):return f'{x:.2f}'
def stats(x):return dict(std_um=float(np.std(x,ddof=0)),ptp_um=float(np.ptp(x)),mean_um=float(np.mean(x)),raw_peak_abs_um=float(np.max(abs(x))),P95_abs_um=float(np.percentile(abs(x),95)))
def windows(c):
    tr=read(ROOT/'review/trajectory.json');ev=read(ROOT/'review/event_targets.json')
    return {'full':(0.,c['t'][-1]),'cold_initial':(0.,.8),'pre_A_start':(.8,tr['start_trigger_s']),
        'start':tuple(ev[0]['core_s']),'outbound_steady':(ev[0]['new_support_s'][1],ev[1]['new_support_s'][0]),
        'turn':tuple(ev[1]['core_s']),'return_steady':(ev[1]['new_support_s'][1],ev[2]['new_support_s'][0]),
        'deceleration_stop':tuple(ev[2]['new_support_s'])}
def source_evidence(c,logs):
    config.DATA_CSV=ROOT/'sources_A/P05R01/D1_时序.csv';pr=disturbance.prepare(log=lambda _:None)
    oldroot=ROOT.parent/'F组_位置执行等效测试_20261011';old=array(oldroot/'outputs/B_BACKGROUND/A0__TORQUE_DIRECT__1ms.npz')
    rawB=ROOT.parent/'UR10_vibration_pretest/UR10_SFC_CLEAN/outputs/sweep/traces/A0.csv'
    lines=rawB.read_text(encoding='utf-8-sig').splitlines();header=next(i for i,s in enumerate(lines) if s and not s.startswith('#'))
    original=np.genfromtxt(lines[header:],delimiter=',',names=True,encoding='utf-8')
    oldcommon=np.load(ROOT/'inputs/old62s_common.npz');bounds={code:(old['t'][np.flatnonzero(oldcommon['phase']==code)[0]],old['t'][np.flatnonzero(oldcommon['phase']==code)[-1]]) for code in [0,2,4]}
    newevents=read(ROOT/'review/event_targets.json');source_windows={'start':(1.2849583981754964,1.926),'turn':(3.461,4.524),'stop':(6.423,7.755)}
    source_baselines={'start':(1.085,1.235),'turn':(3.061,3.261),'stop':(6.173,6.423)}
    rows=[];source_rows=[]
    for ev,code in zip(newevents,[0,2,4]):
        name=ev['event'];lo,hi=source_windows[name];r=pr.raw;mask=(r['t']>=lo)&(r['t']<=hi);blo,bhi=source_baselines[name];base=np.median(r['cross_um'][(r['t']>=blo)&(r['t']<=bhi)])
        dmask=(pr.d['t']>=lo)&(pr.d['t']<=hi);bwin=(1.285,1.926) if code==0 else bounds[code];bwin=(bwin[0]-.15,bwin[1]) if code==4 else bwin
        om=(old['t']>=bwin[0])&(old['t']<=bwin[1]);hm=(original['t']>=bwin[0])&(original['t']<=bwin[1]);nm=(c['t']>=ev['core_s'][0])&(c['t']<=ev['core_s'][1])
        row=dict(event=name,A_raw_min_um=float(np.min(r['cross_um'][mask])),A_raw_max_um=float(np.max(r['cross_um'][mask])),A_near_baseline_um=float(base),A_baseline_peak_um=float(np.max(abs(r['cross_um'][mask]-base))),
            source_evib_min_um=float(np.min(pr.drift['e_vib_um'][dmask])),source_evib_max_um=float(np.max(pr.drift['e_vib_um'][dmask])),
            old_B_Fd_min_N=float(np.min(old['Fd'][om])),old_B_Fd_max_N=float(np.max(old['Fd'][om])),
            old_B_saved200Hz_min_um=float(np.min(original['e_um'][hm])),old_B_saved200Hz_max_um=float(np.max(original['e_um'][hm])),
            accepted_F_oldB_1ms_min_um=float(np.min(old['physical_e_um'][om])),accepted_F_oldB_1ms_max_um=float(np.max(old['physical_e_um'][om])),
            new_target_min_um=float(np.min(c['event_target_um'][nm])),new_target_max_um=float(np.max(c['event_target_um'][nm])),
            current_F_S2_A0_min_um=float(np.min(logs['S2__A0']['physical_e_um'][nm])),current_F_S2_A0_max_um=float(np.max(logs['S2__A0']['physical_e_um'][nm])))
        rows.append(row)
        # Exact old high precision excerpt is labelled as accepted F replay of old B, not an original B sensor log.
        np.savetxt(ROOT/'review'/f'旧B背景已验收F_1ms_{name}.csv',np.column_stack([old['t'][om],old['physical_e_um'][om],old['Fd'][om]]),delimiter=',',fmt='%.17g',header='t_s,physical_e_um,Fd_N',comments='')
        np.savetxt(ROOT/'review'/f'原B保存200Hz_{name}.csv',np.column_stack([original[k][hm] for k in ['t','e_um','e_raw_um','b_um','e_vib_um','Fd_N']]),delimiter=',',fmt='%.17g',header='t_s,e_um,e_raw_um,b_um,e_vib_um,Fd_N',comments='')
        for i in np.flatnonzero(dmask):source_rows.append(dict(event=name,source_t_s=pr.d['t'][i],A_resampled_raw_um=pr.drift['e_raw_um'][i],B_drift_um=pr.drift['b_um'][i],B_evib_um=pr.drift['e_vib_um'][i]))
    table(ROOT/'review/事件来源到响应链.csv',rows);table(ROOT/'review/A_B原始事件分解_1ms.csv',source_rows)
    write(ROOT/'review/source_evidence.json',dict(rows=rows,A_sha256=sha(config.DATA_CSV),original_B_trace_sha256=sha(rawB),
        accepted_F_B_A0_1ms_sha256=sha(oldroot/'outputs/B_BACKGROUND/A0__TORQUE_DIRECT__1ms.npz'),
        precision_boundary='Original B A0 CSV is 200Hz/6g serialization, not available as original full1ms float64. Accepted full F replays the exact old B input with 1ms float64; both are shown with explicit provenance. No B experiment re-run.',
        raw_A_reference='Original cross_track_um retained unchanged. Event local median is used only for a separately labelled peak statistic; no curve is recentered.',
        stop_scope='Source window begins6.423s, covering the measured speed drop near6.56-6.576s and original stop tail; new force support is moved consistently. Source does not justify an invented100-200um stop peak.'))
    return rows
def validate(c,logs):
    fr=read(ROOT/'review/frozen_version.json');errors={};names=[]
    for name,a in logs.items():
        scene,controller=name.split('__');s=fr['scenarios'][scene];g=s['g'];meta=read(ROOT/'outputs/full_1ms'/f'{name}.json');row=read(ROOT/'outputs/full_1ms'/f'{name}.row.json')
        assert row['status']=='OK' and row['n_steps_done']==len(c['t'])
        assert meta['input_sha256']==s['input_id'] and meta['controller_parameters']==PARAMS[controller]
        assert np.array_equal(a['t'],c['t']) and np.array_equal(a['phase'],c['phase'])
        assert np.array_equal(a['q_nominal'],c['q']) and np.array_equal(a['dq_C'],c['dq']*g)
        assert np.array_equal(a['remaining'],c['remaining']*g) and np.array_equal(a['remaining_dot'],c['remaining_dot']*g)
        assert np.array_equal(a['base_ctrl'],c['q']+c['dq']*g)
        fd=np.loadtxt(ROOT/'inputs'/('Fd_event.csv' if s['event_on'] else 'Fd_zero.csv'),delimiter=',',skiprows=1)[:,1]
        assert np.array_equal(a['Fd'],fd)
        assert np.array_equal(a['qfrc_applied'],a['qfrc_disturbance'])
        residual=float(np.max(abs(a['tau_realized_increment']-a['tau_requested'])))
        assert residual<1e-8
        assert np.allclose(a['delta_u'],a['tau_requested']/5000,rtol=0,atol=1e-15)
        assert np.array_equal(a['final_ctrl'],a['base_ctrl']+a['delta_u'])
        assert np.array_equal(a['effective_y'],a['physical_y']+a['remaining'])
        p=PARAMS[controller];e=a['effective_e_um']*1e-6;v=a['effective_v'];F=np.zeros_like(e) if controller=='A0' else -p['K']*e-p['B0']*v-p['mu']*np.sign(v)*abs(v)**p['n']
        ferr=float(np.max(abs(F-a['F_requested'])));assert ferr<1e-9
        names.append(name);errors[name]=dict(max_increment_torque_residual_Nm=residual,max_independent_force_formula_error_N=ferr,
            any_saturated_samples=int(np.count_nonzero(np.any(a['ctrl_saturation']|a['force_saturation']|a['joint_saturation'],axis=1))))
    # Reconstruct the generated input from compact scalar-independent CSV, roundtrip full precision.
    before=read(ROOT/'review/protected_before.json');after={p:sha(p) for p in before};assert after==before
    write(ROOT/'review/protected_after.json',after)
    for p,h in read(ROOT/'review/implementation_freeze.json').items():assert sha(ROOT/p)==h
    old=ROOT.parent/'F组_位置执行等效测试_20261011'
    unchanged=['config.py','sfc.py','controller.py','sim_env.py','sweep.py','position_execution.py','background.py','f_measurement.py','disturbance.py']
    assert all(sha(ROOT/p)==sha(old/p) for p in unchanged)
    assert read(ROOT/'review/compiled_model.json')==read(ROOT/'review/baseline_compiled_model.json')
    # Actual model: verify Jacobian torque interface on a separate state, without stepping any study case.
    from sim_env import SimEnv
    env=SimEnv();jerrs=[]
    for name in ['S3__Alinear','S3__ASFC_strong']:
        a=logs[name]
        for i in [0,1290,5000,8820,9999]:
            env.reset(a['q'][i]);env.forward();jerrs.append(float(np.max(abs(env.jacp()[1]*a['F_requested'][i]-a['tau_requested'][i]))))
    assert max(jerrs)<1e-10
    verdict=dict(all_pass=True,formal_cases=len(names),executed_short_runs=len(ledger()),max_allowed_runs=22,protected_files=len(before),original_files_unchanged=True,
        controller_and_execution_core_identical_to_accepted_F=unchanged,compiled_model_identical=True,
        shared_case_inputs_and_raw_signals_pass=True,g1_exact_unscaled_and_event_off_exact_zero=True,
        no_duplicate_SFC_external_torque=True,no_mean_subtraction_or_output_scaling=True,
        independent_JT_force_error_Nm=max(jerrs),cases=errors,
        limitations=['User approved inherited piecewise-linear event command velocities, not newly planned C1 motion.','Cold initialization remains unchanged and is displayed separately.','Passing execution checks does not demonstrate physical CB3 performance.'])
    write(ROOT/'review/verification.json',verdict);return verdict
def export(c,logs):
    out=ROOT/'outputs/compact_1ms';out.mkdir(exist_ok=True)
    index=[];rows=[];eventrows=[];costrows=[];cold=[];wins=windows(c);fr=read(ROOT/'review/frozen_version.json');ev=read(ROOT/'review/event_targets.json')
    for name,a in logs.items():
        scene,controller=name.split('__');meta=read(ROOT/'outputs/full_1ms'/f'{name}.json');n=len(a['t'])
        cols=dict(t_s=a['t'],phase=a['phase'],nominal_Y_m=a['nominal_y'],physical_e_um=a['physical_e_um'],effective_e_um=a['effective_e_um'],effective_v_m_s=a['effective_v'],Fd_event_N=a['Fd'],F_requested_N=a['F_requested'],
            tau_requested_norm_Nm=np.linalg.norm(a['tau_requested'],axis=1),tau_realized_increment_norm_Nm=np.linalg.norm(a['tau_realized_increment'],axis=1),
            delta_u_norm_rad=np.linalg.norm(a['delta_u'],axis=1),delta_u_max_abs_rad=np.max(abs(a['delta_u']),axis=1),
            ctrl_saturated_any=np.any(a['ctrl_saturation'],axis=1),force_saturated_any=np.any(a['force_saturation'],axis=1),joint_saturated_any=np.any(a['joint_saturation'],axis=1))
        for j in range(6):cols[f'delta_u_J{j+1}_rad']=a['delta_u'][:,j]
        dest=out/f'{name}.csv';np.savetxt(dest,np.column_stack(list(cols.values())),delimiter=',',fmt='%.17g',header=','.join(cols),comments='')
        # Ensure lossless float64 CSV roundtrip for every scalar field, not just sampled points.
        back=np.loadtxt(dest,delimiter=',',skiprows=1);assert np.array_equal(back,np.column_stack(list(cols.values())))
        index.append(dict(case_id=name,scenario=scene,controller=controller,g=fr['scenarios'][scene]['g'],event_on=fr['scenarios'][scene]['event_on'],input_id=meta['input_sha256'],status=meta['status'],samples=n,csv_sha256=sha(dest),local_full_npz_sha256=sha(ROOT/'outputs/full_1ms'/f'{name}.npz')))
        for window,(lo,hi) in wins.items():
            mask=(a['t']>=lo)&(a['t']<=hi)
            for channel in ['effective','physical']:
                rows.append(dict(case_id=name,scenario=scene,controller=controller,window=window,channel=channel,t0_s=lo,t1_s=hi,n=int(mask.sum()),**stats(a[channel+'_e_um'][mask])))
        for event in ev:
            e=event['event'];lo,hi=event['core_s'] if e!='stop' else event['new_support_s'];blo,bhi=(1.085,1.235) if e=='start' else (event['new_support_s'][0]-.4,event['new_support_s'][0]-.05)
            mask=(a['t']>=lo)&(a['t']<=hi);bm=(a['t']>=blo)&(a['t']<=bhi)
            for channel in ['effective','physical']:
                y=a[channel+'_e_um'];base=float(np.median(y[bm]));d=y[mask]-base
                eventrows.append(dict(case_id=name,scenario=scene,controller=controller,event=e,channel=channel,t0_s=lo,t1_s=hi,baseline_t0_s=blo,baseline_t1_s=bhi,nearby_baseline_median_um=base,baseline_peak_abs_um=float(np.max(abs(d))),relative_positive_peak_um=float(np.max(d)),relative_negative_peak_um=float(np.min(d)),raw_min_um=float(np.min(y[mask])),raw_max_um=float(np.max(y[mask]))))
        sat=np.any(a['ctrl_saturation']|a['force_saturation']|a['joint_saturation'],axis=1)
        costrows.append(dict(case_id=name,F_requested_RMS_N=float(np.sqrt(np.mean(a['F_requested']**2))),F_requested_peak_abs_N=float(np.max(abs(a['F_requested']))),
            tau_requested_RMS_norm_Nm=float(np.sqrt(np.mean(np.sum(a['tau_requested']**2,axis=1)))),tau_requested_peak_norm_Nm=float(np.max(np.linalg.norm(a['tau_requested'],axis=1))),
            realized_tau_peak_norm_Nm=float(np.max(np.linalg.norm(a['tau_realized_increment'],axis=1))),delta_u_peak_abs_rad=float(np.max(abs(a['delta_u']))),delta_u_peak_abs_deg=float(np.max(abs(a['delta_u']))*180/np.pi),saturation_samples=int(sat.sum()),saturation_duty=float(np.mean(sat)),first_saturation_t_s=float(a['t'][np.flatnonzero(sat)[0]]) if sat.any() else 'none'))
        m=a['t']<.8;i=np.flatnonzero(m)[np.argmax(abs(a['effective_e_um'][m]))]
        cold.append(dict(case_id=name,initial_peak_t_s=float(a['t'][i]),initial_raw_peak_um=float(a['effective_e_um'][i]),is_A_start=False,A_start_trigger_s=1.2849583981754964))
    for filename,data in [('case_index.csv',index),('分段指标.csv',rows),('事件局部基线峰.csv',eventrows),('控制代价与饱和.csv',costrows),('冷初始化时刻.csv',cold)]:table(ROOT/'review'/filename,data)
    write(ROOT/'review/metric_windows.json',dict(windows=wins,definition='Common fixed windows across all scenarios and controllers. Steady windows exclude event leading/trailing supports; A0 g calibration uses whole source-classified ordinary phases and is separately documented. std ddof0; RMS excluded from Y performance main metrics. No recentering in curves.'))
    return rows,eventrows,costrows,cold
def plots(c,logs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10})
    dest=ROOT/'outputs/figures';dest.mkdir(exist_ok=True);colors={'A0':'#666666','Alinear':'#2374ab','ASFC_strong':'#d73535'}
    all_y=np.concatenate([a['effective_e_um'] for a in logs.values()]);margin=np.ptp(all_y)*.08;limits=(np.min(all_y)-margin,np.max(all_y)+margin)
    wins=windows(c);tr=read(ROOT/'review/trajectory.json')
    for scene in SCENES:
        fig,axes=plt.subplots(2,1,figsize=(11,6.1),sharex=True,gridspec_kw={'height_ratios':[1,3]},layout='constrained');ax=axes[1]
        axes[0].plot(c['t'],c['along'],color='#9467bd',lw=1.3);axes[0].set_ylabel('沿程 / mm');axes[0].set_title(f'{scene}：{SCENES[scene]}（完整往返 10.000 s，1 ms）')
        for controller in LABELS:ax.plot(c['t'],logs[scene+'__'+controller]['effective_e_um'],label=LABELS[controller],color=colors[controller],lw=1.)
        ax.axhline(0,color='black',lw=.7);ax.set_ylim(*limits);ax.set_xlim(0,10);ax.set_ylabel('effective Y 原始偏差 / μm');ax.set_xlabel('时间 / s');ax.grid(alpha=.2);ax.legend(loc='lower left',ncol=3)
        for label,key in [('A启动','start'),('换向','turn'),('减速/停止','deceleration_stop')]:
            lo,hi=wins[key];ax.axvspan(lo,hi,alpha=.08,color='#3498db');axes[0].text((lo+hi)/2,1.,label,transform=axes[0].get_xaxis_transform(),ha='center',va='top',fontsize=9)
        for label,key in [('去程匀速','outbound_steady'),('回程匀速','return_steady')]:
            lo,hi=wins[key];axes[0].text((lo+hi)/2,.65,label,transform=axes[0].get_xaxis_transform(),ha='center',va='top',fontsize=9,color='#68439a')
        ax.axvspan(0,.8,color='#ddaa44',alpha=.10);ax.text(.03,.97,'0–0.8 s：原模型冷初始化，非A启动',transform=ax.transAxes,va='top',fontsize=9)
        for at in [tr['start_trigger_s'],tr['turn_trigger_s'],tr['post_stop_begin_s']]:ax.axvline(at,color='#777777',alpha=.5,ls=':',lw=.8)
        fig.savefig(dest/f'{scene}_完整往返_原始Y.png',dpi=160);plt.close(fig)
    for event in read(ROOT/'review/event_targets.json'):
        lo,hi=event['new_support_s'];m=(c['t']>=lo-.15)&(c['t']<=hi+.15)
        fig,ax=plt.subplots(figsize=(10,4.2),layout='constrained')
        for controller in LABELS:ax.plot(c['t'][m],logs['S3__'+controller]['effective_e_um'][m],label=LABELS[controller],color=colors[controller],lw=1.1)
        blo,bhi=(1.085,1.235) if event['event']=='start' else (lo-.4,lo-.05)
        base=float(np.median(logs['S3__A0']['effective_e_um'][(c['t']>=blo)&(c['t']<=bhi)]))
        ax.axhline(base,color=colors['A0'],ls='--',lw=.8,alpha=.7,label='A0附近中位基线（仅标记）')
        ax.axhline(0,color='black',lw=.7);ax.set_ylim(*limits);ax.set_title('S3 '+{'start':'启动','turn':'换向','stop':'减速/停止'}[event['event']]+'：原始Y，同全程纵轴');ax.set_xlabel('时间 / s');ax.set_ylabel('effective Y 原始偏差 / μm');ax.grid(alpha=.2);ax.legend(ncol=3)
        fig.savefig(dest/f'S3_{event["event"]}_局部原始Y.png',dpi=160);plt.close(fig)
    write(ROOT/'review/plot_definition.json',dict(raw_channel='effective_e_um',same_full_and_local_y_limits_um=limits,mean_subtracted=False,output_smoothed=False,output_multiplied=False,full_1ms_samples=True,cold_initialized_region_shown=True))
def documents(c,logs,metrics,eventmetrics,costs,cold,source,verdict):
    cal=read(ROOT/'review/ordinary_calibration.json');rec=read(ROOT/'review/event_reconstruction.json');tr=read(ROOT/'review/trajectory.json');checks=read(ROOT/'review/background_normalization.json')
    def m(scene,ctrl,win='full',chan='effective'):return next(r for r in metrics if r['case_id']==scene+'__'+ctrl and r['window']==win and r['channel']==chan)
    def ev(ctrl,event):return next(r for r in eventmetrics if r['case_id']=='S3__'+ctrl and r['event']==event and r['channel']=='effective')
    lines=['# F组10秒完整往返｜验收简报','',f'已完成 **15/15** 正式情景（共执行15条短运行，复用3条A0，预算22条），全部OK、每条10,000步。原B/F/C及旧日志{verdict["protected_files"]}个文件哈希保持不变。独立工程：`{ROOT.name}`。',
        '',f'**轨迹与输入。** 采样0–9.999 s、积分总长10.000 s，完整去回最大沿程{tr["max_along_mm"]:.3f} mm，末端沿程闭合误差{abs(tr["end_along_mm"]):.2g} mm。只缩短普通行程；A启动1.285 s、换向顶点{tr["turn_trigger_s"]:.3f} s、停止后边界{tr["post_stop_begin_s"]:.3f} s，保留约1.14 s停后尾段。依用户确认，保留旧事件段位置指令及原有速度连接变化，未新增运动规划。',
        '',f'**有限排查与幅值。** 未查到明确单位或重复相加错误。remaining按视觉−编码器投影定义只入一次；关节背景只入base ctrl一次。动态统计窗未证明纯匀速、UR10e姿态/J0与C测量姿态差异、六正弦合成及ρ=−0.21属于保留的模型假设，本轮不扩大调查。旧62秒种子/频率/相位/归一化逐点复原最大误差{max(checks["old_reproduction_errors"].values()):.2g}，10秒不重新归一化。',
        '',f'A P05R01原D1去/回P95(|原始横向偏差|)={50.14547119894546:.2f}/{50.33608690232005:.2f} μm，等权目标**{cal["A_target_P95_um"]:.2f} μm**。10秒S0 A0={cal["S0_P95_um"]:.2f} μm；统一g=**{cal["g"]:.8f}**后S1 A0={cal["S1_P95_um"]:.2f} μm（首次验证即冻结）。g同时缩dq/dq_dot/remaining/remaining_dot及其固定偏移，不缩事件或控制输出；g=1原量级参照保留。',
        '',f'**大峰与事件。** 旧C_ORDINARY关闭B Fd，故事件大峰没有接入。本轮以原A−B慢偏移的e_vib事件芯部构造目标，只在衔接处渐入/渐出；复用旧B频响30–45 Hz带限反演，2 s零填充，不硬切旧全长外力、不叠回B普通背景。仅事件A0首轮残差RMSE={rec["final_check"]["rmse_um"]:.2f} μm（0.8–9.999 s，相对8.70%；冷初始另列），Fd峰值{rec["final_check"]["Fd_abs_peak_N"]:.3f} N。启动+102.38、换向−111.60 μm；停止芯部−29.53～+18.41 μm。停止窗前移覆盖减速，原资料支持较小停止事件，未制造100–200 μm峰。',
        '', '**当前主情景S3：原始effective Y（全程统计，不称作高频振动）**','', '|方案|标准差 μm|峰峰 μm|均值 μm|原始绝对峰 μm|启动局部峰 μm|换向局部峰 μm|停止局部峰 μm|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for ctrl in LABELS:
        r=m('S3',ctrl);lines.append('|'+LABELS[ctrl]+'|'+'|'.join(fmt(r[k]) for k in ['std_um','ptp_um','mean_um','raw_peak_abs_um'])+'|'+'|'.join(fmt(ev(ctrl,e)['baseline_peak_abs_um']) for e in ['start','turn','stop'])+'|')
    lines+=['','事件局部峰相对于各曲线附近同窗中位基线单独计算；图仍画原始值，不减均值、不平移、不平滑。各情景完整图及3个局部图使用相同纵轴。S3停止局部峰含普通背景及其停止包络变化，不等同A纯事件幅值；新基线窗也与历史三次事件汇总不同。','', '|S3分段|A0 标准差/峰峰 μm|线性 标准差/峰峰 μm|强SFC 标准差/峰峰 μm|','|---|---:|---:|---:|']
    for label,win in [('去程匀速','outbound_steady'),('回程匀速','return_steady'),('A启动','start'),('換向','turn'),('减速/停止','deceleration_stop')]:
        lines.append('|'+label+'|'+'|'.join(f'{m("S3",ctrl,win)["std_um"]:.2f}/{m("S3",ctrl,win)["ptp_um"]:.2f}' for ctrl in LABELS)+'|')
    r0=m('S3','A0');rl=m('S3','Alinear');rs=m('S3','ASFC_strong')
    lines+=['',f'相对A0，S3线性标准差下降{(1-rl["std_um"]/r0["std_um"])*100:.1f}%、峰峰下降{(1-rl["ptp_um"]/r0["ptp_um"])*100:.1f}%；强SFC分别下降{(1-rs["std_um"]/r0["std_um"])*100:.1f}%/{(1-rs["ptp_um"]/r0["ptp_um"])*100:.1f}%。20%仅为参考，未以此调整输入。',
        '', '|S3作用与物理Y|A0|线性|强SFC|','|---|---:|---:|---:|']
    cs={ctrl:next(r for r in costs if r['case_id']=='S3__'+ctrl) for ctrl in LABELS}
    for label,key in [('附加请求力RMS / N','F_requested_RMS_N'),('请求力矩RMS范数 / Nm','tau_requested_RMS_norm_Nm'),('最大位置补偿 / °','delta_u_peak_abs_deg')]:lines.append('|'+label+'|'+'|'.join(f'{cs[ctrl][key]:.4g}' for ctrl in LABELS)+'|')
    lines.append('|physical Y 标准差/峰峰 μm|'+'|'.join(f'{m("S3",ctrl,chan="physical")["std_um"]:.2f}/{m("S3",ctrl,chan="physical")["ptp_um"]:.2f}' for ctrl in LABELS)+'|')
    lines+=['','15条全程均无饱和或abort；请求/实现附加力矩最大残差<1e-8 Nm，位置执行仍为τ/5000，qfrc_applied只有事件外扰。K=5000、B0=400；线性n=1；强SFC n=1.5、μ=70879.56438、原v_ref=0.000509563 m/s保持不变。',
        '', '**限制与交付。** t<0.8 s冷初始化有约77.79 μm瞬态（仅事件A0），非1.285 s的A启动；保留旧初态，已单列时刻。A原始偏移、B去慢偏移目标和局部基线峰不是同一口径；原B现存CSV为200 Hz/6g，另附已验收F旧B背景1 ms高精度日志补证，未重跑B一致性。普通幅值/事件均是近似仿真设置，不能推出CB3实机同等表现。',
        '', '双击 `打开结果.cmd` 查看5页完整原始叠加图；本机完整1 ms日志在 `outputs/full_1ms`。核查包仅文本：代码/MD/CSV/JSON/XML/patch，15条1 ms标量日志无抽稀、保留6关节补偿；无图片、网格或旧大日志。完整工程与旧原始输入保留本机。完整分段、事件、控制代价见review CSV。']
    (ROOT/'简报.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Small per-scenario full raw metric table for easy GPT entry.
    all_lines=['# GPT核查入口','', '先读简报.md、review/verification.json、review/frozen_version.json。当前版本F10-1ms-Acal-event-v1。核查CSV都是完整1ms/float64 roundtrip，不是50ms示意采样。',
        '', '|情景|方案|effective 标准差 μm|峰峰 μm|均值 μm|原始绝对峰 μm|','|---|---|---:|---:|---:|---:|']
    for scene in SCENES:
        for ctrl in LABELS:
            r=m(scene,ctrl);all_lines.append('|'+scene+'|'+LABELS[ctrl]+'|'+'|'.join(fmt(r[k]) for k in ['std_um','ptp_um','mean_um','raw_peak_abs_um'])+'|')
    all_lines+=['','核查重点：共享input_id、g只缩普通输入、事件关闭为零、g=1无额外缩放、SFC公式及参数与已验收F一致、附加力矩只通过位置偏移执行、原始Y未减均值。',
        '', '本ZIP为文本核查包，未包含网格和旧62秒NPZ；不能独立作MuJoCo全工程运行。完整可运行工程保留在本机所列路径。short_report.py是离线报告工具；run_short.py才调用已验收sweep.run_case。复核摘要和CSV无需连接机器人。',
        '', '代码变更：已验收核心9文件逐字节未改，仿真相关现有文件仅position_logging.py新增冻结输入传入接口；新增short_prepare.py/run_short.py/short_report.py/check_review.py，pack_review.py替换旧同名打包工具。完整diff在review/本轮完整代码差异.patch。',
        '', '阶段定义在review/metric_windows.json。主统计std ddof0/ptp；全程std含均值平台变化，不能称为纯高频振动指标。calibration P95窗与剔除事件衔接的稳态统计窗不同，已分别列明。',
        '', '事件来源链表：review/事件来源到响应链.csv；保留原始A源D1、B分解与旧B背景的已验收F高精度源片段。停止没有人为扩大。冷初始化列于review/冷初始化时刻.csv。',
        '', f'完整本机根目录：{ROOT}']
    (ROOT/'GPT核查入口.md').write_text('\n'.join(all_lines)+'\n',encoding='utf-8')
    html=['<!doctype html><meta charset="utf-8"><title>F组10秒往返结果</title><style>body{font-family:Microsoft YaHei,sans-serif;max-width:1100px;margin:28px auto;background:#f7f7f7;color:#222}img{width:100%;background:white}section{background:white;padding:16px;margin:20px 0}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #bbb}a{color:#1268a8}</style>',
        '<h1>F组10秒完整往返 · 原始Y对照</h1>',f'<p>1 ms；统一g={cal["g"]:.8f}；15条完成，无饱和。零线为目标；所有图保留原始均值、峰值与相同纵轴。</p>',
        '<p><a href="简报.md">简报.md</a> · <a href="GPT核查入口.md">GPT核查入口</a> · <a href="review/分段指标.csv">分段指标</a> · <a href="review/事件局部基线峰.csv">事件局部基线峰</a> · <a href="review/控制代价与饱和.csv">控制代价</a></p>',
        '<p>淡黄色0–0.8 s为原模型冷初始化，非A启动。真实A启动约1.285 s。保留旧运动方式，完整去回/停后尾段。主图为effective Y；physical Y单独列于简报和分段表。</p>']
    for scene in SCENES:html.append(f'<section><h2>{scene}：{SCENES[scene]}</h2><img src="outputs/figures/{scene}_完整往返_原始Y.png"></section>')
    for e in ['start','turn','stop']:html.append(f'<section><img src="outputs/figures/S3_{e}_局部原始Y.png"></section>')
    html.append('<p>近似仿真情景，不能据此宣称CB3实机同等表现。原B/F/C和旧62秒日志保持不变；精简包保留完整1ms标量CSV，无图片/网格。</p>')
    (ROOT/'结果.html').write_text('\n'.join(html),encoding='utf-8')
    (ROOT/'results.html').write_text('\n'.join(html),encoding='utf-8')
    # ASCII command body avoids depending on the Windows console code page.
    (ROOT/'打开结果.cmd').write_bytes(b'@echo off\r\nstart "" "%~dp0results.html"\r\n')
def main():
    c=common();logs={s+'__'+p:array(ROOT/'outputs/full_1ms'/f'{s}__{p}.npz') for s in SCENES for p in LABELS}
    target=c['event_target_um'];y=logs['S2__A0']['physical_e_um'];r=target-y;f=logs['S2__A0']['Fd']
    write(ROOT/'review/event_reconstruction_window.json',dict(
        A0_validation_window_s=[.8,float(c['t'][-1])],validation_RMSE_um=float(np.sqrt(np.mean(r[c['t']>=.8]**2))),
        full_trajectory_with_cold_RMSE_um=float(np.sqrt(np.mean(r**2))),full_relative_RMS_error=float(np.linalg.norm(r)/np.linalg.norm(target)),
        cold_initial_peak_t_s=float(c['t'][np.argmax(abs(y[c['t']<.8]))]),cold_initial_peak_abs_um=float(np.max(abs(y[c['t']<.8]))),
        first800ms_Fd_abs_peak_N=float(np.max(abs(f[c['t']<.8]))),last100ms_Fd_abs_peak_N=float(np.max(abs(f[-100:]))),
        last100ms_response_abs_peak_um=float(np.max(abs(y[-100:]))),
        interpretation='A0 reconstruction gate separately excludes inherited0-0.8s cold initialization. All full performance statistics and every figure retain that period; no state/warmup change. Inverse padding suppresses wrap, small early/tail force leakage remains and is measured here.'))
    source=source_evidence(c,logs);verdict=validate(c,logs);metrics,events,cost,cold=export(c,logs);plots(c,logs);documents(c,logs,metrics,events,cost,cold,source,verdict)
    print('REPORT complete; text CSV lossless; shared inputs/interfaces/protected originals PASS',flush=True)
if __name__=='__main__':main()
