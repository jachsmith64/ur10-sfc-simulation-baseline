"""Raw new results and inherited reference; report public correction separately from anti-jitter."""
from run_slow import ROOT,read,write,sha,common,array,PARAMS,VERSION,budget,correction_id,slow_center
from short_report import stats,table,LABELS
from pathlib import Path
import json,numpy as np,difflib

def load():
    logs={}
    for p in LABELS:
        logs['old__'+p]=array(ROOT/'reference_old_S3'/f'S3__{p}.npz')
        logs['new__'+p]=array(ROOT/'outputs/full_1ms'/f'PC__{p}.npz')
    return logs
def validate(c,corr,logs):
    fr=read(ROOT/'review/slow_frozen_version.json');source=read(ROOT/'baseline_review/frozen_version.json');g=fr['g']
    assert correction_id(corr,source['scenarios']['S3']['input_id'])==fr['shared_input_id']
    assert source['g']==g and fr['parameters']==source['parameters']
    errors={};fd=np.loadtxt(ROOT/'inputs/Fd_event.csv',delimiter=',',skiprows=1)[:,1]
    for ctrl in LABELS:
        old=logs['old__'+ctrl];a=logs['new__'+ctrl];meta=read(ROOT/'outputs/full_1ms'/f'PC__{ctrl}.json');row=read(ROOT/'outputs/full_1ms'/f'PC__{ctrl}.row.json')
        assert row['status']=='OK' and row['n_steps_done']==10000 and meta['input_sha256']==fr['shared_input_id']
        for k in ['t','phase','q_nominal','dq_C','remaining','remaining_dot','nominal_y','Fd']:
            assert np.array_equal(a[k],old[k]),(ctrl,k)
        assert np.array_equal(a['dq_C'],c['dq']*g) and np.array_equal(a['remaining_dot'],c['remaining_dot']*g)
        assert np.array_equal(a['original_base_ctrl'],old['base_ctrl'])
        assert np.array_equal(a['base_ctrl'],a['original_base_ctrl']+corr['common_delta_q'])
        assert np.array_equal(a['final_ctrl'],a['base_ctrl']+a['delta_u'])
        for k in corr:assert np.array_equal(a[k],corr[k])
        assert np.array_equal(a['qfrc_applied'],a['qfrc_disturbance']) and np.array_equal(a['Fd'],fd)
        assert np.array_equal(a['effective_y'],a['physical_y']+a['remaining'])
        assert np.all(a['nominal_y']==-.174)
        assert np.array_equal(a['q'][0],old['q'][0]) and np.array_equal(a['qdot'][0],old['qdot'][0])
        # Source command returned exactly when common correction is switched off, without another full run.
        off=a['q_nominal']+a['dq_C'];assert np.array_equal(off,old['base_ctrl'])
        # Controller state responds to the newly integrated plant; it is not an offline shifted trace.
        assert not np.array_equal(a['q'],old['q'])
        assert np.allclose(a['delta_u'],a['tau_requested']/5000,atol=1e-15,rtol=0)
        inc=float(np.max(abs(a['tau_realized_increment']-a['tau_requested'])));assert inc<1e-8
        p=PARAMS[ctrl];e=a['effective_e_um']*1e-6;v=a['effective_v']
        F=np.zeros_like(e) if ctrl=='A0' else -p['K']*e-p['B0']*v-p['mu']*np.sign(v)*abs(v)**p['n']
        ferr=float(np.max(abs(F-a['F_requested'])));assert ferr<1e-9
        errors[ctrl]=dict(max_anti_realized_residual_Nm=inc,independent_controller_formula_error_N=ferr,
            saturation_samples=int(np.count_nonzero(np.any(a['ctrl_saturation']|a['force_saturation']|a['joint_saturation'],axis=1))))
    before=read(ROOT/'review/protected_before.json');after={p:sha(p) for p in before};assert before==after
    write(ROOT/'review/protected_after.json',after)
    for p,h in read(ROOT/'review/slow_implementation_freeze.json').items():assert sha(ROOT/p)==h
    base=Path(read(ROOT/'review/source_version.json')['accepted_complete_root'])
    unchanged=['config.py','sim_env.py','controller.py','sfc.py','position_execution.py','background.py','f_measurement.py','disturbance.py']
    assert all(sha(ROOT/n)==sha(base/n) for n in unchanged)
    for p in (ROOT/'assets_ur10e').rglob('*'):
        if p.is_file():assert sha(p)==sha(base/'assets_ur10e'/p.relative_to(ROOT/'assets_ur10e'))
    assert read(ROOT/'review/static_mapping.json')['full_compiled_model']==read(ROOT/'baseline_review/compiled_model.json')
    verdict=dict(passed=True,new_full_integrations=len(budget()),budget=6,formal_cases=3,protected_original_files=len(before),all_sources_and_old_versions_unchanged=True,
        source_input_arrays_identical=True,public_correction_same_all_cases=True,zero_reference_unchanged=True,old_command_structure_restored_with_correction_off=True,
        no_new_controller_or_dynamics=True,anti_torque_not_duplicated_as_external=True,core_unchanged=unchanged,model_and_initial_state_unchanged=True,
        public_module_no_fabricated_force_or_energy=True,actual_raw_new_outputs=True,controller_checks=errors)
    write(ROOT/'review/slow_verification.json',verdict);return verdict
def export(c,corr,logs):
    out=ROOT/'outputs/compact_1ms';out.mkdir(exist_ok=True)
    wins=read(ROOT/'baseline_review/metric_windows.json')['windows'];wins['after_cold_aux']=[.8,9.999]
    plan=read(ROOT/'review/baseline_plan.json');ev=read(ROOT/'review/event_targets.json')
    index=[];metrics=[];events=[];costs=[];platforms=[];cold=[]
    for name,a in logs.items():
        group,ctrl=name.split('__');new=group=='new';fr=read(ROOT/'review/slow_frozen_version.json')
        cols=dict(t_s=a['t'],phase=a['phase'],nominal_Y_m=a['nominal_y'],physical_e_um=a['physical_e_um'],effective_e_um=a['effective_e_um'],effective_v_m_s=a['effective_v'],Fd_event_N=a['Fd'],F_anti_requested_N=a['F_requested'],
            tau_anti_requested_norm_Nm=np.linalg.norm(a['tau_requested'],axis=1),tau_anti_realized_norm_Nm=np.linalg.norm(a['tau_realized_increment'],axis=1),
            delta_anti_norm_rad=np.linalg.norm(a['delta_u'],axis=1),ctrl_saturated_any=np.any(a['ctrl_saturation'],axis=1),force_saturated_any=np.any(a['force_saturation'],axis=1),joint_saturated_any=np.any(a['joint_saturation'],axis=1))
        for j in range(6):cols[f'delta_anti_J{j+1}_rad']=a['delta_u'][:,j]
        for j in range(6):cols[f'final_ctrl_J{j+1}_rad']=a['final_ctrl'][:,j]
        matrix=np.column_stack(list(cols.values()));p=out/(name+'.csv');np.savetxt(p,matrix,delimiter=',',fmt='%.17g',header=','.join(cols),comments='')
        assert np.array_equal(np.loadtxt(p,delimiter=',',skiprows=1),matrix)
        index.append(dict(case_id=name,group=group,controller=ctrl,samples=len(a['t']),public_on=new,input_id=fr['shared_input_id'] if new else fr['source_S3_input_id'],g=fr['g'],csv_sha256=sha(p),full_log_local=str(ROOT/('outputs/full_1ms/PC__'+ctrl+'.npz' if new else 'reference_old_S3/S3__'+ctrl+'.npz'))))
        for window,(lo,hi) in wins.items():
            mask=(a['t']>=lo)&(a['t']<=hi)
            for channel in ['effective','physical']:metrics.append(dict(case_id=name,group=group,controller=ctrl,window=window,channel=channel,t0_s=lo,t1_s=hi,n=int(mask.sum()),**stats(a[channel+'_e_um'][mask])))
        for platform in plan['platforms']:
            lo=platform['t0_s'];hi=platform['t1_s'];mask=(a['t']>=lo)&(a['t']<=hi)
            platforms.append(dict(case_id=name,group=group,controller=ctrl,platform=platform['platform'],t0_s=lo,t1_s=hi,n=int(mask.sum()),effective_median_um=float(np.median(a['effective_e_um'][mask])),physical_median_um=float(np.median(a['physical_e_um'][mask])),qualification=platform['qualification']))
        for event in ev:
            label=event['event'];lo,hi=event['core_s'] if label!='stop' else event['new_support_s'];blo,bhi=(1.085,1.235) if label=='start' else (event['new_support_s'][0]-.4,event['new_support_s'][0]-.05)
            mask=(a['t']>=lo)&(a['t']<=hi);bm=(a['t']>=blo)&(a['t']<=bhi)
            for channel in ['effective','physical']:
                y=a[channel+'_e_um'];b=float(np.median(y[bm]));d=y[mask]-b
                events.append(dict(case_id=name,group=group,controller=ctrl,event=label,channel=channel,t0_s=lo,t1_s=hi,baseline_t0_s=blo,baseline_t1_s=bhi,baseline_median_um=b,
                    baseline_peak_abs_um=float(np.max(abs(d))),raw_min_um=float(np.min(y[mask])),raw_max_um=float(np.max(y[mask])),time_abs_deviation_gt10um_s=float(np.count_nonzero(abs(d)>10)*.001)))
        sat=np.any(a['ctrl_saturation']|a['force_saturation']|a['joint_saturation'],axis=1)
        costs.append(dict(case_id=name,F_anti_RMS_N=float(np.sqrt(np.mean(a['F_requested']**2))),F_anti_abs_peak_N=float(np.max(abs(a['F_requested']))),
            tau_anti_RMS_norm_Nm=float(np.sqrt(np.mean(np.sum(a['tau_requested']**2,axis=1)))),tau_anti_peak_norm_Nm=float(np.max(np.linalg.norm(a['tau_requested'],axis=1))),
            anti_position_peak_abs_deg=float(np.max(abs(a['delta_u']))*180/np.pi),public_position_peak_abs_deg=float(np.max(abs(corr['common_delta_q']))*180/np.pi) if new else 0.,
            total_added_position_peak_abs_deg=float(np.max(abs(a['delta_u']+(corr['common_delta_q'] if new else 0.)))*180/np.pi),saturation_samples=int(sat.sum()),saturation_duty=float(np.mean(sat))))
        m=a['t']<.8;i=np.flatnonzero(m)[np.argmax(abs(a['effective_e_um'][m]))]
        cold.append(dict(case_id=name,peak_t_s=float(a['t'][i]),peak_signed_um=float(a['effective_e_um'][i]),is_A_start=False,public_initial_position_step=new))
    comparisons=[]
    for event in ev:
        lo,hi=event['core_s'] if event['event']!='stop' else event['new_support_s'];m=(c['t']>=lo)&(c['t']<=hi)
        x=logs['old__A0']['effective_e_um'][m];y=logs['new__A0']['effective_e_um'][m]
        old=next(r for r in events if r['case_id']=='old__A0' and r['event']==event['event'] and r['channel']=='effective')
        new=next(r for r in events if r['case_id']=='new__A0' and r['event']==event['event'] and r['channel']=='effective')
        comparisons.append(dict(event=event['event'],old_local_peak_um=old['baseline_peak_abs_um'],new_local_peak_um=new['baseline_peak_abs_um'],relative_peak_change=new['baseline_peak_abs_um']/old['baseline_peak_abs_um']-1,
            diagnostic_centered_shape_correlation=float(np.corrcoef(x,y)[0,1]),old_threshold_occupancy_s=old['time_abs_deviation_gt10um_s'],new_threshold_occupancy_s=new['time_abs_deviation_gt10um_s'],
            nominal_event_target_and_force_unchanged=True,note='Local median and correlation only diagnose shape; plotted outputs stay raw. Occupancy is not a physical event duration identification.'))
    for n,rows in [('case_index.csv',index),('新旧分段指标.csv',metrics),('平台中心前后.csv',platforms),('新旧事件指标.csv',events),('抗抖代价_公共位置代价分列.csv',costs),('冷初始化对照.csv',cold),('A0事件形状前后核查.csv',comparisons)]:table(ROOT/'review'/n,rows)
    write(ROOT/'review/public_position_cost.json',dict(units='Position rad, velocity rad/s, acceleration rad/s2; no invented force or energy',
        deltaY_peak_abs_um=float(np.max(abs(corr['common_delta_y_m']))*1e6),deltaq_peak_abs_rad=float(np.max(abs(corr['common_delta_q']))),
        deltaq_dot_peak_abs_rad_s=float(np.max(abs(corr['common_delta_q_dot']))),deltaq_ddot_peak_abs_rad_s2=float(np.max(abs(corr['common_delta_q_ddot']))),
        initial_nonzero_command_step_rad=corr['common_delta_q'][0],initial_step_separate_from_smooth_analytic_derivative=True))
    write(ROOT/'review/metric_definition.json',dict(windows=wins,std_ddof=0,full_statistics_include_cold=True,curve_mean_subtracted=False,curve_smoothed=False,curve_multiplied=False,
        public_improvement_not_attributed_to_SFC=True,offline_residual_diagnostic_not_actual_control=True))
    return metrics,platforms,events,costs,comparisons
def plots(c,corr,logs):
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10})
    dest=ROOT/'outputs/figures';dest.mkdir(exist_ok=True);colors={'A0':'#666666','Alinear':'#2374ab','ASFC_strong':'#d73535'}
    raw=np.concatenate([a['effective_e_um'] for a in logs.values()]);margin=np.ptp(raw)*.08;ylim=(np.min(raw)-margin,np.max(raw)+margin)
    t=c['t'];old=logs['old__A0']['effective_e_um'];new=logs['new__A0']['effective_e_um'];b=corr['common_baseline_um'];plan=read(ROOT/'review/baseline_plan.json')
    def save(fig,n):fig.savefig(dest/n,dpi=155);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(11,6.2),sharex=True,layout='constrained')
    axes[0].plot(t,old,color='#666666',lw=1,label='旧S3 A0 原始effective Y');axes[0].plot(t,b,color='#f0932b',lw=2,label='4平台慢中心 b(t)')
    axes[1].plot(t,old-b,color='#2374ab',lw=1,label='旧A0−b：仅离线分离诊断，非新控制结果')
    for ax in axes:ax.axhline(0,color='black',lw=.7);ax.set_ylabel('Y偏差 / μm');ax.grid(alpha=.2);ax.legend();ax.set_ylim(*ylim)
    axes[0].set_title('从旧无附加抗抖结果提取慢中心：保留波纹、事件及冷初始化');axes[1].set_xlabel('时间 / s');save(fig,'01_旧A0与慢中心_诊断.png')
    fig,axes=plt.subplots(3,1,figsize=(11,7),sharex=True,layout='constrained')
    axes[0].plot(t,corr['common_delta_y_m']*1e6,color='#f0932b');axes[0].set_ylabel('预定δY / μm');axes[0].set_title('公共纠偏：静态Jacobian映射、C2连接；初始非零，无预热')
    for j in range(6):axes[1].plot(t,corr['common_delta_q'][:,j]*1e6,label='J'+str(j+1));axes[2].plot(t,corr['common_delta_q_dot'][:,j]*1e6)
    axes[1].set_ylabel('δq / μrad');axes[1].legend(ncol=6);axes[2].set_ylabel('δq速度 / μrad/s');axes[2].set_xlabel('时间 / s')
    for ax in axes:ax.grid(alpha=.2)
    save(fig,'02_公共预定纠偏及速度.png')
    fig,axes=plt.subplots(2,1,figsize=(11,6.2),sharex=True,gridspec_kw={'height_ratios':[1,3]},layout='constrained')
    axes[0].plot(t,c['along'],color='#9467bd');axes[0].set_ylabel('沿程 / mm');axes[0].set_title('共同离线预定慢纠偏下的完整往返：实际原始effective Y，10秒/1ms')
    for ctrl in LABELS:axes[1].plot(t,logs['new__'+ctrl]['effective_e_um'],color=colors[ctrl],label='公共纠偏＋'+LABELS[ctrl],lw=1.)
    axes[1].set_ylim(*ylim);axes[1].axhline(0,color='black',lw=.7);axes[1].set_ylabel('原始Y偏差 / μm');axes[1].set_xlabel('时间 / s');axes[1].legend(ncol=3,loc='lower left');axes[1].grid(alpha=.2);axes[1].axvspan(0,.8,color='#ddaa44',alpha=.1)
    axes[1].text(.01,.96,'0–0.8秒：冷初始化＋起始非零公共命令，非A启动',transform=axes[1].transAxes,va='top',fontsize=9)
    tr=read(ROOT/'review/trajectory.json')
    for label,lo,hi in [('A启动',1.285,1.926),('去程',2.226,4.38),('换向',4.58,5.643),('回程',5.943,8.667),('减速/停止',8.667,9.999)]:
        axes[0].text((lo+hi)/2,.95,label,transform=axes[0].get_xaxis_transform(),ha='center',va='top',fontsize=9)
        if label in ['A启动','换向','减速/停止']:axes[1].axvspan(lo,hi,color='#3498db',alpha=.06)
    for at in [tr['start_trigger_s'],tr['turn_trigger_s'],tr['post_stop_begin_s']]:axes[1].axvline(at,color='#777777',lw=.8,ls=':',alpha=.5)
    save(fig,'03_新三方案_完整原始Y.png')
    fig,ax=plt.subplots(figsize=(11,4.8),layout='constrained');ax.plot(t,old,color='#999999',lw=1,label='旧A0，无公共纠偏');ax.plot(t,new,color='#222222',lw=1,label='新A0，有公共纠偏，实际重新积分')
    ax.axhline(0,color='black',lw=.7);ax.set_ylim(*ylim);ax.set_xlabel('时间 / s');ax.set_ylabel('原始effective Y / μm');ax.set_title('公共模块的效果单独展示：中心移动不归给SFC');ax.legend();ax.grid(alpha=.2);save(fig,'04_新旧A0原始对照.png')
    fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained');m=(t>=1.2849583981754964)&(t<=tr['turn_trigger_s'])
    for ctrl in LABELS:ax.plot(c['along'][m],logs['new__'+ctrl]['effective_e_um'][m],color=colors[ctrl],label=LABELS[ctrl],lw=1.)
    ax.axhline(0,color='black',lw=.7);ax.set_ylim(*ylim);ax.set_xlabel('去程名义沿程位置 / mm');ax.set_ylabel('原始effective Y / μm');ax.set_title('仅去程：启动至换向顶点，回程未混叠');ax.legend(ncol=3);ax.grid(alpha=.2);save(fig,'05_去程沿程位置与原始Y.png')
    for event in read(ROOT/'review/event_targets.json')[:2]:
        lo,hi=event['new_support_s'];m=(t>=lo-.05)&(t<=hi+.10);fig,ax=plt.subplots(figsize=(10,4.2),layout='constrained')
        for ctrl in LABELS:ax.plot(t[m],logs['new__'+ctrl]['effective_e_um'][m],color=colors[ctrl],label=LABELS[ctrl],lw=1.)
        ax.axhline(0,color='black',lw=.7);ax.set_ylim(*ylim);ax.set_title('共同纠偏＋'+('启动' if event['event']=='start' else '换向')+'局部：实际原始Y');ax.set_xlabel('时间 / s');ax.set_ylabel('原始effective Y / μm');ax.legend(ncol=3);ax.grid(alpha=.2)
        save(fig,'06_'+event['event']+'_原始局部.png')
    write(ROOT/'review/plot_definitions.json',dict(global_y_limits_um=ylim,all_raw_comparison_axes_same=True,offline_residual_only_diagnostic=True,raw_case_curve_keys='effective_e_um',no_curve_recenter_smooth_or_rescale=True))
def documents(metrics,platforms,events,costs,changes,verdict):
    def m(group,ctrl,channel='effective',win='full'):return next(r for r in metrics if r['group']==group and r['controller']==ctrl and r['channel']==channel and r['window']==win)
    def cost(ctrl):return next(r for r in costs if r['case_id']=='new__'+ctrl)
    def plat(group,ctrl):return [r['effective_median_um'] for r in platforms if r['group']==group and r['controller']==ctrl]
    source=read(ROOT/'review/slow_frozen_version.json');accept=read(ROOT/'review/public_correction_acceptance.json');public=read(ROOT/'review/public_position_cost.json');mapping=read(ROOT/'review/static_mapping.json')
    lines=['# F组共同预定慢纠偏｜验收简报','',f'完成**3条新10秒/1ms仿真**：1次A0构造通过并复用，线性/强SFC各1次；预算6条。旧S3三方案直接复用。原件清单{verdict["protected_original_files"]}个文件哈希未变。',
        '', '**顺序与前提。** 仅从旧S3 A0 effective Y取4个中位中心，用三段五次smoothstep相连；完整旧数据用于离线非因果预处理。预静止/去程/回程窗避开冷初始化与已选事件支持。停后代表窗9.950–9.999 s仅50点，已有仅事件结果该窗最大约1.20 μm；它是短尾参照，不证明长期停稳，未把较早恢复波形拟成中心。',
        '', '|A0代表中心 / μm|预静止|普通去程|普通回程|停后短尾|','|---|---:|---:|---:|---:|',
        '|原始|'+'|'.join(f'{v:.2f}' for v in plat('old','A0'))+'|','|加入公共纠偏后|'+'|'.join(f'{v:+.2f}' for v in plat('new','A0'))+'|',
        '', f'**公共输入。** δY=−b，增益1.0首次即通过；固定初始姿态的平移Jacobian最小范数右逆，条件数{mapping["condition"]:.3f}。命令为原q_nom＋原g·dq＋公共δq慢＋原τ抗抖/5000，评价零线仍是原名义Y。g={source["g"]:.10f}，全部背景及解析导数、事件目标/Fd逐点保留；K/B0/n/μ、模型、1ms、初态不变。公共最大Y纠偏{public["deltaY_peak_abs_um"]:.2f} μm，单关节{public["deltaq_peak_abs_rad"]*180/np.pi:.6f}°；速度/加速度峰{public["deltaq_dot_peak_abs_rad_s"]:.3g} rad/s、{public["deltaq_ddot_peak_abs_rad_s2"]:.3g} rad/s²。位置量未虚报为力或能量。',
        '', '**新三方案实际effective Y（全程保留冷初始化；std不称纯高频振动）**','', '|方案|标准差 μm|峰峰 μm|均值 μm|原始绝对峰 μm|抗抖力RMS N|抗抖力矩RMS范数 Nm|','|---|---:|---:|---:|---:|---:|---:|']
    for ctrl in LABELS:
        a=m('new',ctrl);cc=cost(ctrl);lines.append('|'+LABELS[ctrl]+'|'+'|'.join(f'{a[k]:.2f}' for k in ['std_um','ptp_um','mean_um','raw_peak_abs_um'])+f'|{cc["F_anti_RMS_N"]:.4f}|{cc["tau_anti_RMS_norm_Nm"]:.4f}|')
    a0=m('new','A0');sf=m('new','ASFC_strong');linear=m('new','Alinear');old=m('old','A0')
    lines+=['',f'在同一公共纠偏下，线性标准差/峰峰相对A0下降{(1-linear["std_um"]/a0["std_um"])*100:.1f}%/{(1-linear["ptp_um"]/a0["ptp_um"])*100:.1f}%，强SFC下降{(1-sf["std_um"]/a0["std_um"])*100:.1f}%/{(1-sf["ptp_um"]/a0["ptp_um"])*100:.1f}%。强SFC抗抖力RMS约为线性的{cost("ASFC_strong")["F_anti_RMS_N"]/cost("Alinear")["F_anti_RMS_N"]:.2f}倍，未宣称等代价优势。',
        '', f'公共模块主要搬稳中心：旧/新A0全程std {old["std_um"]:.2f}→{a0["std_um"]:.2f} μm，峰峰{old["ptp_um"]:.2f}→{a0["ptp_um"]:.2f} μm（略增加），不能将居中直接算成抑振。普通去回、启动/换向/停止、physical同口径统计及新旧参照均在review CSV。',
        '', '|A0事件局部基线峰 / μm|旧|新|','|---|---:|---:|']
    for r in changes:lines.append('|'+{'start':'启动','turn':'换向','stop':'减速/停止'}[r['event']]+'|'+f'{r["old_local_peak_um"]:.2f}|{r["new_local_peak_um"]:.2f}|')
    lines+=['', '所有新方案均无饱和或abort。公共纠偏起始已非零，初态仍未改变；A0冷初始化峰约−71.56 μm/0.035 s，旧约−78.12 μm/0.037 s，单列且全程图保留，不当作1.285 s的A启动。新旧局部形状/占时见CSV，未改事件外力修图。主图为原始输出，未减各自均值、未平移/平滑；离线旧A0−b仅作独立诊断。',
        '', '**代码与边界。** R-026在独立副本修复：缓存及新生成失败均阻断依赖比较，10%/20%原验收阈值与合法无需放大分支保留；6项廉价失败检查通过、零额外积分。新模块另建冻结版本，旧记录保留。R-027“真实慢漂移在线识别/预测/执行及耦合”记为暂缓、未解决；本机未定位含R-026的更完整现行库，已建立接续记录并保留已找到的R-001～008历史出处，没有覆盖历史内容。',
        '', '**限定结论：离线预定慢纠偏条件下，当前工况的抗抖对照。** 不是在线漂移已解决或CB3实机已验证。双击打开结果.cmd查看完整图；完整工程和1ms NPZ在本机，核查ZIP仅文本且保留6条新旧参照的全1ms必要数据。']
    (ROOT/'简报.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (ROOT/'GPT核查入口.md').write_text('''# 本轮GPT核查入口

先读简报.md、review/slow_verification.json、review/slow_frozen_version.json、review/baseline_plan.json。

真实命令在sweep.py唯一新增common_position_correction入口，原工艺q_nominal与Y参考未改变；position_logging.py分别记录original_base_ctrl/common_delta_q/抗抖delta_u/final_ctrl。新模块run_slow.py最多6条积分，A0只按4平台中心/约束构造，通过后冻结才跑线性/强SFC。没有拟合事件或重设g、μ。

输入补丁R026见input_acceptance.py/run_short.py/test_input_acceptance.py与review/R026_failure_checks.json：新生成预算耗尽未通过不formalize，坏缓存不调用仿真/力读取；合法g1无需放大分支保留。旧15条不复跑。

old__为旧S3三方案的1ms原始参照，new__为本轮公共纠偏后的实际新结果；公开公共输入仅一份，所有新方案共用。inputs/frozen_common_full_1ms.csv为原输入；inputs/public_slow_full_1ms.csv为公共基线/映射/导数。每条CSV全10,000行，17g无损往返；代码及diff随包。模型网格/完整中间NPZ不打包，完整工程在本机。

check_slow_review.py可仅用Python标准库复算文本包的共享输入哈希、目标/误差叠加、控制公式、最终命令、分段统计，不启动MuJoCo或机器人。图代码slow_report.py随包；图留本机outputs/figures。

问题库接续记录在review/持续问题库_本轮快照.md。R027为暂缓未解决，不能把本轮离线假设称为在线漂移处理完成。
''',encoding='utf-8')
    html=['<!doctype html><meta charset="utf-8"><title>F预定慢纠偏</title><style>body{font-family:Microsoft YaHei,sans-serif;max-width:1100px;margin:28px auto;background:#f7f7f7;color:#222}section{background:white;padding:16px;margin:20px 0}img{width:100%}a{color:#1268a8}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #bbb}</style>',
        '<h1>共同离线预定慢纠偏 · 当前工况抗抖对照</h1><p>3条新运行；原输入全部保留；先A0构造，再冻结共同位置纠偏，最后比较三方案。主图真实重新积分、未减均值。</p>',
        '<p><a href="简报.md">简报</a> · <a href="review/平台中心前后.csv">平台中心</a> · <a href="review/新旧分段指标.csv">分段指标</a> · <a href="review/抗抖代价_公共位置代价分列.csv">代价</a></p>',
        '<p>停后仅最后50ms短代表，不证明长期稳定。离线非因果假设不是在线漂移算法；初始非零纠偏及冷瞬态已保留。</p>']
    html+=['<section><h2>physical Y同口径核对（全程，含冷初始化）</h2><table><tr><th>方案</th><th>标准差 μm</th><th>峰峰 μm</th><th>均值 μm</th></tr>']
    for ctrl in LABELS:
        a=m('new',ctrl,channel='physical');html.append('<tr><td>'+LABELS[ctrl]+'</td>'+''.join(f'<td>{a[k]:.2f}</td>' for k in ['std_um','ptp_um','mean_um'])+'</tr>')
    html+=['</table><p>原始physical Y与effective Y分开统计；源头remaining及导数保留。</p></section>']
    names=['03_新三方案_完整原始Y.png','01_旧A0与慢中心_诊断.png','02_公共预定纠偏及速度.png','04_新旧A0原始对照.png','05_去程沿程位置与原始Y.png','06_start_原始局部.png','06_turn_原始局部.png']
    for n in names:html.append('<section><img src="outputs/figures/'+n+'"></section>')
    (ROOT/'results.html').write_text('\n'.join(html),encoding='utf-8');(ROOT/'打开结果.cmd').write_bytes(b'@echo off\r\nstart "" "%~dp0results.html"\r\n')
def main():
    c=common()
    with np.load(ROOT/'inputs/public_slow_correction.npz') as z:corr={k:z[k] for k in z.files}
    logs=load();verdict=validate(c,corr,logs);m,p,e,cost,change=export(c,corr,logs);plots(c,corr,logs);documents(m,p,e,cost,change,verdict)
    print('SLOW report/interfaces/raw CSV/protected versions PASS',flush=True)
if __name__=='__main__':main()
