"""Raw report for one explicitly validated batch; never integrates, reuses, or substitutes cases."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
from batch_identity import read,sha,atomic_json
PLOT_VERSION='F10-run-report-v2'
LABELS={'A0':'无附加抗抖','Alinear':'原线性','ASFC_strong':'强SFC'}
def table(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def stat(x):return dict(std_um=float(np.std(x,ddof=0)),ptp_um=float(np.ptp(x)),mean_um=float(np.mean(x)),raw_peak_abs_um=float(np.max(abs(x))))
def render(d,root=None,validated_before_completion=False):
    from run_slow import ROOT,arrays,inputs_from_run,validate_case,require,CONTROLLERS
    root=Path(root or ROOT);d=Path(d);validation=read(d/'validation.json');gen=read(d/'generation.json');rid=gen['run_id'];ctx=inputs_from_run(d)
    require(validation['passed'] is True and validation['run_id']==rid,'No accepted batch to report')
    if not validated_before_completion:require((d/'completed.json').is_file() and read(d/'completed.json')['completed'] is True,'Historical batch not complete')
    logs={};ids={}
    for ctrl in CONTROLLERS:
        p=d/'full_1ms'/f'{ctrl}.npz';v=next(x for x in validation['cases'] if x['case_id']==ctrl)
        require(sha(p)==v['data_sha256'],'Data changed before plotting: '+ctrl)
        validate_case(p,read(p.with_suffix('.row.json')),ctx,rid,gen['generation_id'],ctrl);logs[ctrl]=arrays(p);ids[ctrl]=sha(p)
    (d/'csv_1ms').mkdir(exist_ok=True);(d/'figures').mkdir(exist_ok=True)
    scalars=['t','model_time_s','phase','nominal_y','physical_y','effective_y','physical_e_um','effective_e_um','physical_v','effective_v','remaining','remaining_dot','Fd','F_spring','F_linear','F_shear','F_requested','updated']
    matrices=['q_nominal','dq_C','common_delta_q','delta_u','final_ctrl','q','qdot','tau_requested','qfrc_actuator','base_qfrc_actuator','tau_realized_increment','qfrc_disturbance','qfrc_applied']
    text_schema={};csvhash={}
    for ctrl,a in logs.items():
        columns={k:a[k] for k in scalars}
        for key in matrices:
            for j in range(6):columns[key+'_J'+str(j+1)]=a[key][:,j]
        for key in ['ctrl_saturation','force_saturation','joint_saturation']:columns[key+'_any']=np.any(a[key],axis=1)
        matrix=np.column_stack(list(columns.values()));p=d/'csv_1ms'/f'{ctrl}.csv'
        np.savetxt(p,matrix,delimiter=',',fmt='%.17g',header=','.join(columns),comments='')
        require(np.array_equal(np.loadtxt(p,delimiter=',',skiprows=1),matrix),'CSV precision mismatch')
        text_schema[ctrl]=dict(columns=list(columns),samples=len(a['t']),precision='17g exact roundtrip',data_sha256=ids[ctrl],csv_sha256=sha(p));csvhash[ctrl]=sha(p)
    atomic_json(d/'csv_schema.json',dict(run_id=rid,plot_version=PLOT_VERSION,cases=text_schema,
        units='t/model_time:s; q and offsets:rad; qdot:rad/s; Y/remaining:m; e:um; velocity:m/s; F:N; torque vectors:Nm; flags:bool',
        torque_meaning='qfrc_actuator is total realized actuator torque. tau_realized_increment is same-state counterfactual anti-only increment, not compensation position precision.'))
    wins=dict(read(d/'frozen_inputs/metric_windows.json')['windows']);wins['after_cold_aux']=[.8,9.999]
    metrics=[];costs=[];events=[]
    for ctrl,a in logs.items():
        for window,(lo,hi) in wins.items():
            mask=(a['t']>=lo)&(a['t']<=hi)
            for channel in ['physical','effective']:
                metrics.append(dict(run_id=rid,case_id=ctrl,data_sha256=ids[ctrl],plot_version=PLOT_VERSION,channel=channel,window=window,t0_s=lo,t1_s=hi,n=int(mask.sum()),sample_rule='pre-step full1ms; full includes cold',**stat(a[channel+'_e_um'][mask])))
        sat=np.any(a['ctrl_saturation']|a['force_saturation']|a['joint_saturation'],axis=1)
        costs.append(dict(run_id=rid,case_id=ctrl,data_sha256=ids[ctrl],plot_version=PLOT_VERSION,F_RMS_N=float(np.sqrt(np.mean(a['F_requested']**2))),F_peak_N=float(np.max(abs(a['F_requested']))),
            tau_requested_RMS_norm_Nm=float(np.sqrt(np.mean(np.sum(a['tau_requested']**2,axis=1)))),tau_requested_peak_norm_Nm=float(np.max(np.linalg.norm(a['tau_requested'],axis=1))),
            actuator_total_RMS_norm_Nm=float(np.sqrt(np.mean(np.sum(a['qfrc_actuator']**2,axis=1)))),anti_position_peak_rad=float(np.max(abs(a['delta_u']))),public_position_peak_rad=float(np.max(abs(a['common_delta_q']))),
            saturation_samples=int(sat.sum()),saturation_duty=float(np.mean(sat))))
        for ev in read(d/'frozen_inputs/event_targets.json'):
            lo,hi=ev['core_s'] if ev['event']!='stop' else ev['new_support_s'];blo,bhi=(1.085,1.235) if ev['event']=='start' else (ev['new_support_s'][0]-.4,ev['new_support_s'][0]-.05)
            for channel in ['physical','effective']:
                x=a[channel+'_e_um'];m=(a['t']>=lo)&(a['t']<=hi);b=float(np.median(x[(a['t']>=blo)&(a['t']<=bhi)]));dx=x[m]-b
                events.append(dict(run_id=rid,case_id=ctrl,data_sha256=ids[ctrl],plot_version=PLOT_VERSION,event=ev['event'],channel=channel,t0_s=lo,t1_s=hi,baseline_t0_s=blo,baseline_t1_s=bhi,
                    baseline_um=b,local_peak_abs_um=float(np.max(abs(dx))),threshold10um_occupancy_s=float(np.count_nonzero(abs(dx)>10)*.001)))
    table(d/'metrics.csv',metrics);table(d/'costs.csv',costs);table(d/'events.csv',events)
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10})
    colors={'A0':'#666666','Alinear':'#2476ad','ASFC_strong':'#da3838'};t=ctx['c']['t'];along=ctx['c']['along'];figrows=[]
    def save(fig,name,channel,window,xunit):
        fig.suptitle(rid+' · '+PLOT_VERSION,fontsize=8);fig.savefig(d/'figures'/name,dpi=145);plt.close(fig)
        info=dict(run_id=rid,generation_id=gen['generation_id'],figure=name,figure_sha256=sha(d/'figures'/name),case_data_sha256=ids,csv_sha256=csvhash,
            channel=channel,window_s=window,x_unit=xunit,y_unit='um',sample_rule='pre-step 1ms; cold included in full',plot_version=PLOT_VERSION,plot_source_sha256=sha(Path(__file__)),raw_no_shift_smooth_rescale=True)
        atomic_json(d/'figures'/Path(name).with_suffix('.json').name,info);figrows.append(info)
    for channel in ['effective','physical']:
        raw=np.concatenate([a[channel+'_e_um'] for a in logs.values()]);pad=np.ptp(raw)*.08;ylim=(np.min(raw)-pad,np.max(raw)+pad)
        fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True,gridspec_kw={'height_ratios':[1,3]},layout='constrained')
        axes[0].plot(t,along,color='#9062b5');axes[0].set_ylabel('沿程 / mm');axes[0].set_title(channel+' Y完整往返：实际新积分，原始偏差')
        for ctrl,a in logs.items():axes[1].plot(t,a[channel+'_e_um'],color=colors[ctrl],label=LABELS[ctrl],lw=1)
        axes[1].axhline(0,color='black',lw=.7);axes[1].set_ylim(*ylim);axes[1].axvspan(0,.8,color='#ddaa44',alpha=.1);axes[1].set_xlabel('时间 / s');axes[1].set_ylabel('原始Y偏差 / μm');axes[1].grid(alpha=.2);axes[1].legend(ncol=3,loc='lower left')
        axes[1].text(.01,.96,'0–0.8s冷初始化＋非零公共起始指令；非A启动',transform=axes[1].transAxes,va='top',fontsize=9)
        for name,at in [('A启动',1.2849583981754964),('换向',4.981),('停后',8.858)]:axes[0].axvline(at,color='#777',ls=':',lw=.8);axes[0].text(at,.92,name,transform=axes[0].get_xaxis_transform(),ha='center',va='top')
        save(fig,'full_'+channel+'.png',channel,[0,9.999],'s')
    fig,ax=plt.subplots(figsize=(10,4.6),layout='constrained');m=(t>=1.2849583981754964)&(t<=4.981)
    for ctrl,a in logs.items():ax.plot(along[m],a['effective_e_um'][m],color=colors[ctrl],label=LABELS[ctrl],lw=1)
    ax.axhline(0,color='black',lw=.7);ax.set_xlabel('仅去程名义沿程 / mm');ax.set_ylabel('原始effective Y / μm');ax.set_title('仅去程，不混叠回程');ax.legend(ncol=3);ax.grid(alpha=.2);save(fig,'outbound_effective.png','effective',[1.2849583981754964,4.981],'mm')
    for ev in read(d/'frozen_inputs/event_targets.json')[:2]:
        lo,hi=ev['new_support_s'];m=(t>=lo-.05)&(t<=hi+.10);fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
        for ctrl,a in logs.items():ax.plot(t[m],a['effective_e_um'][m],color=colors[ctrl],label=LABELS[ctrl],lw=1)
        ax.axhline(0,color='black',lw=.7);ax.set_xlabel('时间 / s');ax.set_ylabel('原始effective Y / μm');ax.set_title('启动局部' if ev['event']=='start' else '换向局部');ax.legend(ncol=3);ax.grid(alpha=.2);save(fig,'local_'+ev['event']+'.png','effective',[lo-.05,hi+.10],'s')
    atomic_json(d/'plot_manifest.json',dict(run_id=rid,generation_id=gen['generation_id'],plot_version=PLOT_VERSION,source_sha256=sha(Path(__file__)),figures=figrows,
        metric_tables={n:sha(d/n) for n in ['metrics.csv','costs.csv','events.csv']},all_three_from_this_run=True))
    main=[r for r in metrics if r['window']=='full' and r['channel']=='effective'];cost={r['case_id']:r for r in costs}
    lines=['# F组理想执行基线：本批结果','',f'run_id：`{rid}`；生成版本：`{gen["version"]}`；3条本批实际积分，各10000步。所有数据来自本批，未复用历史结果。',
        '', '|方案|std μm|峰峰 μm|均值 μm|绝对峰 μm|请求力RMS N|请求力矩RMS范数 Nm|','|---|---:|---:|---:|---:|---:|---:|']
    for r in main:
        cc=cost[r['case_id']];lines.append('|'+LABELS[r['case_id']]+'|'+'|'.join(f'{r[k]:.2f}' for k in ['std_um','ptp_um','mean_um','raw_peak_abs_um'])+f'|{cc["F_RMS_N"]:.4f}|{cc["tau_requested_RMS_norm_Nm"]:.4f}|')
    lines+=['',f'强SFC请求力RMS为线性的{cost["ASFC_strong"]["F_RMS_N"]/cost["Alinear"]["F_RMS_N"]:.2f}倍，本轮未做等代价机制比较。饱和样本：'+str({k:v['saturation_samples'] for k,v in cost.items()})+'。',
        '', '原g、关节背景、remaining及导数、事件、公共纠偏、模型和初态保持；原目标零线未移动。主全程包含冷初始化，辅助窗另列。图为原始输出，std不称纯高频振动。physical/effective分别统计。',
        '', '完整1ms NPZ和有符号向量CSV分别在full_1ms、csv_1ms。实际q与最终目标之差是总跟踪误差，不是补偿独有误差；同状态tau_realized_increment只是瞬时抗抖力矩增量，不是位置执行精度。R017尚无新执行模型，CB3未验证。',
        '', '新运行：`python run_slow.py new`，每次新run_id；仅查看/重画本批：`python run_slow.py show --run-id '+rid+'` / `python run_slow.py redraw --run-id '+rid+'`。']
    (d/'简报.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    html=['<!doctype html><meta charset="utf-8"><title>'+rid+'</title><style>body{font-family:Microsoft YaHei;max-width:1120px;margin:28px auto;background:#fafafa}img{width:100%}section{background:white;padding:16px;margin:18px 0}</style>',
        '<h1>指定批次的实际新基线</h1><p>run_id：'+rid+'；'+gen['version']+'。查看此页不会运行仿真。</p>',
        '<p>三组全部校验通过后生成。原始曲线，公共纠偏/原误差/事件全部保持；强SFC代价较高，未做等代价比较。</p>',
        '<p><a href="简报.md">简报</a> · <a href="metrics.csv">指标</a> · <a href="costs.csv">代价</a> · <a href="plot_manifest.json">图表数据身份</a></p>']
    for f in figrows:html.append('<section><img src="figures/'+f['figure']+'"></section>')
    (d/'results.html').write_text('\n'.join(html),encoding='utf-8')
    return d/'results.html'

def compare_batches(root,first,second):
    from run_slow import arrays,require,open_history,CONTROLLERS
    root=Path(root);open_history(first,root);open_history(second,root);rows=[]
    d1=root/'runs'/first;d2=root/'runs'/second;old=root/'historical/accepted_full_1ms'
    require(read(d1/'generation.json')['generation_id']==read(d2/'generation.json')['generation_id'],'Environment/source identity differs across batches')
    for ctrl in CONTROLLERS:
        a=arrays(d1/'full_1ms'/f'{ctrl}.npz');b=arrays(d2/'full_1ms'/f'{ctrl}.npz')
        require(set(a)==set(b),'Fields differ across batches')
        for relation,other in [('batch2',b),('accepted_old',arrays(old/f'PC__{ctrl}.npz'))]:
            for k in sorted(set(a)&set(other)):
                require(a[k].shape==other[k].shape,'Compared shape mismatch')
                err=float(np.max(abs(a[k].astype(float)-other[k].astype(float))))
                tol=1e-8 if k.endswith('_um') else 1e-9 if 'tau' in k or 'qfrc' in k or k.startswith('F') else 1e-12
                rows.append(dict(first_run=first,second_run=second,case_id=ctrl,comparison=relation,array=k,max_abs_difference=err,tolerance=tol,passed=err<=tol,
                    first_data_sha256=sha(d1/'full_1ms'/f'{ctrl}.npz'),other_data_sha256=sha(d2/'full_1ms'/f'{ctrl}.npz') if relation=='batch2' else sha(old/f'PC__{ctrl}.npz')))
                require(err<=tol,'Unexplained baseline response difference '+ctrl+'/'+k)
    table(root/'review/两批及旧结果逐数组复现.csv',rows)
    info=dict(passed=True,run_ids=[first,second],actual_integrations=6,source_environment_identical=True,array_checks=len(rows),max_abs_difference=max(r['max_abs_difference'] for r in rows),
        tolerances='Yum1e-8; force/torque1e-9; other float1e-12; flags exact; metadata timestamp/run_id/container excluded',old_results_only_explicit_reference=True)
    atomic_json(root/'review/reproducibility.json',info);return info
