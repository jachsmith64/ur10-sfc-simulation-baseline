"""Read-only numerical review of uploaded slow-correction package; no simulation."""
from pathlib import Path
import ast,csv,json,hashlib,xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parent
P=ROOT/'slow_review_20261012/F组_预定慢漂移补偿_20261012'
OLD=ROOT/'short_review_20261012/F组_10秒往返_A事件恢复_20261012'
OUT=ROOT/'outputs/slow_review';OUT.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def array(p):return np.genfromtxt(p,delimiter=',',names=True,encoding='utf-8-sig')
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stat(y):return dict(mean_um=float(np.mean(y)),std_um=float(np.std(y)),ptp_um=float(np.ptp(y)),raw_peak_abs_um=float(np.max(abs(y))),P95_abs_um=float(np.percentile(abs(y),95)))

result={'scope':'saved complete 1ms data and static code; no MuJoCo or hardware','checks':{}}
manifest=read(P/'review/text_package_manifest.json')['files']
assert all((P/r['path']).stat().st_size==r['bytes'] and sha(P/r['path'])==r['sha256'] for r in manifest)
for p in P.glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
metadata={str(p.relative_to(P)):read(p) for p in P.rglob('*.json')}
unchanged=[p.name for p in P.glob('*.py') if (OLD/p.name).exists() and p.read_bytes()==(OLD/p.name).read_bytes()]
assert all((P/'assets_ur10e'/n).read_bytes()==(OLD/'assets_ur10e'/n).read_bytes() for n in ['scene.xml','ur10e.xml'])
assert read(P/'review/protected_before.json')==read(P/'review/protected_after.json')
result['checks'].update(manifest_payloads=len(manifest),parsed_python_files=len(list(P.glob('*.py'))),parsed_json_files=len(metadata),unchanged_sources=unchanged,protected_manifest_entries=len(read(P/'review/protected_before.json')))
c=array(P/'inputs/frozen_common_full_1ms.csv');pub=array(P/'inputs/public_slow_full_1ms.csv');fd=array(P/'inputs/Fd_event.csv')
previous=array(OLD/'inputs/frozen_common_full_1ms.csv')
assert c.dtype.names==previous.dtype.names and np.array_equal(c,previous)
assert (P/'inputs/Fd_event.csv').read_bytes()==(OLD/'inputs/Fd_event.csv').read_bytes()
fr=read(P/'review/slow_frozen_version.json');oldfr=read(OLD/'review/frozen_version.json')
assert fr['g']==oldfr['g'] and fr['parameters']==oldfr['parameters']
assert read(P/'baseline_review/frozen_version.json')==oldfr
g=fr['g'];t=c['t'];assert np.array_equal(t,np.arange(10000)*.001)
result['checks']['old_input_and_event_unchanged']=True
plan=read(P/'review/baseline_plan.json');b=np.full(10000,plan['levels_um'][0]);bd=np.zeros(10000);bdd=np.zeros(10000)
for j,(lo,hi) in enumerate(plan['connections_s']):
    u=np.clip((t-lo)/(hi-lo),0,1);d=plan['levels_um'][j+1]-plan['levels_um'][j]
    b+=d*u**3*(10-15*u+6*u*u)
    bd+=d*30*u*u*(1-u)**2/(hi-lo)
    bdd+=d*60*u*(1-u)*(1-2*u)/(hi-lo)**2
assert np.max(abs(b-pub['common_baseline_um']))<2e-14
assert np.max(abs(pub['common_delta_y_m']+b*1e-6))<1e-18
assert np.max(abs(pub['common_delta_y_dot_m_s']+bd*1e-6))<1e-18
assert np.max(abs(pub['common_delta_y_ddot_m_s2']+bdd*1e-6))<1e-18
M=np.array(read(P/'review/static_mapping.json')['M_rad_per_m'])
J=np.array(read(P/'review/static_mapping.json')['J0_translation_m_per_rad'])
assert np.allclose(M,np.linalg.pinv(J)@np.array([0.,1.,0.]),atol=1e-15,rtol=0)

# Kinematic model reconstructed from XML; no MuJoCo import.
def quat(q):
    w,x,y,z=np.array(q)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def rotation(axis,a):
    v=np.array(axis)/np.linalg.norm(axis);x,y,z=v
    K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+np.sin(a)*K+(1-np.cos(a))*(K@K)
def numbers(s):return np.fromstring(s,sep=' ')
xml=ET.parse(P/'assets_ur10e/ur10e.xml').getroot();chain=[];node=xml.find('worldbody/body')
while node is not None:chain.append(node);node=node.find('body')
def fk(q,with_jac=False):
    R=np.eye(3);pos=np.zeros(3);orig=[];axes=[];qi=0
    for node in chain:
        pos+=R@numbers(node.get('pos','0 0 0'));R=R@quat(numbers(node.get('quat','1 0 0 0')))
        joint=node.find('joint')
        if joint is not None:
            axis=numbers(joint.get('axis','0 1 0'));orig.append(pos.copy());axes.append(R@axis);R=R@rotation(axis,q[qi]);qi+=1
    site=chain[-1].find('site');pos+=R@numbers(site.get('pos','0 0 0'))
    Jac=np.column_stack([np.cross(ax,pos-o) for ax,o in zip(axes,orig)])
    return (pos,Jac) if with_jac else pos
q=np.column_stack([c['q_J'+str(j)] for j in range(1,7)])
p0,J0=fk(q[0],True);assert np.max(abs(J0-J))<2e-15
correction=np.column_stack([pub['common_delta_q_J'+str(j)] for j in range(1,7)])
assert np.max(abs(correction-pub['common_delta_y_m'][:,None]*M))<1e-19
sample=np.arange(0,10000,10)
shifts=np.array([fk(q[i]+correction[i])-fk(q[i]) for i in sample])
result['mapping_diagnostics']=dict(max_geometric_Y_error_um=float(np.max(abs(shifts[:,1]-pub['common_delta_y_m'][sample]))*1e6),max_geometric_X_shift_um=float(np.max(abs(shifts[:,0]))*1e6),max_geometric_Z_shift_um=float(np.max(abs(shifts[:,2]))*1e6),condition=float(np.linalg.cond(J)),public_peak_abs_joint_deg=float(np.max(abs(correction))*180/np.pi))

logs={p.stem:array(p) for p in (P/'outputs/compact_1ms').glob('*.csv')}
metrics=rows(P/'review/新旧分段指标.csv');evrows=rows(P/'review/新旧事件指标.csv');costrows=rows(P/'review/抗抖代价_公共位置代价分列.csv')
maxerr=0.;mchecks=0;events=0;result['metrics']={};result['costs']={}
for name,a in logs.items():
    group,ctrl=name.split('__');new=group=='new';p=fr['parameters'][ctrl]
    assert np.array_equal(a['t_s'],t) and np.array_equal(a['phase'],c['phase']) and np.array_equal(a['Fd_event_N'],fd['Fd_event_N']) and np.all(a['nominal_Y_m']==-.174)
    rem=a['effective_e_um']-a['physical_e_um']-g*c['remaining']*1e6;assert np.max(abs(rem))<2e-11
    v=a['effective_v_m_s'];F=np.zeros_like(v) if ctrl=='A0' else -p['K']*a['effective_e_um']*1e-6-p['B0']*v-p['mu']*np.sign(v)*abs(v)**p['n']
    assert np.max(abs(F-a['F_anti_requested_N']))<1e-14
    delta=np.column_stack([a['delta_anti_J'+str(j)+'_rad'] for j in range(1,7)])
    command=np.column_stack([a['final_ctrl_J'+str(j)+'_rad'] for j in range(1,7)])
    dq=np.column_stack([c['dq_J'+str(j)] for j in range(1,7)])
    assert np.array_equal(q+g*dq+(correction if new else 0)+delta,command)
    assert np.max(abs(5000*np.linalg.norm(delta,axis=1)-a['tau_anti_requested_norm_Nm']))<2e-15
    assert np.max(abs(a['tau_anti_requested_norm_Nm']-a['tau_anti_realized_norm_Nm']))<2e-12
    assert all(not a[k].any() for k in ['ctrl_saturated_any','force_saturated_any','joint_saturated_any'])
    result['metrics'][name]={}
    for r in metrics:
        if r['case_id']!=name:continue
        mask=(t>=float(r['t0_s']))&(t<=float(r['t1_s']));assert mask.sum()==int(r['n']);s=stat(a[r['channel']+'_e_um'][mask])
        maxerr=max(maxerr,max(abs(value-float(r[k])) for k,value in s.items()));mchecks+=1
        result['metrics'][name][r['channel']+'__'+r['window']]=s
    for r in evrows:
        if r['case_id']!=name:continue
        key=r['channel']+'_e_um';mask=(t>=float(r['t0_s']))&(t<=float(r['t1_s']));bm=(t>=float(r['baseline_t0_s']))&(t<=float(r['baseline_t1_s']));base=np.median(a[key][bm]);x=a[key][mask]-base
        assert abs(base-float(r['baseline_median_um']))<1e-12 and abs(np.max(abs(x))-float(r['baseline_peak_abs_um']))<1e-12
        assert abs(np.count_nonzero(abs(x)>10)*.001-float(r['time_abs_deviation_gt10um_s']))<1e-12;events+=1
    cc=dict(F_anti_RMS_N=float(np.sqrt(np.mean(F*F))),F_anti_abs_peak_N=float(np.max(abs(F))),tau_anti_RMS_norm_Nm=float(np.sqrt(np.mean(a['tau_anti_requested_norm_Nm']**2))),tau_anti_peak_norm_Nm=float(np.max(a['tau_anti_requested_norm_Nm'])),anti_position_peak_abs_deg=float(np.max(abs(delta))*180/np.pi))
    stored=next(r for r in costrows if r['case_id']==name);assert max(abs(val-float(stored[k])) for k,val in cc.items())<1e-13
    result['costs'][name]=cc
    if not new:
        olda=array(OLD/'outputs/compact_1ms'/('S3__'+ctrl+'.csv'))
        commoncols=[k for k in a.dtype.names if k in olda.dtype.names]
        assert all(np.array_equal(a[k],olda[k]) for k in commoncols)
        renames={'F_anti_requested_N':'F_requested_N','tau_anti_requested_norm_Nm':'tau_requested_norm_Nm','tau_anti_realized_norm_Nm':'tau_realized_increment_norm_Nm','delta_anti_norm_rad':'delta_u_norm_rad'}
        for key,okey in renames.items():assert np.array_equal(a[key],olda[okey])
        for j in range(1,7):assert np.array_equal(a['delta_anti_J'+str(j)+'_rad'],olda['delta_u_J'+str(j)+'_rad'])
assert maxerr<1e-12
result['checks'].update(saved_case_rows=60000,independent_metric_checks=mchecks,event_checks=events,max_metric_difference=maxerr,prior_S3_logs_match=True,actual_command_equations_verified=True,control_formula_verified=True)
result['SFC_vs_linear']={}
for group in ['new','old']:
    s=result['metrics'][group+'__ASFC_strong'];l=result['metrics'][group+'__Alinear']
    result['SFC_vs_linear'][group]={key:{metric:100*(1-s[key][metric]/l[key][metric]) for metric in ['std_um','ptp_um','raw_peak_abs_um']} for key in s}
    result['SFC_vs_linear'][group]['force_RMS_ratio']=result['costs'][group+'__ASFC_strong']['F_anti_RMS_N']/result['costs'][group+'__Alinear']['F_anti_RMS_N']

# Disentangle faithful command execution from a shifted drawing, using existing logs.
diag={}
for ctrl in ['A0','Alinear','ASFC_strong']:
    old=logs['old__'+ctrl]['effective_e_um'];new=logs['new__'+ctrl]['effective_e_um']
    difference=new-(old-b)
    diag[ctrl]=dict(max_deviation_from_offline_subtraction_um=float(np.max(abs(difference))),pre_rest_median_new_um=float(np.median(new[(t>=.85)&(t<=1.075)])))
result['offline_subtraction_diagnostic']=diag
result['passed']=True
(OUT/'independent_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
font=Path('/workspace/scratch/db9fa86f2df8/_qa/handcopy/fonts/NotoSansCJKsc-Regular.otf')
fm.fontManager.addfont(str(font));plt.rcParams.update({'font.family':fm.FontProperties(fname=font).get_name(),'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors={'A0':'#8b939c','Alinear':'#2272a8','ASFC_strong':'#d04d4d'};labels={'A0':'公共纠偏，无附加抗抖','Alinear':'公共纠偏＋线性','ASFC_strong':'公共纠偏＋强SFC'}
fig,axs=plt.subplots(2,1,figsize=(11.4,7.2),gridspec_kw={'height_ratios':[1.45,1]},layout='constrained')
for ctrl in colors:axs[0].plot(t,logs['new__'+ctrl]['effective_e_um'],lw=1.1,color=colors[ctrl],label=labels[ctrl])
for lo,hi,label in [(0,.8,'初始化'),(1.285,1.926,'A启动'),(4.58,5.643,'换向'),(8.667,9.999,'减速／停止')]:
    axs[0].axvspan(lo,hi,color='#7aa7cf' if lo else '#d4a66d',alpha=.12);axs[0].text((lo+hi)/2,107,label,ha='center',fontsize=9)
axs[0].set(xlim=(0,10),ylim=(-115,122),ylabel='实际effective Y偏差 / 微米',xlabel='时间 / 秒',title='共同预定慢纠偏条件：完整10秒原始响应（零线仍为原名义轨迹）')
axs[0].legend(ncol=3,loc='lower left',fontsize=9)
m=(t>=2.226)&(t<=4.38)
for ctrl in colors:axs[1].plot(c['along'][m],logs['new__'+ctrl]['effective_e_um'][m],lw=1.2,color=colors[ctrl])
axs[1].set(ylabel='原始Y偏差 / 微米',xlabel='普通去程名义沿程位置 / 毫米',title='普通去程局部：中心接近零，波纹仍保留')
for ax in axs:ax.axhline(0,color='#333',lw=.8);ax.grid(alpha=.18)
fig.savefig(OUT/'F慢纠偏_三方案原始响应.png',dpi=170);plt.close(fig)
fig,axs=plt.subplots(2,1,figsize=(11.4,6.6),sharex=True,layout='constrained')
axs[0].plot(t,logs['old__A0']['effective_e_um'],color='#909090',lw=1,label='旧A0：未加公共纠偏');axs[0].plot(t,b,color='#d59232',lw=2,label='从旧A0提取的四平台慢中心')
axs[1].plot(t,logs['new__A0']['effective_e_um'],color='#58616b',lw=1,label='新A0：执行公共位置纠偏后的实际响应')
axs[0].set_title('先确定慢中心，再经位置通道拉回；两条都是未减均值的原始数据')
for ax in axs:ax.set(ylabel='effective Y偏差 / 微米',ylim=(-135,112));ax.axhline(0,color='#333',lw=.8);ax.grid(alpha=.18);ax.legend(loc='lower left',fontsize=9)
axs[1].set_xlabel('时间 / 秒');axs[1].set_xlim(0,10)
fig.savefig(OUT/'F慢纠偏_新旧A0与慢中心.png',dpi=170);plt.close(fig)
print(json.dumps({'checks':result['checks'],'mapping':result['mapping_diagnostics'],'new_effective_full':result['SFC_vs_linear']['new']['effective__full'],'force_ratio':result['SFC_vs_linear']['new']['force_RMS_ratio'],'subtraction':diag},ensure_ascii=False,indent=2))
