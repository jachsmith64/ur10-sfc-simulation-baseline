"""Independent text-package audit. Standard library only; no simulator/hardware import."""
from pathlib import Path
import csv,json,hashlib,math,struct
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
def csvrows(p):
    with (ROOT/p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def floats(row):return {k:float(v) for k,v in row.items()}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def percentile(x,q):
    x=sorted(x);v=(len(x)-1)*q/100;i=int(v);j=min(i+1,len(x)-1);return x[i]+(v-i)*(x[j]-x[i])
def stat(x):
    mean=math.fsum(x)/len(x)
    return dict(mean_um=mean,std_um=math.sqrt(math.fsum((v-mean)**2 for v in x)/len(x)),ptp_um=max(x)-min(x),raw_peak_abs_um=max(map(abs,x)),P95_abs_um=percentile(list(map(abs,x)),95))
def packed(values,kind='d'):return struct.pack('<'+kind*len(values),*values)
def main():
    man=read('review/text_package_manifest.json')
    for f in man['files']:
        p=ROOT/f['path'];assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256'],f['path']
    fr=read('review/frozen_version.json');index=csvrows('review/case_index.csv');ref=csvrows('review/分段指标.csv')
    inp=[floats(x) for x in csvrows('inputs/frozen_common_full_1ms.csv')]
    rawfd=[floats(x) for x in csvrows('inputs/Fd_event.csv')];zero=[floats(x) for x in csvrows('inputs/Fd_zero.csv')]
    assert len(index)==15 and len(inp)==10000 and all(r['status']=='OK' for r in index)
    assert sha(ROOT/'inputs/Fd_event.csv')==fr['event_sha256'] and sha(ROOT/'inputs/Fd_zero.csv')==fr['zero_event_sha256']
    ids={}
    # Rebuild byte-level logical identities without NumPy/MuJoCo. Canonical little-endian
    # float64/int64 equals the frozen arrays from this Windows host; CSV used17g.
    for s,desc in fr['scenarios'].items():
        g=desc['g'];h=hashlib.sha256()
        fields={'t':('t',1),'q':('q',6),'phase':('phase',1),'dq':('dq',6),'dqd':('dqd',6),'remaining':('remaining',1),'remaining_dot':('remaining_dot',1)}
        for key,(label,n) in fields.items():
            vals=[r[label if n==1 else label+'_J'+str(j+1)]*(g if key in ['dq','dqd','remaining','remaining_dot'] else 1.) for r in inp for j in range(n)]
            h.update(key.encode());h.update(packed([int(v) for v in vals],'q') if key=='phase' else packed(vals))
        h.update((fr['event_sha256'] if desc['event_on'] else fr['zero_event_sha256']).encode())
        ids[s]=h.hexdigest();assert ids[s]==desc['input_id']
    max_rel=0.;checks=0
    for item in index:
        name=item['case_id'];s=item['scenario'];ctrl=item['controller'];desc=fr['scenarios'][s];g=desc['g']
        assert item['input_id']==ids[s]
        p=ROOT/'outputs/compact_1ms'/f'{name}.csv';assert sha(p)==item['csv_sha256']
        data=[floats(r) for r in csvrows(str(p.relative_to(ROOT)))];assert len(data)==len(inp)
        pars=fr['parameters'][ctrl]
        for a,b,f,z in zip(data,inp,rawfd,zero):
            assert a['t_s']==b['t']==f['t_s']==z['t_s'] and a['phase']==b['phase']
            assert a['Fd_event_N']==(f['Fd_event_N'] if desc['event_on'] else z['Fd_event_N'])
            assert a['nominal_Y_m']==-.174
            residual=abs(a['effective_e_um']-a['physical_e_um']-g*b['remaining']*1e6);max_rel=max(max_rel,residual);assert residual<1e-8
            e=a['effective_e_um']*1e-6;v=a['effective_v_m_s']
            force=0. if ctrl=='A0' else -pars['K']*e-pars['B0']*v-pars['mu']*math.copysign(abs(v)**pars['n'],v)
            assert abs(force-a['F_requested_N'])<1e-9
            delta=[a['delta_u_J'+str(j+1)+'_rad'] for j in range(6)]
            assert abs(math.sqrt(math.fsum(d*d for d in delta))-a['delta_u_norm_rad'])<1e-12
            assert abs(5000*a['delta_u_norm_rad']-a['tau_requested_norm_Nm'])<1e-10
            assert abs(a['tau_requested_norm_Nm']-a['tau_realized_increment_norm_Nm'])<1e-8
            assert all(a[k]==0 for k in ['ctrl_saturated_any','force_saturated_any','joint_saturated_any'])
        for r in ref:
            if r['case_id']!=name:continue
            lo=float(r['t0_s']);hi=float(r['t1_s']);x=[a[r['channel']+'_e_um'] for a in data if lo<=a['t_s']<=hi]
            assert len(x)==int(r['n'])
            for key,value in stat(x).items():assert abs(value-float(r[key]))<1e-8,(name,r['window'],key)
            checks+=1
    # All case entries point to a single immutable input per scenario; g1/zero checked above.
    result=dict(pass_all=True,cases=15,full_1ms_rows=150000,phase_channel_metric_checks=checks,max_physical_plus_remaining_residual_um=max_rel,
        logical_frozen_input_ids_recomputed_without_simulator=True,requires_only_python_standard_library=True,images_or_meshes_required=False)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result
if __name__=='__main__':main()
