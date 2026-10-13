"""Independent text audit: Python standard library only; no simulator or hardware imports."""
from pathlib import Path
import csv,json,hashlib,math,struct,statistics
ROOT=Path(__file__).resolve().parent

def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
def rows(p):
    with (ROOT/p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def floats(p):return [{k:float(v) for k,v in r.items()} for r in rows(p)]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def packed(x,kind='d'):return struct.pack('<'+kind*len(x),*x)
def stat(x):
    mean=math.fsum(x)/len(x);a=sorted(map(abs,x));u=(len(a)-1)*.95;i=int(u);j=min(i+1,len(a)-1)
    return dict(mean_um=mean,std_um=math.sqrt(math.fsum((v-mean)**2 for v in x)/len(x)),ptp_um=max(x)-min(x),raw_peak_abs_um=max(a),P95_abs_um=a[i]+(u-i)*(a[j]-a[i]))
def baseline(t,plan):
    levels=plan['levels_um'];b=levels[0];bd=bdd=0.
    for j,(lo,hi) in enumerate(plan['connections_s']):
        if t>hi:b=levels[j+1]
        elif t>=lo:
            T=hi-lo;u=(t-lo)/T;d=levels[j+1]-levels[j]
            b=levels[j]+d*(10*u**3-15*u**4+6*u**5)
            bd=d*(30*u*u-60*u**3+30*u**4)/T
            bdd=d*(60*u-180*u*u+120*u*u*u)/(T*T)
    return b,bd,bdd

def main():
    manifest=read('review/text_package_manifest.json')
    for item in manifest['files']:
        p=ROOT/item['path'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],item['path']
    fr=read('review/slow_frozen_version.json');source=read('baseline_review/frozen_version.json');g=fr['g']
    assert g==source['g'] and fr['parameters']==source['parameters']
    assert sha(ROOT/'baseline_review/frozen_version.json')==fr['source_version_sha256']
    for name,h in read('review/slow_implementation_freeze.json').items():assert sha(ROOT/name)==h,name
    inp=floats('inputs/frozen_common_full_1ms.csv');fd=floats('inputs/Fd_event.csv');pub=floats('inputs/public_slow_full_1ms.csv')
    assert len(inp)==len(fd)==len(pub)==10000
    assert sha(ROOT/'inputs/Fd_event.csv')==source['event_sha256']
    h=hashlib.sha256()
    fields={'t':1,'q':6,'phase':1,'dq':6,'dqd':6,'remaining':1,'remaining_dot':1}
    for key,n in fields.items():
        vals=[r[key if n==1 else key+'_J'+str(j+1)]*(g if key in ['dq','dqd','remaining','remaining_dot'] else 1.) for r in inp for j in range(n)]
        h.update(key.encode());h.update(packed([int(v) for v in vals],'q') if key=='phase' else packed(vals))
    h.update(source['event_sha256'].encode());source_id=h.hexdigest()
    assert source_id==fr['source_S3_input_id']==source['scenarios']['S3']['input_id']
    h=hashlib.sha256(source_id.encode());schema=read('review/public_input_text_schema.json')
    for key in sorted(schema):
        vals=[r[col] for r in pub for col in schema[key]['columns']]
        h.update(key.encode());h.update(packed(vals))
    assert h.hexdigest()==fr['shared_input_id']
    plan=read('review/baseline_plan.json');M=read('review/static_mapping.json')['M_rad_per_m'];gain=read('review/public_correction_acceptance.json')['gain']
    for a,b in zip(inp,pub):
        assert a['t']==b['t_s']
        center,bd,bdd=baseline(a['t'],plan)
        assert abs(center-b['common_baseline_um'])<1e-11
        for key,expected in [('common_delta_y_m',-gain*center*1e-6),('common_delta_y_dot_m_s',-gain*bd*1e-6),('common_delta_y_ddot_m_s2',-gain*bdd*1e-6)]:assert abs(b[key]-expected)<1e-15
        for key,ykey in [('common_delta_q','common_delta_y_m'),('common_delta_q_dot','common_delta_y_dot_m_s'),('common_delta_q_ddot','common_delta_y_ddot_m_s2')]:
            for j in range(6):assert b[key+'_J'+str(j+1)]==b[ykey]*M[j]
    index=rows('review/case_index.csv');metrics=rows('review/新旧分段指标.csv');events=rows('review/新旧事件指标.csv');platforms=rows('review/平台中心前后.csv')
    assert len(index)==6;checks=eventchecks=0;max_residual=0.;data_by_case={}
    for item in index:
        name=item['case_id'];new=item['group']=='new';ctrl=item['controller'];p=fr['parameters'][ctrl]
        assert item['input_id']==(fr['shared_input_id'] if new else source_id)
        path=ROOT/'outputs/compact_1ms'/f'{name}.csv';assert sha(path)==item['csv_sha256']
        data=floats(str(path.relative_to(ROOT)));assert len(data)==10000;data_by_case[name]=data
        for a,b,f,c in zip(data,inp,fd,pub):
            assert a['t_s']==b['t']==f['t_s']==c['t_s'] and a['phase']==b['phase'] and a['nominal_Y_m']==-.174
            assert a['Fd_event_N']==f['Fd_event_N']
            residual=abs(a['effective_e_um']-a['physical_e_um']-g*b['remaining']*1e6);max_residual=max(max_residual,residual);assert residual<1e-8
            e=a['effective_e_um']*1e-6;v=a['effective_v_m_s']
            F=0. if ctrl=='A0' else -p['K']*e-p['B0']*v-p['mu']*math.copysign(abs(v)**p['n'],v)
            assert abs(F-a['F_anti_requested_N'])<1e-9
            delta=[a['delta_anti_J'+str(j+1)+'_rad'] for j in range(6)]
            assert abs(math.sqrt(math.fsum(x*x for x in delta))-a['delta_anti_norm_rad'])<1e-12
            assert abs(5000*a['delta_anti_norm_rad']-a['tau_anti_requested_norm_Nm'])<1e-10
            assert abs(a['tau_anti_requested_norm_Nm']-a['tau_anti_realized_norm_Nm'])<1e-8
            for j,d in enumerate(delta,1):
                command=b['q_J'+str(j)]+b['dq_J'+str(j)]*g
                if new:command+=c['common_delta_q_J'+str(j)]
                assert command+d==a['final_ctrl_J'+str(j)+'_rad']
            assert all(a[k]==0 for k in ['ctrl_saturated_any','force_saturated_any','joint_saturated_any'])
        for r in metrics:
            if r['case_id']!=name:continue
            x=[a[r['channel']+'_e_um'] for a in data if float(r['t0_s'])<=a['t_s']<=float(r['t1_s'])];assert len(x)==int(r['n'])
            for key,v in stat(x).items():assert abs(v-float(r[key]))<1e-8,(name,r['window'],key)
            checks+=1
        for r in events:
            if r['case_id']!=name:continue
            key=r['channel']+'_e_um';b=statistics.median(a[key] for a in data if float(r['baseline_t0_s'])<=a['t_s']<=float(r['baseline_t1_s']))
            x=[a[key] for a in data if float(r['t0_s'])<=a['t_s']<=float(r['t1_s'])]
            assert abs(b-float(r['baseline_median_um']))<1e-10
            assert abs(max(abs(v-b) for v in x)-float(r['baseline_peak_abs_um']))<1e-8
            assert abs(sum(abs(v-b)>10 for v in x)*.001-float(r['time_abs_deviation_gt10um_s']))<1e-12
            eventchecks+=1
        for r in platforms:
            if r['case_id']!=name:continue
            x=[a for a in data if float(r['t0_s'])<=a['t_s']<=float(r['t1_s'])];assert len(x)==int(r['n'])
            for channel in ['physical','effective']:assert statistics.median(a[channel+'_e_um'] for a in x)==float(r[channel+'_median_um'])
    old=data_by_case['old__A0']
    for p in plan['platforms']:
        assert statistics.median(a['effective_e_um'] for a in old if p['t0_s']<=a['t_s']<=p['t1_s'])==p['effective_median_um']
    ledger=read('review/slow_run_budget.json');assert len(ledger)==3 and all(x['status']=='OK' and x['n_steps_done']==10000 for x in ledger)
    result=dict(passed=True,cases=6,full_1ms_case_rows=60000,new_integrations=3,budget=6,metric_checks=checks,event_checks=eventchecks,
        max_physical_plus_remaining_residual_um=max_residual,source_and_public_input_ids_rebuilt=True,public_baseline_mapping_derivatives_recomputed=True,
        actual_final_commands_checked=True,requires_only_python_standard_library=True)
    print(json.dumps(result,ensure_ascii=False,indent=2));return result
if __name__=='__main__':main()
