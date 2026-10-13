"""Standard-library ZIP verification; no extraction, simulation or hardware."""
import csv,hashlib,io,json,math,statistics,sys,zipfile
def need(c,msg):
    if not c:raise RuntimeError(msg)
def close(x,y,tol=1e-8):need(abs(x-y)<=tol,'Numeric mismatch: '+str((x,y,tol)))
def check(path):
    with zipfile.ZipFile(path) as z:
        def js(n):return json.loads(z.read(n))
        def rows(n):return list(csv.DictReader(io.StringIO(z.read(n).decode('utf-8-sig'))))
        m=js('package_manifest.json');need(set(z.namelist())==set(m['files'])|{'package_manifest.json'},'Unexpected/missing package entries')
        for n,h in m['files'].items():need(hashlib.sha256(z.read(n)).hexdigest()==h,'Package hash mismatch '+n)
        index=js('review/data_deduplication.json');ids=index['run_ids'];logs={};checks=0
        for ctrl in ['A0','Alinear','ASFC_strong']:
            p=index['cases'][ctrl]['package_csv'];a=rows(p);need(len(a)==10000 and len(a[0])==99,'Incomplete signed-vector CSV')
            h=hashlib.sha256(z.read(p)).hexdigest()
            for rid in ids:need(js('runs/'+rid+'/csv_schema.json')['cases'][ctrl]['csv_sha256']==h,'Deduplicated second batch identity mismatch')
            a=[{k:float(v) for k,v in r.items()} for r in a];logs[ctrl]=a
            for i,r in enumerate(a):
                need(all(math.isfinite(v) for v in r.values()),'Nonfinite CSV');close(r['t'],i*.001,1e-12);close(r['model_time_s'],r['t'],1e-9)
                close(r['effective_y']-r['physical_y'],r['remaining'],1e-15)
                close(r['effective_v']-r['physical_v'],r['remaining_dot'],1e-15)
                for j in range(1,7):
                    s='_J'+str(j)
                    close(r['final_ctrl'+s],r['q_nominal'+s]+r['dq_C'+s]+r['common_delta_q'+s]+r['delta_u'+s],1e-12)
                    close(r['delta_u'+s],r['tau_requested'+s]/5000,1e-12)
                    close(r['qfrc_actuator'+s]-r['base_qfrc_actuator'+s],r['tau_realized_increment'+s],1e-10)
                    close(r['qfrc_applied'+s],r['qfrc_disturbance'+s],1e-12)
        for ctrl,a in logs.items():
            for i,r in enumerate(a):
                for k in ['remaining','remaining_dot','Fd','phase']+[q+'_J'+str(j) for q in ['q_nominal','dq_C','common_delta_q'] for j in range(1,7)]:need(r[k]==logs['A0'][i][k],'Background differs across controllers')
        for rid in ids:
            state=js('runs/'+rid+'/completed.json');need(state['completed'] and state['integrations_started']==3,'Wrong actual batch completion')
            for r in rows('runs/'+rid+'/metrics.csv'):
                x=[a[r['channel']+'_e_um'] for a in logs[r['case_id']] if float(r['t0_s'])<=a['t']<=float(r['t1_s'])]
                need(len(x)==int(r['n']),'Metric window count')
                for k,v in [('mean_um',statistics.mean(x)),('std_um',statistics.pstdev(x)),('ptp_um',max(x)-min(x)),('raw_peak_abs_um',max(map(abs,x)))]:close(float(r[k]),v)
                checks+=1
            for r in rows('runs/'+rid+'/costs.csv'):
                a=logs[r['case_id']];close(float(r['F_RMS_N']),math.sqrt(statistics.mean(x['F_requested']**2 for x in a)),1e-10)
                close(float(r['tau_requested_RMS_norm_Nm']),math.sqrt(statistics.mean(sum(x['tau_requested_J'+str(j)]**2 for j in range(1,7)) for x in a)),1e-10)
            for r in rows('runs/'+rid+'/events.csv'):
                a=logs[r['case_id']];key=r['channel']+'_e_um';b=statistics.median(x[key] for x in a if float(r['baseline_t0_s'])<=x['t']<=float(r['baseline_t1_s']))
                x=[x[key]-b for x in a if float(r['t0_s'])<=x['t']<=float(r['t1_s'])]
                close(float(r['baseline_um']),b);close(float(r['local_peak_abs_um']),max(map(abs,x)));close(float(r['threshold10um_occupancy_s']),sum(abs(v)>10 for v in x)*.001,1e-12)
        repro=js('review/reproducibility.json');need(repro['passed'] and repro['actual_integrations']==6 and repro['max_abs_difference']==0,'Reproducibility evidence')
        print(json.dumps(dict(passed=True,package_files=len(m['files']),csv_rows=30000,signed_columns=99,metric_rows_recomputed=checks,second_batch_csv_sha_verified=True,real_integrations=0),ensure_ascii=False))
if __name__=='__main__':check(sys.argv[1])
