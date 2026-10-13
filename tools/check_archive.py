"""Offline check of saved arrays, attached curves and model references. No MuJoCo import."""
from pathlib import Path
import ast, hashlib, json, xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
ENGINE=ROOT/'simulation/engineering_baseline_v2'
OLD=ROOT/'archive/approved_slow_v1'
CASES=['A0','Alinear','ASFC_strong']
RUNS=['20261012T123028491525Z_9f7f6d3a','20261012T123042308025Z_05af114f']
def need(ok,message):
    if not ok:raise RuntimeError(message)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def inspect():
    identity=read(ROOT/'approved_figure/认可图_版本身份.json')
    need(sha(ROOT/'approved_figure/F慢纠偏_三方案原始响应.png')==identity['approved_png_sha256'],'Approved PNG hash')
    need(sha(ROOT/'approved_figure/F慢纠偏_三方案原始响应_数据.csv')==identity['plot_data_sha256'],'Approved CSV hash')
    need(sha(ROOT/'reference/slow_review_independent.py')==identity['original_audit_plot_script_sha256'],'Original plot source hash')
    data=np.genfromtxt(ROOT/'approved_figure/F慢纠偏_三方案原始响应_数据.csv',delimiter=',',names=True,encoding='utf-8-sig')
    t=np.arange(10000)*.001;need(len(data)==10000 and np.array_equal(data['t_s'],t),'Attached exact10000 pre-step1ms points')
    common=arrays(ENGINE/'inputs/frozen_common.npz');corr=arrays(ENGINE/'inputs/public_slow_correction.npz')
    need(np.array_equal(data['along_mm'],common['along']),'Approved along position')
    freeze=read(ENGINE/'historical/public_slow_review/slow_frozen_version.json');g=freeze['g']
    need(g==0.6634406421952753,'Frozen g')
    need(sha(ENGINE/'inputs/frozen_common.npz')==read(ENGINE/'baseline_review/frozen_version.json')['common_sha256'],'Common input identity')
    need(sha(ENGINE/'inputs/public_slow_correction.npz')==freeze['public_correction_sha256'],'Correction identity')
    comparisons=0;maximum=0.;metrics={};required_fields=None;generation_matches=0;model_references=0
    for root in [ENGINE,OLD]:
        for p in root.glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
        for p in (root/'assets_ur10e').glob('*.xml'):
            xml=ET.parse(p).getroot()
            for inc in xml.findall('.//include'):need((p.parent/inc.attrib['file']).is_file(),'Missing model include')
            meshdir=xml.find('compiler').get('meshdir','') if xml.find('compiler') is not None else ''
            for mesh in xml.findall('.//mesh'):
                if mesh.get('file'):need((p.parent/meshdir/mesh.get('file')).is_file(),'Missing required mesh');model_references+=1
    source=read(ROOT/'review/source_copy_manifest.json')
    for rel,row in source['copied_files'].items():need(sha(ROOT/rel)==row['sha256'],'Copied source changed: '+rel)
    provenance=read(ROOT/'third_party/ur10e/provenance.json')
    need(len(provenance['mesh_checks'])==20,'Expected20 model meshes')
    for mesh in provenance['mesh_checks']:need(sha(ENGINE/'assets_ur10e/assets'/mesh['path'])==mesh['sha256'],'Mesh identity')
    for ctrl in CASES:
        first=arrays(ENGINE/'runs'/RUNS[0]/'full_1ms'/f'{ctrl}.npz')
        old=arrays(OLD/'outputs/full_1ms'/f'PC__{ctrl}.npz')
        second=arrays(ENGINE/'runs'/RUNS[1]/'full_1ms'/f'{ctrl}.npz')
        need(set(first)==set(second)==set(old),'Saved array fields differ')
        for other in [second,old]:
            for key in first:
                need(first[key].shape==other[key].shape,'Saved shape differs')
                err=float(np.max(abs(first[key].astype(float)-other[key].astype(float))))
                need(err==0.,'Saved array mismatch: '+ctrl+'/'+key);maximum=max(maximum,err);comparisons+=1
        required_fields=len(first)
        for k,a in first.items():need(a.shape[0]==10000 and np.isfinite(a).all(),'Saved invalid field: '+k)
        need(np.array_equal(first['t'],t) and np.array_equal(first['q_nominal'],common['q']),'Nominal reference')
        need(np.array_equal(first['dq_C'],common['dq']*g) and np.array_equal(first['remaining'],common['remaining']*g),'Source layers')
        for k,a in corr.items():need(np.array_equal(first[k],a),'Public correction mismatch')
        need(np.array_equal(first['base_ctrl'],first['q_nominal']+first['dq_C']+first['common_delta_q']),'Base command equation')
        need(np.array_equal(first['final_ctrl'],first['base_ctrl']+first['delta_u']),'Final command equation')
        need(np.max(abs(first['effective_e_um']-first['physical_e_um']-first['remaining']*1e6))<3e-11,'Effective/physical relation')
        y=data[ctrl+'_effective_e_um'];need(np.array_equal(y,first['effective_e_um']),'Approved curve exact match: '+ctrl)
        stat=dict(std_um=float(np.std(y)),ptp_um=float(np.ptp(y)))
        need(max(abs(stat[k]-identity['case_metrics'][ctrl][k]) for k in stat)<1e-12,'Approved raw statistics')
        metrics[ctrl]=stat
    for rid in RUNS:
        d=ENGINE/'runs'/rid;done=read(d/'completed.json');need(done['completed'] is True and done['status']=='SUCCESS','Batch incomplete')
        gen=read(d/'generation.json')
        for rel,row in gen['files'].items():
            p=ENGINE/rel
            if sha(p)!=row['sha256'] and rel in ['run_slow.py','test_engineering.py']:
                p=ENGINE/'review/积分时源码补存'/rel
            need(p.is_file() and sha(p)==row['sha256'],'Generation source unavailable: '+rel);generation_matches+=1
        for case in done['cases']:need(sha(d/'full_1ms'/f"{case['case_id']}.npz")==case['data_sha256'],'Batch data hash')
        for key,value in read(d/'input_identity.json')['run_input_snapshots'].items():need(sha(d/'frozen_inputs'/key)==value,'Run input snapshot')
    for p in list((ROOT/'tools').glob('*.py'))+[ROOT/'approved_figure/redraw_approved.py']:ast.parse(p.read_text(encoding='utf-8'))
    return dict(passed=True,scope='Saved arrays/static source/assets only; no MuJoCo import or new integration, no hardware.',new_integrations=0,
        approved_png_sha256=identity['approved_png_sha256'],approved_csv_sha256=identity['plot_data_sha256'],samples_per_case=10000,
        g=g,metrics=metrics,array_comparisons=comparisons,max_abs_difference=maximum,fields_per_npz=required_fields,
        generation_file_checks=generation_matches,model_reference_checks=model_references,meshes_per_model=20,
        copied_source_files=len(source['copied_files']),existing_integrations=6,
        not_run=['New simulation/reconstruction/parameter scanning','Fresh dependency installation on another PC','Robot interface or action'],
        history_only_source_changes=['run_slow.py','test_engineering.py; integration-time originals preserved and SHA-checked'])
if __name__=='__main__':print(json.dumps(inspect(),ensure_ascii=False,indent=2))
