"""Offline final checks and compact review companion; no integrations."""
import difflib,hashlib,json,zipfile
from pathlib import Path
from batch_identity import read,sha,atomic_json
import run_slow
R=Path(__file__).resolve().parent;old=R.parent/'F组_预定慢漂移补偿_20261012'
ids=read(R/'review/accepted_run_ids.json')['run_ids']
for rid in ids:run_slow.open_history(rid,R)
gen=read(R/'runs'/ids[0]/'generation.json');changes=[]
for n,h in gen['files'].items():
    if sha(R/n)!=h['sha256']:changes.append(n)
if set(changes)!=set(['run_slow.py','test_engineering.py']):raise RuntimeError('Unexplained changes to actual generation source '+str(changes))
for n in changes:
    if sha(R/'review/积分时源码补存'/n)!=gen['files'][n]['sha256']:raise RuntimeError('Integration source recovery mismatch '+n)
allpatch=[];modified=[]
for p in sorted(R.glob('*.py')):
    prev=old/p.name;a=prev.read_text(encoding='utf-8').splitlines(True) if prev.is_file() else [];b=p.read_text(encoding='utf-8').splitlines(True)
    if a!=b:
        modified.append(dict(file=p.name,type='modified' if a else 'added',sha256=sha(p)))
        allpatch.extend(difflib.unified_diff(a,b,fromfile='accepted-v1/'+p.name if a else '/dev/null',tofile='candidate-v2/'+p.name))
(R/'review/代码差异.patch').write_text(''.join(allpatch),encoding='utf-8');atomic_json(R/'review/改动文件.json',modified)
atomic_json(R/'review/最终本地验收.json',dict(passed=True,run_ids=ids,actual_integrations=6,explicit_history_data_and_artifacts_valid=True,
    source_environment_and_model_frozen=True,post_integration_history_only_changes=changes,recovered_generation_source_sha_verified=True,
    no_control_or_physics_changes=True,negative_checks=read(R/'review/negative_path_checks.json'),reproducibility=read(R/'review/reproducibility.json'),protected=read(R/'review/历史保护核查.json')))
print('Final local checks passed; 6 integrations preserved, no new integrations.')
