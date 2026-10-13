"""Per-run provenance and atomic status writes; no cache/reuse policy."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,platform,subprocess,sys,uuid
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic_json(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp-'+uuid.uuid4().hex)
    try:
        tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,p)
    finally:
        if tmp.exists():tmp.unlink()
def safe_id(run_id):
    import re
    if not isinstance(run_id,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,96}',run_id):raise ValueError('Invalid run_id')
    return run_id
def new_id():return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:8]
def new_directory(root,run_id=None):
    run_id=safe_id(run_id or new_id());d=Path(root)/'runs'/run_id;d.mkdir(parents=True,exist_ok=False);(d/'full_1ms').mkdir();return run_id,d
def files_snapshot(root):
    root=Path(root);chosen=set(root.glob('*.py'))|set(root.glob('*.json'))
    for folder in ['assets_ur10e','inputs']:chosen.update(p for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    chosen.update(root/'historical/public_slow_review'/n for n in ['slow_frozen_version.json','public_correction_acceptance.json','ordinary_calibration.json','event_reconstruction.json','baseline_plan.json','static_mapping.json','metric_windows.json','event_targets.json','trajectory.json'])
    chosen.add(root/'baseline_review/frozen_version.json')
    return {p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(chosen)}
def git_info(root):
    exe=Path('D:/Software/Git/cmd/git.exe')
    if not exe.exists():exe=Path('C:/Users/kinli/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe')
    def run(*args):
        r=subprocess.run([str(exe),'-c','core.quotepath=false','-C',str(root),*args],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=30)
        return dict(exit_code=r.returncode,output=r.stdout.strip(),error=r.stderr.strip())
    try:
        top=run('rev-parse','--show-toplevel');head=run('rev-parse','HEAD');status=run('status','--porcelain=v1','--untracked-files=normal')
        return dict(root=top,commit=head,status=status,dirty=bool(status['output']) if status['exit_code']==0 else None,
            note='Parent repository has untracked F candidates. Actual source/assets hashes identify this implementation; Git HEAD alone is not this F version.')
    except Exception as e:return dict(available=False,error=str(e))
def provenance(root,version,input_id):
    import numpy,mujoco
    files=files_snapshot(root);env=dict(python=sys.version,python_executable=sys.executable,numpy=numpy.__version__,mujoco=mujoco.__version__,
        numpy_module=numpy.__file__,mujoco_module=mujoco.__file__,platform=platform.platform(),dt_s=.001,control_update_steps=1)
    canonical=json.dumps(dict(files=files,environment=env,version=version,input_id=input_id),sort_keys=True,ensure_ascii=False,separators=(',',':')).encode('utf-8')
    return dict(version=version,generation_id=hashlib.sha256(canonical).hexdigest(),files=files,environment=env,git=git_info(root),
        inputs_are_explicitly_inherited=True,result_reuse=False,created_utc=datetime.now(timezone.utc).isoformat())
