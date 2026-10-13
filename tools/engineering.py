"""Portable wrapper. Frozen scientific modules stay byte-identical."""
from pathlib import Path
import argparse, hashlib, importlib.metadata, json, os, platform, shutil, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
ENGINE=ROOT/'simulation/engineering_baseline_v2'
RUNS=['20261012T123028491525Z_9f7f6d3a','20261012T123042308025Z_05af114f']
def portable_git(root):
    exe=shutil.which('git')
    if not exe:return dict(available=False,error='git not on PATH; file hashes still recorded')
    def run(*args):
        p=subprocess.run([exe,'-c','core.quotepath=false','-C',str(root),*args],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=30)
        return dict(exit_code=p.returncode,output=p.stdout.strip(),error=p.stderr.strip())
    state=run('status','--porcelain=v1','--untracked-files=normal')
    return dict(root=run('rev-parse','--show-toplevel'),commit=run('rev-parse','HEAD'),status=state,
                dirty=bool(state['output']) if state['exit_code']==0 else None,
                archive_launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                note='Portable wrapper changes Git executable lookup only; frozen simulation modules and inputs are unchanged.')
def main():
    parser=argparse.ArgumentParser(description='check/show are historical only; new runs three actual simulations. Work on a copy.')
    parser.add_argument('action',choices=['check','show','redraw','new'])
    parser.add_argument('--run-id');parser.add_argument('--retry-of')
    args=parser.parse_args()
    if args.action in ['show','redraw'] and not args.run_id:parser.error('--run-id is required')
    if args.action=='new':
        marker=ROOT/'local_records/工作副本身份.json'
        if not marker.exists():raise RuntimeError('This is the fixed snapshot. Run tools/snapshot.py copy first; run new inside that independent copy.')
        if sys.version_info[:2]!=(3,12):raise RuntimeError('New simulation requires the documented Python3.12 environment.')
        expected={'numpy':'2.5.3','mujoco':'3.15.0','matplotlib':'3.11.2'}
        for name,version in expected.items():
            if importlib.metadata.version(name)!=version:raise RuntimeError('Dependency differs from accepted environment: '+name+'=='+version)
    sys.path.insert(0,str(ENGINE))
    import batch_identity
    batch_identity.git_info=portable_git
    import run_slow
    if args.action=='new':
        rid=run_slow.new_run(run_id=args.run_id,retry_of=args.retry_of)
        print('NEW ACTUAL SIMULATIONS:',rid)
    elif args.action=='check':
        import mujoco
        def forbidden(*a,**kw):raise RuntimeError('History check must not integrate MuJoCo.')
        mujoco.mj_step=forbidden
        paths=[str(run_slow.open_history(rid)) for rid in RUNS]
        print(json.dumps(dict(passed=True,historical_runs=RUNS,reports=paths,new_integrations=0),ensure_ascii=False,indent=2))
    else:
        path=run_slow.open_history(args.run_id,redraw=args.action=='redraw')
        print('HISTORICAL DATA ONLY:',path)
        if hasattr(os,'startfile'):os.startfile(str(path))
        else:
            import webbrowser
            webbrowser.open(path.as_uri())
if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    try:main()
    except Exception as e:print('FAILED (no fallback):',type(e).__name__,str(e),file=sys.stderr);raise SystemExit(2)
