"""Validate the immutable payload or copy it to a new experiment, using stdlib only."""
from pathlib import Path
from datetime import datetime
import argparse, hashlib, json, shutil, uuid
ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def verify(root=ROOT):
    root=Path(root).resolve();record=json.loads((root/'FILE_SHA256.json').read_text(encoding='utf-8'))
    for rel,row in record['files'].items():
        path=(root/rel).resolve()
        if not path.is_relative_to(root) or not path.is_file():raise RuntimeError('Missing/unsafe file: '+rel)
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise RuntimeError('Changed file: '+rel)
    return dict(passed=True,files=len(record['files']),bytes=sum(r['bytes'] for r in record['files'].values()))
def ignored(directory,names):
    return [n for n in names if n in ('.git','.venv','__pycache__','local_records') or n.endswith('.pyc')]
def copy_work(base,task):
    result=verify();base=Path(base).expanduser().resolve()
    if base.is_relative_to(ROOT):raise RuntimeError('Destination must be outside the fixed snapshot.')
    task=''.join(c for c in task if c.isalnum() or c in ('_','-'))[:40] or '新实验'
    destination=base/('工作副本_从慢纠偏基线出发_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+task+'_'+uuid.uuid4().hex[:6])
    base.mkdir(parents=True,exist_ok=True)
    shutil.copytree(ROOT,destination,ignore=ignored)
    verify(destination)
    info=dict(source=str(ROOT),destination=str(destination),source_manifest_sha256=sha(ROOT/'FILE_SHA256.json'),
              verified_files=result['files'],new_simulations=0,note='Independent copy; preserved accepted historical runs. Use tools/engineering.py new only when intended.')
    (destination/'local_records').mkdir()
    (destination/'local_records/工作副本身份.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    return info
if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Verify fixed payload; copy creates a unique independent directory and never runs simulation.')
    sub=parser.add_subparsers(dest='action',required=True)
    sub.add_parser('verify');p=sub.add_parser('copy');p.add_argument('--base',type=Path,default=ROOT.parent);p.add_argument('--task',default='新实验')
    args=parser.parse_args()
    if hasattr(__import__('sys').stdout,'reconfigure'):__import__('sys').stdout.reconfigure(encoding='utf-8')
    print(json.dumps(verify() if args.action=='verify' else copy_work(args.base,args.task),ensure_ascii=False,indent=2))
