from pathlib import Path
import sys,os,time,json,hashlib
P=Path(__file__).resolve().parent;B=P/'01_工程';E=P/'evidence';R=P/'results';RUN=P/'runs132'
sys.dont_write_bytecode=True;os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';sys.path.insert(0,str(B))
import e_bootstrap
import numpy as np,mujoco
from batch_identity import atomic_json,new_id
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def clean(x):
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def save(p,x):atomic_json(Path(p),clean(x))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrays(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def source_id():return {p.relative_to(P).as_posix():sha(p) for p in list(P.glob('b132*.py'))+list(B.glob('*.py'))+list(B.glob('*.json'))}
class Budget:
 def __enter__(self):
  self.path=E/'physical_attempt_ledger.json';self.data=read(self.path);self.active=None;self.original=mujoco.mj_step
  def counted(m,d,*a,**kw):
   if self.active is None:raise RuntimeError('Unidentified physical integration attempt')
   if self.data['used_physical_steps']>=180000:raise RuntimeError('180000step budget exhausted')
   self.data['used_physical_steps']+=1;self.active['physical_steps']+=1
   try:return self.original(m,d,*a,**kw)
   finally:
    if self.data['used_physical_steps']%1000==0:save(self.path,self.data)
  mujoco.mj_step=counted;self.other={k:getattr(mujoco,k) for k in ['mj_step1','mj_step2']}
  def refuse(*a,**kw):raise RuntimeError('Split integration prohibited')
  for k in self.other:setattr(mujoco,k,refuse)
  return self
 def start(self,condition,method,path):
  if self.data['used_physical_steps']+10000>180000:raise RuntimeError('Insufficient remaining budget for10s attempt')
  self.active=dict(condition=condition,method=method,path=str(path),physical_steps=0,status='RUNNING');self.data['attempts'].append(self.active);save(self.path,self.data)
 def finish(self,row):self.active.update(status=row['status'],valid_steps=row['n_steps_done'],reason=row['note']);save(self.path,self.data);self.active=None
 def __exit__(self,*exc):
  mujoco.mj_step=self.original
  for k,v in self.other.items():setattr(mujoco,k,v)
  if exc[0] and self.active:self.active.update(status='FAILED',reason=str(exc[1]))
  save(self.path,self.data)
