"""Frozen local predictor. Receives current state and known nominal only.
There is no background, remaining, Fd, future measurement, or environment handle
in the prediction interface. Scalar and batched SFC use the inherited pure function.
"""
import numpy as np
from sfc import force_terms
from gamma_config import DEFAULT_W_GAMMA
H=.001;N=160;TD=.032;ETA=.1;DP=5e-6
def predict(current,known,paths,targets,gamma_targets,old_target,include_sfc,params):
    w_gamma = float(params.get('w_gamma', DEFAULT_W_GAMMA))
    if not np.isfinite(w_gamma) or w_gamma <= 0:
        raise ValueError('w_gamma must be finite and positive')
    # Nine candidates share one M/J/T/h_r and one initial physical state.
    count=len(targets);q=np.tile(current['q'],(count,1));qd=np.tile(current['qd'],(count,1))
    gam=np.full(count,current['gamma']);q0=current['q'];J=current['J'];kp=current['kp'];kd=current['kd'];Tp=current['Tp']
    M=current['M'];D=kd+current['damping'];A=np.linalg.solve(M+H*np.diag(D),M);B=np.linalg.solve(M+H*np.diag(D),H*np.eye(6))
    b=current['y_meas']-current['y_fk'];bd=current['v_meas']-J@current['qd']
    offset=current['base']-known[0]
    errors=np.empty((count,N+1));vels=np.empty_like(errors);taus=np.empty((count,N+1,6));cmds=np.empty_like(taus)
    reasons=[None]*count
    for i in range(N+1):
        decay=np.exp(-i*H/TD)
        y=current['y_fk']+(q-q0)@J+b+TD*(1-decay)*bd
        v=qd@J+decay*bd;e=y-current['y_nominal']
        F=force_terms(e,v,params['K'],params['B0'],params['mu'],params['n'],'ASFC',gam)['F_total'] if include_sfc else np.zeros(count)
        with np.errstate(over='ignore',invalid='ignore'):
            command=known[i]+offset+paths[:,i,0,None]*Tp+(F[:,None]*J)/kp
            tau=kp*(command-q)-kd*qd
        errors[:,i]=e;vels[:,i]=v;taus[:,i]=tau;cmds[:,i]=command
        ctrl_bad=np.any((command<current['ctrl_lo'])|(command>current['ctrl_hi']),axis=1)
        tau_bad=np.any((tau<current['tau_lo'])|(tau>current['tau_hi']),axis=1)
        finite=np.isfinite(e)&np.isfinite(v)&np.all(np.isfinite(tau),axis=1)
        for c in range(count):
            if reasons[c] is None:
                if not finite[c]:reasons[c]=f'nonfinite prediction at {i}'
                elif ctrl_bad[c]:reasons[c]=f'compiled ctrlrange at {i}'
                elif tau_bad[c]:reasons[c]=f'compiled actuator/joint force boundary at {i}'
        if i<N:
            qd=qd@A.T+(kp*(command-q)-current['bias'])@B.T
            q=q+H*qd
            gam+=np.clip(gamma_targets-gam,-H,H)
    limits=current['tau_scale'];vs=params['v_ref']
    with np.errstate(over='ignore',invalid='ignore'):
        stage=(errors[:,1:]/50e-6)**2+.2*(vels[:,1:]/vs)**2+.05*np.sum((taus[:,1:]/limits)**2,axis=2)+.02*np.sum((np.diff(taus,axis=1)/H/(limits/.050))**2,axis=2)
        score=np.mean(stage,axis=1)+2*((errors[:,-1]/50e-6)**2+.2*(vels[:,-1]/vs)**2)+.01*((targets-old_target)/DP)**2+w_gamma*((gamma_targets-1)/ETA)**2
    for c in range(count):
        if reasons[c] is not None or not np.isfinite(score[c]):
            score[c]=np.inf;reasons[c]=reasons[c] or 'nonfinite score'
    return dict(scores=score,reasons=reasons,e=errors,v=vels,tau=taus,cmd=cmds)
def select(result,targets,gammas,old):
    legal=[i for i,v in enumerate(result['scores']) if np.isfinite(v)]
    if not legal:return None
    # Equality ties only; no hidden numerical score tolerance.
    return min(legal,key=lambda i:(result['scores'][i],abs(gammas[i]-1),targets[i]!=old,abs(targets[i]-old),i))
