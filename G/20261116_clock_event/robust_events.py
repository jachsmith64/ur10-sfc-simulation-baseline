"""One fixed offline detector. Raw trajectories and raw metrics remain separate."""
import numpy as np


def detector_grid(t,v,policy,max_gap):
    dt=policy["grid_s"]
    g=np.arange(np.ceil(t[0]/dt),np.floor(t[-1]/dt)+1,dtype=int)*dt
    right=np.searchsorted(t,g,side="right")
    left=np.maximum(right-1,0)
    right=np.minimum(right,len(t)-1)
    allowed=(t[right]-t[left]<=max_gap+1e-9)|(g==t[left])
    y=np.interp(g,t,v)
    filtered=np.full_like(g,np.nan)
    half=round(policy["median_half_width_s"]/dt)
    chunks=np.split(np.flatnonzero(allowed),np.flatnonzero(np.diff(np.flatnonzero(allowed))!=1)+1)
    for ix in chunks:
        if not len(ix):
            continue
        pad=np.pad(y[ix],half,mode="edge")
        f=np.median(np.lib.stride_tricks.sliding_window_view(pad,2*half+1),axis=-1)
        # Guard non-native boundaries of an unavailable interval.
        if ix[0]!=0:
            f[:half]=np.nan
        if ix[-1]!=len(g)-1:
            f[-half:]=np.nan
        filtered[ix]=f
    return g,filtered


def candidates(g,y,level,lo,hi,rising):
    s=(y-level)*(1 if rising else -1)
    ok=np.flatnonzero((s[:-1]<0)&(s[1:]>=0)&(g[1:]>=lo)&(g[:-1]<=hi))
    for i in ok:
        ts=g[i]+(g[i+1]-g[i])*(-s[i])/(s[i+1]-s[i])
        if lo<=ts<=hi:
            yield float(ts),[float(g[i]),float(g[i+1])]


def detect(t,x,v,rec,policy,max_gap):
    g,f=detector_grid(t,v,policy,max_gap)
    win=rec["windows_s"]
    speed=rec["speed_mm_s"]
    start,turn=win["start"][0],win["turn"][0]
    span=policy["min_persistence_s"]
    hy=policy["persistence_hysteresis_fraction"]*speed
    decision={}

    def persist(level,lo,hi,rising):
        sign=1 if rising else -1
        all_hits=list(candidates(g,f,level,lo,hi,rising))
        for stamp,bracket in all_hits:
            q=f[(g>=stamp)&(g<=stamp+span)]
            if len(q)>=int(span/policy["grid_s"])-1 and np.all(sign*(q-level)>=-hy):
                return dict(time_s=stamp,bracket_s=bracket,rule="crossing and elapsed persistence",alternatives=len(all_hits))
        return dict(time_s=None,bracket_s=None,rule="no persistent crossing",alternatives=len(all_hits))

    for label,ratio in (("start10",.1),("start50",.5),("start80",.8),("start90",.9)):
        decision[label]=persist(ratio*speed,start-.10,start+.68,True)
    band=policy["zero_prepost_confirm_s"]
    edge=policy["zero_prepost_speed_fraction"]*speed
    zeros=[]
    for stamp,bracket in candidates(g,f,0,turn-.035,turn+.52,False):
        left=f[(g>=stamp-band)&(g<stamp)]
        right=f[(g>stamp)&(g<=stamp+band)]
        if len(left) and len(right) and np.isfinite(left).all() and np.isfinite(right).all() and left.max()>=edge and right.min()<=-edge:
            zeros.append((stamp,bracket))
    if len(zeros)==1:
        zero,zbr=next(iter(zeros))
        decision["zero"]=dict(time_s=zero,bracket_s=zbr,rule="single confirmed sign exchange")
        for key,ratio in (("decel50",.5),("decel80",.8),("decel90",.9)):
            level=ratio*speed
            hits=list(candidates(g,f,level,max(turn-.15,zero-.36),zero,False))
            valid=[]
            for ts,br in hits:
                forward=f[(g>=ts)&(g<=zero)]
                before=f[(g>=ts-.06)&(g<ts)]
                if len(forward) and len(before) and np.isfinite(forward).all() and np.isfinite(before).all():
                    if np.max(forward)<=level+hy and np.any(before>=level):
                        valid.append((ts,br))
            decision[key]=(dict(time_s=valid[-1][0],bracket_s=valid[-1][1],
                                rule="last connected descent before confirmed zero",alternatives=len(hits))
                           if valid else dict(time_s=None,bracket_s=None,rule="no connected descent",alternatives=len(hits)))
        for label,ratio in (("return50",.5),("return80",.8),("return90",.9)):
            decision[label]=persist(-ratio*speed,zero,turn+.65,False)
    else:
        for key in ("zero","decel50","decel80","decel90","return50","return80","return90"):
            decision[key]=dict(time_s=None,bracket_s=None,rule=f"ambiguous main zero: {len(zeros)} confirmations")
    selection=(t>=turn-.03)&(t<=turn+.56)
    index=np.flatnonzero(selection)[np.argmax(x[selection])]
    decision["turn_xmax"]=dict(time_s=float(t[index]),bracket_s=t[max(0,index-1):index+2][[0,-1]].tolist(),rule="raw position maximum")
    return decision,g,f


def spans(events):
    def delta(a,b):
        x,y=events[a]["time_s"],events[b]["time_s"]
        return None if x is None or y is None or x>y else (y-x)*1000
    return dict(rise_10_90=delta("start10","start90"),reversal_90_90=delta("decel90","return90"),
                reversal_50_50=delta("decel50","return50"),reversal_80_80=delta("decel80","return80"))
