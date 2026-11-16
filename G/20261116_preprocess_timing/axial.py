"""Measure X event timing and velocity shape on the original source host axis."""
import numpy as np


def crossing(t,v,level,lo,hi,rising):
    """Return the threshold bracket and a local linear time; flag a large RTDE gap."""
    d=(v-level)*(1 if rising else -1)
    edges=np.flatnonzero((t[1:]>=lo)&(t[:-1]<=hi)&(d[:-1]<0)&(d[1:]>=0))
    for i in edges:
        a,b=float(t[i]),float(t[i+1])
        x=a+(b-a)*(-d[i])/(d[i+1]-d[i])
        if not lo<=x<=hi:
            continue
        # Reject transient threshold spikes without shifting the crossing.
        after=(t>t[i+1]) & (t<=b+.045)
        if after.any() and np.median(d[after])<0:
            continue
        return dict(time_s=float(x) if b-a<=.040 else None,bracket_s=[a,b],
                    unresolved_gap=bool(b-a>.040))
    return dict(time_s=None,bracket_s=None,unresolved_gap=False)


def events(t,x,v,record):
    win=record["windows_s"]
    speed=record["speed_mm_s"]
    turn=win["turn"][0]
    start=win["start"][0]
    stop=win["stop"][0]
    limits = [
        ("start10",.1*speed,start-.10,start+.58,True),
        ("start50",.5*speed,start-.08,start+.62,True),
        ("start90",.9*speed,start-.08,start+.68,True),
        ("decel90",.9*speed,turn-.035,turn+.44,False),
        ("zero",0.,turn-.03,turn+.52,False),
        ("return50",-.5*speed,turn-.02,turn+.55,False),
        ("return90",-.9*speed,turn-.02,turn+.62,False),
        ("stop90",-.9*speed,stop+.02,stop+.62,True),
        ("stop10",-.1*speed,stop+.02,min(stop+.72,win["stop"][1]+.08),True),
    ]
    result={key:crossing(t,v,f,lo,hi,rising) for key,f,lo,hi,rising in limits}
    turn_mask=(t>=turn-.03)&(t<=turn+.56)
    result["turn_xmax"]=dict(time_s=float(t[turn_mask][np.argmax(x[turn_mask])]))
    return result


def measure(src,record,t,x,v):
    ht=src["host_time_s"]
    x_a=src["actual_tcp_pose"][:,0]
    v_a=src["actual_tcp_speed"][:,0]*1e3
    source=events(ht,x_a,v_a,record)
    model=events(t,x,v,record)
    diff={}
    for key in source:
        at,bt=source[key]["time_s"],model[key]["time_s"]
        diff[key]=1000*(bt-at) if at is not None and bt is not None else None
    durations={}
    for name,left,right in (("rise_10_90","start10","start90"),("reversal_90_90","decel90","return90"),
                            ("stop_90_10","stop90","stop10")):
        durations[name]={}
        for label,series in (("A",source),("candidate",model)):
            a,b=series[left]["time_s"],series[right]["time_s"]
            durations[name][label+"_ms"]=(b-a)*1000 if a is not None and b is not None and b>=a else None
    shapes={}
    for name in ("start","turn","outbound","return_straight","stop"):
        lo,hi=record["windows_s"][name]
        keep=(ht>=lo)&(ht<hi)&(ht>=t[0])&(ht<=t[-1])
        ts=ht[keep]
        ev=np.interp(ts,t,v)-v_a[keep]
        ex=np.interp(ts,t,x)-x_a[keep]
        shapes[name]=dict(samples=int(keep.sum()),v_RMSE_mm_s=float(np.sqrt(np.mean(ev**2))),
                          v_peak_error_mm_s=float(np.max(abs(ev))),x_RMSE_um=float(np.sqrt(np.mean(ex**2))*1e6))
    keys=("start50","turn_xmax","return50")
    gate=all(diff[k] is not None and abs(diff[k])<=50 for k in keys)
    return dict(source=source,candidate=model,delay_ms=diff,durations=durations,shape=shapes,gate_50ms=gate,
                gate_markers=keys,shape_acceptance="Inspect unshifted X/Vx and independent duration/shape errors; 50 ms gate alone is insufficient.")
