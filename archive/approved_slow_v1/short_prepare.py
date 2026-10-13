from pathlib import Path
import json,hashlib,csv,ast
import numpy as np
import config,disturbance,record
from sim_env import SimEnv,build_nominal_joint_trajectory
from background import Background,smooth,smooth_dot
from position_execution import compiled_mapping
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=serial),encoding='utf-8')
def envelope(t,phase):
    w=np.ones_like(t);wd=np.zeros_like(t)
    for code,rising in [(0,True),(4,False)]:
        ix=np.flatnonzero(phase==code);lo=t[ix[0]];T=t[ix[-1]]-lo;u=(t[ix]-lo)/T
        w[ix]=smooth(u) if rising else 1-smooth(u);wd[ix]=smooth_dot(u,T) if rising else -smooth_dot(u,T)
    return w,wd

def short_template(p):
    d=p.d;c=p.cuts;al=d['along_mm'];cr=p.drift['e_vib_um']
    units={name:(al[a:b+1]-al[a],cr[a:b+1],np.arange(a,b+1)) for name,a,b in
        [('start',0,c.iA),('mid_fwd',c.iA,c.iB),('turn_fwd',c.iB,c.iP),('turn_ret',c.iP,c.iC),('mid_ret',c.iC,c.iD-1),('stop',c.iD,len(al)-1)]}
    def assemble(extra,full=False):
        parts=[];idxparts=[];segments=[];base=0.;size=0;return_correction=0.
        def place(name,loc,idx):
            nonlocal base,size
            # Shared endpoint belongs to previous segment; omit duplicated seam timestamp.
            start=0 if not parts else 1
            vals=base+loc[start:];inds=idx[start:]
            if len(vals):segments.append((name,size,size+len(vals)));parts.append(vals);idxparts.append(inds);size+=len(vals)
            base+=loc[-1]
        a,e,i=units['start'];place('start',a,i)
        a,e,i=units['mid_fwd'];place('mid_fwd',a,i)
        if extra:place('mid_fwd',a[:extra+1],i[:extra+1])
        for name in ['turn_fwd','turn_ret']:
            a,e,i=units[name];place(name,a,i)
        a,e,i=units['mid_ret'];need=base+units['stop'][0][-1]
        chunks=[];idxs=[]
        while need>abs(a[-1])+1e-12:
            chunks.append(a);idxs.append(i);need-=abs(a[-1])
        take=int(np.argmin(abs(abs(a)-need)))
        take=max(1,take);partial=a[:take+1].copy()
        return_correction=-need-partial[-1]
        # Only ordinary return endpoint rounding (<=one 1ms displacement) is distributed smoothly.
        partial+=return_correction*smooth(np.linspace(0,1,len(partial)))
        chunks.append(partial);idxs.append(i[:take+1])
        for loc,idx in zip(chunks,idxs):place('mid_ret',loc,idx)
        a,e,i=units['stop'];place('stop',a,i)
        if full:return np.concatenate(parts),np.concatenate(idxparts),segments,return_correction
        return size
    choices=[(abs(assemble(k)-10001),k,assemble(k)) for k in range(len(units['mid_fwd'][0]))]
    _,extra,n=min(choices)
    along,source_idx,segments,correction=assemble(extra,True)
    t=np.arange(n)*.001;phase=record.phase_codes_from_segments(segments)
    assert abs(along[-1]-along[0])<1e-10
    assert 9.9<=t[-1]<=12.0
    b={name:(int(ix[0]),int(ix[-1])+1) for name in ['start','fwd_steady','turn','ret_steady','stop'] for ix in [np.flatnonzero(phase==config.DRIFT_PHASE_CODE[name])]}
    # A shared endpoint remains in the preceding phase. Event time refers to that endpoint,
    # one tick before the first new sample, matching the source cut rather than shifting events.
    turn=t[b['turn'][0]]-.001;stop=t[b['stop'][0]]-.001
    meta=dict(samples=n,last_sample_s=t[-1],integration_duration_s=n*.001,max_along_mm=along.max(),end_along_mm=along[-1],
        source_duration_s=p.d['t'][-1],ordinary_extra_fwd_steps=extra,return_rounding_correction_mm=correction,
        strategy='Only steady repetition/length reduced; unchanged source start/turn/stop samples, durations, amplitudes. Omit shared duplicate seam samples. Quintic sub-step closure correction only on ordinary return.',
        segments=segments,phase_bounds=b,start_trigger_s=1.2849583981754964,turn_trigger_s=turn+(c.iP-c.iB)*.001,
        turn_begin_s=turn,stop_begin_s=stop,post_stop_begin_s=stop+(6.613756461197408-c.iD*.001),
        source_stop_speed_evidence='UR speed ~-5 at6.56085s then~-2 at6.57596s; original iD=6.573s within sampling bracket. Stop event support begins 0.15s earlier, covers deceleration.',
        all_events_not_time_compressed=True,
        ordinary_repeat_note='Return repetition ends one source tick before iD, excluding the half-speed deceleration tick from repeated ordinary segments. Genuine final deceleration is retained in stop.',
        velocity_limitation='Original event commands are piecewise linear sampled positions, not a C1 planner. No new event planning. Ordinary joins have matching sampled constant-speed slopes; tiny closure correction has zero endpoint derivative.')
    return t,along,phase,source_idx,meta

def prepare():
    config.DATA_CSV=ROOT/'sources_A/P05R01/D1_时序.csv'
    assert sha(config.DATA_CSV)=='d8d76ea3a259f8dc631547391ed0ec546b39014c608b90d16e229b68d8ad99fe'
    pr=disturbance.prepare(log=lambda x:None)
    t,along,phase,source_idx,meta=short_template(pr)
    q=build_nominal_joint_trajectory(along,config.Y_NOMINAL,config.Z_HOLD)
    w,wd=envelope(t,phase)
    old=np.load(ROOT/'inputs/old62s_common.npz');params=json.loads((ROOT/'inputs/仿真误差参数.json').read_text(encoding='utf-8'));cfg=json.loads((ROOT/'f_config.json').read_text(encoding='utf-8'))
    env=SimEnv();env.reset(q[0]);J0=env.jacp()[1]
    bg=Background(cfg,params,old['t'],np.isin(old['phase'],[1,3]),J0)
    old_values=bg.value(old['t'],old['weight'],old['weight_dot'])
    restore_errors={key:float(np.max(abs(value-old[key]))) for key,value in zip(['dq','dqd','remaining','remaining_dot'],old_values)}
    assert max(restore_errors.values())<1e-14,restore_errors
    dq,dqd,rem,remd=bg.value(t,w,wd)
    # Derivative verification uses the same analytic envelope, never sampled future controls.
    def evaluate(x):
        ww=np.ones_like(x);dd=np.zeros_like(x)
        for code,rising in [(0,True),(4,False)]:
            ix=np.flatnonzero(phase==code);lo=t[ix[0]];T=t[ix[-1]]-lo;m=(x>=lo)&(x<=t[ix[-1]]);u=(x[m]-lo)/T
            ww[m]=smooth(u) if rising else 1-smooth(u);dd[m]=smooth_dot(u,T) if rising else -smooth_dot(u,T)
        return bg.value(x,ww,dd)
    probes=np.array([.123,.543,.749,1.6,3.2,meta['stop_begin_s']+.15]);eps=1e-5;a=evaluate(probes);pl=evaluate(probes+eps);mi=evaluate(probes-eps)
    deriv=dict(dq_rad_s=float(np.max(abs((pl[0]-mi[0])/(2*eps)-a[1]))),remaining_m_s=float(np.max(abs((pl[2]-mi[2])/(2*eps)-a[3]))))
    assert max(deriv.values())<1e-8
    norm=dict(old62s_reference_retained=True,seed=cfg['seed'],J0=J0,old_reproduction_errors=restore_errors,derivative_errors=deriv,
        banks=[dict(w=b.w,phi=b.phi,a=b.a,offset=b.offset,scale=b.scale) for b in bg.banks+[bg.res_bank]],
        joint_projection_scale=bg.joint_projection_scale,res_mix_center=bg.res_mix_center,res_mix_scale=bg.res_mix_scale,
        short_joint_std_rad=np.std(dq[np.isin(phase,[1,3])],axis=0),short_remaining_std_um=np.std(rem[np.isin(phase,[1,3])])*1e6,
        note='Normalization is from old62s motion masks; no re-normalization to new10s std. Signals keep same seed/frequency/phase at same absolute t; motion envelope changes at relocated stop.')
    # One-time event targets, original e_vib core untouched; only leading/trailing joins tapered.
    event=np.zeros_like(t);parts={};evmeta=[]
    specs=[('start',1.085,2.226,0.,1.2849583981754964,1.926),
        ('turn',meta['turn_begin_s']-.2,meta['turn_begin_s']+1.063+.3,3.461-meta['turn_begin_s'],meta['turn_begin_s'],meta['turn_begin_s']+1.063),
        ('stop',meta['stop_begin_s']-.15,t[-1],6.573-meta['stop_begin_s'],meta['stop_begin_s']-.02,meta['post_stop_begin_s']+.6)]
    for name,lo,hi,shift,corelo,corehi in specs:
        mask=(t>=lo)&(t<=hi);values=np.interp(t+shift,pr.d['t'],pr.drift['e_vib_um'])
        gain=smooth((t-lo)/.05)*(1-smooth((t-(hi-.15))/.15));gain*=mask
        parts[name]=values*gain;event+=parts[name]
        cm=(t>=corelo)&(t<=corehi)
        assert np.allclose(gain[cm],1.,rtol=0,atol=1e-12)
        evmeta.append(dict(event=name,new_support_s=[lo,hi],core_s=[corelo,corehi],source_shift_s=shift,
            peak_positive_um=float(np.max(parts[name])),peak_negative_um=float(np.min(parts[name])),core_original_preserved=True,
            note='e_vib=A raw-b; preserves core sign/duration/shape, source noise may remain; leading50ms/trailing150ms joins only. Stop includes deceleration and >1s original tail; last150ms taper prevents FFT endpoint wrap.'))
    assert np.max(abs(event[t<.8]))==0 and abs(event[-1])<1e-12
    np.savez_compressed(ROOT/'inputs/frozen_common.npz',t=t,along=along,phase=phase,q=q,dq=dq,dqd=dqd,remaining=rem,remaining_dot=remd,weight=w,weight_dot=wd,
        source_index=source_idx,event_target_um=event,**{'target_'+k+'_um':v for k,v in parts.items()})
    write(ROOT/'review/trajectory.json',meta);write(ROOT/'review/background_normalization.json',norm);write(ROOT/'review/event_targets.json',evmeta)
    mapping=compiled_mapping(env.model);assert mapping['simple_mapping_applicable'];write(ROOT/'review/compiled_model.json',mapping)
    rows=[]
    for lo,hi,phase_name in [(1.926,3.461,'outbound'),(4.524,6.573,'return')]:
        m=(pr.raw['t']>=lo)&(pr.raw['t']<=hi);x=pr.raw['cross_um'][m]
        rows.append(dict(phase=phase_name,t0=lo,t1=hi,n=int(m.sum()),P95_abs_um=float(np.percentile(abs(x),95)),max_abs_um=float(np.max(abs(x))),mean_um=float(np.mean(x)),std_um=float(np.std(x))))
    write(ROOT/'review/A_ordinary_reference.json',dict(source_sha256=sha(config.DATA_CSV),raw_frames_no_resampling=True,windows=rows,target_P95_um=float(np.mean([r['P95_abs_um'] for r in rows])),interpolated_example_in_task_is_not_fixed_truth=True))
    # Limited unit/window/duplicate audit; no video reprocessing or new identification.
    unit_errors={}
    for key,p in params['joints'].items():
        for stat in ['bias_center','static_demeaned_std','dynamic_demeaned_std']:
            if 'deg' in p[stat] and 'rad' in p[stat]:unit_errors[key+'_'+stat]=abs(p[stat]['rad']['value']-np.deg2rad(p[stat]['deg']['value']))
    remaining=params['remaining_endpoint'];audit=dict(deg_rad_rounding_errors=unit_errors,mm_to_m_and_um_to_m_correct=True,remaining=remaining,
        duplicate_check='Background only consumes remaining_endpoint visual-actual projection. C local single-joint residuals are not additionally injected. Six-joint dq enters base ctrl once; residual enters SharedMeasurement once.',
        source_checks='C quantified residual has saved pointwise visual-actual identity. Dynamic joint std comes from move phases and is not demonstrated pure constant-speed noise; no new source reprocessing in this round.',
        J0=J0,J0_fixed_usage='For generated residual correlation only; physical motion uses current model kinematics. Calibration camera strong axis near Z is treated as non-motion Y per user, not a new spatial calibration.',
        scenario_assumptions=['six independent sine banks with old multipliers1.26-1.34','fixed sampled trial bias added once, not accumulated each frame','finite sine correlations and UR10e posture differ from measured C posture','remaining12um is equivalent observed output, not flexibility dynamics','old start amplitude envelope includes pre-rest; preserved rather than inventing new initialization'],
        confirmed_encoding_error_found=False,scope='One static audit completed; plausible model/measurement differences recorded, no root-cause hunt.')
    write(ROOT/'review/ordinary_limited_audit.json',audit)
    # Save text source target and corrected stop evidence for independent review.
    np.savetxt(ROOT/'inputs/10秒轨迹与事件目标.csv',np.column_stack([t,along,phase,event]),delimiter=',',fmt='%.17g',header='t_s,along_mm,phase,event_target_um',comments='')
    with (ROOT/'sources_A/P05R01/D1_时序.csv').open(encoding='utf-8-sig') as f:source=list(csv.DictReader(f))
    with (ROOT/'review/停止边界原始证据.csv').open('w',newline='',encoding='utf-8-sig') as f:
        cols=['time_s','phase','nominal_along_mm','ur_speed_mm_s','vision_speed_mm_s','cross_track_um'];out=csv.DictWriter(f,fieldnames=cols);out.writeheader()
        for r in source:
            if 6.40<=float(r['time_s'])<=6.75:out.writerow({k:r[k] for k in cols})
    print('PREPARED duration',t[-1],'s; excursion',along.max(),'mm; A P95 target',np.mean([r['P95_abs_um'] for r in rows]),flush=True)
