"""Build separate replay-target and evaluation knots from the fixed cabinet clock."""
import numpy as np


def groups(cabinet):
    if np.any(np.diff(cabinet)<-1e-8):
        raise ValueError("Cabinet time runs backwards")
    return np.split(np.arange(len(cabinet)),np.flatnonzero(np.diff(cabinet)>1e-8)+1)


def select(same_stamp,field):
    chosen=[]
    result=[]
    for inds in same_stamp:
        i=int(inds[0])
        if not np.all(field[inds]==field[i]):
            result.extend((int(j),"excluded_conflicting_timestamp",None) for j in inds)
            continue
        chosen.append(i)
        result.append((i,"kept",i))
        result.extend((int(j),"merged_identical_payload",i) for j in inds[1:])
    return np.array(chosen),sorted(result)


def contradictory_plateau_rows(t,idx,pose,speed,policy):
    """Exclude isolated phase-inconsistent rows anchored by the adjacent straight segment."""
    x=pose[idx,0]*1000
    lag=np.diff(t)
    slopes=np.diff(x)/lag
    rules=policy["singleton_target_consistency"]
    evidence=[]
    for i in range(2,len(idx)-2):
        if t[i+1]-t[i-1]>rules["max_bridge_s"]+1e-9 or max(lag[i-2],lag[i+1])>rules["max_bridge_s"]+1e-9:
            continue
        bridge=(x[i+1]-x[i-1])/(t[i+1]-t[i-1])
        aim=np.sign(bridge)*speed
        knots=[slopes[i-2],bridge,slopes[i+1]]
        bad=max(abs(slopes[i-1]-bridge),abs(slopes[i]-bridge))
        if aim and max(abs(v-aim) for v in knots)<=rules["plateau_tolerance_fraction"]*speed and bad>rules["local_bad_slope_departure_fraction"]*speed:
            evidence.append(dict(excluded_source_index=int(idx[i]),trusted_source_endpoints=idx[[i-1,i+1]].tolist(),
                  mapped_bracket_s=t[[i-1,i+1]].tolist(),interior_velocities_mm_s=slopes[i-1:i+1].tolist(),
                  adjacent_and_bridge_mm_s=[float(v) for v in knots], reason="isolated phase-discordant target on independently bounded plateau"))
    excluded={v["excluded_source_index"] for v in evidence}
    if any(i in excluded for row in evidence for i in row["trusted_source_endpoints"]):
        raise ValueError("Overlapping target repairs are not supported")
    return excluded,evidence


def map_clocks(src,rec,policy):
    h=src["host_time_s"]
    cab=src["cabinet_time_s"]
    target=src["target_tcp_pose"]
    observed=np.column_stack([src["actual_tcp_pose"],src["actual_tcp_speed"]])
    cluster=groups(cab)
    intercept_residual=h-(cab-cab[0])
    anchor=np.array([c[0] for c in cluster if len(c)==1])
    offset=float(np.median(intercept_residual[anchor]))
    mapped=(cab-cab[0])+offset
    qt,td=select(cluster,target)
    actual_i,ad=select(cluster,observed)
    discord,field_consistency=contradictory_plateau_rows(mapped[qt],qt,target,rec["speed_mm_s"],policy)
    qt=np.array([i for i in qt if i not in discord])
    T=mapped[qt]
    nominal=target[qt].copy()
    raw_nominal=nominal.copy()
    max_gap=policy["long_gap_s"]
    allowed=policy["quiet_nominal_gap_max_translation_m"]
    gaps=[]
    blocking=[]
    for i in np.flatnonzero(np.diff(T)>max_gap+1e-9):
        lo,hi=T[i:i+2]
        movement=float(np.linalg.norm(raw_nominal[i+1,:3]-raw_nominal[i,:3]))
        intersects=[key for key in ("start","turn") if lo<rec["windows_s"][key][1] and hi>rec["windows_s"][key][0]]
        rejoin=i+2<len(T) and T[i+2]-hi<=max_gap+1e-9
        # Blocks touching inferred startup or reversal are never filled.
        valid=not intersects and movement<=allowed and rejoin
        if valid:
            nominal[i+1]=nominal[i]
        r=dict(start_s=float(lo),end_s=float(hi),seconds=float(hi-lo),left_source_index=int(qt[i]),
               right_source_index=int(qt[i+1]),observed_tip_displacement_um=movement*1e6,
               intersects_critical=intersects,accepted_bounded_target_hold=bool(valid),
               right_return_index=int(qt[i+2]) if rejoin else None,
               rejoin_s=[float(hi),float(T[i+2])] if rejoin else None,
               actual_capture_gap_not_filled=True,
               interpretation="Bounded nominal hold; no actual motion inferred.")
        gaps.append(r)
        if not valid:
            blocking.append(r)
    dt_actual=np.diff(mapped[actual_i])
    actual_gaps=[dict(source_indices=actual_i[i:i+2].tolist(),mapped_s=mapped[actual_i[i:i+2]].tolist(),
                      seconds=float(dt_actual[i]))
                 for i in np.flatnonzero(dt_actual>max_gap+1e-9)]
    for g in actual_gaps:
        lo,hi=g["mapped_s"]
        if any(lo<rec["windows_s"][k][1] and hi>rec["windows_s"][k][0] for k in ("start","turn")):
            blocking.append(g|{"kind":"unavailable_actual_critical"})
    decisions=[]
    suppressed={g["right_source_index"] for g in gaps if g["accepted_bounded_target_hold"]}
    for (i,action,representative),(j,act_action,act_i) in zip(td,ad,strict=True):
        assert i==j
        decisions.append(dict(index=i,host_s=float(h[i]),cabinet_s=float(cab[i]),mapped_host_s=float(mapped[i]),
            target_action=("excluded_local_field_inconsistency" if i in discord else "bounded_hold_boundary" if i in suppressed else action),
            target_rep_index=None if i in discord else representative,
            actual_action=act_action,actual_rep_index=act_i))
    d=1000*(h[anchor]-mapped[anchor])
    offsets=[float(np.median(intercept_residual[ix])*1000) for ix in np.array_split(anchor,4)]
    receipt=dict(clock_scale=1.,cabinet_origin_s=float(cab[0]),offset_s=offset,singleton_count=int(len(anchor)),
                 residual_quantile_ms=np.percentile(d,[0,5,25,50,75,95,100]).tolist(),
                 offset_by_time_quartile_ms=offsets,largest_abs_residual_ms=float(np.max(abs(d))),
                 arrival_uncertainty="No direct sample-time anchor. Residuals include arrival variation, getter order and unknown constant delay. Fixed offset is an estimate.",
                 actual_mapping="actual TCP pose and speed are validated jointly within equal stamps; other actual fields and camera are not shifted",
                 target_conflict_rows=int(sum(a=="excluded_conflicting_timestamp" for _,a,_ in td)),
                 actual_conflict_rows=int(sum(a=="excluded_conflicting_timestamp" for _,a,_ in ad)),
                 target_identical_rows_merged=int(sum(a=="merged_identical_payload" for _,a,_ in td)),
                 actual_identical_rows_merged=int(sum(a=="merged_identical_payload" for _,a,_ in ad)),
                 field_consistency_exclusions=field_consistency,long_target_gaps=gaps,actual_gaps=actual_gaps,blocking_gaps=blocking,
                 decisions=decisions)
    arrays=dict(target_index=qt,target_mapped_host_s=T,derived_target_tcp_pose=nominal,
                all_mapped_host_s=mapped,actual_index=actual_i,actual_mapped_host_s=mapped[actual_i],
                actual_original_host_s=h[actual_i],target_original_host_s=h[qt])
    return arrays,receipt
