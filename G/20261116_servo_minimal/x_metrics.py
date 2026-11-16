"""Key position, timing, shape and hard-stop tests for the isolated X pair."""
import numpy as np


def track(src,rec,t,x,v,nom_x):
    ht=src["host_time_s"]
    actual_x=src["actual_tcp_pose"][:,0]
    actual_v=src["actual_tcp_speed"][:,0]*1000
    name_sign={"outbound":1,"return_straight":-1}
    result={}
    for key,direction in name_sign.items():
        lo,hi=rec["windows_s"][key]
        ok=(ht>=lo)&(ht<hi)
        points=ht[ok]
        ex=(np.interp(points,t,x)-actual_x[ok])*1000
        err_nom=(np.interp(points,t,x)-np.interp(points,t,nom_x))*1000
        ev=np.interp(points,t,v)-actual_v[ok]
        result[key]=dict(count=int(ok.sum()),actual_x_error_bias_mm=float(np.median(ex)),
            actual_x_RMSE_mm=float(np.sqrt(np.mean(ex**2))),nominal_x_error_bias_mm=float(np.median(err_nom)),
            nominal_tracking_scale_ms=float(-1000*np.median(err_nom)*direction/rec["speed_mm_s"]),
            actual_v_RMSE_mm_s=float(np.sqrt(np.mean(ev**2))))
    return result


def overshoot(src,rec,t,x,v):
    ht=src["host_time_s"]
    actual_x=src["actual_tcp_pose"][:,0]
    actual_v=src["actual_tcp_speed"][:,0]*1000
    peaks={}
    for name in ("start","turn","stop"):
        lo,hi=rec["windows_s"][name]
        ma=(ht>=lo)&(ht<hi)
        mb=(t>=lo)&(t<hi)
        pa=float(np.max(abs(actual_v[ma])))
        pb=float(np.max(abs(v[mb])))
        peaks[name]=dict(A_abs_peak_mm_s=pa,model_abs_peak_mm_s=pb,
                         excess_over_A_mm_s=max(0.,pb-pa))
    return dict(speed=peaks,far_X_excess_over_A_mm=max(0.,float((max(x)-max(actual_x))*1000)),
                near_X_excess_over_A_mm=max(0.,float((min(actual_x)-min(x))*1000)))


def gate(original,candidate,setting,config):
    required=("start50","turn_xmax","return50")
    timing={key:candidate["axis"]["delay_ms"][key] for key in required}
    gate_time=all(delay is not None and abs(delay)<=config["episode_gate_delay_ms"] for delay in timing.values())
    duration_errors={}
    for k in ("rise_10_90","reversal_90_90"):
        old,new=(entry["axis"]["durations"][k] for entry in (original,candidate))
        a=old["A_ms"]
        b,c=old["candidate_ms"],new["candidate_ms"]
        duration_errors[k]=dict(original_ms=abs(b-a) if a is not None and b is not None else None,
                                candidate_ms=abs(c-a) if a is not None and c is not None else None)
    duration_pass=all(v["candidate_ms"] is not None and v["original_ms"] is not None and
                      v["candidate_ms"]<v["original_ms"] for v in duration_errors.values())
    shape_rmse={k:[o["axis"]["shape"][k]["v_RMSE_mm_s"] for o in (original,candidate)] for k in ("start","turn")}
    shape_pass=all(b<a for a,b in shape_rmse.values())
    speed_overshoot_change={k:candidate["overshoot"]["speed"][k]["excess_over_A_mm_s"]-
                             original["overshoot"]["speed"][k]["excess_over_A_mm_s"]
                            for k in ("start","turn","stop")}
    spatial_overshoot_change={k:candidate["overshoot"][k]-original["overshoot"][k]
                              for k in ("far_X_excess_over_A_mm","near_X_excess_over_A_mm")}
    overshoot_pass=(max(speed_overshoot_change.values())<=setting*config["maximum_new_speed_overshoot_fraction_nominal"] and
                    max(spatial_overshoot_change.values())<=config["maximum_new_spatial_overshoot_mm"])
    protection=candidate["status"]=="OK" and candidate["guard_clear"] and not any(candidate["saturation_rows"].values())
    return dict(timing_ms=timing,timing_pass=gate_time,duration_errors=duration_errors,
                duration_pass=duration_pass,speed_RMSE=shape_rmse,shape_pass=shape_pass,
                speed_overshoot_change_mm_s=speed_overshoot_change,
                spatial_overshoot_change_mm=spatial_overshoot_change,overshoot_pass=overshoot_pass,
                protection_pass=protection,pass_all=gate_time and duration_pass and shape_pass and overshoot_pass and protection)
