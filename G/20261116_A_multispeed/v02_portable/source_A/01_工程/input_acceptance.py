"""R-026: explicit numeric input acceptance, including caches; no simulations."""
import math
class InputRejected(RuntimeError):pass
def number(x,label):
    if isinstance(x,bool):raise InputRejected(label+' is boolean, not numeric')
    try:v=float(x)
    except (ValueError,TypeError):raise InputRejected(label+' missing/invalid')
    if not math.isfinite(v):raise InputRejected(label+' nonfinite')
    return v
def calibration_verdict(record):
    try:
        g=number(record.get('g'),'g');target=number(record.get('A_target_P95_um'),'target')
        base=number(record.get('S0_P95_um'),'base');output=number(record.get('S1_P95_um'),'output')
        tol=number(record.get('check_tolerance_relative'),'tolerance')
        if not 0<g<=1 or min(target,base)<=0 or output<0:raise InputRejected('invalid calibration range')
        if tol!=.10:raise InputRejected('original10% calibration tolerance changed')
        if record.get('accepted') is False:raise InputRejected('cached calibration explicitly rejected')
        close=abs(output/target-1)<=.10
        no_amplification=g==1. and base<=target and output<=target*1.10
        passed=close or no_amplification
        return dict(passed=passed,reason='within original10% tolerance' if close else ('legal g1 no-amplification branch' if no_amplification else 'outside tolerance'),
            tolerance_relative=.10,g=g,relative_difference=output/target-1,no_amplification_branch=no_amplification)
    except InputRejected as e:return dict(passed=False,reason=str(e),tolerance_relative=.10)
def event_verdict(record):
    try:
        error=number(record.get('final_check',{}).get('relative_rms_error'),'event relative residual')
        if error<0:raise InputRejected('negative residual')
        passed=error<=.20 and record.get('sufficient_for_approximate_comparison') is True
        return dict(passed=passed,reason='original20% gate passed' if passed else 'event gate or explicit acceptance failed',relative_rms_error=error,threshold=.20)
    except InputRejected as e:return dict(passed=False,reason=str(e),threshold=.20)
def require_calibration(record):
    v=calibration_verdict(record)
    if not v['passed']:raise InputRejected('Calibration input rejected: '+v['reason'])
    return v
def require_event(record):
    v=event_verdict(record)
    if not v['passed']:raise InputRejected('Event input rejected: '+v['reason'])
    return v
