"""Transparent synchronous timers around the inherited coordinator operations.
Wrappers return original objects/results and are always restored. No numerical,
planning-state, or candidate-selection change. Files are written after each case.
"""
import time,weakref
import e_coupling as inherited
from gamma_config import DEFAULT_W_GAMMA
class TimedCoordinator(inherited.Coordinator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.stage_times={}
    def current(self,*args,**kwargs):
        start=time.perf_counter();result=super().current(*args,**kwargs)
        self.stage_times['model_s']+=time.perf_counter()-start;return result
    def before_step(self,i,env,measurement,ctrl,base):
        if not (self.planning and i%32==0):return super().before_step(i,env,measurement,ctrl,base)
        self.stage_times={k:0. for k in ['model_s','ruckig_generation_s','trajectory_sampling_s','prediction_scoring_s']}
        original_plan=inherited.Plan;original_predict=inherited.predict
        weak_owner=weakref.ref(self)
        def timed_plan(*args,**kwargs):
            start=time.perf_counter()
            try:plan=original_plan(*args,**kwargs)
            finally:self.stage_times['ruckig_generation_s']+=time.perf_counter()-start
            original_preview=type(plan).preview
            weak_plan=weakref.ref(plan)
            def timed_preview(*args,**kwargs):
                owner=weak_owner();active_plan=weak_plan()
                start=time.perf_counter()
                try:return original_preview(active_plan,*args,**kwargs)
                finally:owner.stage_times['trajectory_sampling_s']+=time.perf_counter()-start
            plan.preview=timed_preview;return plan
        def timed_predict(*args,**kwargs):
            start=time.perf_counter()
            try:return original_predict(*args,**kwargs)
            finally:self.stage_times['prediction_scoring_s']+=time.perf_counter()-start
        inherited.Plan=timed_plan;inherited.predict=timed_predict
        start=time.perf_counter()
        try:super().before_step(i,env,measurement,ctrl,base)
        finally:inherited.Plan=original_plan;inherited.predict=original_predict
        total=time.perf_counter()-start
        timing=dict(self.stage_times)
        timing.update(legacy_total_s=self.decisions[-1]['seconds'],full_coordinator_s=total,
            bookkeeping_other_s=max(0.,total-sum(self.stage_times.values())),full_over32ms=total>.032,legacy_over32ms=self.decisions[-1]['seconds']>.032,
            meaning='Other includes selection/record construction/timing overhead; file writes after case are measured separately')
        self.decisions[-1]['timing_parts']=timing
        self.decisions[-1]['w_gamma_used']=self.params.get('w_gamma',DEFAULT_W_GAMMA)
        self.records[i].update(coordination_full_s=total,**self.stage_times)
