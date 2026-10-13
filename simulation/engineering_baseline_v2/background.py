"""Smooth exogenous finite sine background. No feedback or controller input."""
import numpy as np

def smooth(u):
    u=np.clip(np.asarray(u),0.,1.)
    return u**3*(10-15*u+6*u*u)

def smooth_dot(u,T):
    a=np.asarray(u); c=np.clip(a,0.,1.)
    return np.where((a>0)&(a<1),30*c*c*(1-c)*(1-c)/T,0.)

class SineBank:
    def __init__(self,rng,tau,t,mask):
        self.w=2*np.pi*(1/(2*np.pi*tau))*np.array([.4,.7,1.,1.3,1.9])
        self.phi=rng.uniform(0,2*np.pi,5); self.a=np.array([.6,1.,1.,.7,.4])
        values,der=self.raw(t); self.offset=float(np.mean(values[mask])); self.scale=float(np.std(values[mask]))
    def raw(self,t):
        angles=np.asarray(t)[...,None]*self.w+self.phi
        return np.sum(self.a*np.sin(angles),axis=-1),np.sum(self.a*self.w*np.cos(angles),axis=-1)
    def value(self,t):
        v,d=self.raw(t)
        return (v-self.offset)/self.scale,d/self.scale

class Background:
    def __init__(self,cfg,params,t,move_mask,J0):
        self.cfg=cfg; self.params=params; self.J0=np.asarray(J0)
        rng=np.random.default_rng(cfg['seed'])
        js=[params['joints'][f'J{i}'] for i in range(1,7)]
        self.centers=np.array([a['bias_center']['rad']['value'] for a in js])
        ranges=np.array([a['trial_bias_range_rad']['value'] for a in js])
        self.bias=self.centers if cfg['bias_scenario']=='center' else rng.uniform(ranges[:,0],ranges[:,1])
        self.fixed_trial_offset=self.bias-self.centers
        self.static=np.array([a['static_demeaned_std']['rad']['value'] for a in js])*cfg['static_joint_scale']
        self.dynamic=np.array([a['dynamic_demeaned_std']['rad']['value'] for a in js])
        self.banks=[SineBank(rng,cfg['joint_tau_s'],t,move_mask) for _ in js]
        self.res_bank=SineBank(rng,cfg['remaining_tau_s'],t,move_mask)
        x=np.column_stack([b.value(t)[0] for b in self.banks])@ (self.J0*self.dynamic)
        self.joint_projection_scale=float(np.std(x[move_mask]))
        independent=self.res_bank.value(t)[0]
        correlated=cfg['motion_rho']*x/self.joint_projection_scale+np.sqrt(1-cfg['motion_rho']**2)*independent
        self.res_mix_center=float(np.mean(correlated[move_mask])); self.res_mix_scale=float(np.std(correlated[move_mask]))
    def value(self,t,weight,weight_dot,enabled=True):
        t=np.asarray(t); w=np.asarray(weight); wd=np.asarray(weight_dot)
        z=np.stack([b.value(t)[0] for b in self.banks],axis=-1)
        zd=np.stack([b.value(t)[1] for b in self.banks],axis=-1)
        sigma=self.static+w[...,None]*(self.dynamic-self.static)
        sigmad=wd[...,None]*(self.dynamic-self.static)
        onset=smooth(t/.75); onsetd=smooth_dot(t/.75,.75)
        scales=np.array(self.cfg['joint_input_scale'])
        noise=z*sigma*scales; noised=(zd*sigma+z*sigmad)*scales
        dq=onset[...,None]*(self.bias+noise)
        dqd=onsetd[...,None]*(self.bias+noise)+onset[...,None]*noised
        # Correlation uses only frozen generated background, never controlled plant error.
        proj=(z*self.dynamic)@self.J0/self.joint_projection_scale
        projd=(zd*self.dynamic)@self.J0/self.joint_projection_scale
        eta,etad=self.res_bank.value(t)
        rho=self.cfg['motion_rho']; mixed=(rho*proj+np.sqrt(1-rho*rho)*eta-self.res_mix_center)/self.res_mix_scale
        mixedd=(rho*projd+np.sqrt(1-rho*rho)*etad)/self.res_mix_scale
        std_move=self.params['remaining_endpoint']['demeaned_std_m']['value']
        std_static=self.cfg['static_remaining_std_um']*1e-6
        # Static is deliberately a reduced equivalent-output scenario, not motion rho.
        signal=(1-w)*eta+w*mixed
        signald=(1-w)*etad+w*mixedd+wd*(mixed-eta)
        sig=std_static+w*(std_move-std_static); sigd=wd*(std_move-std_static)
        center=self.params['remaining_endpoint']['signed_center']['value']*1e-6
        rem=onset*(center+sig*signal)
        remd=onsetd*(center+sig*signal)+onset*(sigd*signal+sig*signald)
        if not enabled: dq=dq*0; dqd=dqd*0; rem=rem*0; remd=remd*0
        return dq,dqd,rem,remd
