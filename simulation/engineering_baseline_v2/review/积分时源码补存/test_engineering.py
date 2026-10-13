"""Isolated negative paths; no MuJoCo step or real integration is invoked."""
import sys,tempfile,shutil,json,unittest
from pathlib import Path
from unittest.mock import patch
import run_slow as r
import numpy as np
from batch_identity import atomic_json,read,files_snapshot
import run_report

class EngineeringChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=str(r.ROOT/'review'));self.root=Path(self.tmp.name)
        for n in ['inputs','baseline_review']:shutil.copytree(r.ROOT/n,self.root/n)
        shutil.copytree(r.ROOT/'historical/public_slow_review',self.root/'historical/public_slow_review')
        (self.root/'assets_ur10e').mkdir();(self.root/'assets_ur10e/scene.xml').write_text('<mujoco/>',encoding='utf-8')
        (self.root/'entry.py').write_text('# isolated fixture',encoding='utf-8')
        self.calls=[];self.ctx=r.load_inputs(self.root)
    def tearDown(self):self.tmp.cleanup()
    def fake(self,*args,**kw):
        ctrl=args[0];logger=kw['precise_logger'];self.calls.append((logger.meta['run_id'],ctrl))
        a=r.arrays(r.ROOT/'historical/accepted_full_1ms'/f'PC__{ctrl}.npz')
        n=len(a['t']);row=dict(status='OK',note='',case_id=ctrl,n_steps_done=n,n_steps_total=n)
        mode=getattr(self,'failure_mode',None)
        if mode in ['FAILED','INVALID']:row.update(status=mode,note='injected failure')
        if mode=='short':a={k:v[:-1] for k,v in a.items()}
        if mode=='nan':a['q'][17,2]=np.nan
        if mode=='missing':del a['tau_requested']
        if mode=='time':a['t'][20]+=.0002
        if mode=='identity':logger.meta['input_sha256']='wrong'
        if mode=='row_count':row['n_steps_done']=9999
        if mode=='meta_count':n=9999
        np.savez_compressed(logger.path,**a)
        meta=dict(logger.meta,status=row['status'],n_steps_done=row['n_steps_done'],logger_samples=n,n_steps_requested=10000)
        atomic_json(logger.path.with_suffix('.json'),meta);return row
    def batch(self):return r.new_run(self.root,make_report=True)
    def test_old_valid_corrupt_and_changed_xml_always_call_current_integration(self):
        old=self.root/'outputs/full_1ms';old.mkdir(parents=True)
        (old/'PC__A0.npz').write_bytes(b'not an npz');(old/'PC__A0.row.json').write_text('{"status":"OK","n_steps_done":10000}')
        def report(d,**kw):(Path(d)/'results.html').write_text('fixture successful report',encoding='utf-8')
        with patch.object(r.sweep,'run_case',side_effect=self.fake),patch.object(run_report,'render',side_effect=report) as renderer:
            first=self.batch();(old/'PC__A0.npz').write_bytes((r.ROOT/'historical/accepted_full_1ms/PC__A0.npz').read_bytes())
            (self.root/'assets_ur10e/scene.xml').write_text('<mujoco><!-- changed XML identity --></mujoco>',encoding='utf-8');second=self.batch()
        self.assertNotEqual(first,second);self.assertEqual(len(self.calls),6);self.assertEqual(renderer.call_count,2)
        self.assertNotEqual(read(self.root/'runs'/first/'generation.json')['generation_id'],read(self.root/'runs'/second/'generation.json')['generation_id'])
        self.assertEqual([c for rid,c in self.calls if rid==second],list(r.CONTROLLERS))
    def test_failed_invalid_short_nonfinite_missing_time_identity_counts(self):
        for mode in ['FAILED','INVALID','short','nan','missing','time','identity','row_count','meta_count']:
            with self.subTest(mode=mode):
                self.failure_mode=mode
                with patch.object(r.sweep,'run_case',side_effect=self.fake),patch.object(run_report,'render') as renderer:
                    with self.assertRaises(r.RunRejected):self.batch()
                state=sorted((self.root/'runs').glob('*/state.json'),key=lambda p:p.stat().st_mtime_ns)[-1]
                self.assertFalse(read(state)['completed']);self.assertEqual(read(state)['status'],'FAILED')
                self.assertFalse((state.parent/'completed.json').exists());self.assertFalse((state.parent/'results.html').exists());renderer.assert_not_called()
    def test_directory_collision_never_overwrites_or_calls_simulator(self):
        (self.root/'runs/collision').mkdir(parents=True);p=self.root/'runs/collision/keep.txt';p.write_text('keep')
        with patch.object(r.sweep,'run_case') as integrate:
            with self.assertRaises(FileExistsError):r.new_run(self.root,'collision')
            integrate.assert_not_called()
        self.assertEqual(p.read_text(),'keep')
    def test_integration_exception_no_success_and_nonzero_cli(self):
        with patch.object(r,'new_run',side_effect=RuntimeError('injected integration error')):self.assertEqual(r.cli(['new']),2)
        with patch.object(r.sweep,'run_case',side_effect=RuntimeError('injected integration error')),patch.object(run_report,'render') as render:
            with self.assertRaises(RuntimeError):self.batch()
        d=next((self.root/'runs').iterdir());self.assertFalse((d/'completed.json').exists());self.assertEqual(read(d/'state.json')['status'],'FAILED');render.assert_not_called()
    def test_later_case_failure_stops_batch_and_report(self):
        fake=self.fake
        def fail_second(*args,**kw):
            if args[0]=='Alinear':self.failure_mode='INVALID'
            return fake(*args,**kw)
        with patch.object(r.sweep,'run_case',side_effect=fail_second),patch.object(run_report,'render') as render:
            with self.assertRaises(r.RunRejected):self.batch()
        self.assertEqual([c for rid,c in self.calls],['A0','Alinear']);render.assert_not_called()
    def test_history_corruption_no_simulation_or_fallback(self):
        def report(d,**kw):(Path(d)/'results.html').write_text('fixture',encoding='utf-8')
        with patch.object(r.sweep,'run_case',side_effect=self.fake),patch.object(run_report,'render',side_effect=report):rid=self.batch()
        # Current inputs and XML need not remain accessible to view an explicit completed historical batch.
        shutil.rmtree(self.root/'inputs')
        with patch.object(r.sweep,'run_case') as integrate:
            self.assertTrue(r.open_history(rid,self.root).is_file());integrate.assert_not_called()
            (self.root/'runs'/rid/'full_1ms/A0.npz').write_bytes(b'corrupt')
            with self.assertRaises(r.RunRejected):r.open_history(rid,self.root)
            with self.assertRaises(r.RunRejected):r.open_history('missing',self.root)
            integrate.assert_not_called()
    def test_source_change_during_batch_rejects_completion(self):
        fake=self.fake
        def mutate(*args,**kw):
            row=fake(*args,**kw)
            if args[0]=='ASFC_strong':(self.root/'entry.py').write_text('# modified while integrating')
            return row
        with patch.object(r.sweep,'run_case',side_effect=mutate),patch.object(run_report,'render') as render:
            with self.assertRaises(r.RunRejected):self.batch()
        render.assert_not_called();d=next((self.root/'runs').iterdir());self.assertFalse((d/'completed.json').exists())
    def test_actual_report_from_accepted_fixture(self):
        with patch.object(r.sweep,'run_case',side_effect=self.fake):rid=self.batch()
        d=self.root/'runs'/rid
        self.assertTrue((d/'completed.json').is_file());self.assertTrue((d/'results.html').is_file())
        self.assertEqual(len(read(d/'plot_manifest.json')['figures']),5)
        self.assertEqual(len(list((d/'csv_1ms').glob('*.csv'))),3)
    def test_prefix_and_input_failure_protection(self):
        with patch.object(r,'load_inputs',side_effect=RuntimeError('input rejected')),patch.object(r.sweep,'run_case') as integrate:
            with self.assertRaises(RuntimeError):self.batch()
            integrate.assert_not_called()
        p=self.root/'prefix.npz';l=r.PreciseLogger(p,10000,'C_ORDINARY','POSITION_EQUIV',1,self.ctx['input_id'],input_data=self.ctx['c'],common_correction=self.ctx['corr'])
        l.done=4;l.finish(dict(status='FAILED',n_steps_done=3,note='step failed after fourth pre-step sample'))
        self.assertTrue(all(len(x)==3 for x in r.arrays(p).values()));self.assertEqual(read(p.with_suffix('.json'))['logger_samples'],3)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(EngineeringChecks))
    atomic_json(r.ROOT/'review/negative_path_checks.json',dict(passed=result.wasSuccessful(),test_groups=result.testsRun,real_integrations=0,
        optimization=sys.flags.optimize,negative_variants=9,no_MuJoCo_step=True,tests_source_sha256=r.sha(Path(__file__))))
    raise SystemExit(0 if result.wasSuccessful() else 1)
