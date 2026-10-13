"""Cheap R026 failure-path checks; all run_case paths mocked, zero integrations."""
from pathlib import Path
import sys,json,tempfile,unittest
from unittest.mock import patch,Mock
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'F组首次接入_20261010/.runtime'));sys.path.insert(0,str(ROOT))
import numpy as np
import run_short as r
from input_acceptance import InputRejected,require_calibration,require_event
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v),encoding='utf-8')
class Guards(unittest.TestCase):
    def test_numeric_calibration_failure(self):
        good=dict(g=.5,A_target_P95_um=50,S0_P95_um=100,S1_P95_um=50,check_tolerance_relative=.1)
        self.assertTrue(require_calibration(good)['passed'])
        for change in [dict(g=0),dict(g=1.2),dict(S1_P95_um=60),dict(S1_P95_um=float('nan')),dict(check_tolerance_relative=.5),dict(accepted=False),dict(S0_P95_um=0)]:
            with self.subTest(change=change),self.assertRaises(InputRejected):require_calibration(dict(good,**change))
    def test_legal_no_amplification(self):
        v=require_calibration(dict(g=1.,A_target_P95_um=100.,S0_P95_um=20.,S1_P95_um=20.,check_tolerance_relative=.1))
        self.assertTrue(v['no_amplification_branch'])
    def test_event_failure_numeric_and_flag(self):
        self.assertTrue(require_event(dict(final_check=dict(relative_rms_error=.2),sufficient_for_approximate_comparison=True))['passed'])
        for err,flag in [(.200001,True),(.05,False),(float('nan'),True),(-.1,True),(.05,None)]:
            with self.subTest(error=err,flag=flag),self.assertRaises(InputRejected):require_event(dict(final_check=dict(relative_rms_error=err),sufficient_for_approximate_comparison=flag))
    def test_cached_failures_stop_before_any_run_or_fd_use(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'review') as d:
            root=Path(d);save(root/'review/ordinary_calibration.json',dict(g=.5,A_target_P95_um=50,S0_P95_um=100,S1_P95_um=90,check_tolerance_relative=.1))
            save(root/'review/event_reconstruction.json',dict(final_check=dict(relative_rms_error=.3),sufficient_for_approximate_comparison=False))
            with patch.object(r,'ROOT',root),patch.object(r,'run') as run,patch.object(r.np,'loadtxt') as load:
                with self.assertRaises(InputRejected):r.calibrate({},None)
                with self.assertRaises(InputRejected):r.reconstruct({})
                run.assert_not_called();load.assert_not_called()
    def test_generated_calibration_exhaustion_never_formalizes(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'review') as d:
            root=Path(d);p=root/'fake.npz';save(p.with_suffix('.row.json'),dict(status='OK'))
            save(root/'review/A_ordinary_reference.json',dict(target_P95_um=50))
            with patch.object(r,'ROOT',root),patch.object(r,'run',return_value=p) as run,patch.object(r,'ordinary_p95',side_effect=[100,90,90,90]),patch.object(r,'formalize') as formal:
                with self.assertRaises(InputRejected):r.calibrate({},None)
                self.assertEqual(run.call_count,4);formal.assert_not_called()
                self.assertFalse(r.read(root/'review/ordinary_calibration.json')['accepted'])
    def test_generated_event_exhaustion_never_freezes_or_formalizes(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory(dir=ROOT/'review') as d:
            root=Path(d);p=root/'fake.npz';save(p.with_suffix('.row.json'),dict(status='OK'))
            (root/'reference_B_packet').mkdir();(root/'reference_B_packet/plant_frf.npz').write_bytes(b'test placeholder, no plant measurements')
            fd=SimpleNamespace(path=root/'candidate.csv',sha256='test');c=dict(t=np.arange(10)*.001,event_target_um=np.zeros(10))
            with patch.object(r,'ROOT',root),patch.object(r.disturbance,'load_plant',return_value=object()),patch.object(r,'inverse_padded',return_value=np.zeros(10)),patch.object(r,'fd_object',return_value=fd),patch.object(r,'run',return_value=p) as run,patch.object(r,'array',return_value=dict(physical_e_um=np.zeros(10))),patch.object(r,'event_check',return_value=dict(relative_rms_error=.21)),patch.object(r,'formalize') as formal,patch.object(r.shutil,'copy2') as copy:
                with self.assertRaises(InputRejected):r.reconstruct(c)
                self.assertEqual(run.call_count,3);formal.assert_not_called();copy.assert_not_called()
                self.assertFalse(r.read(root/'review/event_reconstruction.json')['sufficient_for_approximate_comparison'])
if __name__=='__main__':
    res=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Guards))
    save(ROOT/'review/R026_failure_checks.json',dict(passed=res.wasSuccessful(),unit_tests=res.testsRun,numeric_rejection_subcases=12,new_integrations=0,
        cached_and_generated_paths_exercised=True,failed_inputs_not_formalized_or_used=True,original_thresholds=dict(calibration_relative=.10,event_relative=.20),legal_no_amplification_preserved=True))
    raise SystemExit(0 if res.wasSuccessful() else 1)
