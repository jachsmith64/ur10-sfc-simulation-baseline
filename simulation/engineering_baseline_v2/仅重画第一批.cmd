@echo off
cd /d "%~dp0"
"C:\Users\kinli\Documents\Codex\2026-10-02\UR10_CB3_single_joint_pretest_local_acceptance\.venv\Scripts\python.exe" run_slow.py redraw --run-id 20261012T123028491525Z_9f7f6d3a
if errorlevel 1 echo FAILED: no successful result or fallback.
pause
