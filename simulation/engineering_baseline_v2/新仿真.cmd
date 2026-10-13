@echo off
cd /d "%~dp0"
"C:\Users\kinli\Documents\Codex\2026-10-02\UR10_CB3_single_joint_pretest_local_acceptance\.venv\Scripts\python.exe" run_slow.py new
if errorlevel 1 echo FAILED: no successful result or fallback.
pause
