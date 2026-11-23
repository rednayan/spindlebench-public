@echo off
rem GPU track, unattended: train every job of `run_all.py --track gpu` on the NVIDIA GPU, then score
rem every finished run for selection. Restartable: run this file again after a reboot; finished runs are
rem skipped, interrupted ones are archived and rerun from scratch.
rem   Start hidden (PowerShell):  Start-Process cmd -ArgumentList '/c', '<path>\run_gpu_grid.cmd' -WindowStyle Hidden
rem   Workers: default 4, the RAM limit on the RTX 4070 machine (about 4.5 GB RAM per run, 32 GB in all).
rem   Priority (optional, in quotes): run_gpu_grid.cmd 4 "lam0_gpu,student_mv,student_ds" runs those jobs first.
rem   Logs: results\student\logs\run_all_gpu.out, selection_score.out, train_<run>.out
cd /d "%~dp0"
set WORKERS=%1
if "%WORKERS%"=="" set WORKERS=4
set FIRST=
if not "%~2"=="" set FIRST=--first %~2
echo %date% %time% start, %WORKERS% GPU workers %FIRST% >> ..\..\results\student\logs\run_all_gpu.out
uv run --no-group cpu --group cuda python -W ignore run_all.py --track gpu --workers %WORKERS% %FIRST% >> ..\..\results\student\logs\run_all_gpu.out 2>&1
echo %date% %time% training finished >> ..\..\results\student\logs\run_all_gpu.out
uv run --no-group cpu --group cuda python -W ignore selection.py score --workers 6 >> ..\..\results\student\logs\selection_score.out 2>&1
echo %date% %time% ALL DONE >> ..\..\results\student\logs\run_all_gpu.out
