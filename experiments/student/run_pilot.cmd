@echo off
rem rules4-fix pilot on the GPU (run_all.py --pilot): 3 variants x seeds 0, 1, passes 1-5 each.
rem Reads cache and results from the main checkout, writes runs to ..\runs\student_pilot (both next to it).
rem   run_pilot.cmd [workers]   workers = machine-wide limit on training processes, the grid's included:
rem                             4 = only free grid slots (waits for the grid); 5 = one run beside the grid
rem                             (about 4.5 GB RAM per run, 32 GB in all).
rem   Logs: <main>\results\student\logs\run_pilot.out, train_student_rules4_fix*_gpu_s*.out
cd /d "%~dp0"
set MAIN=%~dp0..\..\..\spindlebench-private
set SPINDLEBENCH_RESULTS=%MAIN%\results
set SPINDLEBENCH_CACHE=%MAIN%\cache
set SPINDLEBENCH_RUNS=%~dp0..\..\..\runs\student_pilot
set SPINDLEBENCH_RUNS_MAIN=%~dp0..\..\..\runs\student
set WORKERS=%1
if "%WORKERS%"=="" set WORKERS=4
echo %date% %time% pilot start, %WORKERS% workers >> "%SPINDLEBENCH_RESULTS%\student\logs\run_pilot.out"
uv run --no-group cpu --group cuda python -W ignore run_all.py --track gpu --pilot --workers %WORKERS% >> "%SPINDLEBENCH_RESULTS%\student\logs\run_pilot.out" 2>&1
echo %date% %time% pilot finished >> "%SPINDLEBENCH_RESULTS%\student\logs\run_pilot.out"
