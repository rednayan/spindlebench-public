@echo off
rem fixA full runs on the GPU (run_all.py --fix: 5 seeds, full schedule), then selection scoring of them.
rem Reads cache and results from the main checkout, writes runs to ..\runs\student_fix (both next to it).
rem   run_fix.cmd [workers] [wait]   wait = "wait": first wait until the main grid log says "training
rem                                  finished" (the grid's selection scoring then runs on the CPU beside it,
rem                                  hence 3 workers by default).
rem   Logs: <main>\results\student\logs\run_fix.out, selection_score_fix.out, train_student_rules4_fixA_gpu_s*.out
cd /d "%~dp0"
set MAIN=%~dp0..\..\..\spindlebench-private
set SPINDLEBENCH_RESULTS=%MAIN%\results
set SPINDLEBENCH_CACHE=%MAIN%\cache
set SPINDLEBENCH_RUNS=%~dp0..\..\..\runs\student_fix
set LOG=%SPINDLEBENCH_RESULTS%\student\logs\run_fix.out
set WORKERS=%1
if "%WORKERS%"=="" set WORKERS=3
if /i not "%~2"=="wait" goto start
echo %date% %time% waiting for the main grid to finish training >> "%LOG%"
:wait
findstr /c:"training finished" "%SPINDLEBENCH_RESULTS%\student\logs\run_all_gpu.out" >nul 2>&1 && goto start
ping -n 301 127.0.0.1 >nul
goto wait
:start
echo %date% %time% fixA start, %WORKERS% workers >> "%LOG%"
uv run --no-group cpu --group cuda python -W ignore run_all.py --track gpu --fix --workers %WORKERS% >> "%LOG%" 2>&1
echo %date% %time% fixA training finished >> "%LOG%"
uv run --no-group cpu --group cuda python -W ignore selection.py score --workers 4 >> "%SPINDLEBENCH_RESULTS%\student\logs\selection_score_fix.out" 2>&1
echo %date% %time% fixA ALL DONE >> "%LOG%"
