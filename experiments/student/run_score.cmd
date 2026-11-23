@echo off
rem Selection scoring with the network on the GPU (selection.py --device cuda), for runs that already exist.
rem   run_score.cmd <runs folder name under ..\runs> [workers]   e.g. run_score.cmd student 6
rem Reads cache and results from the main checkout. Log: <main>\results\student\logs\selection_score_<name>.out
rem Restarts itself (up to 30 times, 1 min apart) when selection.py ends with an error, e.g. when its
rem processes are killed from outside; finished runs are skipped, so a restart loses only the runs in progress.
cd /d "%~dp0"
set MAIN=%~dp0..\..\..\spindlebench-private
set SPINDLEBENCH_RESULTS=%MAIN%\results
set SPINDLEBENCH_CACHE=%MAIN%\cache
set SPINDLEBENCH_RUNS=%~dp0..\..\..\runs\%1
set WORKERS=%2
if "%WORKERS%"=="" set WORKERS=6
set LOG=%SPINDLEBENCH_RESULTS%\student\logs\selection_score_%1.out
set TRY=0
:again
set /a TRY+=1
echo %date% %time% scoring start (GPU), %WORKERS% workers, try %TRY% >> "%LOG%"
uv run --no-group cpu --group cuda python -W ignore selection.py score --device cuda --workers %WORKERS% >> "%LOG%" 2>&1
if not errorlevel 1 goto done
echo %date% %time% selection.py ended with exit code %errorlevel% >> "%LOG%"
if %TRY% geq 30 goto done
ping -n 61 127.0.0.1 >nul
goto again
:done
echo %date% %time% scoring finished >> "%LOG%"
