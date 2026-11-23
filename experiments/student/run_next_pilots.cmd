@echo off
REM Unattended 1-seed GPU pilot suite for the RTX 4070 machine (restartable: skips finished runs).
REM Run from experiments\student\:
REM     run_next_pilots.cmd

setlocal enabledelayedexpansion
set PY=uv run --no-group cpu --group cuda python
set RUNS=..\..\..\runs\student

echo === [1/5] Arm A: Faint-Copy + PU + Wake/REM (15 passes) ===
if not exist "%RUNS%\student_faint_pu_gpu_s0\done.json" (
    %PY% train.py --arm student_faint_pu --seed 0 --track gpu --passes 15
    if errorlevel 1 exit /b 1
) else (
    echo   (already done, skipping)
)

echo === [2/5] Arm C: C4-M1 Contralateral Witness + O2 Alpha Veto + Faint-Copy + PU + Wake/REM (15 passes) ===
if not exist "%RUNS%\student_faint_pu_c4witness_gpu_s0\done.json" (
    %PY% train.py --arm student_faint_pu --seed 0 --track gpu --passes 15 --mc-witness --tag c4witness
    if errorlevel 1 exit /b 1
) else (
    echo   (already done, skipping)
)

echo === [3/5] Arm B Stage 1: SNR-Invariant VICReg SSL Pretraining (15 passes) ===
if not exist "%RUNS%\ssl_vicreg_gpu_s0\done.json" (
    %PY% pretrain_ssl.py --seed 0 --track gpu --passes 15
    if errorlevel 1 exit /b 1
) else (
    echo   (already done, skipping)
)

echo === [4/5] Arm B Stage 2: Frozen Linear Probes (5 passes each) ===
if not exist "%RUNS%\student_ds_ssl_probe_gpu_s0\done.json" (
    %PY% train.py --arm student_ds --seed 0 --track gpu --passes 5 --init-from "%RUNS%\ssl_vicreg_gpu_s0" --freeze-trunk --tag ssl_probe
    if errorlevel 1 exit /b 1
) else (
    echo   (student_ds_ssl_probe already done, skipping)
)
if not exist "%RUNS%\student_faint_pu_ssl_probe_gpu_s0\done.json" (
    %PY% train.py --arm student_faint_pu --seed 0 --track gpu --passes 5 --init-from "%RUNS%\ssl_vicreg_gpu_s0" --freeze-trunk --tag ssl_probe
    if errorlevel 1 exit /b 1
) else (
    echo   (student_faint_pu_ssl_probe already done, skipping)
)

echo === [5/5] Evaluate all pilots on the 3-axis Step 1 trade-off curve (40 val nights, v0..v3) ===
%PY% eval_faint_curve.py --reuse-baselines --device cuda --runs student_ds_gpu_s0,student_rules4_fixA_gpu_s0,student_faint_pu_gpu_s0:15,student_faint_pu_c4witness_gpu_s0:15,student_ds_ssl_probe_gpu_s0:5,student_faint_pu_ssl_probe_gpu_s0:5
if errorlevel 1 exit /b 1

echo === All pilots complete! Check results\student\faint_curve\faint_tradeoff.png ===
endlocal
