# Keep Windows awake while the student training runs (no admin rights needed).
# Holds ES_SYSTEM_REQUIRED (the same request a video player makes) as long as a run_all.py, train.py or
# selection.py process is alive, then exits after 15 min with none of them. The display may still turn off.
#   Start hidden:  Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','<path>\keep_awake.ps1' -WindowStyle Hidden
#   Check:         powercfg /requests   (as admin)   or   Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | ? CommandLine -match keep_awake
Add-Type -Namespace Win32 -Name Power -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
'@
$ES_CONTINUOUS = [uint32]"0x80000000"; $ES_SYSTEM_REQUIRED = [uint32]"0x00000001"
$pattern = 'run_all\.py|train\.py|selection\.py'
$idle = 0
while ($idle -lt 15) {
    $busy = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match $pattern }
    if ($busy) { $idle = 0 } else { $idle += 1 }
    [void][Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
    Start-Sleep -Seconds 60
}
[void][Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS)
