# Unit tests for problem detection and ETA. Plain script, no Pester needed.
#   powershell -ExecutionPolicy Bypass -File scripts\laptop_watchdog\tests\Test-Detection.ps1
. (Join-Path $PSScriptRoot '..\Watchdog.Lib.ps1')
$env:WATCHDOG_HOME = Join-Path ([IO.Path]::GetTempPath()) 'wd-test-nonexistent'   # force default config
$cfg = Read-WdConfig
$fixture = Get-Content (Join-Path $PSScriptRoot 'fixtures\sample_active.json') -Raw
$idle = Get-Content (Join-Path $PSScriptRoot 'fixtures\sample_idle.json') -Raw
$script:fail = 0; $script:n = 0

function Mk([scriptblock]$Mutate, [string]$At, [string]$Json = $fixture) {
    $j = $Json | ConvertFrom-Json
    if ($Mutate) { & $Mutate $j }
    $res = [pscustomobject]@{ ok = $true; error = $null; attempts = 1; json_text = ($j | ConvertTo-Json -Depth 30 -Compress) }
    New-WdSnapshot -Alias 'x' -Result $res -Now ([DateTimeOffset]::Parse($At)) -Cfg $cfg
}
function MkFail([string]$At) {
    New-WdSnapshot -Alias 'x' -Result ([pscustomobject]@{ ok = $false; error = 'ssh/remote exit code 255'; attempts = 3 }) -Now ([DateTimeOffset]::Parse($At)) -Cfg $cfg
}
function Probs($Snap, $Hist) { $p = Get-WdProblems -S $Snap -History $Hist -Cfg $cfg; foreach ($x in $p) { "$($x.severity):$($x.code)" } }
function Check([string]$Name, $Got, [string[]]$Expect, [string[]]$Absent = @()) {
    $script:n++
    $got = @($Got)
    $ok = $true
    foreach ($e in $Expect) { if ($got -notcontains $e) { $ok = $false } }
    foreach ($a in $Absent) { if ($got -contains $a) { $ok = $false } }
    if ($ok) { Write-Host "PASS  $Name" -ForegroundColor Green }
    else { $script:fail++; Write-Host "FAIL  $Name  got=[$($got -join ',')] expected=[$($Expect -join ',')] absent=[$($Absent -join ',')]" -ForegroundColor Red }
}

# neutral baseline: window closed, pipeline not running (fixture state), no history
$base = Mk $null '2026-10-12T14:20:00+02:00'
Check 'baseline: rearmed fixture gives info only' (Probs $base @()) @('info:rearmed') @('error:not_running', 'error:stuck', 'warning:check_failed')

# --- unreachable
$f1 = MkFail '2026-10-13T01:10:00+02:00'
Check 'one failed night check = warning' (Probs $f1 @()) @('warning:check_failed') @('error:unreachable')
$h = @((MkFail '2026-10-13T00:10:00+02:00'), (MkFail '2026-10-13T00:40:00+02:00'))
Check '3 consecutive failed night checks = error' (Probs $f1 $h) @('warning:check_failed', 'error:unreachable')
$hday = @((MkFail '2026-10-13T10:00:00+02:00'), (MkFail '2026-10-13T10:30:00+02:00'))
Check 'day failures do not trigger the error' (Probs (MkFail '2026-10-13T11:00:00+02:00') $hday) @('warning:check_failed') @('error:unreachable')
$hbreak = @((MkFail '2026-10-13T00:10:00+02:00'), $base, (MkFail '2026-10-13T00:40:00+02:00'))
Check 'a success resets the failure streak' (Probs $f1 $hbreak) @('warning:check_failed') @('error:unreachable')

# --- simple health flags
Check 'rearmed_cron=false -> no rearm info' (Probs (Mk { param($j) $j.rearmed_cron = $false } '2026-10-12T14:20:00+02:00') @()) @() @('info:rearmed')
Check 'gave_up = error' (Probs (Mk { param($j) $j.health.gave_up = $true } '2026-10-12T14:20:00+02:00') @()) @('error:gave_up')
Check 'failures>0 = warning' (Probs (Mk { param($j) $j.health.failures = 2 } '2026-10-12T14:20:00+02:00') @()) @('warning:failures') @('error:gave_up')
Check 'cron silent (heartbeat 1000 s) = warning' (Probs (Mk { param($j) $j.health.heartbeat_age_seconds = 1000 } '2026-10-12T14:20:00+02:00') @()) @('warning:cron_silent')
Check 'heartbeat 120 s is fine' (Probs (Mk { param($j) $j.health.heartbeat_age_seconds = 120 } '2026-10-12T14:20:00+02:00') @()) @() @('warning:cron_silent')
Check 'inactive host = info' (Probs (Mk $null '2026-10-12T14:20:00+02:00' $idle) @()) @('info:inactive')
Check 'schema mismatch = error' (Probs (Mk { param($j) $j.schema = 2 } '2026-10-12T14:20:00+02:00') @()) @('error:schema')

# --- should be running but is not
$openIdle = { param($j) $j.health.gpu_window.state = 'open'; $j.health.gpu_window.seconds = 20000; $j.health.gpu_mem_used_mb = 0; $j.health.pipeline_running = $false }
$p1 = Mk $openIdle '2026-10-12T18:30:00+02:00'
$p2 = Mk $openIdle '2026-10-12T19:00:00+02:00'
Check 'idle in open window, first check = no error yet' (Probs $p1 @()) @() @('error:not_running')
Check 'idle in open window, 2 consecutive = error' (Probs $p2 @($p1)) @('error:not_running')
$short = Mk { param($j) & $openIdle $j; $j.health.gpu_window.seconds = 600 } '2026-10-12T19:00:00+02:00'
Check 'less than 30 min left = no error' (Probs $short @($p1)) @() @('error:not_running')
$busy = Mk { param($j) & $openIdle $j; $j.health.gpu_mem_used_mb = 15000 } '2026-10-12T19:00:00+02:00'
Check 'GPU memory in use = no error' (Probs $busy @($p1)) @() @('error:not_running')
$done = Mk { param($j) & $openIdle $j; $j.all_done = $true } '2026-10-12T19:00:00+02:00'
Check 'all_done = no error' (Probs $done @($p1)) @() @('error:not_running')

# --- stuck
$run = { param($j) $j.health.pipeline_running = $true; $j.health.pipeline_uptime_seconds = 90000 }
$r0 = Mk $run '2026-10-13T20:00:00+02:00'
$r1 = Mk $run '2026-10-13T21:00:00+02:00'
$r2 = Mk $run '2026-10-13T22:00:00+02:00'
$r3 = Mk $run '2026-10-13T23:00:00+02:00'
Check 'running, unchanged 1 h = not stuck yet' (Probs $r1 @($r0)) @() @('error:stuck')
Check 'running, unchanged 3 h (> 2 x epoch 1376 s and > 1 h) = stuck' (Probs $r3 @($r0, $r1, $r2)) @('error:stuck')
$moved = Mk { param($j) & $run $j; $j.progress.validate.epochs_done = 38; $j.progress.validate.live = 'Epoch 38: 1%' } '2026-10-13T23:00:00+02:00'
Check 'progress moved = not stuck' (Probs $moved @($r0, $r1, $r2)) @() @('error:stuck')
$notRun = Mk $null '2026-10-13T21:30:00+02:00'
Check 'a stopped check breaks the stuck streak' (Probs $r3 @($r0, $notRun, $r2)) @() @('error:stuck')

# --- crash vs cleanup
$prevM = Mk { param($j) $j.progress.validate.exception_marker_at = '2026-10-09T00:33:02+02:00' } '2026-10-12T14:00:00+02:00'
Check 'new marker Tue 00:33 = expected cleanup (info)' (Probs (Mk { param($j) $j.progress.validate.exception_marker_at = '2026-10-13T00:33:05+02:00' } '2026-10-13T00:40:00+02:00') @($prevM)) @('info:cleanup_kill') @('warning:crash')
Check 'new marker Tue 15:12 = crash (warning)' (Probs (Mk { param($j) $j.progress.validate.exception_marker_at = '2026-10-13T15:12:05+02:00' } '2026-10-13T16:00:00+02:00') @($prevM)) @('warning:crash') @('info:cleanup_kill')
Check 'new marker Sat 00:33 = crash (weekend is spared)' (Probs (Mk { param($j) $j.progress.validate.exception_marker_at = '2026-10-17T00:33:05+02:00' } '2026-10-17T00:40:00+02:00') @($prevM)) @('warning:crash')
Check 'same marker = nothing' (Probs (Mk { param($j) $j.progress.validate.exception_marker_at = '2026-10-09T00:33:02+02:00' } '2026-10-12T14:30:00+02:00') @($prevM)) @() @('warning:crash', 'info:cleanup_kill')

# --- ETA. Windows: Mon-Thu 18:00 -> 08:00, Fri 18:00 -> Mon 08:00; weekday 00:33 kill costs 10 min restart gap + 1 epoch (1376 s here)
function EtaAt($Remaining, [string]$From, $C = $cfg) {
    $j = $fixture | ConvertFrom-Json; $j.remaining_gpu_seconds = $Remaining
    Get-WdEta -Latest $j -At ([DateTimeOffset]::Parse($From)) -Cfg $C
}
function Fin($e, [string]$fmt = 'yyyy-MM-dd HH') { ([DateTimeOffset]::Parse($e.finish)).LocalDateTime.ToString($fmt) }
Check 'ETA 3 h on Monday 14:20 = Monday 21:00 (before the kill)' @((Fin (EtaAt (3*3600) '2026-10-12T14:20:00+02:00') 'yyyy-MM-dd HH:mm')) @('2026-10-12 21:00')
$e = EtaAt (10*3600) '2026-10-12T14:20:00+02:00'
Check 'ETA 10 h from Monday = Tue 04h (18:00 + 10 h + kill penalty), 1 window' @((Fin $e), $e.windows) @('2026-10-13 04', '1')
$e = EtaAt (24*3600) '2026-10-12T14:20:00+02:00'
Check 'ETA 24 h from Monday = Wed 05h, 2 windows' @((Fin $e), $e.windows) @('2026-10-14 05', '2')
$e = EtaAt (60*3600) '2026-10-14T10:00:00+02:00'
Check 'ETA 60 h from Wed morning = Sun 03h, 3 windows (Wed, Thu, Fri block)' @((Fin $e), $e.windows) @('2026-10-18 03', '3')
$noRearm = $cfg | ConvertTo-Json -Depth 10 | ConvertFrom-Json; $noRearm.eta.assume_rearm = $false
$e = EtaAt (10*3600) '2026-10-12T14:20:00+02:00' $noRearm
Check 'ETA without re-arm: windows end at 00:33, 10 h finishes Tue 21h' @((Fin $e)) @('2026-10-13 21')
$j = $fixture | ConvertFrom-Json; $j.remaining_gpu_seconds = $null
Check 'ETA null when remaining unknown' @([string]($null -eq (Get-WdEta -Latest $j -At ([DateTimeOffset]::Now) -Cfg $cfg))) @('True')

Write-Host ""
if ($script:fail -eq 0) { Write-Host "ALL $($script:n) TESTS PASSED" -ForegroundColor Green; exit 0 }
else { Write-Host "$($script:fail) / $($script:n) FAILED" -ForegroundColor Red; exit 1 }
