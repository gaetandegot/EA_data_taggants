# Laptop monitor of the multi-seed robustness experiment (scripts/queue_phi_seed.sh on tarse / frontal / temporal).
# Self-contained: does not use or modify the ImageNet watchdog (scripts/laptop_watchdog) except that -Setup disables
# its scheduled tasks (they would re-arm the ImageNet runs, which compete for the GPUs) and -Uninstall re-enables them.
#
#   .\PhiWatch.ps1 -Setup [-Aliases gpu,gpu2,gpu3]   once: start the seed on each machine (seed chosen from the host
#                                                   name: tarse 1, frontal 2, temporal 3), install the night tasks,
#                                                   run a first check, open the dashboard
#   .\PhiWatch.ps1                                  one check of every machine + dashboard (what the tasks run)
#   .\PhiWatch.ps1 -Open                            open the dashboard
#   .\PhiWatch.ps1 -Uninstall                       remove the tasks (and re-enable the ImageNet watchdog tasks)
#
# Each check runs `bash ~/EA/EA_data_taggants/scripts/phi_tick.sh` over SSH: it relaunches the machine's queue when
# it should be running but is not (e.g. after the 00:33 cleanup) and prints the run state as one JSON line.
# Data, log and dashboard: %LOCALAPPDATA%\PhiWatch (dashboard.html: open it in a browser, it reloads itself).
# ASCII only (Windows PowerShell 5.1 reads BOM-less files as ANSI).
param([switch]$Setup, [switch]$Uninstall, [switch]$Open, [string[]]$Aliases)
$ErrorActionPreference = 'Stop'
$home_ = Join-Path $env:LOCALAPPDATA 'PhiWatch'
$app = Join-Path $home_ 'app'
$cfgFile = Join-Path $home_ 'config.json'
$dash = Join-Path $home_ 'dashboard.html'
$logFile = Join-Path $home_ 'phiwatch.log'
$tick = 'bash ~/EA/EA_data_taggants/scripts/phi_tick.sh'
$taskNight = 'PhiWatch-Night'; $taskWeekend = 'PhiWatch-Weekend'
$oldTasks = 'TaggantsWatchdog-Night', 'TaggantsWatchdog-Weekend', 'TaggantsWatchdog-Day'
New-Item -ItemType Directory -Force (Join-Path $home_ 'data') | Out-Null

function Log([string]$msg) {
    $line = '{0} {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    try { Add-Content -Path $logFile -Value $line -Encoding UTF8 } catch {}
    Write-Host $line
}

function Split-Aliases($list) {
    # `powershell -File x.ps1 -Aliases a,b,c` passes ONE string "a,b,c" (no array parsing with -File): split it
    return @(@($list) | ForEach-Object { "$_" -split '[,\s]+' } | Where-Object { $_ })
}

function Get-Aliases {
    if ($Aliases) { return Split-Aliases $Aliases }
    if (Test-Path $cfgFile) { return Split-Aliases (Get-Content $cfgFile -Raw | ConvertFrom-Json).aliases }
    return @('gpu', 'gpu2', 'gpu3')
}

# Run one remote command on every alias in parallel; returns @{alias = @{ok; out; err; code}}
function Invoke-Remote([string[]]$Names, [string]$Cmd, [int]$Timeout = 150) {
    $jobs = @{}
    foreach ($a in $Names) {
        $jobs[$a] = Start-Job -ArgumentList $a, $Cmd, $Timeout -ScriptBlock {
            param($alias, $cmd, $timeout)
            $res = $null
            for ($attempt = 1; $attempt -le 3; $attempt++) {
                $psi = New-Object Diagnostics.ProcessStartInfo
                $psi.FileName = 'ssh.exe'
                $psi.Arguments = "-o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=10 -o ServerAliveCountMax=3 $alias `"$cmd`""
                $psi.UseShellExecute = $false; $psi.CreateNoWindow = $true
                $psi.RedirectStandardOutput = $true; $psi.RedirectStandardError = $true; $psi.RedirectStandardInput = $true
                $code = -1; $out = ''; $err = ''
                try {
                    $p = [Diagnostics.Process]::Start($psi)
                    $p.StandardInput.Close()
                    $so = $p.StandardOutput.ReadToEndAsync(); $se = $p.StandardError.ReadToEndAsync()
                    if ($p.WaitForExit($timeout * 1000)) { $code = $p.ExitCode; $out = $so.Result; $err = $se.Result }
                    else { try { $p.Kill() } catch {}; $err = "timeout after ${timeout}s" }
                } catch { $code = -2; $err = $_.Exception.Message }
                $res = [pscustomobject]@{ ok = ($code -eq 0); out = $out; err = $err.Trim(); code = $code }
                if ($res.ok -or ($code -ne 255 -and $code -ne -1 -and $code -ne -2)) { break }  # retry SSH-level failures only
                Start-Sleep -Seconds 15
            }
            $res
        }
    }
    $r = @{}
    foreach ($a in $Names) {
        $null = Wait-Job $jobs[$a] -Timeout ($Timeout * 3 + 60)
        $x = @(Receive-Job $jobs[$a] -ErrorAction SilentlyContinue) | Select-Object -Last 1
        if (-not $x) { $x = [pscustomobject]@{ ok = $false; out = ''; err = 'no result (job timeout)'; code = -3 } }
        $r[$a] = $x
        Remove-Job $jobs[$a] -Force -ErrorAction SilentlyContinue
    }
    return $r
}

function JsonStr([string]$s) {  # JSON string literal
    if ($null -eq $s) { return 'null' }
    $b = New-Object Text.StringBuilder
    [void]$b.Append('"')
    foreach ($c in $s.ToCharArray()) {
        switch ($c) {
            '"' { [void]$b.Append('\"') } '\' { [void]$b.Append('\\') }
            "`n" { [void]$b.Append('\n') } "`r" { [void]$b.Append('\r') } "`t" { [void]$b.Append('\t') }
            default { if ([int]$c -lt 32 -or [int]$c -gt 126) { [void]$b.Append(('\u{0:x4}' -f [int]$c)) } else { [void]$b.Append($c) } }
        }
    }
    [void]$b.Append('"')
    return $b.ToString()
}

function Invoke-Check {
    $names = Get-Aliases
    Log "check start ($($names -join ', '))"
    $res = Invoke-Remote $names $tick
    $now = [DateTimeOffset]::Now.ToString('o')
    foreach ($a in $names) {
        $x = $res[$a]
        $status = 'null'; $err = $null
        if ($x.ok) {
            $line = ($x.out -split "`r?`n" | Where-Object { $_.Trim() } | Select-Object -Last 1)
            try { $null = $line | ConvertFrom-Json -ErrorAction Stop; $status = $line.Trim() } catch { $err = 'invalid JSON from phi_tick' }
        } else { $err = "ssh exit $($x.code): $($x.err)" }
        $rec = '{{"t":{0},"alias":{1},"ok":{2},"error":{3},"status":{4}}}' -f (JsonStr $now), (JsonStr $a), $(if ($status -ne 'null') { 'true' } else { 'false' }), (JsonStr $err), $status
        $f = Join-Path $home_ "data\$a.jsonl"
        Add-Content -Path $f -Value $rec -Encoding UTF8
        if ((Get-Item $f).Length -gt 20MB) { Set-Content $f (Get-Content $f -Tail 5000 -Encoding UTF8) -Encoding UTF8 }
        if ($err) { Log "$a FAILED: $err" }
        else {
            $s = $status | ConvertFrom-Json
            if (-not $s.active) { Log "$a ($($s.host)): no phi run registered" }
            else { Log ("{0} ({1}) seed {2}: stage {3} running={4} relaunched={5} failures={6}" -f $a, $s.host, $s.seed, $s.current_stage, $s.health.pipeline_running, $s.relaunched, $s.health.failures) }
        }
    }
    Write-Dashboard $names
    Log "check end; dashboard: $dash"
}

function Write-Dashboard([string[]]$Names) {
    $parts = @()
    foreach ($a in $Names) {
        $f = Join-Path $home_ "data\$a.jsonl"
        $lines = @()
        if (Test-Path $f) { $lines = @(Get-Content $f -Tail 400 -Encoding UTF8 | Where-Object { $_.Trim() }) }
        $parts += ('{0}:[{1}]' -f (JsonStr $a), ($lines -join ','))
    }
    $tasks = @()
    foreach ($n in $taskNight, $taskWeekend) {
        $t = Get-ScheduledTask -TaskName $n -ErrorAction SilentlyContinue
        $next = $null
        if ($t) { $i = Get-ScheduledTaskInfo -TaskName $n; if ($i.NextRunTime -and $i.NextRunTime.Year -gt 2000) { $next = $i.NextRunTime.ToString('o') } }
        $tasks += ('{{"name":{0},"state":{1},"next":{2}}}' -f (JsonStr $n), (JsonStr $(if ($t) { [string]$t.State } else { 'not installed' })), (JsonStr $next))
    }
    $data = '{{"generated":{0},"aliases":[{1}],"history":{{{2}}},"tasks":[{3}]}}' -f (JsonStr ([DateTimeOffset]::Now.ToString('o'))), (($Names | ForEach-Object { JsonStr $_ }) -join ','), ($parts -join ','), ($tasks -join ',')
    $tpl = [IO.File]::ReadAllText((Join-Path $PSScriptRoot 'dashboard.template.html'), [Text.Encoding]::UTF8)
    $html = $tpl.Replace('/*__DATA__*/null', $data.Replace('</', '<\/'))
    [IO.File]::WriteAllText($dash, $html, (New-Object Text.UTF8Encoding($false)))
}

function Install-Tasks {
    New-Item -ItemType Directory -Force $app | Out-Null
    foreach ($f in 'PhiWatch.ps1', 'dashboard.template.html', 'run-hidden.vbs') { Copy-Item (Join-Path $PSScriptRoot $f) (Join-Path $app $f) -Force }
    $action = New-ScheduledTaskAction -Execute 'wscript.exe' -Argument ('//B //Nologo "{0}"' -f (Join-Path $app 'run-hidden.vbs'))
    $princ = New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
    $set = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 15) -DontStopIfGoingOnBatteries
    $set.DisallowStartIfOnBatteries = $true
    # every night: every 30 min 18:00-07:30, plus 00:40 and 01:10 (right after the 00:33 cleanup). One explicit trigger
    # per slot: Windows only arms a wake timer for a trigger's start time, not for repetitions.
    $trig = @()
    foreach ($m in 0..27) { $trig += New-ScheduledTaskTrigger -Daily -At (([datetime]'2000-01-01 18:00').AddMinutes(30 * $m).ToString('HH:mm')) }
    $trig += New-ScheduledTaskTrigger -Daily -At '00:40'
    $trig += New-ScheduledTaskTrigger -Daily -At '01:10'
    foreach ($n in $taskNight, $taskWeekend) { if (Get-ScheduledTask -TaskName $n -ErrorAction SilentlyContinue) { Unregister-ScheduledTask -TaskName $n -Confirm:$false } }
    Register-ScheduledTask -TaskName $taskNight -Action $action -Trigger $trig -Settings $set -Principal $princ -Description 'PhiWatch: night checks of the robustness experiment (wakes the laptop on AC power)' | Out-Null
    # weekend daytime (the GPU window is open from Friday 18:00 to Monday 08:00); separate task (48 triggers max)
    $wk = @()
    foreach ($m in 0..19) { $wk += New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday, Sunday -At (([datetime]'2000-01-01 08:00').AddMinutes(30 * $m).ToString('HH:mm')) }
    Register-ScheduledTask -TaskName $taskWeekend -Action $action -Trigger $wk -Settings $set -Principal $princ -Description 'PhiWatch: weekend daytime checks' | Out-Null
    foreach ($n in $taskNight, $taskWeekend) { if (-not (Get-ScheduledTask -TaskName $n -ErrorAction SilentlyContinue)) { throw "task $n was not registered" } }
    Log "tasks installed: $taskNight, $taskWeekend (wake on AC power only)"
    foreach ($n in $oldTasks) {
        if (Get-ScheduledTask -TaskName $n -ErrorAction SilentlyContinue) { Disable-ScheduledTask -TaskName $n | Out-Null; Log "disabled ImageNet watchdog task $n (re-enabled by -Uninstall)" }
    }
}

if ($Uninstall) {
    foreach ($n in $taskNight, $taskWeekend) { if (Get-ScheduledTask -TaskName $n -ErrorAction SilentlyContinue) { Unregister-ScheduledTask -TaskName $n -Confirm:$false; Log "removed $n" } }
    foreach ($n in $oldTasks) { if (Get-ScheduledTask -TaskName $n -ErrorAction SilentlyContinue) { Enable-ScheduledTask -TaskName $n | Out-Null; Log "re-enabled $n" } }
    exit 0
}
if ($Open) { Start-Process $dash; exit 0 }

if ($Setup) {
    $names = Get-Aliases
    ('{{"aliases":[{0}]}}' -f (($names | ForEach-Object { JsonStr $_ }) -join ',')) | Set-Content $cfgFile -Encoding UTF8
    Log "setup: launching the experiment on $($names -join ', ')"
    $res = Invoke-Remote $names 'bash ~/EA/EA_data_taggants/scripts/phi_launch.sh auto' 120
    $bad = 0
    foreach ($a in $names) {
        $x = $res[$a]
        $txt = (($x.out + "`n" + $x.err).Trim() -split "`r?`n" | Where-Object { $_.Trim() }) -join ' | '
        if ($x.ok) { Log "  $a ok: $txt" } else { $bad++; Log "  $a FAILED (exit $($x.code)): $txt" }
    }
    Install-Tasks
    Invoke-Check
    Start-Process $dash
    if ($bad) { Log "setup finished with $bad machine(s) not launched: read the messages above, fix, then re-run -Setup" ; exit 1 }
    exit 0
}

$mutex = New-Object Threading.Mutex($false, 'Global\PhiWatchRun')
if (-not $mutex.WaitOne(0)) { Log 'another check is running, exiting'; exit 0 }
try { Invoke-Check } catch { Log "check crashed: $($_.Exception.Message)" } finally { $mutex.ReleaseMutex() }
