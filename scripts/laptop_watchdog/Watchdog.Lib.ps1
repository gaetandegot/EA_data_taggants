# Shared functions for the laptop watchdog. ASCII only (Windows PowerShell 5.1 reads BOM-less files as ANSI).
$script:Inv = [Globalization.CultureInfo]::InvariantCulture

function Get-WdHome {
    if ($env:WATCHDOG_HOME) { return $env:WATCHDOG_HOME }
    return (Join-Path $env:LOCALAPPDATA 'TaggantsWatchdog')
}

function Get-WdDefaultConfig {
    [ordered]@{
        aliases                = @('gpu', 'gpu2', 'gpu3')
        remote_command         = 'bash ~/EA/EA_data_taggants/scripts/remote_tick.sh'
        check_timeout_seconds  = 120
        network_wait_seconds   = 180
        retry_interval_seconds = 15
        refresh_seconds        = 300
        log_max_bytes          = 5242880
        history_max_lines      = 20000
        timeline_days          = 14
        night_start_hour       = 18
        night_end_hour         = 8
        detection              = [ordered]@{
            fail_error_count        = 3
            min_window_seconds      = 1800
            idle_gpu_mem_mb         = 2000
            stuck_min_seconds       = 3600
            stuck_factor            = 2
            heartbeat_max_age       = 900
            cleanup_kill_time       = '00:33'
            cleanup_kill_tolerance_minutes = 10
        }
        eta                    = [ordered]@{
            weekday_window_start   = '18:00'
            weekday_window_end     = '08:00'
            weekend_window_start   = '18:00'
            weekend_window_end     = '08:00'
            kill_time              = '00:33'
            assume_rearm           = $true
            rearm_gap_seconds      = 600
            lost_seconds_default   = 1400
            max_days               = 90
        }
        schedule               = [ordered]@{
            night_task   = 'TaggantsWatchdog-Night'
            weekend_task = 'TaggantsWatchdog-Weekend'
            day_task     = 'TaggantsWatchdog-Day'
        }
    }
}

function Read-WdConfig {
    $path = Join-Path (Get-WdHome) 'config.json'
    if (-not (Test-Path $path)) { return ([pscustomobject](Get-WdDefaultConfig | ConvertTo-Json -Depth 10 | ConvertFrom-Json)) }
    return (Get-Content $path -Raw -Encoding UTF8 | ConvertFrom-Json)
}

function ConvertTo-WdDate([string]$s) {
    if (-not $s) { return $null }
    try { return [DateTimeOffset]::Parse($s, $script:Inv) } catch { return $null }
}

function Test-WdNight([DateTimeOffset]$t, $Cfg) {
    $h = $t.LocalDateTime.Hour
    return ($h -ge [int]$Cfg.night_start_hour -or $h -lt [int]$Cfg.night_end_hour)
}

# ---------------------------------------------------------------- logging
function Write-WdLog {
    param([string]$Level, [string]$Message)
    $dir = Get-WdHome
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force $dir | Out-Null }
    $file = Join-Path $dir 'watchdog.log'
    $max = 5242880
    try {
        if ((Test-Path $file) -and (Get-Item $file).Length -gt $max) {
            Move-Item $file "$file.1" -Force
        }
    } catch {}
    $line = '{0} {1,-5} {2}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Level.ToUpper(), $Message
    try { Add-Content -Path $file -Value $line -Encoding UTF8 } catch {}
    Write-Host $line
}

# ---------------------------------------------------------------- keep-awake
function Set-WdKeepAwake([bool]$On) {
    if (-not ('WdPower' -as [type])) {
        Add-Type -TypeDefinition 'public static class WdPower { [System.Runtime.InteropServices.DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f); }'
    }
    $ES_CONTINUOUS = [uint32]2147483648; $ES_SYSTEM_REQUIRED = [uint32]1
    if ($On) { [void][WdPower]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) }
    else { [void][WdPower]::SetThreadExecutionState($ES_CONTINUOUS) }
}

# ---------------------------------------------------------------- collection
# One job per alias, all in parallel. Each retries on SSH-level failure (exit 255 / timeout)
# until network_wait_seconds is exhausted.
function Invoke-WdCollect($Cfg, [string[]]$Aliases) {
    if (-not $Aliases) { $Aliases = @($Cfg.aliases) }
    $jobs = [ordered]@{}
    foreach ($a in $Aliases) {
        $argList = @($a, [string]$Cfg.remote_command, [int]$Cfg.network_wait_seconds, [int]$Cfg.retry_interval_seconds, [int]$Cfg.check_timeout_seconds)
        $jobs[$a] = Start-Job -ArgumentList $argList -ScriptBlock {
            param($alias, $cmd, $netWait, $retryEvery, $timeout)
            $sw = [Diagnostics.Stopwatch]::StartNew()
            $attempt = 0
            $res = $null
            do {
                $attempt++
                $psi = New-Object Diagnostics.ProcessStartInfo
                $psi.FileName = 'ssh.exe'
                $psi.Arguments = "-o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=10 -o ServerAliveCountMax=3 $alias `"$cmd`""
                $psi.UseShellExecute = $false; $psi.CreateNoWindow = $true
                $psi.RedirectStandardOutput = $true; $psi.RedirectStandardError = $true; $psi.RedirectStandardInput = $true
                $code = -1; $timedOut = $false; $stdout = ''; $stderr = ''
                try {
                    $p = [Diagnostics.Process]::Start($psi)
                    $p.StandardInput.Close()   # ssh must not wait on stdin
                    $so = $p.StandardOutput.ReadToEndAsync(); $se = $p.StandardError.ReadToEndAsync()
                    if (-not $p.WaitForExit($timeout * 1000)) { $timedOut = $true; try { $p.Kill() } catch {} }
                    else { $code = $p.ExitCode; $stdout = $so.Result; $stderr = $se.Result }
                } catch { $code = -2; $stderr = $_.Exception.Message }
                $res =[ordered]@{ alias = $alias; ok = $false; error = $null; json_text = $null; stderr = $stderr.Trim(); attempts = $attempt; exit = $code }
                if ($timedOut) { $res.error = "timeout after ${timeout}s" }
                elseif ($code -ne 0) { $res.error = "ssh/remote exit code $code" }
                else {
                    $line = ($stdout -split "`r?`n" | Where-Object { $_.Trim() } | Select-Object -Last 1)
                    try { $null = $line | ConvertFrom-Json -ErrorAction Stop; $res.ok = $true; $res.json_text = $line }
                    catch { $res.error = 'invalid JSON on stdout' }
                }
                $retry = (-not $res.ok) -and ($timedOut -or $code -eq 255 -or $code -eq -2) -and (($sw.Elapsed.TotalSeconds + $retryEvery + 5) -lt $netWait)
                if ($retry) { Start-Sleep -Seconds $retryEvery }
            } while ($retry)
            $res.seconds = [math]::Round($sw.Elapsed.TotalSeconds, 1)
            [pscustomobject]$res
        }
    }
    $hardLimit = (Get-Date).AddSeconds([int]$Cfg.network_wait_seconds + [int]$Cfg.check_timeout_seconds + 30)
    $results = [ordered]@{}
    foreach ($a in $Aliases) {
        $left = [math]::Max(1, ($hardLimit - (Get-Date)).TotalSeconds)
        $null = Wait-Job $jobs[$a] -Timeout $left
        if ($jobs[$a].State -eq 'Running') {
            Stop-Job $jobs[$a]
            $results[$a] = [pscustomobject]@{ alias = $a; ok = $false; error = 'job timeout'; json_text = $null; stderr = ''; attempts = 0; exit = -3; seconds = 0 }
        } else {
            $r = @(Receive-Job $jobs[$a] -ErrorAction SilentlyContinue) | Select-Object -Last 1
            if (-not $r) { $r = [pscustomobject]@{ alias = $a; ok = $false; error = 'job returned nothing'; json_text = $null; stderr = ''; attempts = 0; exit = -4; seconds = 0 } }
            $results[$a] = $r
        }
        Remove-Job $jobs[$a] -Force -ErrorAction SilentlyContinue
    }
    return $results
}

# ---------------------------------------------------------------- snapshots
function Get-WdCurrent($Latest) {
    # Compact "where is the run" tuple used for stuck detection.
    $stage = $Latest.current_stage
    if (-not $stage) { return $null }
    $p = $Latest.progress.$stage
    if (-not $p) { return $null }
    $count = $null; $period = $null
    if ($stage -eq 'sign') { $count = $p.keys_done; $period = $p.seconds_per_key }
    else { $count = $p.epochs_done; $period = $p.epoch_seconds }
    [ordered]@{ stage = $stage; count = $count; period = $period; live = $p.live; ckpt = $p.checkpoint_at }
}

function New-WdSnapshot {
    param([string]$Alias, $Result, [DateTimeOffset]$Now, $Cfg)
    $s = [ordered]@{
        t = $Now.ToString('o'); alias = $Alias; ok = [bool]$Result.ok; error = $Result.error
        night = (Test-WdNight $Now $Cfg); attempts = $Result.attempts
    }
    if ($Result.ok) {
        $j = $Result.json_text | ConvertFrom-Json
        $s.schema = $j.schema; $s.host = $j.host; $s.checked_at = $j.checked_at
        $s.active = [bool]$j.active; $s.rearmed_cron = [bool]$j.rearmed_cron
        if ($j.active) {
            $s.run_name = $j.run_name; $s.current_stage = $j.current_stage; $s.all_done = [bool]$j.all_done
            $s.remaining_gpu_seconds = $j.remaining_gpu_seconds
            $s.health = $j.health; $s.progress = $j.progress
            $s.cur = Get-WdCurrent $j
        }
    }
    return [pscustomobject]$s
}

function Read-WdHistory([string]$Alias, [int]$Last = 3000) {
    $file = Join-Path (Get-WdHome) "data\$Alias\history.jsonl"
    if (-not (Test-Path $file)) { return @() }
    $out = New-Object System.Collections.ArrayList
    foreach ($l in (Get-Content $file -Tail $Last -Encoding UTF8)) {
        if ($l.Trim()) { try { [void]$out.Add(($l | ConvertFrom-Json)) } catch {} }
    }
    return , $out.ToArray()
}

# ---------------------------------------------------------------- detection (pure)
function New-WdProblem($Sev, $Code, $Msg) { [pscustomobject]@{ severity = $Sev; code = $Code; msg = $Msg } }

function Test-WdShouldRun($S, $D) {
    if (-not ($S.ok -and $S.active -and $S.health)) { return $false }
    $h = $S.health
    if ($S.all_done) { return $false }
    $gw = $h.gpu_window
    if (-not $gw -or $gw.state -ne 'open') { return $false }
    if ($null -eq $gw.seconds -or [double]$gw.seconds -lt [double]$D.min_window_seconds) { return $false }
    if ($null -eq $h.gpu_mem_used_mb -or [double]$h.gpu_mem_used_mb -ge [double]$D.idle_gpu_mem_mb) { return $false }
    return (-not [bool]$h.pipeline_running)
}

function Test-WdCleanupTime($Marker, $D) {
    $m = ConvertTo-WdDate $Marker
    if (-not $m) { return $false }
    $l = $m.LocalDateTime
    if ($l.DayOfWeek -in 'Saturday', 'Sunday') { return $false }
    $parts = ([string]$D.cleanup_kill_time).Split(':')
    $target = $l.Date.AddHours([int]$parts[0]).AddMinutes([int]$parts[1])
    return ([math]::Abs(($l - $target).TotalMinutes) -le [double]$D.cleanup_kill_tolerance_minutes)
}

function Get-WdMarkers($S) {
    $r = @{}
    if ($S -and $S.progress) {
        foreach ($st in 'pretrain', 'sign', 'validate') {
            $p = $S.progress.$st
            if ($p -and $p.exception_marker_at) { $r[$st] = [string]$p.exception_marker_at }
        }
    }
    return $r
}

# $S: current snapshot; $History: earlier snapshots of the same alias (ascending, without $S).
function Get-WdProblems {
    param($S, $History, $Cfg)
    $D = $Cfg.detection
    $now = ConvertTo-WdDate $S.t
    $out = New-Object System.Collections.ArrayList
    $History = @($History)

    if (-not $S.ok) {
        [void]$out.Add((New-WdProblem 'warning' 'check_failed' "check failed: $($S.error)"))
        # trailing run of failed checks including this one; only night checks count towards the error
        $nightFails = 1
        if (-not $S.night) { $nightFails = 0 }
        for ($i = $History.Count - 1; $i -ge 0; $i--) {
            if ($History[$i].ok) { break }
            if ($History[$i].night) { $nightFails++ }
        }
        if ($nightFails -ge [int]$D.fail_error_count) {
            [void]$out.Add((New-WdProblem 'error' 'unreachable' "$nightFails consecutive failed night checks"))
        }
        return , $out.ToArray()
    }

    if ($S.schema -ne 1) { [void]$out.Add((New-WdProblem 'error' 'schema' "unexpected schema $($S.schema) (expected 1)")) }
    if ($S.rearmed_cron) { [void]$out.Add((New-WdProblem 'info' 'rearmed' 'cron re-armed (crontab had been wiped)')) }
    if (-not $S.active) {
        [void]$out.Add((New-WdProblem 'info' 'inactive' 'no active run'))
        return , $out.ToArray()
    }

    $h = $S.health
    if ($h.gave_up) { [void]$out.Add((New-WdProblem 'error' 'gave_up' "supervisor gave up after $($h.failures)/$($h.max_failures) failures")) }
    elseif ([int]$h.failures -gt 0) { [void]$out.Add((New-WdProblem 'warning' 'failures' "failures counter at $($h.failures)/$($h.max_failures)")) }

    # should be running but is not (2 consecutive checks)
    $prev = $null
    for ($i = $History.Count - 1; $i -ge 0; $i--) { if ($History[$i].ok) { $prev = $History[$i]; break } }
    if ((Test-WdShouldRun $S $D) -and $prev -and (Test-WdShouldRun $prev $D)) {
        [void]$out.Add((New-WdProblem 'error' 'not_running' 'GPU window open, GPU idle, pipeline not running (2 consecutive checks)'))
    }

    # stuck: running, nothing moved for a long time
    if ($h.pipeline_running -and $S.cur) {
        $since = $now
        for ($i = $History.Count - 1; $i -ge 0; $i--) {
            $e = $History[$i]
            if (-not ($e.ok -and $e.active -and $e.health -and $e.health.pipeline_running -and $e.cur)) { break }
            if ($e.cur.stage -ne $S.cur.stage -or $e.cur.count -ne $S.cur.count -or $e.cur.live -ne $S.cur.live -or $e.cur.ckpt -ne $S.cur.ckpt) { break }
            $since = ConvertTo-WdDate $e.t
        }
        $limit = [double]$D.stuck_min_seconds
        if ($S.cur.period) { $limit = [math]::Max($limit, [double]$D.stuck_factor * [double]$S.cur.period) }
        $idle = ($now - $since).TotalSeconds
        if ($idle -gt $limit) {
            [void]$out.Add((New-WdProblem 'error' 'stuck' ("pipeline running but no progress for {0:N1} h (limit {1:N1} h)" -f ($idle / 3600), ($limit / 3600))))
        }
    }

    # cron not firing
    if ($h.cron_installed -and $null -ne $h.heartbeat_age_seconds -and [double]$h.heartbeat_age_seconds -gt [double]$D.heartbeat_max_age) {
        [void]$out.Add((New-WdProblem 'warning' 'cron_silent' "cron installed but heartbeat is $([int]$h.heartbeat_age_seconds) s old"))
    }

    # crash vs expected cleanup kill
    if ($prev) {
        $pm = Get-WdMarkers $prev
        $cm = Get-WdMarkers $S
        foreach ($st in $cm.Keys) {
            if ($pm[$st] -ne $cm[$st]) {
                if (Test-WdCleanupTime $cm[$st] $D) { [void]$out.Add((New-WdProblem 'info' 'cleanup_kill' "expected cleanup kill ($st, exception marker $($cm[$st]))")) }
                else { [void]$out.Add((New-WdProblem 'warning' 'crash' "unexpected exception marker in $st at $($cm[$st])")) }
            }
        }
    }
    return , $out.ToArray()
}

# ---------------------------------------------------------------- ETA
# Usable GPU windows: Mon-Thu nights 18:00 -> 08:00, Friday 18:00 -> Monday 08:00. Every weekday 00:33 the school kills the
# run. With assume_rearm (the laptop re-arms at ~00:40) the run resumes and only loses a restart gap + about one epoch;
# without it the window effectively ends at the kill.
function Get-WdEta {
    param($Latest, [DateTimeOffset]$At, $Cfg)
    if (-not $Latest -or -not $Latest.active -or $Latest.all_done) { return $null }
    if ($null -eq $Latest.remaining_gpu_seconds) { return $null }
    $E = $Cfg.eta
    $total = [double]$Latest.remaining_gpu_seconds
    $remaining = $total
    $lost = [double]$E.lost_seconds_default
    $cur = Get-WdCurrent $Latest
    if ($cur -and $cur.period) { $lost = [double]$cur.period }
    $rearm = $true; if ($null -ne $E.assume_rearm) { $rearm = [bool]$E.assume_rearm }
    $gap = [double]$E.rearm_gap_seconds
    $now = $At.LocalDateTime
    $wdS = [TimeSpan]::Parse($E.weekday_window_start); $wdE = [TimeSpan]::Parse($E.weekday_window_end)
    $weS = [TimeSpan]::Parse($E.weekend_window_start); $weE = [TimeSpan]::Parse($E.weekend_window_end)
    $kt = [TimeSpan]::Parse($E.kill_time)
    $day = $now.Date.AddDays(-3)
    $limit = $now.AddDays([int]$E.max_days)
    $windows = 0
    while ($day -lt $limit) {
        $start = $null; $end = $null; $kill = $null
        switch ($day.DayOfWeek) {
            'Friday' { $start = $day + $weS; $end = $day.AddDays(3) + $weE; $kill = $day.AddDays(3) + $kt }   # Fri evening -> Mon morning
            'Saturday' {}
            'Sunday' {}
            default { $start = $day + $wdS; $end = $day.AddDays(1) + $wdE; $kill = $day.AddDays(1) + $kt }
        }
        if ($start -and $end -gt $now) {
            $s = if ($start -gt $now) { $start } else { $now }
            $finish = $null
            if ($rearm) {
                if ($s.AddSeconds($remaining) -le $kill) { $finish = $s.AddSeconds($remaining) }
                else {
                    $loss = 0; if ($kill -gt $s) { $loss = $gap + $lost }
                    $avail = ($end - $s).TotalSeconds
                    if ($remaining + $loss -le $avail) { $finish = $s.AddSeconds($remaining + $loss) }
                    else { $remaining = [math]::Max(0, $remaining - ($avail - $loss)) }
                }
            } else {
                if ($kill -gt $s) {
                    $avail = ($kill - $s).TotalSeconds
                    if ($remaining -le $avail) { $finish = $s.AddSeconds($remaining) } else { $remaining = $remaining - $avail + $lost }
                }
            }
            $windows++
            if ($finish) {
                return [pscustomobject]@{ finish = (New-Object DateTimeOffset($finish)).ToString('o'); windows = $windows; remaining_seconds = $total; lost_per_kill = $lost }
            }
        }
        $day = $day.AddDays(1)
    }
    return [pscustomobject]@{ finish = $null; windows = $windows; remaining_seconds = $total; lost_per_kill = $lost }
}

# ---------------------------------------------------------------- timeline (remote events + laptop-observed)
function Get-WdTimeline {
    param($Latest, $History, [DateTimeOffset]$Now, $Cfg)
    $cut = $Now.AddDays(-[int]$Cfg.timeline_days)
    $ev = New-Object System.Collections.ArrayList
    $remoteRearms = @()
    if ($Latest -and $Latest.events) {
        foreach ($e in $Latest.events) {
            $d = ConvertTo-WdDate $e.at
            if ($d -and $d -ge $cut) { [void]$ev.Add([pscustomobject]@{ at = $e.at; kind = [string]$e.kind; msg = [string]$e.msg; source = 'remote'; severity = 'info' }) }
            if ($e.kind -eq 'rearm' -and $d) { $remoteRearms += $d }
        }
    }
    $History = @($History)
    $prev = $null
    foreach ($s in $History) {
        $d = ConvertTo-WdDate $s.t
        if (-not $d -or $d -lt $cut) { $prev = $s; continue }
        if (-not $s.ok) {
            [void]$ev.Add([pscustomobject]@{ at = $s.t; kind = 'check_failed'; msg = [string]$s.error; source = 'laptop'; severity = 'warning' })
        } else {
            if ($s.rearmed_cron) {
                $dup = $false
                foreach ($r in $remoteRearms) { if ([math]::Abs(($r - $d).TotalSeconds) -lt 180) { $dup = $true } }
                if (-not $dup) { [void]$ev.Add([pscustomobject]@{ at = $s.t; kind = 'rearm'; msg = 'cron re-armed (crontab had been wiped)'; source = 'laptop'; severity = 'info' }) }
            }
            if ($prev -and $prev.ok -and $prev.active -and $s.active -and $prev.health -and $s.health -and $prev.health.pipeline_running -and -not $s.health.pipeline_running -and -not $s.all_done -and $s.health.gpu_window.state -eq 'open') {
                [void]$ev.Add([pscustomobject]@{ at = $s.t; kind = 'killed'; msg = 'pipeline stopped between two checks while the window is open (inferred)'; source = 'laptop'; severity = 'info' })
            }
        }
        $prevCodes = @(); if ($prev -and $prev.problems) { $prevCodes = @($prev.problems | ForEach-Object { $_.code }) }
        foreach ($p in @($s.problems)) {
            if ($p -and $p.severity -ne 'info' -and ($prevCodes -notcontains $p.code)) {
                [void]$ev.Add([pscustomobject]@{ at = $s.t; kind = 'problem'; msg = [string]$p.msg; source = 'laptop'; severity = [string]$p.severity })
            }
        }
        $prev = $s
    }
    return , @($ev | Sort-Object { ConvertTo-WdDate $_.at } -Descending)
}
