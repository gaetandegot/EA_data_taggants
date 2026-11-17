# One watchdog run: query the machines, store results, detect problems, regenerate the dashboard.
# Called by the scheduled tasks and by install.ps1 -RunNow.
param([switch]$NoDashboard)

$ErrorActionPreference = 'Continue'
. (Join-Path $PSScriptRoot 'Watchdog.Lib.ps1')
. (Join-Path $PSScriptRoot 'Dashboard.ps1')

$wdHome = Get-WdHome
$cfg = Read-WdConfig
$started = [DateTimeOffset]::Now
$mutex = New-Object System.Threading.Mutex($false, 'Global\TaggantsWatchdogRun')
if (-not $mutex.WaitOne(0)) { Write-WdLog 'warn' 'another run is in progress, exiting'; exit 0 }

try {
    Set-WdKeepAwake $true
    Write-WdLog 'info' "run start (aliases: $($cfg.aliases -join ', '))"
    $results = Invoke-WdCollect $cfg
    $now = [DateTimeOffset]::Now

    foreach ($alias in $cfg.aliases) {
        $res = $results[$alias]
        $dir = Join-Path $wdHome "data\$alias"
        New-Item -ItemType Directory -Force $dir | Out-Null
        $history = Read-WdHistory $alias
        $snap = New-WdSnapshot -Alias $alias -Result $res -Now $now -Cfg $cfg
        $problems = Get-WdProblems -S $snap -History $history -Cfg $cfg
        $snap | Add-Member -NotePropertyName problems -NotePropertyValue @($problems) -Force

        if ($res.ok) { [IO.File]::WriteAllText((Join-Path $dir 'latest.json'), $res.json_text, (New-Object Text.UTF8Encoding($false))) }
        Add-Content -Path (Join-Path $dir 'history.jsonl') -Value ($snap | ConvertTo-Json -Depth 20 -Compress) -Encoding UTF8

        if (-not $res.ok) {
            Write-WdLog 'warn' "$alias FAILED after $($res.attempts) attempt(s) in $($res.seconds)s: $($res.error)"
            if ($res.stderr) { Write-WdLog 'warn' "$alias stderr: $(($res.stderr -split "`n")[0].Trim())" }
        } elseif (-not $snap.active) {
            Write-WdLog 'info' "$alias ($($snap.host)) ok: no active run"
        } else {
            $c = $snap.cur
            $h = $snap.health
            Write-WdLog 'info' ("{0} ({1}) ok: stage={2} {3} running={4} cron={5} fails={6}/{7} window={8} remaining={9}h rearmed={10}" -f $alias, $snap.host, $snap.current_stage, $c.count, $h.pipeline_running, $h.cron_installed, $h.failures, $h.max_failures, $h.gpu_window.state, $(if ($null -ne $snap.remaining_gpu_seconds) { '{0:N1}' -f ([double]$snap.remaining_gpu_seconds / 3600) } else { 'n/a' }), $snap.rearmed_cron)
        }
        foreach ($p in $problems) {
            $lvl = switch ($p.severity) { 'error' { 'error' } 'warning' { 'warn' } default { 'info' } }
            Write-WdLog $lvl "$alias [$($p.severity)] $($p.code): $($p.msg)"
        }

        # trim history
        $hf = Join-Path $dir 'history.jsonl'
        if ((Get-Item $hf).Length -gt 30MB) {
            $keep = Get-Content $hf -Tail ([int]$cfg.history_max_lines) -Encoding UTF8
            Set-Content $hf $keep -Encoding UTF8
        }
    }

    $okCount = @($results.Values | Where-Object { $_.ok }).Count
    $runInfo = [ordered]@{ started = $started.ToString('o'); finished = [DateTimeOffset]::Now.ToString('o'); ok = $okCount; total = @($cfg.aliases).Count }
    New-Item -ItemType Directory -Force (Join-Path $wdHome 'data') | Out-Null
    [IO.File]::WriteAllText((Join-Path $wdHome 'data\last_run.json'), ($runInfo | ConvertTo-Json -Compress), (New-Object Text.UTF8Encoding($false)))

    if (-not $NoDashboard) {
        try { Write-WdDashboard -Cfg $cfg; Write-WdLog 'info' "dashboard written: $(Join-Path $wdHome 'dashboard.html')" }
        catch { Write-WdLog 'error' "dashboard failed: $($_.Exception.Message)" }
    }
    Write-WdLog 'info' "run end: $okCount/$(@($cfg.aliases).Count) checks ok"
} catch {
    Write-WdLog 'error' "run crashed: $($_.Exception.Message) at $($_.InvocationInfo.ScriptLineNumber)"
} finally {
    Set-WdKeepAwake $false
    $mutex.ReleaseMutex()
}
