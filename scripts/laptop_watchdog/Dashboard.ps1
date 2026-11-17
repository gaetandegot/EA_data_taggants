# Builds dashboard.html: one self-contained file, data embedded inline, rendered client-side.
function Get-WdTaskInfo($Cfg) {
    $r = @()
    foreach ($n in @($Cfg.schedule.night_task, 'TaggantsWatchdog-Weekend', $Cfg.schedule.day_task)) {
        try {
            $t = Get-ScheduledTask -TaskName $n -ErrorAction Stop
            $i = Get-ScheduledTaskInfo -TaskName $n -ErrorAction Stop
            $next = $null; if ($i.NextRunTime -and $i.NextRunTime.Year -gt 2000) { $next = $i.NextRunTime.ToString('o') }
            $last = $null; if ($i.LastRunTime -and $i.LastRunTime.Year -gt 2000) { $last = $i.LastRunTime.ToString('o') }
            $r += [pscustomobject]@{ name = $n; state = [string]$t.State; next = $next; last = $last; last_result = $i.LastTaskResult; wake = [bool]$t.Settings.WakeToRun }
        } catch {
            $r += [pscustomobject]@{ name = $n; state = 'not installed'; next = $null; last = $null; last_result = $null; wake = $false }
        }
    }
    return , $r
}

function Write-WdDashboard {
    param($Cfg)
    $wdHome = Get-WdHome
    $now = [DateTimeOffset]::Now
    $machines = New-Object System.Collections.ArrayList
    $failures = New-Object System.Collections.ArrayList

    foreach ($alias in $Cfg.aliases) {
        $dir = Join-Path $wdHome "data\$alias"
        $latestFile = Join-Path $dir 'latest.json'
        $latestText = 'null'; $latest = $null
        if (Test-Path $latestFile) {
            $latestText = [IO.File]::ReadAllText($latestFile, [Text.Encoding]::UTF8).Trim()
            try { $latest = $latestText | ConvertFrom-Json } catch { $latestText = 'null' }
        }
        $history = Read-WdHistory $alias 3000
        $last = $null; if ($history.Count -gt 0) { $last = $history[$history.Count - 1] }
        $lastProblems = @(); if ($last -and $last.problems) { $lastProblems = @($last.problems) }
        $eta = $null; if ($latest) { $eta = Get-WdEta -Latest $latest -At $now -Cfg $Cfg }
        $timeline = Get-WdTimeline -Latest $latest -History $history -Now $now -Cfg $Cfg
        foreach ($h in ($history | Where-Object { -not $_.ok } | Select-Object -Last 5)) {
            [void]$failures.Add([pscustomobject]@{ alias = $alias; t = $h.t; error = $h.error })
        }
        $status = [pscustomobject]@{
            last_check = $(if ($last) { $last.t } else { $null })
            last_ok    = $(if ($last) { [bool]$last.ok } else { $null })
            error      = $(if ($last) { $last.error } else { $null })
            checks     = $history.Count
        }
        $json = '{"alias":' + (ConvertTo-Json -InputObject ([string]$alias)) +
        ',"status":' + (ConvertTo-Json -InputObject $status -Compress) +
        ',"problems":' + (ConvertTo-Json -InputObject @($lastProblems) -Depth 5 -Compress) +
        ',"eta":' + $(if ($eta) { ConvertTo-Json -InputObject $eta -Compress } else { 'null' }) +
        ',"timeline":' + (ConvertTo-Json -InputObject @($timeline) -Depth 5 -Compress) +
        ',"latest":' + $latestText + '}'
        [void]$machines.Add($json)
    }

    $lastRun = $null
    $lrFile = Join-Path $wdHome 'data\last_run.json'
    if (Test-Path $lrFile) { $lastRun = Get-Content $lrFile -Raw -Encoding UTF8 | ConvertFrom-Json }
    $watch = [pscustomobject]@{
        generated  = $now.ToString('o')
        last_run   = $lastRun
        tasks      = (Get-WdTaskInfo $Cfg)
        failures   = @($failures | Sort-Object t -Descending | Select-Object -First 10)
        log_path   = (Join-Path $wdHome 'watchdog.log')
        html_path  = (Join-Path $wdHome 'dashboard.html')
        refresh    = [int]$Cfg.refresh_seconds
    }
    $data = '{"watchdog":' + (ConvertTo-Json -InputObject $watch -Depth 6 -Compress) + ',"machines":[' + ($machines -join ',') + ']}'
    $data = $data.Replace('</', '<\/').Replace([string][char]0x2028, ' ').Replace([string][char]0x2029, ' ')

    $tpl = [IO.File]::ReadAllText((Join-Path $PSScriptRoot 'dashboard.template.html'), [Text.Encoding]::UTF8)
    $html = $tpl.Replace('__REFRESH__', [string][int]$Cfg.refresh_seconds).Replace('/*__DATA__*/null', $data)
    $out = Join-Path $wdHome 'dashboard.html'
    $tmp = "$out.tmp"
    [IO.File]::WriteAllText($tmp, $html, (New-Object Text.UTF8Encoding($false)))
    Move-Item $tmp $out -Force
}
