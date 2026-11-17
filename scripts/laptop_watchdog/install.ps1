# Installs / updates / removes the laptop watchdog (scheduled tasks, config, desktop shortcut).
#   .\install.ps1                 install or update (idempotent)
#   .\install.ps1 -RunNow         run once in the foreground
#   .\install.ps1 -ApplyPower     apply the AC power settings (wake timers, lid, no hibernate on AC)
#   .\install.ps1 -SleepTest      create a one-shot wake task 5 min ahead
#   .\install.ps1 -SleepTestResult  check whether that task ran while the laptop was asleep
#   .\install.ps1 -Uninstall [-Purge]
param(
    [switch]$Uninstall, [switch]$Purge, [switch]$RunNow, [switch]$SleepTest, [switch]$SleepTestResult,
    [switch]$ApplyPower, [string[]]$Aliases
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Watchdog.Lib.ps1')
$wdHome = Get-WdHome
$app = Join-Path $wdHome 'app'
$nightName = 'TaggantsWatchdog-Night'; $dayName = 'TaggantsWatchdog-Day'; $testName = 'TaggantsWatchdog-SleepTest'; $weekendName = 'TaggantsWatchdog-Weekend'
$shortcut = Join-Path ([Environment]::GetFolderPath('Desktop')) 'Taggants Watchdog.lnk'

function Remove-WdTask($name) { if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) { Unregister-ScheduledTask -TaskName $name -Confirm:$false } }

function Assert-WdTask($name) {
    $t = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
    if (-not $t) { throw "Task $name was NOT registered (see the error above)" }
    Write-Host "  ${name}: $(@($t.Triggers).Count) triggers"
}

function Sync-WdApp {
    New-Item -ItemType Directory -Force $app | Out-Null
    foreach ($f in 'Watchdog.Lib.ps1', 'Run-Watchdog.ps1', 'Dashboard.ps1', 'dashboard.template.html', 'run-hidden.vbs') {
        Copy-Item (Join-Path $PSScriptRoot $f) (Join-Path $app $f) -Force
    }
}

function Get-WdAction {
    New-ScheduledTaskAction -Execute 'wscript.exe' -Argument ('//B //Nologo "{0}"' -f (Join-Path $app 'run-hidden.vbs'))
}

function New-RepeatTrigger([string]$At, [int]$Hours, [string[]]$Days) {
    if ($Days) { $t = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $Days -At $At } else { $t = New-ScheduledTaskTrigger -Daily -At $At }
    if ($Hours -gt 0) {
        $rep = (New-ScheduledTaskTrigger -Once -At $At -RepetitionInterval (New-TimeSpan -Minutes 30) -RepetitionDuration (New-TimeSpan -Hours $Hours)).Repetition
        $t.Repetition = $rep
    }
    return $t
}

if ($Uninstall) {
    foreach ($n in $nightName, $dayName, $weekendName, $testName) { Remove-WdTask $n }
    if (Test-Path $shortcut) { Remove-Item $shortcut -Force }
    if ($Purge -and (Test-Path $wdHome)) { Remove-Item $wdHome -Recurse -Force; Write-Host "Purged $wdHome" }
    Write-Host 'Tasks removed.'
    & powercfg /waketimers
    exit 0
}

if ($ApplyPower) {
    Write-Host 'Applying AC power settings to the current scheme...'
    & powercfg /setacvalueindex SCHEME_CURRENT SUB_SLEEP RTCWAKE 1
    & powercfg /setdcvalueindex SCHEME_CURRENT SUB_SLEEP RTCWAKE 0
    & powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 1
    & powercfg /change hibernate-timeout-ac 0
    & powercfg /setactive SCHEME_CURRENT
    Write-Host 'Done. Battery behaviour is unchanged.'
    exit 0
}

if ($SleepTest) {
    Sync-WdApp
    $at = (Get-Date).AddMinutes(5)
    Remove-WdTask $testName
    $set = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 10) -DontStopIfGoingOnBatteries
    $set.DisallowStartIfOnBatteries = $true
    $princ = New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $testName -Action (Get-WdAction) -Trigger (New-ScheduledTaskTrigger -Once -At $at) -Settings $set -Principal $princ | Out-Null
    New-Item -ItemType Directory -Force $wdHome | Out-Null
    @{ scheduled = $at.ToString('o'); created = (Get-Date).ToString('o') } | ConvertTo-Json | Set-Content (Join-Path $wdHome 'sleeptest.json') -Encoding UTF8
    Write-Host "Sleep test armed for $($at.ToString('HH:mm:ss'))."
    Write-Host 'Now: plug in the charger, close the lid (laptop goes to sleep), wait ~10 minutes, open the lid, then run:'
    Write-Host '  .\install.ps1 -SleepTestResult'
    exit 0
}

if ($SleepTestResult) {
    $f = Join-Path $wdHome 'sleeptest.json'
    if (-not (Test-Path $f)) { Write-Host 'No sleep test recorded.'; exit 1 }
    $t = Get-Content $f -Raw | ConvertFrom-Json
    $sched = [DateTimeOffset]::Parse($t.scheduled)
    $log = Join-Path $wdHome 'watchdog.log'
    $hit = @()
    if (Test-Path $log) { $hit = Get-Content $log -Encoding UTF8 | Where-Object { $_ -match '^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) INFO\s+run start' } | ForEach-Object { [datetime]::ParseExact(($_ -replace '^(\S+ \S+).*', '$1'), 'yyyy-MM-dd HH:mm:ss', $null) } | Where-Object { ($_ - $sched.LocalDateTime).TotalMinutes -ge -1 -and ($_ - $sched.LocalDateTime).TotalMinutes -le 15 } }
    Write-Host "Scheduled: $($sched.LocalDateTime)"
    if ($hit) { Write-Host "Run logged at: $($hit[0])  (delay $([math]::Round(($hit[0] - $sched.LocalDateTime).TotalSeconds)) s)" } else { Write-Host 'No run logged in the 15 min after the scheduled time.' }
    Write-Host '--- powercfg /lastwake'; & powercfg /lastwake
    Write-Host '--- Power-Troubleshooter events (wake source) in the last 30 min'
    try { Get-WinEvent -FilterHashtable @{ LogName = 'System'; ProviderName = 'Microsoft-Windows-Power-Troubleshooter'; StartTime = (Get-Date).AddMinutes(-30) } -ErrorAction Stop | Select-Object TimeCreated, Message | Format-List } catch { Write-Host '(none)' }
    Write-Host 'Compare the run time with the Sleep/Wake events above: a run logged while the system was asleep proves the timed wake worked.'
    exit 0
}

if ($RunNow) {
    Sync-WdApp
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $app 'Run-Watchdog.ps1')
    exit $LASTEXITCODE
}

# ---------------- default: install / update
New-Item -ItemType Directory -Force (Join-Path $wdHome 'data') | Out-Null
Sync-WdApp

$cfgFile = Join-Path $wdHome 'config.json'
if (Test-Path $cfgFile) {
    $cfg = Get-Content $cfgFile -Raw | ConvertFrom-Json
    # add keys introduced by newer versions without touching the user's values
    $def = Get-WdDefaultConfig | ConvertTo-Json -Depth 10 | ConvertFrom-Json
    foreach ($p in $def.PSObject.Properties) {
        if ($null -eq $cfg.($p.Name)) { $cfg | Add-Member -NotePropertyName $p.Name -NotePropertyValue $p.Value }
        elseif ($p.Value -is [pscustomobject]) {
            foreach ($q in $p.Value.PSObject.Properties) { if ($null -eq $cfg.($p.Name).($q.Name)) { $cfg.($p.Name) | Add-Member -NotePropertyName $q.Name -NotePropertyValue $q.Value } }
        }
    }
} else {
    $cfg = Get-WdDefaultConfig | ConvertTo-Json -Depth 10 | ConvertFrom-Json
}
if ($Aliases) { $cfg.aliases = @($Aliases) }
$cfg | ConvertTo-Json -Depth 10 | Set-Content $cfgFile -Encoding UTF8
Write-Host "Config: $cfgFile (aliases: $($cfg.aliases -join ', '))"

$user = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$princ = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
$weekdays = 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'

# Night task: wakes the laptop, only on AC power
$nightTriggers = @(
    (New-RepeatTrigger '00:40' 0 $weekdays),
    (New-RepeatTrigger '01:10' 0 $weekdays),
    (New-RepeatTrigger '18:05' 0 $null)       # window opening, every day
)
# Every 30 min from 18:00 to 07:30, every night. One explicit trigger per slot: Windows only arms a wake timer for a
# trigger's start time, not for the occurrences of a repetition (verified with powercfg /waketimers).
foreach ($m in 0..27) {
    $slot = [datetime]'2000-01-01 18:00'
    $at = $slot.AddMinutes(30 * $m).ToString('HH:mm')
    $nightTriggers += (New-RepeatTrigger $at 0 $null)
}
$nightSet = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 10) -DontStopIfGoingOnBatteries
$nightSet.DisallowStartIfOnBatteries = $true
Remove-WdTask $nightName
Register-ScheduledTask -TaskName $nightName -Action (Get-WdAction) -Trigger $nightTriggers -Settings $nightSet -Principal $princ -Description 'Taggants watchdog: night and weekend checks (wakes the laptop on AC power)' | Out-Null
Assert-WdTask $nightName

# Weekend task: the GPU window is open from Friday 18:00 to Monday 08:00, so wake every 30 min on Saturday and Sunday from
# 08:00 to 17:30 as well (the night task already covers 18:00-07:30). Separate task: Task Scheduler allows 48 triggers max.
$weekendTriggers = @()
foreach ($m in 0..19) {
    $at = ([datetime]'2000-01-01 08:00').AddMinutes(30 * $m).ToString('HH:mm')
    $weekendTriggers += (New-RepeatTrigger $at 0 @('Saturday', 'Sunday'))
}
Remove-WdTask $weekendName
Register-ScheduledTask -TaskName $weekendName -Action (Get-WdAction) -Trigger $weekendTriggers -Settings $nightSet -Principal $princ -Description 'Taggants watchdog: weekend daytime checks (wakes the laptop on AC power)' | Out-Null
Assert-WdTask $weekendName

# Day task: dashboard refresh only, no wake, battery allowed
$daySet = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 10) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Remove-WdTask $dayName
$dayTriggers = @()
# Explicit triggers (08:00 to 17:30) rather than one repeating trigger: repetitions proved unreliable (see the night task).
foreach ($m in 0..19) {
    $at = ([datetime]'2000-01-01 08:00').AddMinutes(30 * $m).ToString('HH:mm')
    $dayTriggers += (New-RepeatTrigger $at 0 $null)
}
Register-ScheduledTask -TaskName $dayName -Action (Get-WdAction) -Trigger $dayTriggers -Settings $daySet -Principal $princ -Description 'Taggants watchdog: daytime refresh (no wake)' | Out-Null
Assert-WdTask $dayName
Write-Host "Tasks registered: $nightName and $weekendName (wake, AC only), $dayName (no wake)"

# Desktop shortcut to the dashboard
try {
    $sh = (New-Object -ComObject WScript.Shell).CreateShortcut($shortcut)
    $sh.TargetPath = Join-Path $wdHome 'dashboard.html'
    $sh.Description = 'Taggants watchdog dashboard'
    $sh.Save()
    Write-Host "Shortcut: $shortcut"
} catch { Write-Host "Shortcut failed: $($_.Exception.Message)" }
