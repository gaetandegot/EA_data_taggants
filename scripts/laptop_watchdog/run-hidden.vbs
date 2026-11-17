' Launches Run-Watchdog.ps1 with no visible window at all (a plain powershell.exe -WindowStyle Hidden still flashes a console).
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
dir = fso.GetParentFolderName(WScript.ScriptFullName)
cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & dir & "\Run-Watchdog.ps1"""
WScript.Quit sh.Run(cmd, 0, True)
