' Launches PhiWatch.ps1 (one check + dashboard) with no visible window (powershell -WindowStyle Hidden still flashes).
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
dir = fso.GetParentFolderName(WScript.ScriptFullName)
cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & dir & "\PhiWatch.ps1"""
WScript.Quit sh.Run(cmd, 0, True)
