import os
import subprocess
import win32com.client
import pythoncom
import win32com.shell.shell
import win32event
import sys
def get_resource_path(relative_path):
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.getcwd()
    return base_path+'\\'+relative_path
while not os.path.exists(get_resource_path("flight-sim.lnk")):
    desktop = get_resource_path('.') # path to where you want to put the .lnk
    path = os.path.join(desktop, 'flight-sim.lnk')
    target = get_resource_path(r"\program_files\runwheel.vbs")
    shell = win32com.client.Dispatch("WScript.Shell")
    wdir=get_resource_path(r'\program_files')
    shortcut = shell.CreateShortCut(path)
    shortcut.Targetpath = target
    shortcut.WorkingDirectory=wdir
    shortcut.WindowStyle = 1 # 7 - Minimized, 3 - Maximized, 1 - Normal
    shortcut.save()
se_ret = win32com.shell.shell.ShellExecuteEx(fMask=0x140, lpFile=get_resource_path("flight-sim.lnk"), nShow=1)
win32event.WaitForSingleObject(se_ret['hProcess'], -1)
