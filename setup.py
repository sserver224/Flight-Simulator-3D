import os
import shutil
import subprocess
import win32com.client
import pythoncom
from tkinter import *
from tkinter import messagebox
from threading import Thread
from tkinter.ttk import *
def get_resource_path(relative_path):
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)
def close():
    if not root.busy:
        root.destroy()
def copy_files():
    try:
        a.config(text='Copying program files')
        src = get_resource_path('program_files\\')
        dest = os.getenv('LOCALAPPDATA')+"\\flight-sim\\"
        shutil.copytree(src, dest) 
        a.config(text='Making Start Menu shortcuts')
        if not os.path.exists(os.getenv('APPDATA')+"\\Microsoft\\Windows\\Start Menu\\Programs\\Flight_Simulator_v0.1_by_sserver"):
            os.mkdir(os.getenv('APPDATA')+"\\Microsoft\\Windows\\Start Menu\\Programs\\Flight_Simulator_v0.1_by_sserver")
        pythoncom.CoInitialize() # remove the '#' at the beginning of the line if running in a thread.
        desktop = os.getenv('APPDATA')+"\\Microsoft\\Windows\\Start Menu\\Programs\\Flight_Simulator_v0.1_by_sserver" # path to where you want to put the .lnk
        path = os.path.join(desktop, 'Flight Simulator v0.1 by sserver (Normal Controls).lnk')
        target = os.getenv('LOCALAPPDATA')+"\\flight-sim\\run.vbs"
        shell = win32com.client.Dispatch("WScript.Shell")
        wdir=os.getenv('LOCALAPPDATA')+"\\flight-sim"
        shortcut = shell.CreateShortCut(path)
        shortcut.Targetpath = target
        shortcut.IconLocation=get_resource_path('plane.ico')
        shortcut.WorkingDirectory=wdir
        shortcut.WindowStyle = 1 # 7 - Minimized, 3 - Maximized, 1 - Normal
        shortcut.save()
        path = os.path.join(desktop, 'Flight Simulator v0.1 by sserver (Racing Wheel Controls).lnk')
        target = os.getenv('LOCALAPPDATA')+"\\flight-sim\\runwheel.vbs"
        shell = win32com.client.Dispatch("WScript.Shell")
        shortcut = shell.CreateShortCut(path)
        shortcut.IconLocation=get_resource_path('plane.ico')
        shortcut.Targetpath = target
        shortcut.WorkingDirectory=wdir
        shortcut.WindowStyle = 1 # 7 - Minimized, 3 - Maximized, 1 - Normal
        shortcut.save()
        a.config(text='Done')
        messagebox.showinfo('Install Complete', 'Flight Simulator v0.1 is now installed.')
        root.destroy()
    except OSError as e:
        a.config(text=str(e))
        messagebox.showerror('OSError', str(e))
        os._exit(1)
    except Exception as e:
        a.config(text='Unknown exception')
        messagebox.showerror('Error', str(e))
        os._exit(1)
def remove_files():
    errors=0
    a.config(text='Removing Start Menu shortcuts')
    try:
        shutil.rmtree(os.getenv('APPDATA')+"\\Microsoft\\Windows\\Start Menu\\Programs\\Flight_Simulator_v0.1_by_sserver")
    except:
        errors+=1
    a.config(text='Removing program files')
    try:
        shutil.rmtree(os.getenv('LOCALAPPDATA')+"\\flight-sim\\")
    except:
        errors+=1
    a.config(text='Done')
    if errors==2:
        messagebox.showerror("Could Not Uninstall, 'Flight Simulator v0.1 could not be uninstall.\nCheck if other programs are locking the files.")
    elif errors==1:
        messagebox.showinfo('Uninstall Complete', 'Flight Simulator v0.1 is now uninstalled.\nSome elements could not be removed. These can be removed manually.')
    else:
        messagebox.showinfo('Uninstall Complete', 'Flight Simulator v0.1 is now uninstalled.')
    root.destroy()
def install():
    root.busy=True
    b.config(state=DISABLED)
    w=Thread(target=copy_files, daemon=True)
    w.start()
def uninstall():
    root.busy=True
    b.config(state=DISABLED)
    w=Thread(target=remove_files, daemon=True)
    w.start()
root=Tk()
root.title('Install Flight Simulator v0.1 by sserver')
root.protocol('WM_DELETE_WINDOW', close)
root.resizable(False, False)
root.busy=False
root.iconbitmap(get_resource_path('plane.ico'))
a=Label(root, text='Installing to AppData directory. For best results, wait at least 10 seconds before installing.')
a.pack()
b=Button(root, text='Install', command=install)
b.pack()
if os.path.exists(os.getenv('LOCALAPPDATA')+"\\flight-sim\\") or os.path.exists(os.getenv('APPDATA')+"\\Microsoft\\Windows\\Start Menu\\Programs\\Flight_Simulator_v0.1_by_sserver"):
    b.config(text='Uninstall', command=uninstall)
    a.config(text='Flight Simulator v0.1 by sserver is already installed.')
from PIL import Image, ImageTk
img = Image.open(get_resource_path('screenshot.gif'))
scn = ImageTk.PhotoImage(img)
Label(root, image=scn).pack()
root.mainloop()
