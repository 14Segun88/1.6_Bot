import sys
import subprocess
import json

if not sys.platform.startswith("win"):
    print("[check_windows.py] Windows-only diagnostics utility (skipped on Linux/POSIX).")
    sys.exit(0)

import ctypes
from ctypes import wintypes

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
psapi = ctypes.windll.psapi

print("=== CHECKING PROCESSES & WINDOWS ===")
# List top level windows
def get_windows():
    wins = []
    def cb(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            buf = ctypes.create_string_buffer(256)
            user32.GetWindowTextA(hwnd, buf, 256)
            title = buf.value.decode("cp1251", errors="ignore")
            
            cls_buf = ctypes.create_string_buffer(256)
            user32.GetClassNameA(hwnd, cls_buf, 256)
            cls_name = cls_buf.value.decode("latin1", errors="ignore")
            
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            
            pname = ""
            h_proc = kernel32.OpenProcess(0x0410, False, pid.value)
            if h_proc:
                nb = ctypes.create_string_buffer(256)
                psapi.GetModuleBaseNameA(h_proc, None, nb, 256)
                kernel32.CloseHandle(h_proc)
                pname = nb.value.decode("latin1", errors="ignore")
            
            if title or pname in ["hl.exe", "cstrike.exe", "python.exe", "cmd.exe"]:
                wins.append((hwnd, pid.value, pname, cls_name, title))
        return True
    
    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows(WNDENUMPROC(cb), 0)
    return wins

cur_fg = user32.GetForegroundWindow()
print(f"Foreground HWND: {cur_fg}")

all_wins = get_windows()
for hwnd, pid, pname, cls, title in all_wins:
    is_fg = (hwnd == cur_fg)
    if any(k in pname.lower() or k in title.lower() or k in cls.lower() for k in ["hl", "cs", "strike", "python", "cmd", "gold"]):
        fg_str = " [ACTIVE FOREGROUND]" if is_fg else ""
        print(f"PID={pid:5d} | Proc={pname:12s} | HWND={hwnd} | Class={cls:16s} | Title='{title}'{fg_str}")
