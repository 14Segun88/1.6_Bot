import ctypes, struct, time, math
from ctypes import wintypes

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

pid = wintypes.DWORD()
h = user32.FindWindowA(b"SDL_app", None) or user32.FindWindowA(b"Valve001", None)
user32.GetWindowThreadProcessId(h, ctypes.byref(pid))
hProc = kernel32.OpenProcess(0x1010, False, pid.value)

class MODULEENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD), ("th32ModuleID", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD), ("GlblcntUsage", wintypes.DWORD),
        ("ProccntUsage", wintypes.DWORD), ("modBaseAddr", ctypes.c_void_p),
        ("modBaseSize", wintypes.DWORD), ("hModule", wintypes.HMODULE),
        ("szModule", ctypes.c_wchar * 256), ("szExePath", ctypes.c_wchar * 260)
    ]

snap = kernel32.CreateToolhelp32Snapshot(0x08 | 0x10, pid.value)
client_base = None
me32 = MODULEENTRY32W()
me32.dwSize = ctypes.sizeof(MODULEENTRY32W)
if kernel32.Module32FirstW(snap, ctypes.byref(me32)):
    while True:
        if me32.szModule.lower() == "client.dll":
            client_base = int(me32.modBaseAddr)
            break
        if not kernel32.Module32NextW(snap, ctypes.byref(me32)): break
kernel32.CloseHandle(snap)

print(f"Tracking client.dll + 0x199E3C for 10 ticks...")
for _ in range(10):
    buf = (ctypes.c_char * 24)()
    kernel32.ReadProcessMemory(hProc, ctypes.c_void_p(client_base + 0x199E3C), buf, 24, None)
    x, y, z, fx, fy, fz = struct.unpack_from("<6f", buf, 0)
    yaw = math.degrees(math.atan2(fy, fx))
    pitch = -math.degrees(math.asin(max(-1.0, min(1.0, fz))))
    print(f"  Pos: ({x:7.1f}, {y:7.1f}, {z:7.1f}) | Pitch: {pitch:6.1f} deg, Yaw: {yaw:6.1f} deg")
    time.sleep(0.1)
