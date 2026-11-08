"""
Diagnostic: dump raw floats from client.dll + 0x199E3C and hw.dll dummy table.
Run on Windows while CS 1.6 is in a match:  python diag_memory.py
"""
import ctypes, struct, time, math
from ctypes import wintypes

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

pid = wintypes.DWORD()
h = user32.FindWindowA(b"SDL_app", None) or user32.FindWindowA(b"Valve001", None)
print(f"Window handle: {h}")
user32.GetWindowThreadProcessId(h, ctypes.byref(pid))
print(f"PID: {pid.value}")
hProc = kernel32.OpenProcess(0x1010, False, pid.value)
print(f"OpenProcess handle: {hProc} (0 = failed, run as Administrator)")


class MODULEENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD), ("th32ModuleID", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD), ("GlblcntUsage", wintypes.DWORD),
        ("ProccntUsage", wintypes.DWORD), ("modBaseAddr", ctypes.c_void_p),
        ("modBaseSize", wintypes.DWORD), ("hModule", wintypes.HMODULE),
        ("szModule", ctypes.c_wchar * 256), ("szExePath", ctypes.c_wchar * 260)
    ]


snap = kernel32.CreateToolhelp32Snapshot(0x08 | 0x10, pid.value)
bases = {}
me32 = MODULEENTRY32W()
me32.dwSize = ctypes.sizeof(MODULEENTRY32W)
if kernel32.Module32FirstW(snap, ctypes.byref(me32)):
    while True:
        n = me32.szModule.lower()
        if n in ("client.dll", "hw.dll"):
            bases[n] = (int(me32.modBaseAddr), me32.szExePath)
        if not kernel32.Module32NextW(snap, ctypes.byref(me32)):
            break
kernel32.CloseHandle(snap)
for k, (b, p) in bases.items():
    print(f"{k}: base=0x{b:X} path={p}")

client_base = bases.get("client.dll", (None,))[0]
hw_base = bases.get("hw.dll", (None,))[0]

print("\n--- client.dll + 0x199E3C (raw 6 floats), 10 ticks ---")
for _ in range(10):
    buf = (ctypes.c_char * 24)()
    ok = kernel32.ReadProcessMemory(hProc, ctypes.c_void_p(client_base + 0x199E3C), buf, 24, None) if client_base else 0
    vals = struct.unpack_from("<6f", buf, 0)
    x, y, z, fx, fy, fz = vals
    norm = math.sqrt(fx * fx + fy * fy + fz * fz)
    print(f"  ok={ok} raw={['%.4f' % v for v in vals]} fwd_norm={norm:.4f}")
    time.sleep(0.1)

print("\n--- hw.dll + 0x288970 dummy table, slots 5..15 ---")
if hw_base:
    for idx in range(5, 16):
        buf = (ctypes.c_char * 12)()
        ok = kernel32.ReadProcessMemory(hProc, ctypes.c_void_p(hw_base + 0x288970 + idx * 0xE0), buf, 12, None)
        print(f"  slot {idx}: ok={ok} {['%.1f' % v for v in struct.unpack_from('<3f', buf, 0)]}")

