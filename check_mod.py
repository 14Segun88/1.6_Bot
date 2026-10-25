import ctypes
from ctypes import wintypes
import struct
import subprocess

kernel32 = ctypes.windll.kernel32
psapi = ctypes.windll.psapi

psapi.GetModuleBaseNameA.argtypes = [wintypes.HANDLE, ctypes.c_void_p, ctypes.c_char_p, wintypes.DWORD]
psapi.GetModuleBaseNameA.restype = wintypes.DWORD

out = subprocess.check_output(["tasklist", "/FI", "IMAGENAME eq hl.exe", "/FO", "CSV"]).decode("latin-1")
pid = int([l.split(",")[1].strip('" \r\n') for l in out.splitlines() if "hl.exe" in l][0])
hProc = kernel32.OpenProcess(0x1F0FFF, False, pid)

class MODULEINFO(ctypes.Structure):
    _fields_ = [
        ("lpBaseOfDll", ctypes.c_void_p),
        ("SizeOfImage", wintypes.DWORD),
        ("EntryPoint", ctypes.c_void_p),
    ]

psapi.GetModuleInformation.argtypes = [wintypes.HANDLE, ctypes.c_void_p, ctypes.POINTER(MODULEINFO), wintypes.DWORD]
psapi.GetModuleInformation.restype = wintypes.BOOL

hMods = (ctypes.c_void_p * 1024)()
cbNeeded = wintypes.DWORD()
psapi.EnumProcessModulesEx(hProc, hMods, ctypes.sizeof(hMods), ctypes.byref(cbNeeded), 0x03)
mod_count = cbNeeded.value // ctypes.sizeof(ctypes.c_void_p)

modules = []
for i in range(mod_count):
    hMod = hMods[i]
    if not hMod: continue
    mod_name = (ctypes.c_char * 260)()
    psapi.GetModuleBaseNameA(hProc, hMod, mod_name, 260)
    info = MODULEINFO()
    psapi.GetModuleInformation(hProc, hMod, ctypes.byref(info), ctypes.sizeof(info))
    name = mod_name.value.decode("latin-1", errors="ignore")
    base = info.lpBaseOfDll or 0
    modules.append((name, base, info.SizeOfImage))

print("Modules in hl.exe:")
target_addr = 0x03DFBBBC
found_mod = None
for name, base, size in modules:
    if base <= target_addr < base + size:
        found_mod = (name, base, target_addr - base)
    if any(k in name.lower() for k in ["hw.dll", "client.dll", "hl.exe", "mp.dll"]):
        print("  %-15s Base: 0x%08X, Size: 0x%08X" % (name, base, size))

if found_mod:
    print("\n>>> Target 0x%08X is inside %s (offset +0x%X) <<<" % (target_addr, found_mod[0], found_mod[2]))
else:
    print("\nTarget 0x%08X is in dynamic memory." % target_addr)

buf = (ctypes.c_char * 128)()
kernel32.ReadProcessMemory(hProc, ctypes.c_void_p(target_addr - 16), buf, 128, None)
floats = struct.unpack_from("<32f", buf, 0)
print("\nNearby float values:")
for j in range(0, len(floats), 4):
    print("  %+03d: %.1f, %.1f, %.1f, %.1f" % (j*4-16, floats[j], floats[j+1], floats[j+2], floats[j+3]))
