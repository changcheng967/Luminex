#!/usr/bin/env python3
# Check what a release binary needs from the system, read from its headers,
# against what the release notes promise for it: the System column of the
# Binaries table (RELEASE_TEMPLATE.md).
#
#   check-deps.py BINARY ASSET
#
# ASSET is the binary's release name: luminex-linux-x86-64,
# luminex-windows-x86-64.exe or luminex-macos-arm64. A binary that needs more
# than its row says fails the build. v6.6.0's first binaries needed macOS 26,
# as clang builds for the runner's macOS without a deployment target, and on
# Windows the Visual C++ Redistributable, for the DLL runtime; their notes
# said neither. A limit here and its row in the notes change together.
import struct
import sys

# Linux: the newest symbol version each library may be asked for. The Linux
# binary is built on Ubuntu 22.04 by clang 23 with LLVM's libc++ and libc++abi
# and libgcc linked in, so it needs glibc 2.35, Ubuntu 22.04's, and none of
# the C++ runtime libraries (libstdc++ too, for a build that took it).
LINUX = {"GLIBC": (2, 35)}
LINUX_NO_LIBS = ("libstdc++.so.6", "libgcc_s.so.1", "libc++.so.1", "libc++abi.so.1", "libunwind.so.1")
# macOS 13.3, CMakeLists.txt's deployment target: the first with std::to_chars
# for floating point, which std::format needs.
MACOS_MIN = (13, 3, 0)
# Windows: no DLL runtime, so nothing to install.
WINDOWS_NO_DLLS = ("msvcp", "vcruntime", "concrt", "ucrtbase", "api-ms-win-crt-")


class Binary:
    def __init__(self, path):
        self.f = open(path, "rb")

    def read(self, off, n):
        self.f.seek(off)
        b = self.f.read(n)
        if len(b) != n:
            raise ValueError(f"short read at {off}")
        return b

    def cstr(self, off):
        out = b""
        while True:
            self.f.seek(off)
            chunk = self.f.read(256)
            if not chunk:
                raise ValueError(f"unterminated string at {off}")
            j = chunk.find(b"\0")
            if j >= 0:
                return (out + chunk[:j]).decode()
            out += chunk
            off += len(chunk)


def version(s):
    """'GLIBC_2.38' -> ('GLIBC', (2, 38)); None for names like GLIBC_PRIVATE."""
    prefix, _, num = s.rpartition("_")
    parts = num.split(".")
    if not prefix or not all(p.isdigit() for p in parts):
        return None
    return prefix, tuple(int(p) for p in parts)


def elf_needs(b):
    h = b.read(0, 64)
    if h[4] != 2 or h[5] != 1:
        raise ValueError("not a 64-bit little-endian ELF")
    phoff, = struct.unpack_from("<Q", h, 0x20)
    phentsize, phnum = struct.unpack_from("<HH", h, 0x36)
    loads, dyn = [], None
    for i in range(phnum):
        ptype, _, off, vaddr, _, filesz, _, _ = struct.unpack(
            "<IIQQQQQQ", b.read(phoff + i * phentsize, 56))
        if ptype == 1:
            loads.append((vaddr, off, filesz))
        elif ptype == 2:
            dyn = (off, filesz)
    if dyn is None:
        return [], {}

    def to_off(v):
        for vaddr, off, filesz in loads:
            if vaddr <= v < vaddr + filesz:
                return v - vaddr + off
        raise ValueError(f"address {v:#x} outside the file")

    tags, needed = {}, []
    raw = b.read(*dyn)
    for k in range(0, len(raw), 16):
        tag, val = struct.unpack_from("<qQ", raw, k)
        if tag == 0:
            break
        if tag == 1:
            needed.append(val)
        else:
            tags[tag] = val
    strtab = to_off(tags[5])
    libs = [b.cstr(strtab + n) for n in needed]
    versions = {}
    if 0x6FFFFFFE in tags:
        vn = to_off(tags[0x6FFFFFFE])
        for _ in range(tags.get(0x6FFFFFFF, 0)):
            _, cnt, file, aux, nxt = struct.unpack("<HHIII", b.read(vn, 16))
            lib = b.cstr(strtab + file)
            a = vn + aux
            for _ in range(cnt):
                _, _, _, name, anext = struct.unpack("<IHHII", b.read(a, 16))
                versions.setdefault(lib, set()).add(b.cstr(strtab + name))
                a += anext
            if nxt == 0:
                break
            vn += nxt
    return libs, versions


def check_linux(b):
    libs, versions = elf_needs(b)
    limits = LINUX
    errors, newest = [], {}
    for lib, names in versions.items():
        for name in names:
            v = version(name)
            if v is None:
                continue
            prefix, num = v
            if num > newest.get(prefix, ()):
                newest[prefix] = num
            if prefix in limits and num > limits[prefix]:
                errors.append(f"{lib} {name}, newer than {prefix}_{'.'.join(map(str, limits[prefix]))}")
    errors += [f"links {lib}, which this build carries inside" for lib in libs if lib in LINUX_NO_LIBS]
    summary = ", ".join(f"{p}_{'.'.join(map(str, n))}" for p, n in sorted(newest.items()))
    return f"libraries {', '.join(libs)}; newest versions {summary}", errors


def check_macos(b):
    magic, = struct.unpack("<I", b.read(0, 4))
    if magic != 0xFEEDFACF:
        raise ValueError(f"not a thin 64-bit Mach-O (magic {magic:#x})")
    ncmds, sizeofcmds = struct.unpack("<II", b.read(16, 8))
    lc = b.read(32, sizeofcmds)
    off, minos, dylibs = 0, None, []
    for _ in range(ncmds):
        cmd, size = struct.unpack_from("<II", lc, off)
        if cmd == 0x32:  # LC_BUILD_VERSION
            v, = struct.unpack_from("<I", lc, off + 12)
            minos = (v >> 16, (v >> 8) & 0xFF, v & 0xFF)
        elif cmd == 0x24:  # LC_VERSION_MIN_MACOSX
            v, = struct.unpack_from("<I", lc, off + 8)
            minos = (v >> 16, (v >> 8) & 0xFF, v & 0xFF)
        elif cmd in (0xC, 0x80000018):  # LC_LOAD_DYLIB, LC_LOAD_WEAK_DYLIB
            noff, = struct.unpack_from("<I", lc, off + 8)
            dylibs.append(lc[off + noff:off + size].split(b"\0")[0].decode())
        off += size
    if minos is None:
        return "no minimum macOS version in the load commands", ["no LC_BUILD_VERSION"]
    shown = ".".join(map(str, minos))
    errors = []
    if minos > MACOS_MIN:
        errors.append(f"needs macOS {shown}, newer than {'.'.join(map(str, MACOS_MIN))}")
    return f"macOS {shown} or newer; dylibs {', '.join(dylibs)}", errors


def check_windows(b):
    lfanew, = struct.unpack("<I", b.read(0x3C, 4))
    coff = b.read(lfanew, 24)
    if coff[:4] != b"PE\0\0":
        raise ValueError("not a PE file")
    nsec, = struct.unpack_from("<H", coff, 6)
    optsize, = struct.unpack_from("<H", coff, 20)
    opt = b.read(lfanew + 24, optsize)
    if struct.unpack_from("<H", opt, 0)[0] != 0x20B:
        raise ValueError("not PE32+")
    dirs = {i: struct.unpack_from("<II", opt, 112 + 8 * i) for i in (1, 13)}
    table = b.read(lfanew + 24 + optsize, 40 * nsec)
    sections = [struct.unpack_from("<8sIIII", table, 40 * i)[1:] for i in range(nsec)]

    def to_off(rva):
        for vsize, va, rawsize, rawptr in sections:
            if va <= rva < va + max(vsize, rawsize):
                return rva - va + rawptr
        raise ValueError(f"RVA {rva:#x} outside the sections")

    dlls = []
    for index, size, name_at in ((1, 20, 12), (13, 32, 4)):  # imports, delay imports
        rva = dirs[index][0]
        if not rva:
            continue
        d = to_off(rva)
        while True:
            entry = b.read(d, size)
            if entry == b"\0" * size:
                break
            name_rva, = struct.unpack_from("<I", entry, name_at)
            dlls.append(b.cstr(to_off(name_rva)))
            d += size
    errors = [f"imports {dll}, the DLL runtime (install the Visual C++ Redistributable)"
              for dll in dlls if dll.lower().startswith(WINDOWS_NO_DLLS)]
    return f"imports {', '.join(dlls)}", errors


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: check-deps.py BINARY ASSET")
    path, asset = sys.argv[1], sys.argv[2]
    b = Binary(path)
    if asset.startswith("luminex-linux-"):
        summary, errors = check_linux(b)
    elif asset.startswith("luminex-macos-"):
        summary, errors = check_macos(b)
    elif asset.startswith("luminex-windows-"):
        summary, errors = check_windows(b)
    else:
        sys.exit(f"{asset}: not a release binary's name")
    print(f"{asset}: {summary}")
    for e in errors:
        print(f"{asset}: {e}", file=sys.stderr)
    if errors:
        sys.exit(f"{asset} needs more from the system than the release notes promise")


if __name__ == "__main__":
    main()
