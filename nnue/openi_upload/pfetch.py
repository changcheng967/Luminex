#!/usr/bin/env python3
"""pfetch.py -- download one file over several connections at once, resumably.

usage: pfetch.py URL OUT SIZE CONN [CONN_FILE]
       pfetch.py --selftest

The file is cut into pieces of PFETCH_PIECE bytes (64 MiB), fetched with HTTP
range requests over CONN connections and written in place. Finished pieces are
listed in OUT.pieces, so a stopped download resumes where it was, and a partial
file left by a plain sequential download (curl -C -) counts as done up to its
last whole piece.

CONN "auto" measures while it downloads: one connection for PFETCH_STEP
seconds, then twice as many, and so on while the speed still grows by 10%. It
keeps the fastest count and writes it to CONN_FILE, so the next files start
there. The bytes fetched while measuring are kept.

Why: one connection to Hugging Face's Xet bridge gave the Blackwell box 1-11
MB/s (about 7 on average), and a t91 month is a 4-18 GB file.

--selftest serves a random file from a local server that redirects like
Hugging Face and caps the speed per connection and in total, then checks a
fresh "auto" download, a resume after a kill, a sequential partial file, and
that a server ignoring ranges is refused.
"""
import hashlib
import os
import sys
import threading
import time
import urllib.request

PIECE = int(os.environ.get("PFETCH_PIECE", 64 << 20))
STEP = float(os.environ.get("PFETCH_STEP", 12))
REPORT = float(os.environ.get("PFETCH_REPORT", 30))
TRIES = int(os.environ.get("PFETCH_TRIES", 12))
MAX_CONN = 64
GAIN = 1.10


def download(url, out, size, conn, conn_file=None):
    name = os.path.basename(out)
    n = (size + PIECE - 1) // PIECE

    def plen(i):
        return min(PIECE, size - i * PIECE)

    log_path = out + ".pieces"
    fresh = set()
    if os.path.exists(log_path):
        with open(log_path) as f:
            done = {int(x) for x in f.read().split()}
    else:
        have = os.path.getsize(out) if os.path.exists(out) else 0
        fresh = set(range(min(have, size) // PIECE))
        done = set(fresh)
    fd = os.open(out, os.O_RDWR | os.O_CREAT, 0o644)
    os.ftruncate(fd, size)
    log = open(log_path, "a")
    if fresh:
        log.write("".join(f"{i}\n" for i in sorted(fresh)))
        log.flush()
    todo = [i for i in range(n - 1, -1, -1) if i not in done]   # pop() takes the lowest piece
    tuning = conn == "auto"
    st = {"target": 1 if tuning else max(1, int(conn)), "active": 0, "recv": 0,
          "failed": False, "errors": 0, "last_error": ""}
    lock = threading.Lock()
    print(f"  pfetch: {name}: {len(done)} of {n} pieces already here, fetching {len(todo)} "
          f"over {'auto' if tuning else st['target']} connections", flush=True)

    def fetch_piece(i):
        s = i * PIECE
        e = s + plen(i) - 1
        for attempt in range(TRIES):
            pos = s
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={s}-{e}", "User-Agent": "pfetch/1"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    if r.status != 206 and not (r.status == 200 and s == 0 and e == size - 1):
                        raise OSError(f"HTTP {r.status} to a range request")
                    while pos <= e:
                        b = r.read(min(1 << 20, e + 1 - pos))
                        if not b:
                            break
                        os.pwrite(fd, b, pos)
                        pos += len(b)
                        with lock:
                            st["recv"] += len(b)
                if pos == e + 1:
                    return True
                raise OSError(f"piece {i} ended at {pos - s} of {e + 1 - s} bytes")
            except Exception as ex:   # any failure: fetch the piece again
                with lock:
                    st["errors"] += 1
                    st["last_error"] = f"{type(ex).__name__}: {ex}"
                if attempt + 1 < TRIES:
                    time.sleep(min(60, 2 ** attempt))
        return False

    def worker():
        while True:
            with lock:
                if st["failed"] or not todo or st["active"] > st["target"]:
                    st["active"] -= 1
                    return
                i = todo.pop()
            ok = fetch_piece(i)
            with lock:
                if ok:
                    done.add(i)
                    log.write(f"{i}\n")
                    log.flush()
                else:
                    st["failed"] = True

    def spawn(k):
        with lock:
            st["active"] += k
        for _ in range(k):
            threading.Thread(target=worker, daemon=True).start()

    t0 = last_t = phase_t = time.time()
    last_recv = phase_recv = 0
    results = []
    cut = False
    spawn(st["target"])
    while True:
        time.sleep(0.1)
        now = time.time()
        with lock:
            active, recv, left, target = st["active"], st["recv"], len(todo), st["target"]
        if active == 0:
            break
        if tuning and now - phase_t >= STEP:
            results.append((target, (recv - phase_recv) / (now - phase_t) / 1e6))
            grows = len(results) == 1 or results[-1][1] >= results[-2][1] * GAIN
            if not left:   # the file ran out of pieces, so connections sat idle: no verdict
                tuning, cut = False, True
            elif grows and target * 2 <= MAX_CONN:
                with lock:
                    st["target"] = target * 2      # first, or the new workers retire at once
                spawn(target)
                phase_t, phase_recv = now, recv
            else:
                best = max(results, key=lambda x: x[1])[0]
                with lock:
                    st["target"] = best
                tuning = False
                print("  pfetch: measured " + ", ".join(f"{c} conn {r:.1f} MB/s" for c, r in results)
                      + f" -> {best} connections", flush=True)
                if conn_file:
                    with open(conn_file, "w") as f:
                        f.write(str(best))
        if now - last_t >= REPORT:
            with lock:
                have = sum(plen(i) for i in done)
            print(f"  ... {name}: {have // 1000000} of {size // 1000000} MB, "
                  f"{(recv - last_recv) / (now - last_t) / 1e6:.1f} MB/s over {active} connections", flush=True)
            last_t, last_recv = now, recv
    os.close(fd)
    log.close()
    secs = max(time.time() - t0, 1e-9)
    if (tuning or cut) and results:
        print("  pfetch: measured " + ", ".join(f"{c} conn {r:.1f} MB/s" for c, r in results)
              + " (the file ended while measuring; the next file measures again)", flush=True)
    missing = n - len(done)
    if missing:
        print(f"  pfetch: FAIL {name}: {missing} of {n} pieces missing after {TRIES} tries each; "
              f"last error: {st['last_error']}", flush=True)
        return 1
    print(f"  pfetch: {name} complete: {st['recv']} bytes fetched in {secs:.0f} s "
          f"({st['recv'] / 1e6 / secs:.1f} MB/s), {st['errors']} failed requests retried", flush=True)
    return 0


def selftest():
    import http.server
    import re
    import shutil
    import subprocess
    import tempfile

    t_start = time.time()
    tmp = tempfile.mkdtemp(prefix="pfetch_selftest_")
    srv = None
    try:
        size = (64 << 20) + 12345                 # not a whole number of pieces
        data = os.urandom(size)
        want = hashlib.sha256(data).hexdigest()
        per_conn, total = 6e6, 30e6               # bytes/s: one connection, all of them together
        gate = {"next": time.time()}
        gate_lock = threading.Lock()

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path.startswith("/r/"):   # like Hugging Face: resolve, then the CDN
                    self.send_response(302)
                    self.send_header("Location", "/f/" + self.path[3:])
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                m = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers.get("Range") or "")
                if m and self.path.startswith("/f/"):
                    s, e = int(m[1]), min(int(m[2]), size - 1)
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {s}-{e}/{size}")
                else:                             # /norange/ ignores Range, as a plain server may
                    s, e = 0, size - 1
                    self.send_response(200)
                self.send_header("Content-Length", str(e - s + 1))
                self.end_headers()
                t0, sent, count = time.time(), 0, e - s + 1
                while sent < count:
                    b = data[s + sent: s + min(sent + (256 << 10), count)]
                    with gate_lock:
                        slot = max(time.time(), gate["next"])
                        gate["next"] = slot + len(b) / total
                    wait = max(slot, t0 + (sent + len(b)) / per_conn) - time.time()
                    if wait > 0:
                        time.sleep(wait)
                    try:
                        self.wfile.write(b)
                    except OSError:
                        return
                    sent += len(b)

            def log_message(self, *args):
                pass

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        env = dict(os.environ, PFETCH_PIECE=str(1 << 20), PFETCH_STEP="0.4", PFETCH_REPORT="1000")

        def start(out, conn, path="/r/src.bin", tries="12"):
            cmd = [sys.executable, os.path.abspath(__file__), base + path, out, str(size), conn, out + ".conn"]
            return subprocess.Popen(cmd, env=dict(env, PFETCH_TRIES=tries), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)

        def run(out, conn, **kw):
            p = start(out, conn, **kw)
            text = p.communicate(timeout=120)[0]
            return p.returncode, text

        def file_sha(path):
            with open(path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()

        def fetched(text):
            m = re.search(r"complete: (\d+) bytes fetched", text)
            return int(m[1]) if m else -1

        def check(ok, what, text=""):
            if not ok:
                raise AssertionError(what + ("\n" + text if text else ""))

        # 1. a fresh download that measures the connection count
        out = os.path.join(tmp, "fresh.bin")
        rc, text = run(out, "auto")
        check(rc == 0 and file_sha(out) == want, "fresh auto download differs", text)
        measured = re.search(r"measured (.*)", text)
        # 2. killed part-way, then resumed
        out = os.path.join(tmp, "resume.bin")
        p = start(out, "4")
        deadline = time.time() + 60
        while time.time() < deadline:
            if os.path.exists(out + ".pieces") and len(open(out + ".pieces").read().split()) >= 12:
                break
            time.sleep(0.05)
        p.kill()
        p.communicate()
        rc, text = run(out, "4")
        check(rc == 0 and file_sha(out) == want, "resumed download differs", text)
        check(0 <= fetched(text) <= size - (12 << 20), "the resume fetched the finished pieces again", text)
        # 3. a partial file from a plain sequential download, 10.5 pieces long
        out = os.path.join(tmp, "prefix.bin")
        with open(out, "wb") as f:
            f.write(data[: (10 << 20) + (512 << 10)])
        rc, text = run(out, "4")
        check(rc == 0 and file_sha(out) == want, "download after a sequential partial file differs", text)
        check(0 <= fetched(text) <= size - (10 << 20), "the sequential partial file was fetched again", text)
        # 4. a server that ignores Range must be refused, not written in place
        rc, text = run(os.path.join(tmp, "norange.bin"), "4", path="/norange/src.bin", tries="1")
        check(rc != 0 and "HTTP 200 to a range request" in text, "a server ignoring ranges was accepted", text)
        print(f"  pfetch self-test: ok in {time.time() - t_start:.1f} s (fresh auto download, resume after a kill, "
              f"sequential partial file, range refusal; local server capped at 6 MB/s a connection, 30 in all: "
              f"{measured[1] if measured else 'file ended while measuring'})", flush=True)
        return 0
    except AssertionError as e:
        print(f"  pfetch self-test: FAIL: {e}", flush=True)
        return 1
    finally:
        if srv is not None:
            srv.shutdown()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        sys.exit(selftest())
    if len(sys.argv) not in (5, 6):
        sys.exit(__doc__)
    sys.exit(download(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4],
                      sys.argv[5] if len(sys.argv) > 5 else None))
