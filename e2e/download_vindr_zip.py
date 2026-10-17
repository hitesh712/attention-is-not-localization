"""Selective VinDr-Mammo download through HTTP range requests on PhysioNet's project ZIP.

PhysioNet serves the individual DICOMs uncompressed (~19 MB each), whereas the project ZIP holds the same files
deflated (a few MB each) and answers byte-range requests. Only the archive's directory and the members that are
needed are fetched, which is several times less data than downloading the same files one by one.

  * credentials: the physionet.org entry of %USERPROFILE%\\_netrc (read by `requests` itself; never printed or logged)
  * PhysioNet answers HTTP basic authentication only for wget, so the client identifies itself as wget
  * every extracted file is verified against PhysioNet's SHA256SUMS.txt; verified files are skipped on re-runs and
    interrupted members resume from the bytes already on disk
  * order = priority, so that a pre-specified subset is complete at any time:
      1 mass images of the official test split      2 300 BI-RADS 1 normal images of the official test split (seed 0)
      3 calcification-only images, test split       4 mass images, training split
      5 calcification-only images, training split

Usage:  python download_vindr_zip.py [--test] [--threads 8] [--max-priority 5]
"""
import os, sys, io, csv, json, time, zlib, struct, random, hashlib, zipfile, threading, argparse, shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

ap = argparse.ArgumentParser()
ap.add_argument('--test', action='store_true', help='fetch the two smallest selected members and stop')
ap.add_argument('--threads', type=int, default=8)
ap.add_argument('--max-priority', type=int, default=5)
args = ap.parse_args()

ZIP = 'https://physionet.org/content/vindr-mammo/get-zip/1.0.0/'
FILES = 'https://physionet.org/files/vindr-mammo/1.0.0/'
DEST = 'D:/vindr_dicom'; LOCAL_CSV = 'D:/pipeline2_mammogram/pipeline2_mammogram/data/vindrmammo'
N_NORMALS, SEED = 300, 0
os.makedirs(DEST, exist_ok=True)
T0 = time.time(); log = lambda *a: print(time.strftime('%Y-%m-%dT%H:%M:%S'), *a, flush=True)
tls = threading.local()


def session():
    if not hasattr(tls, 's'):
        tls.s = requests.Session(); tls.s.headers['User-Agent'] = 'Wget/1.21.4'
    return tls.s


def get_range(start, end, tries=8):
    """bytes [start, end] of the ZIP (inclusive)."""
    buf = b''
    for t in range(tries):
        try:
            with session().get(ZIP, headers={'Range': f'bytes={start + len(buf)}-{end}'}, stream=True, timeout=(30, 120)) as r:
                if r.status_code in (401, 403): raise PermissionError(f'HTTP {r.status_code}')
                if r.status_code != 206: raise IOError(f'HTTP {r.status_code} (range not honoured)')
                for c in r.iter_content(1 << 15): buf += c
            if len(buf) == end - start + 1: return buf
        except PermissionError: raise
        except Exception as e:
            if t == tries - 1: raise
            time.sleep(5 * (t + 1))
    raise IOError('short range')


def sha256_file(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(buf), b''): h.update(b)
    return h.hexdigest()


class NeedRange(Exception):
    pass


class Sparse(io.RawIOBase):
    """Read-only view of the remote ZIP that holds only the byte ranges fetched so far (enough for zipfile to parse the directory)."""
    def __init__(self, size): self.size, self.pos, self.seg = size, 0, []
    def add(self, start, data): self.seg.append((start, data))
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off; return self.pos
    def read(self, n=-1):
        n = self.size - self.pos if n is None or n < 0 else min(n, self.size - self.pos)
        for s, d in self.seg:
            if s <= self.pos and self.pos + n <= s + len(d):
                out = d[self.pos - s:self.pos - s + n]; self.pos += n; return out
        raise NeedRange(self.pos, n)


def zip_directory():
    """{member name: [header offset, compressed size, size, method]} of the remote ZIP, cached on disk."""
    cache = DEST + '/zip_directory.json'
    if os.path.exists(cache): return json.load(open(cache))
    with session().get(ZIP, headers={'Range': 'bytes=0-0'}, stream=True, timeout=(30, 120)) as r:
        if r.status_code in (401, 403): sys.exit(f'Access denied (HTTP {r.status_code}).')
        if r.status_code != 206: sys.exit(f'The ZIP does not answer range requests (HTTP {r.status_code}).')
        size = int(r.headers['Content-Range'].split('/')[1])
    log(f'ZIP size {size / 2**30:.1f} GiB; reading its directory ...')
    sp = Sparse(size); tail = 1 << 17; sp.add(size - tail, get_range(size - tail, size - 1))
    while True:
        try:
            zf = zipfile.ZipFile(sp); break
        except NeedRange as e:
            start, n = e.args; step = -(-n // args.threads)
            parts = [(start + k * step, min(start + (k + 1) * step, start + n) - 1) for k in range(args.threads) if start + k * step < start + n]
            log(f'  directory: {n / 2**20:.1f} MB at offset {start} ({len(parts)} parallel ranges)')
            with ThreadPoolExecutor(args.threads) as ex: data = list(ex.map(lambda p: get_range(*p), parts))
            sp.add(start, b''.join(data))
    D = {i.filename: [i.header_offset, i.compress_size, i.file_size, i.compress_type] for i in zf.infolist() if not i.is_dir()}
    json.dump(D, open(cache, 'w')); log(f'  {len(D)} members')
    return D


def fetch_member(name, rel, entry, expected):
    """Download one deflated member (resumable), inflate it, verify the SHA-256, write DEST/rel."""
    off, csize, usize, method = entry
    out = os.path.join(DEST, rel.replace('/', os.sep)); zpart = out + '.zpart'
    if os.path.exists(out):
        if expected is None or sha256_file(out) == expected: return rel, 0, 'skip'
        os.remove(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    guess = 30 + len(name.encode('utf-8')) + 1024 + csize              # local header + data (the header's extra field is short)
    err = 'incomplete after 10 attempts'
    for t in range(10):
        try:
            have = os.path.getsize(zpart) if os.path.exists(zpart) else 0; total = guess
            if have >= 30:
                with open(zpart, 'rb') as f: hd = f.read(30)
                if hd[:4] != b'PK\x03\x04': os.remove(zpart); have = 0
                else: total = 30 + sum(struct.unpack('<HH', hd[26:30])) + csize
            if have < total:
                with session().get(ZIP, headers={'Range': f'bytes={off + have}-{off + total - 1}'}, stream=True, timeout=(30, 120)) as r:
                    if r.status_code in (401, 403): raise PermissionError(f'HTTP {r.status_code}')
                    if r.status_code != 206: raise IOError(f'HTTP {r.status_code}')
                    with open(zpart, 'ab') as f:
                        for c in r.iter_content(1 << 15):
                            f.write(c); count(len(c))
                continue                                               # re-check the size (and the true header length)
            with open(zpart, 'rb') as f:
                hd = f.read(30); n, m = struct.unpack('<HH', hd[26:30]); f.seek(30 + n + m)
                d = zlib.decompressobj(-15) if method == zipfile.ZIP_DEFLATED else None
                h = hashlib.sha256(); left = csize
                with open(out + '.part', 'wb') as g:
                    while left:
                        c = f.read(min(left, 1 << 20)); left -= len(c)
                        if not c: raise IOError('truncated member')
                        u = d.decompress(c) if d else c; h.update(u); g.write(u)
                    if d: u = d.flush(); h.update(u); g.write(u)
            if expected is not None and h.hexdigest() != expected:
                os.remove(zpart); os.remove(out + '.part'); raise IOError('checksum mismatch')
            os.replace(out + '.part', out); os.remove(zpart)
            return rel, csize, 'ok'
        except PermissionError: raise
        except Exception as e:
            err = f'{type(e).__name__}: {str(e)[:100]}'; time.sleep(min(60, 5 * (t + 1)))
    return rel, 0, 'FAILED: ' + err


NB = [0]; LOCK = threading.Lock()
def count(n):
    with LOCK: NB[0] += n


def main():
    if not requests.utils.get_netrc_auth(ZIP): sys.exit('No physionet.org entry found in the netrc file.')
    open(DEST + '/download.pid', 'w').write(str(os.getpid()))
    # ---- checksums (small file served directly) and the archive directory
    if not os.path.exists(DEST + '/SHA256SUMS.txt'):
        r = session().get(FILES + 'SHA256SUMS.txt', timeout=(30, 300)); r.raise_for_status(); open(DEST + '/SHA256SUMS.txt', 'wb').write(r.content)
    sums = {}
    for line in open(DEST + '/SHA256SUMS.txt', encoding='utf-8'):
        p = line.strip().split(maxsplit=1)
        if len(p) == 2: sums[p[1].lstrip('*').strip()] = p[0]
    Z = zip_directory()
    k0 = next(k for k in Z if k.endswith('.dicom')); pre = k0[:k0.index('images/')]          # top-level folder inside the ZIP
    Zrel = {k[len(pre):]: k for k in Z if k.startswith(pre)}

    # ---- annotation tables: reuse the local copies when they are byte-identical to PhysioNet's, else take them from the ZIP
    for f in ('finding_annotations.csv', 'breast-level_annotations.csv'):
        if os.path.exists(f'{DEST}/{f}') and sha256_file(f'{DEST}/{f}') == sums.get(f): continue
        if os.path.exists(f'{LOCAL_CSV}/{f}') and sha256_file(f'{LOCAL_CSV}/{f}') == sums.get(f):
            shutil.copy2(f'{LOCAL_CSV}/{f}', f'{DEST}/{f}'); log(f'{f}: local copy is identical to PhysioNet (SHA-256)'); continue
        if f in Zrel: log(f'{f}: from the ZIP,', fetch_member(Zrel[f], f, Z[Zrel[f]], sums.get(f))[2])
        else:
            r = session().get(FILES + f, timeout=(30, 900)); r.raise_for_status(); open(f'{DEST}/{f}', 'wb').write(r.content)
            log(f'{f}: downloaded directly,', 'verified' if sha256_file(f'{DEST}/{f}') == sums.get(f) else 'CHECKSUM MISMATCH')

    # ---- selection, in priority order
    find = list(csv.DictReader(open(DEST + '/finding_annotations.csv', encoding='utf-8')))
    bl = list(csv.DictReader(open(DEST + '/breast-level_annotations.csv', encoding='utf-8')))
    key = lambda r: (r['study_id'], r['image_id']); split = {key(r): r['split'] for r in bl}
    has = lambda r, c: c in r['finding_categories'] and r['xmin'] not in ('', None)
    mass = {key(r) for r in find if has(r, 'Mass')}; calc = {key(r) for r in find if has(r, 'Suspicious Calcification')} - mass
    lesion = {key(r) for r in find if 'No Finding' not in r['finding_categories']}
    norm = sorted({key(r) for r in bl if r['split'] == 'test' and r['breast_birads'] == 'BI-RADS 1'} - lesion)
    random.Random(SEED).shuffle(norm); norm = norm[:N_NORMALS]
    groups = [(1, 'mass_test', sorted(k for k in mass if split.get(k) == 'test')), (2, 'normal_test_birads1', sorted(norm)),
              (3, 'calc_test', sorted(k for k in calc if split.get(k) == 'test')),
              (4, 'mass_training', sorted(k for k in mass if split.get(k) != 'test')),
              (5, 'calc_training', sorted(k for k in calc if split.get(k) != 'test'))]
    todo = []
    with open(DEST + '/selection.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['priority', 'set', 'study_id', 'image_id'])
        for pr, tag, ks in groups:
            rels = [f'images/{s}/{i}.dicom' for s, i in ks]; miss = [r for r in rels if r not in Zrel]
            gb = sum(Z[Zrel[r]][1] for r in rels if r in Zrel) / 2**30
            log(f'  priority {pr} {tag:20s} {len(rels):5d} images | {gb:5.2f} GB to download' + (f' | {len(miss)} NOT IN ZIP' if miss else ''))
            for (s, i), r in zip(ks, rels):
                w.writerow([pr, tag, s, i])
                if r in Zrel and pr <= args.max_priority: todo.append((pr, r))
    if args.test: todo = sorted(todo, key=lambda x: Z[Zrel[x[1]]][1])[:2]
    need = sum(Z[Zrel[r]][1] for _, r in todo); disk = sum(Z[Zrel[r]][2] for _, r in todo)
    log(f'{len(todo)} files | {need / 2**30:.2f} GB to download | {disk / 2**30:.1f} GB on disk | {args.threads} connections')

    done = 0; ok_bytes = 0; fails = []; last = time.time(); cur = 0
    with ThreadPoolExecutor(args.threads) as ex:
        futs = {ex.submit(fetch_member, Zrel[r], r, Z[Zrel[r]], sums.get(r)): pr for pr, r in todo}   # submitted in priority order
        for fu in as_completed(futs):
            rel, size, status = fu.result(); done += 1; ok_bytes += Z[Zrel[rel]][1]
            if status.startswith('FAILED'): fails.append((rel, status))
            if time.time() - last > 120 or done == len(todo) or args.test:
                last = time.time(); el = last - T0; rate = NB[0] / max(el, 1)
                eta = (need - ok_bytes) / max(rate, 1) / 3600
                log(f'{done}/{len(todo)} files | {NB[0] / 2**20:8.1f} MB this run | {rate / 1024:6.1f} kB/s | ETA {eta:5.1f} h | failures {len(fails)}')
    for pr, tag, ks in groups:
        n = sum(os.path.exists(f'{DEST}/images/{s}/{i}.dicom') for s, i in ks); log(f'  complete: priority {pr} {tag:20s} {n}/{len(ks)}')
    log('ALL DONE' if not fails else f'{len(fails)} files failed - re-run to retry: ' + '; '.join(f'{r} {s}' for r, s in fails[:5]))


if __name__ == '__main__':
    main()
