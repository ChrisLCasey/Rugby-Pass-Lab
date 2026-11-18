"""Minimal USDC (crate) reader: enough to pull mesh data out of a Meshy export."""
import struct
import numpy as np

# ---------- LZ4 ----------

def lz4_block(src, usize):
    out = bytearray(usize)
    ip = 0
    op = 0
    n = len(src)
    while ip < n:
        token = src[ip]; ip += 1
        lit = token >> 4
        if lit == 15:
            while True:
                b = src[ip]; ip += 1
                lit += b
                if b != 255:
                    break
        out[op:op + lit] = src[ip:ip + lit]
        ip += lit; op += lit
        if ip >= n:
            break
        off = src[ip] | (src[ip + 1] << 8); ip += 2
        ml = token & 15
        if ml == 15:
            while True:
                b = src[ip]; ip += 1
                ml += b
                if b != 255:
                    break
        ml += 4
        start = op - off
        if off >= ml:
            out[op:op + ml] = out[start:start + ml]
        else:
            pat = bytes(out[start:op])
            reps = (ml + off - 1) // off
            out[op:op + ml] = (pat * reps)[:ml]
        op += ml
    return bytes(out[:op])


def fast_decompress(buf, usize):
    """TfFastCompression: first byte = chunk count (0 => single block)."""
    nchunks = buf[0]
    if nchunks == 0:
        return lz4_block(buf[1:], usize)
    pos = 1
    parts = []
    remaining = usize
    CH = 127 * 1024 * 1024
    for _ in range(nchunks):
        csize = struct.unpack_from('<i', buf, pos)[0]; pos += 4
        this = min(remaining, CH)
        parts.append(lz4_block(buf[pos:pos + csize], this))
        pos += csize
        remaining -= this
    return b''.join(parts)


# ---------- integer coding ----------

def int_decode(raw, n, width=4):
    """Usd integer compression -> numpy int64 array (wrapping handled by caller)."""
    fmt = '<i' if width == 4 else '<q'
    common = struct.unpack_from(fmt, raw, 0)[0]
    ncode = (n + 3) // 4
    codes_b = np.frombuffer(raw, dtype=np.uint8, count=ncode, offset=width)
    codes = np.empty(ncode * 4, dtype=np.uint8)
    codes[0::4] = codes_b & 3
    codes[1::4] = (codes_b >> 2) & 3
    codes[2::4] = (codes_b >> 4) & 3
    codes[3::4] = (codes_b >> 6) & 3
    codes = codes[:n]
    sizes = np.array([0, 1, 2, width], dtype=np.int64)[codes]
    if width == 8:
        sizes = np.array([0, 1, 2, 4], dtype=np.int64)[codes]
    base = width + ncode
    offs = np.cumsum(sizes) - sizes
    data = np.frombuffer(raw, dtype=np.uint8, offset=base)
    deltas = np.zeros(n, dtype=np.int64)
    deltas[codes == 0] = common
    m = codes == 1
    if m.any():
        deltas[m] = data[offs[m]].astype(np.int8)
    m = codes == 2
    if m.any():
        o = offs[m]
        v = data[o].astype(np.uint16) | (data[o + 1].astype(np.uint16) << 8)
        deltas[m] = v.astype(np.int16)
    m = codes == 3
    if m.any():
        o = offs[m]
        v = (data[o].astype(np.uint32) | (data[o + 1].astype(np.uint32) << 8)
             | (data[o + 2].astype(np.uint32) << 16) | (data[o + 3].astype(np.uint32) << 24))
        deltas[m] = v.astype(np.int32)
    return np.cumsum(deltas)


TYPE_NAMES = {
    1: 'Bool', 2: 'UChar', 3: 'Int', 4: 'UInt', 5: 'Int64', 6: 'UInt64', 7: 'Half', 8: 'Float',
    9: 'Double', 10: 'String', 11: 'Token', 12: 'AssetPath', 13: 'Matrix2d', 14: 'Matrix3d',
    15: 'Matrix4d', 16: 'Quatd', 17: 'Quatf', 18: 'Quath', 19: 'Vec2d', 20: 'Vec2f', 21: 'Vec2h',
    22: 'Vec2i', 23: 'Vec3d', 24: 'Vec3f', 25: 'Vec3h', 26: 'Vec3i', 27: 'Vec4d', 28: 'Vec4f',
    29: 'Vec4h', 30: 'Vec4i', 31: 'Dictionary', 32: 'TokenListOp', 33: 'StringListOp',
    34: 'PathListOp', 35: 'ReferenceListOp', 36: 'IntListOp', 37: 'Int64ListOp', 38: 'UIntListOp',
    39: 'UInt64ListOp', 40: 'PathVector', 41: 'TokenVector', 42: 'Specifier', 43: 'Permission',
    44: 'Variability', 45: 'VariabilityVector', 46: 'DoubleVector', 47: 'LayerOffsetVector',
    48: 'StringVector', 49: 'ValueBlock', 50: 'Value', 51: 'UnregisteredValue',
    52: 'UnregisteredValueListOp', 53: 'PayloadListOp', 54: 'TimeSamples',
}


class Rep:
    __slots__ = ('raw', 'is_array', 'inlined', 'compressed', 'type', 'payload')

    def __init__(self, raw):
        self.raw = raw
        self.is_array = bool(raw >> 63 & 1)
        self.inlined = bool(raw >> 62 & 1)
        self.compressed = bool(raw >> 61 & 1)
        self.type = (raw >> 48) & 0xFF
        self.payload = raw & ((1 << 48) - 1)

    def __repr__(self):
        return (f"Rep({TYPE_NAMES.get(self.type, self.type)}"
                f"{'[]' if self.is_array else ''}{' inl' if self.inlined else ''}"
                f"{' cmp' if self.compressed else ''} {self.payload:#x})")


class Crate:
    def __init__(self, data):
        self.d = data
        assert data[:8] == b'PXR-USDC', 'not a crate file'
        self.version = tuple(data[8:11])
        toc_off = struct.unpack_from('<q', data, 16)[0]
        nsec = struct.unpack_from('<Q', data, toc_off)[0]
        self.sections = {}
        p = toc_off + 8
        for _ in range(nsec):
            name = data[p:p + 16].split(b'\0')[0].decode()
            start, size = struct.unpack_from('<qq', data, p + 16)
            self.sections[name] = (start, size)
            p += 32
        self._read_tokens()
        self._read_fields()
        self._read_fieldsets()
        self._read_paths()
        self._read_specs()

    # --- helpers
    def u64(self, p):
        return struct.unpack_from('<Q', self.d, p)[0]

    def _cints(self, p, n, width=4):
        csize = self.u64(p); p += 8
        wsize = width + (n + 3) // 4 + n * width
        raw = fast_decompress(self.d[p:p + csize], wsize)
        return int_decode(raw, n, width), p + csize

    # --- sections
    def _read_tokens(self):
        p = self.sections['TOKENS'][0]
        n, usize, csize = struct.unpack_from('<QQQ', self.d, p); p += 24
        raw = fast_decompress(self.d[p:p + csize], usize)
        toks = raw.split(b'\0')[:n]
        self.tokens = [t.decode('utf-8', 'replace') for t in toks]

    def _read_fields(self):
        p = self.sections['FIELDS'][0]
        n = self.u64(p); p += 8
        idx, p = self._cints(p, n)
        csize = self.u64(p); p += 8
        raw = fast_decompress(self.d[p:p + csize], n * 8)
        reps = np.frombuffer(raw, dtype=np.uint64)
        self.fields = [(self.tokens[int(i) & 0xFFFFFFFF], Rep(int(r))) for i, r in zip(idx, reps)]

    def _read_fieldsets(self):
        p = self.sections['FIELDSETS'][0]
        n = self.u64(p); p += 8
        idx, p = self._cints(p, n)
        sets, cur = [], []
        for v in idx:
            v = int(v) & 0xFFFFFFFF
            if v == 0xFFFFFFFF:
                sets.append(cur); cur = []
            else:
                cur.append(v)
        self.fieldsets_flat = idx
        self.fieldset_starts = []
        # record start offsets into flat list for random access by index
        starts, pos = {}, 0
        cur_start = 0
        for i, v in enumerate(idx):
            if i == cur_start:
                starts[i] = True
            if (int(v) & 0xFFFFFFFF) == 0xFFFFFFFF:
                cur_start = i + 1
        self._fs_flat = [int(v) & 0xFFFFFFFF for v in idx]

    def fieldset_at(self, start):
        out = []
        i = start
        while self._fs_flat[i] != 0xFFFFFFFF:
            out.append(self._fs_flat[i]); i += 1
        return out

    def _read_paths(self):
        p = self.sections['PATHS'][0]
        npaths = self.u64(p); p += 8
        nenc = self.u64(p); p += 8
        pidx, p = self._cints(p, nenc)
        etok, p = self._cints(p, nenc)
        jumps, p = self._cints(p, nenc)
        pidx = [int(v) & 0xFFFFFFFF for v in pidx]
        etok = [int(v) for v in etok]   # signed int32 semantics
        etok = [v - (1 << 32) if v >= (1 << 31) else v for v in etok]
        jumps = [int(v) for v in jumps]
        jumps = [v - (1 << 32) if v >= (1 << 31) else v for v in jumps]
        self.paths = [''] * npaths

        def build(cur, parent):
            has_child = False
            has_sib = True
            while has_child or has_sib:
                this = cur; cur += 1
                if parent == '':
                    parent = '/'
                    self.paths[pidx[this]] = '/'
                else:
                    t = etok[this]
                    is_prop = t < 0
                    tok = self.tokens[abs(t)]
                    if is_prop:
                        pth = parent + '.' + tok
                    else:
                        pth = (parent.rstrip('/') + '/' + tok)
                    self.paths[pidx[this]] = pth
                has_child = jumps[this] > 0 or jumps[this] == -1
                has_sib = jumps[this] >= 0
                if has_child:
                    if has_sib:
                        build(this + jumps[this], parent)
                    parent = self.paths[pidx[this]]
            return cur
        import sys
        sys.setrecursionlimit(100000)
        build(0, '')

    def _read_specs(self):
        p = self.sections['SPECS'][0]
        n = self.u64(p); p += 8
        pidx, p = self._cints(p, n)
        fsidx, p = self._cints(p, n)
        stype, p = self._cints(p, n)
        self.specs = []
        for a, b, c in zip(pidx, fsidx, stype):
            path = self.paths[int(a) & 0xFFFFFFFF]
            fs = self.fieldset_at(int(b) & 0xFFFFFFFF)
            self.specs.append((path, int(c), [self.fields[i] for i in fs]))

    # --- values
    def value(self, rep):
        t = rep.type
        d = self.d
        if rep.is_array:
            return self._array(rep)
        if rep.inlined:
            pl = rep.payload
            if t == 11 or t == 12 or t == 10:
                return self.tokens[pl]
            if t == 3:
                return struct.unpack('<i', struct.pack('<I', pl & 0xFFFFFFFF))[0]
            if t == 4:
                return pl & 0xFFFFFFFF
            if t == 8:
                return struct.unpack('<f', struct.pack('<I', pl & 0xFFFFFFFF))[0]
            if t == 9:
                return struct.unpack('<f', struct.pack('<I', pl & 0xFFFFFFFF))[0]
            if t == 1:
                return bool(pl & 1)
            if t in (24, 23, 28, 27, 20, 19, 26, 22):
                comp = {24: 3, 23: 3, 28: 4, 27: 4, 20: 2, 19: 2, 26: 3, 22: 2}[t]
                return [struct.unpack('b', bytes([(pl >> (8 * i)) & 0xFF]))[0] for i in range(comp)]
            if t in (15, 14, 13):
                dim = {15: 4, 14: 3, 13: 2}[t]
                m = np.zeros((dim, dim))
                for i in range(dim):
                    m[i, i] = struct.unpack('b', bytes([(pl >> (8 * i)) & 0xFF]))[0]
                return m
            return ('inline', TYPE_NAMES.get(t, t), pl)
        p = rep.payload
        if t == 15:
            return np.frombuffer(d, dtype='<f8', count=16, offset=p).reshape(4, 4).copy()
        if t == 14:
            return np.frombuffer(d, dtype='<f8', count=9, offset=p).reshape(3, 3).copy()
        if t == 9:
            return struct.unpack_from('<d', d, p)[0]
        if t in (24, 28, 20, 23, 27, 19):
            comp = {24: 3, 28: 4, 20: 2, 23: 3, 27: 4, 19: 2}[t]
            dt = '<f8' if t in (23, 27, 19) else '<f4'
            return np.frombuffer(d, dtype=dt, count=comp, offset=p).copy()
        if t == 41:  # TokenVector
            n = self.u64(p); p += 8
            idx = np.frombuffer(d, dtype='<u4', count=n, offset=p)
            return [self.tokens[i] for i in idx]
        if t == 11 or t == 12:
            return self.tokens[struct.unpack_from('<I', d, p)[0]]
        return ('offset', TYPE_NAMES.get(t, t), p)

    def _array(self, rep):
        t = rep.type
        d = self.d
        if rep.inlined:
            return np.zeros(0)
        p = rep.payload
        n = self.u64(p); p += 8
        comps = {3: 1, 4: 1, 8: 1, 9: 1, 7: 1, 20: 2, 24: 3, 28: 4, 23: 3, 19: 2, 22: 2, 26: 3, 5: 1, 6: 1}
        dts = {3: '<i4', 4: '<u4', 8: '<f4', 9: '<f8', 7: '<f2', 20: '<f4', 24: '<f4', 28: '<f4',
               23: '<f8', 19: '<f8', 22: '<i4', 26: '<i4', 5: '<i8', 6: '<u8'}
        if t == 11 or t == 12:
            idx = np.frombuffer(d, dtype='<u4', count=n, offset=p)
            return [self.tokens[i] for i in idx]
        if t not in comps:
            return ('array', TYPE_NAMES.get(t, t), n)
        c = comps[t]
        if rep.compressed and t in (3, 4):
            csize = self.u64(p); p += 8
            raw = fast_decompress(d[p:p + csize], 4 + (n + 3) // 4 + n * 4)
            v = int_decode(raw, n, 4)
            return (v.astype(np.int64) & 0xFFFFFFFF).astype(np.uint32).astype(np.int32 if t == 3 else np.uint32)
        if rep.compressed and t in (8, 9):
            code = d[p]; p += 1
            if code == ord('i'):
                csize = self.u64(p); p += 8
                raw = fast_decompress(d[p:p + csize], 4 + (n + 3) // 4 + n * 4)
                v = int_decode(raw, n, 4)
                return v.astype(np.int32).astype(np.float32 if t == 8 else np.float64)
            if code == ord('t'):
                nvals = struct.unpack_from('<I', d, p)[0]; p += 4
                lut = np.frombuffer(d, dtype=dts[t], count=nvals, offset=p); p += nvals * lut.itemsize
                csize = self.u64(p); p += 8
                raw = fast_decompress(d[p:p + csize], 4 + (n + 3) // 4 + n * 4)
                idx = int_decode(raw, n, 4)
                return lut[idx.astype(np.int64) & 0xFFFFFFFF]
            raise ValueError('unknown float compression code %r' % code)
        arr = np.frombuffer(d, dtype=dts[t], count=n * c, offset=p)
        return arr.reshape(n, c).copy() if c > 1 else arr.copy()
