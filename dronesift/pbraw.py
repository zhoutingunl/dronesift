"""Schema-less protobuf decoder: returns nested tree of (field, wiretype, value)."""
import struct

def read_varint(b, i):
    r = 0; s = 0
    while True:
        c = b[i]; i += 1
        r |= (c & 0x7f) << s; s += 7
        if not c & 0x80: return r, i

def decode(b, depth=0, maxdepth=8):
    out = []
    i = 0; n = len(b)
    try:
        while i < n:
            key, i = read_varint(b, i)
            f, wt = key >> 3, key & 7
            if wt == 0:
                v, i = read_varint(b, i); out.append((f, 'v', v))
            elif wt == 1:
                v = b[i:i+8]; i += 8; out.append((f, '64', v))
            elif wt == 5:
                v = b[i:i+4]; i += 4; out.append((f, '32', v))
            elif wt == 2:
                ln, i = read_varint(b, i); v = b[i:i+ln]; i += ln
                sub = None
                if depth < maxdepth and ln > 0:
                    sub = decode(v, depth+1, maxdepth)
                out.append((f, 'len', v, sub))
            else:
                return None
    except Exception:
        return None
    return out

def fmt(v, kind):
    if kind == 'v': return str(v)
    if kind == '32':
        f = struct.unpack('<f', v)[0]; i = struct.unpack('<i', v)[0]
        return f"f32={f:.6g} i32={i}"
    if kind == '64':
        f = struct.unpack('<d', v)[0]; i = struct.unpack('<q', v)[0]
        return f"f64={f:.6g} i64={i}"

def dump(tree, indent=0, maxlist=12):
    lines = []
    if tree is None: return lines
    for item in tree:
        f, kind = item[0], item[1]
        pad = '  ' * indent
        if kind == 'len':
            v, sub = item[2], item[3]
            if sub is not None and (len(sub) > 0) and all(isinstance(x, tuple) for x in sub) and sub_valid(sub, v):
                lines.append(f"{pad}{f}: msg({len(v)}B)")
                lines += dump(sub, indent+1, maxlist)
            else:
                try:
                    s = v.decode('utf-8'); printable = all(32 <= ord(c) < 127 for c in s)
                except Exception: printable = False
                if printable and len(s) > 0:
                    lines.append(f"{pad}{f}: str={s!r}")
                else:
                    lines.append(f"{pad}{f}: bytes[{len(v)}]={v[:24].hex()}")
        else:
            lines.append(f"{pad}{f}: {fmt(item[2], kind)}")
    return lines

def sub_valid(sub, raw):
    # heuristic: decoding consumed everything and field numbers reasonable
    return all(1 <= x[0] <= 500 for x in sub)

if __name__ == '__main__':
    # Dump the raw djmd tree of a video, e.g. to re-infer field numbers for a new model.
    import argparse, sys
    from .djmd import packets
    ap = argparse.ArgumentParser(description='schema-less dump of a djmd-like metadata track')
    ap.add_argument('mp4'); ap.add_argument('--stream', type=int, default=1); ap.add_argument('-n', type=int, default=5, help='packets to dump')
    a = ap.parse_args()
    for i, (pts, b) in enumerate(packets(a.mp4, a.stream)):
        print(f'--- pts {pts:.3f} ({len(b)}B)')
        print('\n'.join(dump(decode(b))))
        if i + 1 >= a.n: break
