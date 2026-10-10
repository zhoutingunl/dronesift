"""djmd packet decoding against hand-built protobuf (Air 3S field layout, see djmd.py header)."""
import math
import struct
import pytest
from dronesift import djmd

def varint(v):
    v &= (1 << 64) - 1                              # negative ints travel as 10-byte two's complement
    out = bytearray()
    while True:
        b = v & 0x7f; v >>= 7
        out.append(b | (0x80 if v else 0))
        if not v: return bytes(out)

def tag(f, wt): return varint(f << 3 | wt)
def sub(f, payload): return tag(f, 2) + varint(len(payload)) + payload
def vint(f, x): return tag(f, 0) + varint(int(x))
def f32(f, x): return tag(f, 5) + struct.pack('<f', x)
def f64(f, x): return tag(f, 1) + struct.pack('<d', x)

def build_packet(frame=7, ts_us=123456, iso=200.0, exp=(1, 120), ev=-0.7,
                 vel=(3.0, -4.0, -0.5), ac=(10.0, -5.0, 200.0), lat_rad=0.65, lon_rad=-2.13,
                 alt_mm=60000, rel_mm=25000.0, gim=(20.0, 0.0, 199.0), q=(1.0, 0.0, 0.0, 0.0)):
    hdr = vint(1, frame) + vint(2, ts_us)
    packed = b''.join(varint(x) for x in exp)
    cam = sub(9, f32(1, iso)) + sub(10, sub(1, packed)) + sub(31, f32(1, ev))
    fc = (sub(2, f32(1, vel[0]) + f32(2, vel[1]) + f32(3, vel[2]))
          + sub(3, vint(1, round(ac[0] * 10)) + vint(2, round(ac[1] * 10)) + vint(3, round(ac[2] * 10)))
          + sub(4, sub(1, f64(2, lat_rad) + f64(3, lon_rad)) + vint(2, alt_mm))
          + sub(5, f32(1, rel_mm)))
    gb = (sub(3, vint(1, round(gim[0] * 10)) + vint(2, round(gim[1] * 10)) + vint(3, round(gim[2] * 10)))
          + sub(4, f32(1, q[0]) + f32(2, q[1]) + f32(3, q[2]) + f32(4, q[3])))
    return sub(3, sub(1, hdr) + sub(2, cam) + sub(3, fc) + sub(4, gb))

def test_decode_full_packet():
    r = djmd._decode(build_packet(), pts=1.5)
    assert r['pts'] == 1.5 and r['frame'] == 7 and r['ts_us'] == 123456
    assert r['iso'] == 200.0 and r['exposure'] == 1 / 120 and r['ev'] == pytest.approx(-0.7, abs=1e-6)
    assert (r['vx'], r['vy'], r['vz']) == (3.0, -4.0, -0.5)
    assert (r['ac_pitch'], r['ac_roll'], r['ac_yaw']) == (10.0, -5.0, 200.0)
    assert r['lat'] == pytest.approx(math.degrees(0.65), abs=1e-9)
    assert r['lon'] == pytest.approx(math.degrees(-2.13), abs=1e-9)
    assert r['abs_alt'] == 60.0 and r['rel_alt'] == 25.0
    assert (r['g_pitch'], r['g_roll'], r['g_yaw']) == (20.0, 0.0, 199.0)
    assert r['qw'] == 1.0

def test_decode_negative_deci_degrees():
    r = djmd._decode(build_packet(gim=(-20.0, 0.0, 199.0)), pts=0.0)
    assert r['g_pitch'] == -20.0

def test_decode_skips_non_frame_meta_packets():
    assert djmd._decode(sub(9, b'junk'), 0.0) is None
